#![deny(unsafe_code)]

//! Runs Leone kernels through Apple's Metal command queue.
//!
//! The backend owns shared-storage `MTLBuffer` allocations and dispatches every
//! numerical operation through the native Metal shader. Linux builds expose the
//! same type and return an unsupported-device error.

mod backend;
#[allow(unsafe_code)]
mod ffi;

pub use backend::{
    MetalAttentionBatchStats, MetalAttentionPath, MetalBackend, MetalBuffer, MetalDeviceInfo,
    MetalDeviceMetadata,
};

pub(crate) const SHADER_SOURCE: &[u8] = include_bytes!("../metal/leone.metal");

/// Identifies the shader source embedded in the Metal backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MetalShaderIdentity {
    pub name: &'static str,
    pub sha256: &'static str,
    pub bytes: usize,
}

/// Returns the embedded shader identity without opening a GPU context.
pub fn shader_identity() -> MetalShaderIdentity {
    MetalShaderIdentity {
        name: "leone.metal",
        sha256: env!("LEONE_METAL_SHADER_SHA256"),
        bytes: SHADER_SOURCE.len(),
    }
}

/// Captures host memory fields from one native macOS bridge call.
///
/// The task, sysctl, and host-statistics reads are not atomic.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MetalHostMemorySnapshot {
    /// Resident bytes reported for the current process.
    pub process_resident_bytes: Option<u64>,
    /// Virtual address space bytes for the current process.
    ///
    /// This field is address-space telemetry. It does not count as available
    /// memory or reduce the system-memory estimate.
    pub process_virtual_bytes: Option<u64>,
    /// Physical memory bytes reported by `hw.memsize`.
    pub system_total_bytes: Option<u64>,
    /// Available-memory estimate from one `host_statistics64` snapshot.
    ///
    /// This uses a `free_count + external_page_count - speculative_count`
    /// estimate. XNU defines external pages as file-backed non-swap pages and
    /// includes speculative pages in `free_count`. It is not a promise of
    /// allocatable memory.
    pub system_available_bytes: Option<u64>,
    /// Total bytes minus the available-pages estimate when both are consistent.
    pub system_used_bytes: Option<u64>,
}

/// Returns one host memory observation assembled by a macOS bridge call.
///
/// The task, sysctl, and host-statistics reads are not atomic.
///
/// Non-macOS builds return the Metal bridge's unsupported-platform error.
pub fn host_memory_snapshot() -> Result<MetalHostMemorySnapshot, String> {
    Ok(host_snapshot_from_raw(ffi::host_memory_snapshot()?))
}

fn host_snapshot_from_raw(raw: ffi::HostMemoryInfo) -> MetalHostMemorySnapshot {
    let process_resident_bytes = raw
        .has(ffi::HOST_PROCESS_RESIDENT)
        .then_some(raw.process_resident_bytes);
    let process_virtual_bytes = raw
        .has(ffi::HOST_PROCESS_VIRTUAL)
        .then_some(raw.process_virtual_bytes);
    let system_total_bytes = raw
        .has(ffi::HOST_SYSTEM_TOTAL)
        .then_some(raw.system_total_bytes);
    let estimated_available_bytes = available_page_bytes(raw);
    let system_available_bytes = match (system_total_bytes, estimated_available_bytes) {
        (Some(total), Some(available)) if available <= total => Some(available),
        (Some(_), Some(_)) => None,
        (_, available) => available,
    };
    let system_used_bytes = system_total_bytes
        .zip(system_available_bytes)
        .and_then(|(total, available)| total.checked_sub(available));
    MetalHostMemorySnapshot {
        process_resident_bytes,
        process_virtual_bytes,
        system_total_bytes,
        system_available_bytes,
        system_used_bytes,
    }
}

fn available_page_bytes(raw: ffi::HostMemoryInfo) -> Option<u64> {
    let complete_page_observation = [
        ffi::HOST_SYSTEM_PAGE_SIZE,
        ffi::HOST_SYSTEM_FREE_PAGES,
        ffi::HOST_SYSTEM_FILE_BACKED_PAGES,
        ffi::HOST_SYSTEM_SPECULATIVE_PAGES,
    ]
    .into_iter()
    .all(|field| raw.has(field));
    if !complete_page_observation || raw.system_page_size == 0 {
        return None;
    }
    raw.system_file_backed_pages
        .checked_sub(raw.system_speculative_pages)
        .and_then(|file_backed_pages| raw.system_free_pages.checked_add(file_backed_pages))
        .and_then(|pages| pages.checked_mul(raw.system_page_size))
}

#[cfg(test)]
mod tests {
    use sha2::{Digest, Sha256};

    #[test]
    fn build_identity_matches_embedded_shader() {
        let identity = super::shader_identity();
        assert_eq!(
            identity.sha256,
            format!("{:x}", Sha256::digest(super::SHADER_SOURCE))
        );
        assert_eq!(identity.bytes, super::SHADER_SOURCE.len());
        assert!(identity.bytes > 0);
    }

    #[test]
    fn host_snapshot_converts_available_memory_estimate() {
        let raw = super::ffi::HostMemoryInfo {
            process_resident_bytes: 11,
            process_virtual_bytes: 22,
            system_total_bytes: 4096 * 100,
            system_page_size: 4096,
            system_free_pages: 10,
            system_file_backed_pages: 20,
            system_speculative_pages: 5,
            available: super::ffi::HOST_PROCESS_RESIDENT
                | super::ffi::HOST_PROCESS_VIRTUAL
                | super::ffi::HOST_SYSTEM_TOTAL
                | super::ffi::HOST_SYSTEM_PAGE_SIZE
                | super::ffi::HOST_SYSTEM_FREE_PAGES
                | super::ffi::HOST_SYSTEM_FILE_BACKED_PAGES
                | super::ffi::HOST_SYSTEM_SPECULATIVE_PAGES,
        };
        let snapshot = super::host_snapshot_from_raw(raw);
        assert_eq!(snapshot.process_resident_bytes, Some(11));
        assert_eq!(snapshot.process_virtual_bytes, Some(22));
        assert_eq!(snapshot.system_total_bytes, Some(409_600));
        assert_eq!(snapshot.system_available_bytes, Some(102_400));
        assert_eq!(snapshot.system_used_bytes, Some(307_200));
    }

    #[test]
    fn host_snapshot_rejects_overflow_and_inconsistent_totals() {
        let overflow = super::ffi::HostMemoryInfo {
            system_page_size: 2,
            system_free_pages: u64::MAX,
            system_file_backed_pages: 1,
            system_speculative_pages: 0,
            available: super::ffi::HOST_SYSTEM_PAGE_SIZE
                | super::ffi::HOST_SYSTEM_FREE_PAGES
                | super::ffi::HOST_SYSTEM_FILE_BACKED_PAGES
                | super::ffi::HOST_SYSTEM_SPECULATIVE_PAGES,
            ..Default::default()
        };
        let overflow_snapshot = super::host_snapshot_from_raw(overflow);
        assert_eq!(overflow_snapshot.system_available_bytes, None);
        assert_eq!(overflow_snapshot.system_used_bytes, None);

        let inconsistent = super::ffi::HostMemoryInfo {
            system_total_bytes: 10,
            system_page_size: 2,
            system_free_pages: 6,
            system_file_backed_pages: 1,
            system_speculative_pages: 1,
            available: super::ffi::HOST_SYSTEM_TOTAL
                | super::ffi::HOST_SYSTEM_PAGE_SIZE
                | super::ffi::HOST_SYSTEM_FREE_PAGES
                | super::ffi::HOST_SYSTEM_FILE_BACKED_PAGES
                | super::ffi::HOST_SYSTEM_SPECULATIVE_PAGES,
            ..Default::default()
        };
        let inconsistent_snapshot = super::host_snapshot_from_raw(inconsistent);
        assert_eq!(inconsistent_snapshot.system_total_bytes, Some(10));
        assert_eq!(inconsistent_snapshot.system_available_bytes, None);
        assert_eq!(inconsistent_snapshot.system_used_bytes, None);

        let underflow = super::ffi::HostMemoryInfo {
            system_page_size: 2,
            system_free_pages: 10,
            system_file_backed_pages: 1,
            system_speculative_pages: 2,
            available: super::ffi::HOST_SYSTEM_PAGE_SIZE
                | super::ffi::HOST_SYSTEM_FREE_PAGES
                | super::ffi::HOST_SYSTEM_FILE_BACKED_PAGES
                | super::ffi::HOST_SYSTEM_SPECULATIVE_PAGES,
            ..Default::default()
        };
        let underflow_snapshot = super::host_snapshot_from_raw(underflow);
        assert_eq!(underflow_snapshot.system_available_bytes, None);
    }

    #[test]
    fn host_snapshot_keeps_unavailable_fields_explicit() {
        let raw = super::ffi::HostMemoryInfo {
            process_resident_bytes: 7,
            available: super::ffi::HOST_PROCESS_RESIDENT,
            ..Default::default()
        };
        let snapshot = super::host_snapshot_from_raw(raw);
        assert_eq!(snapshot.process_resident_bytes, Some(7));
        assert_eq!(snapshot.process_virtual_bytes, None);
        assert_eq!(snapshot.system_total_bytes, None);
        assert_eq!(snapshot.system_available_bytes, None);
        assert_eq!(snapshot.system_used_bytes, None);
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn host_snapshot_stub_reports_unsupported_platform() {
        assert_eq!(
            super::host_memory_snapshot().unwrap_err(),
            "Metal backend requires macOS"
        );
    }
}
