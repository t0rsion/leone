use leone::BackendError;

#[cfg(all(feature = "cuda", feature = "metal"))]
compile_error!("select exactly one Runtime driver backend");

#[cfg(not(any(feature = "cuda", feature = "metal")))]
compile_error!("select the Runtime driver cuda or metal feature");

#[cfg(feature = "cuda")]
use leone_cuda::{CudaAttentionBatchPath, CudaBackend};
#[cfg(feature = "metal")]
use leone_metal::{MetalAttentionPath, MetalBackend};

/// Selects one explicit batched attention path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchPath {
    /// Runs one attention row at a time.
    PerRow,
    /// Uses one fixed tile schedule for each row.
    FixedTilePerRow,
    /// Shares prefix reads with a schedule-dependent reduction order.
    SharedReadUnconstrained,
    /// Shares prefix reads with a fixed reduction order.
    SharedReadFixedReduction,
}

/// Counts the batched attention work recorded by one backend.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BatchStats {
    /// Number of batch attention calls.
    pub dispatches: usize,
    /// Number of submitted attention rows.
    pub rows: usize,
    /// Number of dispatched row groups.
    pub groups: usize,
    /// Number of groups containing more than one row.
    pub multi_row_groups: usize,
}

#[cfg(feature = "cuda")]
pub type DriverBackend = CudaBackend;

#[cfg(feature = "metal")]
pub type DriverBackend = MetalBackend;

pub type DriverRuntime = leone::Runtime<DriverBackend>;

pub fn device_metadata(backend: &DriverBackend) -> Result<serde_json::Value, BackendError> {
    #[cfg(feature = "cuda")]
    {
        let metadata = backend.device_info()?;
        Ok(serde_json::json!({
            "name": metadata.name,
            "compute_major": metadata.compute_major,
            "compute_minor": metadata.compute_minor,
            "driver_version": metadata.driver_version,
            "runtime_version": metadata.runtime_version,
            "total_global_mem": metadata.total_global_mem,
        }))
    }
    #[cfg(feature = "metal")]
    {
        let metadata = backend.device_metadata()?;
        Ok(serde_json::json!({
            "name": metadata.device_name,
            "architecture": metadata.architecture_name,
            "os_version": metadata.os_version,
            "compiler_version": metadata.compiler_version,
            "shader_source_hash": metadata.shader_source_hash,
            "fast_math_enabled": metadata.fast_math_enabled,
        }))
    }
}

/// Parses one manifest path name.
pub fn parse_path(value: &str) -> Result<BatchPath, String> {
    match value {
        "per_row" => Ok(BatchPath::PerRow),
        "fixed_tile_per_row" => Ok(BatchPath::FixedTilePerRow),
        "shared_read_unconstrained" => Ok(BatchPath::SharedReadUnconstrained),
        "shared_read_fixed_reduction" => Ok(BatchPath::SharedReadFixedReduction),
        _ => Err(format!("unknown batch path {value}")),
    }
}

/// Returns the stable manifest name for one path.
pub const fn path_name(path: BatchPath) -> &'static str {
    match path {
        BatchPath::PerRow => "per_row",
        BatchPath::FixedTilePerRow => "fixed_tile_per_row",
        BatchPath::SharedReadUnconstrained => "shared_read_unconstrained",
        BatchPath::SharedReadFixedReduction => "shared_read_fixed_reduction",
    }
}

/// Constructs the selected backend with its explicit attention path.
pub fn load_backend(path: BatchPath) -> Result<DriverBackend, BackendError> {
    #[cfg(feature = "cuda")]
    {
        CudaBackend::with_attention_batch_path(0, cuda_path(path))
    }
    #[cfg(feature = "metal")]
    {
        let mut backend = MetalBackend::new()?;
        backend.set_attention_path(metal_path(path));
        backend.set_research_batch_size();
        Ok(backend)
    }
}

/// Reads the backend-native counters through the portable receipt shape.
pub const fn batch_stats(backend: &DriverBackend) -> BatchStats {
    #[cfg(feature = "cuda")]
    {
        let stats = backend.attention_batch_stats();
        BatchStats {
            dispatches: stats.dispatches,
            rows: stats.rows,
            groups: stats.groups,
            multi_row_groups: stats.multi_row_groups,
        }
    }
    #[cfg(feature = "metal")]
    {
        let stats = backend.attention_batch_stats();
        BatchStats {
            dispatches: stats.dispatches,
            rows: stats.rows,
            groups: stats.groups,
            multi_row_groups: stats.multi_row_groups,
        }
    }
}

#[cfg(feature = "cuda")]
const fn cuda_path(path: BatchPath) -> CudaAttentionBatchPath {
    match path {
        BatchPath::PerRow => CudaAttentionBatchPath::PerRow,
        BatchPath::FixedTilePerRow => CudaAttentionBatchPath::FixedTilePerRow,
        BatchPath::SharedReadUnconstrained => CudaAttentionBatchPath::SharedReadUnconstrained,
        BatchPath::SharedReadFixedReduction => CudaAttentionBatchPath::SharedReadFixedReduction,
    }
}

#[cfg(feature = "metal")]
const fn metal_path(path: BatchPath) -> MetalAttentionPath {
    match path {
        BatchPath::PerRow => MetalAttentionPath::PerRow,
        BatchPath::FixedTilePerRow => MetalAttentionPath::FixedTilePerRow,
        BatchPath::SharedReadUnconstrained => MetalAttentionPath::SharedReadUnconstrained,
        BatchPath::SharedReadFixedReduction => MetalAttentionPath::SharedReadFixedReduction,
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_path, path_name, BatchPath};

    #[test]
    fn path_names_round_trip() {
        for path in [
            BatchPath::PerRow,
            BatchPath::FixedTilePerRow,
            BatchPath::SharedReadUnconstrained,
            BatchPath::SharedReadFixedReduction,
        ] {
            assert_eq!(parse_path(path_name(path)), Ok(path));
        }
    }

    #[test]
    fn unknown_path_is_typed_as_a_parse_error() {
        assert_eq!(
            parse_path("unknown"),
            Err("unknown batch path unknown".to_owned())
        );
    }
}
