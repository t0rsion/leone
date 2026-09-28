use super::{
    MemoryAccounting, MemoryAllocation, MemoryBudget, MemoryClass, MemoryError, MemoryReservation,
    MemoryTracker,
};
use leone_gguf::{AllocationBudget, AllocationBudgetError, AllocationGuard, AllocationReservation};

const RESERVATION_METADATA_BYTES: u64 = 128;

/// Tracks checked host bytes used while importing model data.
#[derive(Debug, Clone, Default)]
pub struct HostStaging {
    tracker: MemoryTracker,
}

impl HostStaging {
    /// Creates a host staging ledger with the supplied byte budget.
    pub fn new(budget: MemoryBudget) -> Self {
        Self {
            tracker: MemoryTracker::new(budget),
        }
    }

    /// Reuses an existing checked tracker for host snapshots and staging.
    pub fn from_tracker(tracker: MemoryTracker) -> Self {
        Self { tracker }
    }

    /// Returns a host staging ledger without a byte limit.
    pub fn unlimited() -> Self {
        Self::default()
    }

    /// Changes the budget when all staged bytes fit.
    pub fn set_budget(&self, budget: MemoryBudget) -> Result<(), MemoryError> {
        self.tracker.set_budget(budget)
    }

    /// Reserves host bytes before constructing a staging allocation.
    pub fn reserve(&self, bytes: u64) -> Result<MemoryReservation, MemoryError> {
        self.tracker.reserve(MemoryClass::ContractBuffer, bytes)
    }

    /// Accounts host bytes for the lifetime of the returned allocation token.
    pub fn allocate(&self, bytes: u64) -> Result<MemoryAllocation, MemoryError> {
        self.tracker.allocate(MemoryClass::ContractBuffer, bytes)
    }

    /// Returns the checked host staging accounting snapshot.
    pub fn snapshot(&self) -> MemoryAccounting {
        self.tracker.snapshot()
    }
}

#[derive(Debug)]
struct StagingReservation {
    reservation: MemoryReservation,
    what: &'static str,
    bytes: u64,
}

impl AllocationBudget for HostStaging {
    fn guard_metadata_bytes(&self) -> u64 {
        u64::try_from(
            std::mem::size_of::<MemoryAllocation>()
                + std::mem::size_of::<Box<dyn AllocationGuard>>(),
        )
        .expect("guard metadata size fits in u64")
    }

    fn reservation_metadata_bytes(&self) -> u64 {
        RESERVATION_METADATA_BYTES
    }

    fn reserve(
        &self,
        bytes: u64,
        what: &'static str,
    ) -> Result<Box<dyn AllocationReservation>, AllocationBudgetError> {
        HostStaging::reserve(self, bytes)
            .map(|reservation| {
                Box::new(StagingReservation {
                    reservation,
                    what,
                    bytes,
                }) as Box<dyn AllocationReservation>
            })
            .map_err(|error| AllocationBudgetError::new(what, bytes, error))
    }
}

impl AllocationReservation for StagingReservation {
    fn commit(
        self: Box<Self>,
    ) -> Result<Box<dyn leone_gguf::AllocationGuard>, AllocationBudgetError> {
        self.reservation
            .commit()
            .map(|allocation| Box::new(allocation) as Box<dyn leone_gguf::AllocationGuard>)
            .map_err(|error| AllocationBudgetError::new(self.what, self.bytes, error))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn allocation_denial_leaves_host_staging_empty() {
        let staging = HostStaging::new(MemoryBudget::limited(8).unwrap());
        assert!(matches!(
            staging.allocate(9),
            Err(MemoryError::BudgetExceeded {
                requested: 9,
                budget: 8,
                owned: 0,
                reserved: 0,
            })
        ));
        let snapshot = staging.snapshot();
        assert_eq!(snapshot.live_bytes, 0);
        assert_eq!(snapshot.reserved_bytes, 0);
        assert_eq!(snapshot.peak_live_bytes, 0);
    }

    #[test]
    fn allocation_token_releases_staging_bytes() {
        let staging = HostStaging::new(MemoryBudget::limited(8).unwrap());
        let allocation = staging.allocate(8).unwrap();
        assert_eq!(staging.snapshot().live_bytes, 8);
        drop(allocation);
        assert_eq!(staging.snapshot().live_bytes, 0);
    }

    #[test]
    fn tracker_reuse_shares_host_budget_and_snapshot() {
        let tracker = MemoryTracker::new(MemoryBudget::limited(8).unwrap());
        let staging = HostStaging::from_tracker(tracker.clone());
        let allocation = staging.allocate(8).unwrap();
        assert_eq!(tracker.snapshot().live_bytes, 8);
        drop(allocation);
        assert_eq!(tracker.snapshot().live_bytes, 0);
    }

    #[test]
    fn parser_adapter_metadata_bounds_cover_current_layout() {
        let staging = HostStaging::unlimited();
        assert_eq!(
            staging.guard_metadata_bytes(),
            u64::try_from(
                std::mem::size_of::<MemoryAllocation>()
                    + std::mem::size_of::<Box<dyn AllocationGuard>>(),
            )
            .unwrap()
        );
        assert!(
            std::mem::size_of::<StagingReservation>()
                + std::mem::size_of::<Box<dyn AllocationReservation>>()
                <= RESERVATION_METADATA_BYTES as usize
        );
    }

    #[test]
    fn gguf_import_rejects_before_host_allocation() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        bytes.extend_from_slice(&17_u64.to_le_bytes());
        bytes.extend_from_slice(b"general.alignment");
        bytes.extend_from_slice(&4_u32.to_le_bytes());
        bytes.extend_from_slice(&32_u32.to_le_bytes());
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("leone-host-budget-{unique}.gguf"));
        fs::write(&path, bytes).unwrap();
        let bootstrap_staging = HostStaging::new(MemoryBudget::limited(8_192).unwrap());
        let bootstrap_error = leone_gguf::Gguf::open_with_budget(&path, &bootstrap_staging)
            .expect_err("parser buffer exceeds metadata-only budget");
        assert!(matches!(
            bootstrap_error,
            leone_gguf::Error::AllocationBudget(error)
                if error.what() == "GGUF parser buffer"
                    && error.source_error().downcast_ref::<MemoryError>().is_some()
        ));
        assert_eq!(bootstrap_staging.snapshot().live_bytes, 0);
        assert_eq!(bootstrap_staging.snapshot().reserved_bytes, 0);

        let staging = HostStaging::new(MemoryBudget::limited(16_384).unwrap());
        let result = leone_gguf::Gguf::open_with_budget(&path, &staging);
        assert!(matches!(
            result,
            Err(leone_gguf::Error::AllocationBudget(error))
                if error.what() == "metadata map"
                    && error.source_error().downcast_ref::<MemoryError>().is_some()
        ));
        let snapshot = staging.snapshot();
        assert_eq!(snapshot.live_bytes, 0);
        assert_eq!(snapshot.reserved_bytes, 0);

        let success_staging = HostStaging::new(MemoryBudget::limited(1_000_000).unwrap());
        let gguf = leone_gguf::Gguf::open_with_budget(&path, &success_staging).unwrap();
        assert!(success_staging.snapshot().live_bytes > 0);
        drop(gguf);
        assert_eq!(success_staging.snapshot().live_bytes, 0);
        assert_eq!(success_staging.snapshot().reserved_bytes, 0);
        fs::remove_file(&path).unwrap();
    }
}
