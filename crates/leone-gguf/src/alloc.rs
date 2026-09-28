use std::error::Error as StdError;
use std::fmt;
use thiserror::Error;

/// Describes one checked allocation request made by the GGUF parser.
#[derive(Debug, Error)]
#[error("{what} allocation for {bytes} bytes failed: {source}")]
pub struct AllocationBudgetError {
    what: &'static str,
    bytes: u64,
    #[source]
    source: Box<dyn StdError + Send + Sync>,
}

impl AllocationBudgetError {
    /// Creates an allocation error while preserving the adapter source error.
    pub fn new<E>(what: &'static str, bytes: u64, source: E) -> Self
    where
        E: StdError + Send + Sync + 'static,
    {
        Self {
            what,
            bytes,
            source: Box::new(source),
        }
    }

    /// Returns the allocation label supplied by the parser.
    pub const fn what(&self) -> &'static str {
        self.what
    }

    /// Returns the requested byte count.
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Returns the adapter error that rejected the request.
    pub fn source_error(&self) -> &(dyn StdError + Send + Sync + 'static) {
        self.source.as_ref()
    }
}

/// Owns one committed parser allocation until the parsed container drops.
pub trait AllocationGuard: fmt::Debug + Send + Sync {}

impl<T> AllocationGuard for T where T: fmt::Debug + Send + Sync {}

/// Holds one parser allocation while its physical storage is constructed.
///
/// A caller may keep a reservation pending when it describes a conservative
/// container bound instead of one exact payload.
pub trait AllocationReservation: fmt::Debug + Send {
    /// Commits the reservation after the physical allocation succeeds.
    fn commit(self: Box<Self>) -> Result<Box<dyn AllocationGuard>, AllocationBudgetError>;
}

/// Supplies checked parser allocation reservations without a crate dependency.
pub trait AllocationBudget: Send + Sync {
    /// Returns the checked bytes for one retained guard and its owner slot.
    ///
    /// The bound includes the concrete guard object and one collection slot.
    fn guard_metadata_bytes(&self) -> u64;

    /// Returns the checked bytes for one retained reservation wrapper.
    ///
    /// The parser multiplies this bound by its wrapper capacity before growth.
    fn reservation_metadata_bytes(&self) -> u64;

    /// Reserves bytes before the parser constructs one owned value.
    fn reserve(
        &self,
        bytes: u64,
        what: &'static str,
    ) -> Result<Box<dyn AllocationReservation>, AllocationBudgetError>;
}

#[derive(Debug, Default)]
pub(crate) struct UnlimitedAllocationBudget;

#[derive(Debug)]
struct UnlimitedAllocationReservation;

#[derive(Debug)]
struct UnlimitedAllocationGuard;

impl AllocationBudget for UnlimitedAllocationBudget {
    fn guard_metadata_bytes(&self) -> u64 {
        0
    }

    fn reservation_metadata_bytes(&self) -> u64 {
        0
    }

    fn reserve(
        &self,
        _bytes: u64,
        _what: &'static str,
    ) -> Result<Box<dyn AllocationReservation>, AllocationBudgetError> {
        Ok(Box::new(UnlimitedAllocationReservation))
    }
}

impl AllocationReservation for UnlimitedAllocationReservation {
    fn commit(self: Box<Self>) -> Result<Box<dyn AllocationGuard>, AllocationBudgetError> {
        Ok(Box::new(UnlimitedAllocationGuard))
    }
}
