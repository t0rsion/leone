use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

/// The kind of a runtime state allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StateKind {
    Kv,
}

/// The stable identifier of one state allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AllocationId(u64);

/// The stable identifier of one state transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TransactionId(u64);

/// The lifetime rule attached to a state allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateLifetime {
    Committed,
    Transactional(TransactionId),
}

/// The accounting record for one state allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateAllocation {
    id: AllocationId,
    kind: StateKind,
    bytes: u64,
    lifetime: StateLifetime,
}

impl StateAllocation {
    pub const fn id(self) -> AllocationId {
        self.id
    }

    pub const fn kind(self) -> StateKind {
        self.kind
    }

    pub const fn bytes(self) -> u64 {
        self.bytes
    }

    pub const fn lifetime(self) -> StateLifetime {
        self.lifetime
    }
}

/// An error returned by checked state accounting or transactions.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum StateError {
    #[error("state allocation bytes must be nonzero")]
    ZeroAllocation,
    #[error("state capacity is {capacity_bytes} bytes, requested {requested_bytes} with {used_bytes} used")]
    Capacity {
        capacity_bytes: u64,
        used_bytes: u64,
        requested_bytes: u64,
    },
    #[error("state byte accounting overflowed")]
    SizeOverflow,
    #[error("state allocation {0:?} does not exist")]
    UnknownAllocation(AllocationId),
    #[error("state transaction {0:?} is not active")]
    InactiveTransaction(TransactionId),
    #[error("state identifier space is exhausted")]
    IdentifierExhausted,
}

/// Accounts state kinds, capacity, and transactional lifetime.
#[derive(Debug)]
pub struct StateAllocator {
    capacity_bytes: u64,
    used_bytes: u64,
    next_allocation: u64,
    next_transaction: u64,
    allocations: BTreeMap<AllocationId, StateAllocation>,
    active_transactions: BTreeSet<TransactionId>,
}

impl StateAllocator {
    /// Creates an allocator with an exact byte capacity.
    pub fn new(capacity_bytes: u64) -> Self {
        Self {
            capacity_bytes,
            used_bytes: 0,
            next_allocation: 0,
            next_transaction: 0,
            allocations: BTreeMap::new(),
            active_transactions: BTreeSet::new(),
        }
    }

    pub const fn capacity_bytes(&self) -> u64 {
        self.capacity_bytes
    }

    pub const fn used_bytes(&self) -> u64 {
        self.used_bytes
    }

    pub fn allocations(&self) -> impl Iterator<Item = &StateAllocation> {
        self.allocations.values()
    }

    /// Starts a transaction for state that may be committed or rolled back.
    pub fn begin_transaction(&mut self) -> Result<TransactionId, StateError> {
        let id = TransactionId(self.next_transaction);
        self.next_transaction = self
            .next_transaction
            .checked_add(1)
            .ok_or(StateError::IdentifierExhausted)?;
        self.active_transactions.insert(id);
        Ok(id)
    }

    /// Allocates an accounting record under the selected lifetime rule.
    pub fn allocate(
        &mut self,
        kind: StateKind,
        bytes: u64,
        lifetime: StateLifetime,
    ) -> Result<StateAllocation, StateError> {
        if bytes == 0 {
            return Err(StateError::ZeroAllocation);
        }
        if let StateLifetime::Transactional(transaction) = lifetime {
            self.require_active(transaction)?;
        }
        let next_used = self
            .used_bytes
            .checked_add(bytes)
            .ok_or(StateError::SizeOverflow)?;
        if next_used > self.capacity_bytes {
            return Err(StateError::Capacity {
                capacity_bytes: self.capacity_bytes,
                used_bytes: self.used_bytes,
                requested_bytes: bytes,
            });
        }
        let id = AllocationId(self.next_allocation);
        self.next_allocation = self
            .next_allocation
            .checked_add(1)
            .ok_or(StateError::IdentifierExhausted)?;
        let allocation = StateAllocation {
            id,
            kind,
            bytes,
            lifetime,
        };
        self.allocations.insert(id, allocation);
        self.used_bytes = next_used;
        Ok(allocation)
    }

    /// Frees one allocation and returns its accounting record.
    pub fn free(&mut self, id: AllocationId) -> Result<StateAllocation, StateError> {
        let allocation = self
            .allocations
            .remove(&id)
            .ok_or(StateError::UnknownAllocation(id))?;
        self.used_bytes -= allocation.bytes;
        Ok(allocation)
    }

    /// Commits every allocation owned by an active transaction.
    pub fn commit(&mut self, transaction: TransactionId) -> Result<(), StateError> {
        self.require_active(transaction)?;
        for allocation in self.allocations.values_mut() {
            if allocation.lifetime == StateLifetime::Transactional(transaction) {
                allocation.lifetime = StateLifetime::Committed;
            }
        }
        self.active_transactions.remove(&transaction);
        Ok(())
    }

    /// Removes every allocation owned by an active transaction.
    pub fn rollback(
        &mut self,
        transaction: TransactionId,
    ) -> Result<Vec<StateAllocation>, StateError> {
        self.require_active(transaction)?;
        let ids: Vec<_> = self
            .allocations
            .values()
            .filter(|allocation| allocation.lifetime == StateLifetime::Transactional(transaction))
            .map(|allocation| allocation.id)
            .collect();
        let mut removed = Vec::with_capacity(ids.len());
        for id in ids {
            removed.push(self.free(id)?);
        }
        self.active_transactions.remove(&transaction);
        Ok(removed)
    }

    /// Clears all allocations and transactions without changing capacity.
    pub fn reset(&mut self) {
        self.used_bytes = 0;
        self.allocations.clear();
        self.active_transactions.clear();
    }

    fn require_active(&self, transaction: TransactionId) -> Result<(), StateError> {
        if self.active_transactions.contains(&transaction) {
            Ok(())
        } else {
            Err(StateError::InactiveTransaction(transaction))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::collections::BTreeMap;

    proptest! {
        #[test]
        fn randomized_traces_match_a_cpu_shadow(operations in prop::collection::vec((0_u8..6, 1_u16..512), 1..2_000)) {
            let capacity = 8_192_u64;
            let mut allocator = StateAllocator::new(capacity);
            let mut shadow = Shadow::new(capacity);
            let mut live = Vec::new();
            let mut transaction = None;

            for (opcode, amount) in operations {
                match opcode {
                    0 | 1 => {
                        let lifetime = transaction
                            .map(StateLifetime::Transactional)
                            .unwrap_or(StateLifetime::Committed);
                        let actual = allocator.allocate(StateKind::Kv, u64::from(amount), lifetime);
                        let expected = shadow.allocate(u64::from(amount), lifetime);
                        prop_assert_eq!(actual.as_ref().map(|value| value.bytes()), expected.as_ref().map(|value| value.bytes()));
                        if let Ok(allocation) = actual {
                            live.push(allocation.id());
                        }
                    }
                    2 if !live.is_empty() => {
                        let index = usize::from(amount) % live.len();
                        let id = live.swap_remove(index);
                        prop_assert_eq!(allocator.free(id).map(|value| value.bytes()), shadow.free(id).map(|value| value.bytes()));
                    }
                    3 if transaction.is_none() => {
                        let actual = allocator.begin_transaction().unwrap();
                        let expected = shadow.begin_transaction();
                        prop_assert_eq!(actual, expected);
                        transaction = Some(actual);
                    }
                    4 if transaction.is_some() => {
                        let id = transaction.take().unwrap();
                        allocator.commit(id).unwrap();
                        shadow.commit(id).unwrap();
                    }
                    5 if transaction.is_some() => {
                        let id = transaction.take().unwrap();
                        let removed = allocator.rollback(id).unwrap();
                        let expected = shadow.rollback(id).unwrap();
                        prop_assert_eq!(removed.len(), expected.len());
                        live.retain(|allocation| !removed.iter().any(|value| value.id() == *allocation));
                    }
                    _ => {
                        allocator.reset();
                        shadow.reset();
                        live.clear();
                        transaction = None;
                    }
                }
                prop_assert_eq!(allocator.used_bytes(), shadow.used);
                prop_assert_eq!(allocator.allocations().copied().collect::<Vec<_>>(), shadow.allocations.values().copied().collect::<Vec<_>>());
            }
        }
    }

    struct Shadow {
        capacity: u64,
        used: u64,
        next_allocation: u64,
        next_transaction: u64,
        allocations: BTreeMap<AllocationId, StateAllocation>,
        active: BTreeSet<TransactionId>,
    }

    impl Shadow {
        fn new(capacity: u64) -> Self {
            Self {
                capacity,
                used: 0,
                next_allocation: 0,
                next_transaction: 0,
                allocations: BTreeMap::new(),
                active: BTreeSet::new(),
            }
        }

        fn begin_transaction(&mut self) -> TransactionId {
            let id = TransactionId(self.next_transaction);
            self.next_transaction += 1;
            self.active.insert(id);
            id
        }

        fn allocate(
            &mut self,
            bytes: u64,
            lifetime: StateLifetime,
        ) -> Result<StateAllocation, StateError> {
            if self.used + bytes > self.capacity {
                return Err(StateError::Capacity {
                    capacity_bytes: self.capacity,
                    used_bytes: self.used,
                    requested_bytes: bytes,
                });
            }
            let allocation = StateAllocation {
                id: AllocationId(self.next_allocation),
                kind: StateKind::Kv,
                bytes,
                lifetime,
            };
            self.next_allocation += 1;
            self.used += bytes;
            self.allocations.insert(allocation.id, allocation);
            Ok(allocation)
        }

        fn free(&mut self, id: AllocationId) -> Result<StateAllocation, StateError> {
            let allocation = self
                .allocations
                .remove(&id)
                .ok_or(StateError::UnknownAllocation(id))?;
            self.used -= allocation.bytes;
            Ok(allocation)
        }

        fn commit(&mut self, id: TransactionId) -> Result<(), StateError> {
            if !self.active.remove(&id) {
                return Err(StateError::InactiveTransaction(id));
            }
            for allocation in self.allocations.values_mut() {
                if allocation.lifetime == StateLifetime::Transactional(id) {
                    allocation.lifetime = StateLifetime::Committed;
                }
            }
            Ok(())
        }

        fn rollback(&mut self, id: TransactionId) -> Result<Vec<StateAllocation>, StateError> {
            if !self.active.remove(&id) {
                return Err(StateError::InactiveTransaction(id));
            }
            let ids: Vec<_> = self
                .allocations
                .values()
                .filter(|value| value.lifetime == StateLifetime::Transactional(id))
                .map(|value| value.id)
                .collect();
            ids.into_iter()
                .map(|allocation| self.free(allocation))
                .collect()
        }

        fn reset(&mut self) {
            self.used = 0;
            self.allocations.clear();
            self.active.clear();
        }
    }
}
