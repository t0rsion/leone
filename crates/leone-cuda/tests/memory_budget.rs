use leone::{Backend, BackendError, BufferLayout, MemoryBudget, MemoryError};
use leone_cuda::CudaBackend;

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn cuda_budget_rejection_preserves_buffers_and_recovers_after_release() {
    let mut backend = CudaBackend::with_memory_budget(0, MemoryBudget::Unlimited).unwrap();
    let startup = backend.memory_accounting();
    let tracker = backend.memory_tracker_root();
    let budget_bytes = startup.live_bytes.checked_add(16).unwrap();
    let budget = MemoryBudget::limited(budget_bytes).unwrap();
    backend.set_memory_budget(budget).unwrap();
    let mut source = backend.allocate(BufferLayout::u32(4).unwrap()).unwrap();
    backend.write_u32(&mut source, &[3, 1, 4, 1]).unwrap();
    let before = backend.memory_accounting();

    match backend.clone_buffer(&source) {
        Err(BackendError::Memory(MemoryError::BudgetExceeded {
            requested,
            budget,
            owned,
            reserved,
        })) => assert_eq!(
            (requested, budget, owned, reserved),
            (16, budget_bytes, before.live_bytes, 0)
        ),
        result => panic!("unexpected clone result: {result:?}"),
    }
    assert_eq!(backend.memory_accounting(), before);
    let mut actual = [0; 4];
    backend.read_u32(&source, &mut actual).unwrap();
    assert_eq!(actual, [3, 1, 4, 1]);

    backend
        .set_memory_budget(MemoryBudget::limited(startup.live_bytes + 32).unwrap())
        .unwrap();
    let copy = backend.clone_buffer(&source).unwrap();
    backend.synchronize().unwrap();
    let full = backend.memory_accounting();
    assert_eq!(full.live_bytes, startup.live_bytes + 32);
    assert_eq!(full.reserved_bytes, 0);
    assert!(matches!(
        backend.set_memory_budget(budget),
        Err(BackendError::Memory(MemoryError::BudgetBelowOwned { .. }))
    ));
    assert_eq!(backend.memory_accounting(), full);

    drop(copy);
    backend.set_memory_budget(budget).unwrap();
    drop(source);
    let replacement = backend.allocate(BufferLayout::u32(4).unwrap()).unwrap();
    assert_eq!(
        backend.memory_accounting().live_bytes,
        startup.live_bytes + 16
    );
    drop(replacement);
    let empty = backend.memory_accounting();
    assert_eq!(empty.live_bytes, startup.live_bytes);
    assert_eq!(empty.reserved_bytes, 0);
    assert_eq!(empty.live_allocations, startup.live_allocations);
    assert_eq!(
        empty.allocations - empty.frees,
        startup.allocations - startup.frees
    );

    drop(backend);
    let released = tracker.snapshot();
    assert_eq!(released.live_bytes, 0);
    assert_eq!(released.reserved_bytes, 0);
    assert_eq!(released.live_allocations, 0);
}
