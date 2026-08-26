use leone::{Backend, BufferLayout};
use leone_cuda::CudaBackend;

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn cuda_buffer_clone_is_exact_and_independent() {
    let mut backend = CudaBackend::new(0).unwrap();
    let mut source = backend.allocate(BufferLayout::u32(4).unwrap()).unwrap();
    backend.write_u32(&mut source, &[3, 1, 4, 1]).unwrap();
    let clone = backend.clone_buffer(&source).unwrap();
    backend.write_u32(&mut source, &[5, 9, 2, 6]).unwrap();
    backend.synchronize().unwrap();
    let mut actual = [0; 4];
    backend.read_u32(&clone, &mut actual).unwrap();
    assert_eq!(actual, [3, 1, 4, 1]);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn cuda_buffer_snapshot_restores_exact_physical_bytes() {
    let mut backend = CudaBackend::new(0).unwrap();
    let mut source = backend.allocate(BufferLayout::u32(4).unwrap()).unwrap();
    backend.write_u32(&mut source, &[2, 7, 1, 8]).unwrap();
    let snapshot = backend.download_buffer(&source).unwrap();
    backend.write_u32(&mut source, &[2, 8, 1, 8]).unwrap();
    let restored = backend.restore_buffer(&snapshot).unwrap();
    let mut actual = [0; 4];
    backend.read_u32(&restored, &mut actual).unwrap();
    assert_eq!(actual, [2, 7, 1, 8]);
}
