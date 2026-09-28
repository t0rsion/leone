use half::f16;
use leone::backend::{MemoryClass, MemoryError};
use leone::{
    Backend, BackendError, BufferLayout, BufferSnapshot, MemoryBudget, PrefillPlan,
    PrefillWorkspace, QuantFormat, QuantMatrix,
};
use leone_cuda::{CudaBackend, CudaBuffer, QuantizedMatrixShape};
use leone_gguf::ref_dequant;
use std::error::Error;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const GROWTH_MARGIN_BYTES: u64 = 64 * 1024;

#[test]
#[ignore = "requires an SM89 CUDA GPU"]
fn prefill_reuse_recovers_under_finite_budget_and_preserves_session_data() -> TestResult {
    let fixture = prepare_reuse_fixture()?;
    let mut backend = fixture.backend;
    let gemm = prepare_gemm_fixture(&mut backend)?;
    run_valid_prefill(&mut backend, &gemm)?;
    reject_oversized_current_plan(&mut backend, &gemm, fixture.target_usage)?;
    let observed_sentinel = backend.download_buffer(&fixture.sentinel_buffer)?;
    assert_eq!(observed_sentinel, fixture.sentinel);
    Ok(())
}

struct ReuseFixture {
    backend: CudaBackend,
    sentinel: BufferSnapshot,
    sentinel_buffer: CudaBuffer,
    target_usage: PrefillWorkspace,
}

struct GemmFixture {
    matrix: QuantMatrix,
    shape: QuantizedMatrixShape,
    weights: Vec<u8>,
    device_weights: CudaBuffer,
}

fn prepare_reuse_fixture() -> TestResult<ReuseFixture> {
    let small_plan = prefill_plan(4, 128, 4);
    let large_plan = prefill_plan(64, 4_096, 8);
    let target_usage = probe_prefill_usage(large_plan)?;
    let sentinel_layout = BufferLayout::f32(4)?;
    let sentinel = BufferSnapshot::new(sentinel_layout, f32_bytes(&[1.25_f32, -2.5, 3.75, -4.0]))?;
    let (mut backend, sentinel_buffer) =
        prepare_budgeted_backend(small_plan, target_usage, sentinel_layout, &sentinel)?;
    let small_usage = backend
        .memory_accounting()
        .class(MemoryClass::PrefillScratch)
        .live_bytes;
    assert!(
        target_usage.total_bytes > small_usage + GROWTH_MARGIN_BYTES,
        "the test plans must force replacement coexistence"
    );
    let transitioned_usage = backend.prepare_prefill(large_plan)?;
    assert_eq!(transitioned_usage, target_usage);
    assert_recovered_footprint(&backend, target_usage);
    Ok(ReuseFixture {
        backend,
        sentinel,
        sentinel_buffer,
        target_usage,
    })
}

fn prepare_budgeted_backend(
    small_plan: PrefillPlan,
    target_usage: PrefillWorkspace,
    sentinel_layout: BufferLayout,
    sentinel: &BufferSnapshot,
) -> TestResult<(CudaBackend, CudaBuffer)> {
    let mut backend = CudaBackend::new(0)?;
    let sentinel_buffer =
        backend.restore_buffer_classified(sentinel, MemoryClass::ContractBuffer)?;
    backend.prepare_prefill(small_plan)?;
    let budget = MemoryBudget::limited(
        target_usage
            .total_bytes
            .checked_add(sentinel_layout.bytes() as u64)
            .and_then(|bytes| bytes.checked_add(GROWTH_MARGIN_BYTES))
            .ok_or(MemoryError::ByteOverflow)?,
    )?;
    backend.set_memory_budget(budget)?;
    Ok((backend, sentinel_buffer))
}

fn assert_recovered_footprint(backend: &CudaBackend, target_usage: PrefillWorkspace) {
    let stats = backend
        .memory_accounting()
        .class(MemoryClass::PrefillScratch);
    assert_eq!(stats.live_bytes, target_usage.total_bytes);
    assert_eq!(stats.live_allocations, 8);
    assert!(stats.allocations > stats.live_allocations);
    assert!(stats.frees > 0);
}

fn prepare_gemm_fixture(backend: &mut CudaBackend) -> TestResult<GemmFixture> {
    let matrix = QuantMatrix::new(4, 256, QuantFormat::Q4K)?;
    let shape = QuantizedMatrixShape::new(4, 256, leone_cuda::QuantFormat::Q4K)?;
    let weights = test_weights(shape);
    let device_weights = backend.upload(matrix.layout()?, &weights)?;
    Ok(GemmFixture {
        matrix,
        shape,
        weights,
        device_weights,
    })
}

fn run_valid_prefill(backend: &mut CudaBackend, fixture: &GemmFixture) -> TestResult {
    let tokens = 2;
    let input = test_input(tokens * fixture.matrix.columns());
    let device_input = backend.upload(BufferLayout::f32(input.len())?, &f32_bytes(&input))?;
    let mut device_output = backend.allocate(BufferLayout::f32(tokens * fixture.matrix.rows())?)?;
    backend.prefill_gemm(
        &fixture.device_weights,
        &device_input,
        &mut device_output,
        fixture.matrix,
        tokens,
    )?;
    let actual = read_f32_output(backend, &device_output, tokens * fixture.matrix.rows())?;
    let expected = prefill_cpu_oracle(&fixture.weights, &input, fixture.shape)?;
    assert_close(&actual, &expected, 2e-2, 2e-3);
    Ok(())
}

fn read_f32_output(
    backend: &mut CudaBackend,
    output: &CudaBuffer,
    elements: usize,
) -> TestResult<Vec<f32>> {
    backend.synchronize()?;
    let mut values = vec![0.0_f32; elements];
    backend.read_f32(output, &mut values)?;
    Ok(values)
}

fn reject_oversized_current_plan(
    backend: &mut CudaBackend,
    fixture: &GemmFixture,
    target_usage: PrefillWorkspace,
) -> TestResult {
    let smaller_plan = prefill_plan(2, 64, 4);
    assert_eq!(backend.prepare_prefill(smaller_plan)?, target_usage);
    let tokens = smaller_plan.chunk_tokens() + 1;
    let input = test_input(tokens * fixture.matrix.columns());
    let device_input = backend.upload(BufferLayout::f32(input.len())?, &f32_bytes(&input))?;
    let mut device_output = backend.allocate(BufferLayout::f32(tokens * fixture.matrix.rows())?)?;
    let rejection = backend.prefill_gemm(
        &fixture.device_weights,
        &device_input,
        &mut device_output,
        fixture.matrix,
        tokens,
    );
    assert!(matches!(
        rejection,
        Err(BackendError::Operation {
            operation: "run prefill GEMM",
            ..
        })
    ));
    Ok(())
}

fn prefill_plan(chunk_tokens: usize, context_tokens: usize, max_matrix_rows: usize) -> PrefillPlan {
    PrefillPlan::new(
        chunk_tokens,
        context_tokens,
        4,
        2,
        32,
        256,
        256,
        max_matrix_rows,
    )
    .expect("test prefill plan")
}

fn probe_prefill_usage(plan: PrefillPlan) -> TestResult<leone::PrefillWorkspace> {
    let mut probe = CudaBackend::new(0)?;
    Ok(probe.prepare_prefill(plan)?)
}

fn test_weights(shape: leone_cuda::QuantizedMatrixShape) -> Vec<u8> {
    let mut bytes = (0..shape.bytes())
        .map(|index| (index as u8).wrapping_mul(37).wrapping_add(11))
        .collect::<Vec<_>>();
    for block in bytes.chunks_exact_mut(shape.format().block_bytes()) {
        set_half(block, 0, 0.03);
        set_half(block, 2, 0.02);
    }
    bytes
}

fn test_input(elements: usize) -> Vec<f32> {
    (0..elements)
        .map(|index| ((index % 19) as f32 - 9.0) / 32.0)
        .collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_bits().to_le_bytes())
        .collect()
}

fn prefill_cpu_oracle(
    weights: &[u8],
    input: &[f32],
    shape: leone_cuda::QuantizedMatrixShape,
) -> TestResult<Vec<f64>> {
    let mut output = vec![0.0_f64; input.len() / shape.columns() * shape.rows()];
    for row in 0..shape.rows() {
        let start = row * shape.row_bytes();
        let decoded = ref_dequant::q4_k::dequant_row(
            &weights[start..start + shape.row_bytes()],
            shape.columns(),
        )?;
        for token in 0..input.len() / shape.columns() {
            output[token * shape.rows() + row] = decoded
                .iter()
                .zip(&input[token * shape.columns()..(token + 1) * shape.columns()])
                .map(|(weight, value)| {
                    f64::from(f16::from_f32(*weight).to_f32())
                        * f64::from(f16::from_f32(*value).to_f32())
                })
                .sum();
        }
    }
    Ok(output)
}

fn assert_close(actual: &[f32], expected: &[f64], absolute: f64, relative: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let error = (f64::from(*actual) - expected).abs();
        let bound = absolute + relative * expected.abs();
        assert!(
            error <= bound,
            "prefill output differs at {index}: {actual:e} versus {expected:e}, bound {bound:e}"
        );
    }
}

fn set_half(block: &mut [u8], offset: usize, value: f32) {
    block[offset..offset + 2].copy_from_slice(&f16::from_f32(value).to_bits().to_le_bytes());
}
