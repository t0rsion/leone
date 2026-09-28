use half::f16;
use leone::{
    AttentionShape, Backend, BufferLayout, BufferSnapshot, KvReadSpan, KvReadView, KvWriteSpan,
    Position, PrefillPlan, RopePairing, VectorShape,
};
use leone_cuda::{CudaBackend, CudaBuffer};
use std::error::Error;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

#[derive(Clone, Copy)]
enum CacheKind {
    F16,
    Q8,
}

#[derive(Clone, Copy)]
struct SpanSpec {
    logical_start: usize,
    tokens: usize,
    capacity: usize,
}

struct SpanStorage {
    keys: Vec<CudaBuffer>,
    values: Vec<CudaBuffer>,
}

struct SpanSources {
    keys: Vec<CudaBuffer>,
    values: Vec<CudaBuffer>,
}

struct FusedInputs {
    query: Vec<f32>,
    key: Vec<f32>,
    value: Vec<f32>,
    query_weight: Vec<f32>,
    key_weight: Vec<f32>,
}

struct FusedCase {
    backend: CudaBackend,
    shape: AttentionShape,
    query_shape: VectorShape,
    key_shape: VectorShape,
    start_position: usize,
    positions: usize,
    query: CudaBuffer,
    query_weight: CudaBuffer,
    key: CudaBuffer,
    key_weight: CudaBuffer,
    value: CudaBuffer,
    query_output: CudaBuffer,
    key_output: CudaBuffer,
    key_cache: CudaBuffer,
    value_cache: CudaBuffer,
    expected_query: Vec<f64>,
    expected_key: Vec<f64>,
    expected_key_cache: Vec<f32>,
    expected_value_cache: Vec<f32>,
}

struct BadQkBuffers {
    key_shape: VectorShape,
    key: CudaBuffer,
    weight: CudaBuffer,
    value: CudaBuffer,
    query_output: CudaBuffer,
    key_output: CudaBuffer,
}

struct BadQkSnapshots {
    key: Vec<u8>,
    value: Vec<u8>,
    query_output: Vec<u8>,
    key_output: Vec<u8>,
}

const SPANS: [SpanSpec; 3] = [
    SpanSpec {
        logical_start: 0,
        tokens: 2,
        capacity: 3,
    },
    SpanSpec {
        logical_start: 2,
        tokens: 3,
        capacity: 4,
    },
    SpanSpec {
        logical_start: 5,
        tokens: 4,
        capacity: 5,
    },
];

const SHAPE_MAX_CONTEXT: usize = 14;
const EPSILON: f32 = 1e-6;
const THETA: f32 = 10_000.0;
const HEAD_DIMS: [usize; 3] = [32, 64, 128];
const Q8_BLOCK_ELEMENTS: usize = 32;
const Q8_BLOCK_BYTES: usize = 34;

#[test]
#[ignore = "requires an SM89 CUDA GPU"]
fn f16_span_attention_matches_fp64_decode_prefill_and_verifier_oracles() -> TestResult {
    for head_dim in HEAD_DIMS {
        run_span_attention_case(CacheKind::F16, head_dim)?;
    }
    Ok(())
}

#[test]
#[ignore = "requires an SM89 CUDA GPU"]
fn q8_span_attention_matches_fp64_decode_and_prefill_oracles() -> TestResult {
    for head_dim in HEAD_DIMS {
        run_span_attention_case(CacheKind::Q8, head_dim)?;
    }
    Ok(())
}

fn run_span_attention_case(kind: CacheKind, head_dim: usize) -> TestResult {
    let shape = attention_shape(head_dim)?;
    let projected = shape.projected_kv_elements()?;
    let mapped_tokens = 9;
    let keys = canonical_values(mapped_tokens * projected, 17);
    let values = canonical_values(mapped_tokens * projected, 41);
    let mut backend = CudaBackend::new(0)?;
    let storage = append_spans(&mut backend, kind, shape, &keys, &values)?;
    let logical_keys = logical_cache(kind, &keys, shape);
    let logical_values = logical_cache(kind, &values, shape);
    let query = canonical_values(query_elements(shape), 73);
    let actual = run_decode(&mut backend, &storage, shape, &query)?;
    let expected = attention_oracle(&query, &logical_keys, &logical_values, shape, mapped_tokens);
    assert_close("span decode", &actual, &expected, 4e-4, 3e-3);

    let prefill_start = 2;
    let prefill_tokens = 5;
    let prefill_query = canonical_values(prefill_tokens * query_elements(shape), 101);
    let actual = run_prefill(
        &mut backend,
        &storage,
        shape,
        &prefill_query,
        prefill_start,
        prefill_tokens,
    )?;
    let expected = causal_oracle(
        &prefill_query,
        &logical_keys,
        &logical_values,
        shape,
        prefill_start,
        prefill_tokens,
        true,
    );
    assert_close("span causal prefill", &actual, &expected, 7e-4, 5e-3);

    if matches!(kind, CacheKind::F16) {
        let verifier_start = 1;
        let verifier_tokens = 4;
        let verifier_query = canonical_values(verifier_tokens * query_elements(shape), 137);
        let actual = run_verify_attention(
            &mut backend,
            &storage,
            shape,
            &verifier_query,
            verifier_start,
            verifier_tokens,
        )?;
        let expected = causal_oracle(
            &verifier_query,
            &logical_keys,
            &logical_values,
            shape,
            verifier_start,
            verifier_tokens,
            false,
        );
        assert_close("span verifier attention", &actual, &expected, 7e-4, 5e-3);
    }
    Ok(())
}

#[test]
#[ignore = "requires an SM89 CUDA GPU"]
fn device_span_attention_reports_position_beyond_mapped_prefix() -> TestResult {
    let shape = AttentionShape::new(1, 1, 32, 128)?;
    let mut backend = CudaBackend::new(0)?;
    let query = upload_f32(
        &mut backend,
        &canonical_values(shape.query_elements()?, 307),
    )?;
    let (key_cache, value_cache) = invalid_position_cache(&mut backend, shape)?;
    let position = invalid_position_buffer(&mut backend)?;
    run_invalid_position_attention(
        &mut backend,
        shape,
        &query,
        &key_cache,
        &value_cache,
        &position,
    )
}

#[test]
#[ignore = "requires an SM89 CUDA GPU"]
fn device_span_append_reports_position_out_of_bounds() -> TestResult {
    let case = invalid_span_append_case()?;
    let error = run_invalid_span_append(case, 4, 4)?;
    assert!(matches!(
        error,
        leone::BackendError::PositionOutsideSpan {
            position: 3,
            start: 4,
            end: 8,
        }
    ));
    Ok(())
}

#[test]
#[ignore = "requires an SM89 CUDA GPU"]
fn device_span_append_at_zero_reports_position_outside_span() -> TestResult {
    let case = invalid_span_append_case_with(8, 8)?;
    let error = run_invalid_span_append(case, 0, 8)?;
    assert!(matches!(
        error,
        leone::BackendError::PositionOutsideSpan {
            position: 8,
            start: 0,
            end: 8,
        }
    ));
    Ok(())
}

type InvalidAppendCase = (
    CudaBackend,
    CudaBuffer,
    CudaBuffer,
    CudaBuffer,
    CudaBuffer,
    CudaBuffer,
);

fn invalid_span_append_case() -> TestResult<InvalidAppendCase> {
    invalid_span_append_case_with(4, 3)
}

fn invalid_span_append_case_with(
    capacity: usize,
    position: usize,
) -> TestResult<InvalidAppendCase> {
    let mut backend = CudaBackend::new(0)?;
    let key = invalid_span_source(&mut backend, 311)?;
    let value = invalid_span_source(&mut backend, 313)?;
    let (key_cache, value_cache) = invalid_span_caches(&mut backend, capacity)?;
    let position = invalid_position_buffer_value(&mut backend, position)?;
    Ok((backend, key, value, key_cache, value_cache, position))
}

fn run_invalid_span_append(
    case: InvalidAppendCase,
    start: usize,
    capacity: usize,
) -> TestResult<leone::BackendError> {
    let (mut backend, key, value, mut key_cache, mut value_cache, position) = case;
    let shape = AttentionShape::new(1, 1, 32, 128)?;
    let target = KvWriteSpan::new(&mut key_cache, &mut value_cache, start, capacity)?;
    backend.kv_append_span(&key, &value, target, shape, Position::Device(&position))?;
    let projected = shape.projected_kv_elements()?;
    let mut host = vec![0_u16; projected * capacity];
    Ok(backend
        .read_f16(&key_cache, &mut host)
        .expect_err("invalid device append position"))
}

fn invalid_span_source(backend: &mut CudaBackend, seed: usize) -> TestResult<CudaBuffer> {
    let shape = AttentionShape::new(1, 1, 32, 128)?;
    let projected = shape.projected_kv_elements()?;
    upload_f32(backend, &canonical_values(projected, seed))
}

fn invalid_span_caches(
    backend: &mut CudaBackend,
    capacity: usize,
) -> TestResult<(CudaBuffer, CudaBuffer)> {
    let shape = AttentionShape::new(1, 1, 32, 128)?;
    let projected = shape.projected_kv_elements()?;
    Ok((
        zero_buffer(backend, BufferLayout::f16(projected * capacity)?)?,
        zero_buffer(backend, BufferLayout::f16(projected * capacity)?)?,
    ))
}

fn run_invalid_position_attention(
    backend: &mut CudaBackend,
    shape: AttentionShape,
    query: &CudaBuffer,
    key_cache: &CudaBuffer,
    value_cache: &CudaBuffer,
    position: &CudaBuffer,
) -> TestResult {
    let span = KvReadSpan::new(key_cache, value_cache, 0, 32, 32)?;
    let spans = [span];
    let view = KvReadView::new(&spans)?;
    let mut output = zero_buffer(backend, BufferLayout::f32(shape.query_elements()?)?)?;
    backend.attention_decode_spans(query, view, &mut output, shape, Position::Device(position))?;
    let mut host_output = vec![0_f32; shape.query_elements()?];
    let error = backend
        .read_f32(&output, &mut host_output)
        .expect_err("invalid device position");
    assert!(matches!(
        error,
        leone::BackendError::PositionOutOfBounds {
            position: 33,
            max_context: 32,
        }
    ));
    Ok(())
}

fn invalid_position_cache(
    backend: &mut CudaBackend,
    shape: AttentionShape,
) -> TestResult<(CudaBuffer, CudaBuffer)> {
    let cache_elements = 32 * shape.projected_kv_elements()?;
    let cache_layout = BufferLayout::f16(cache_elements)?;
    let cache_bytes = cache_layout.bytes();
    let zero_cache = BufferSnapshot::new(cache_layout, vec![0_u8; cache_bytes])?;
    Ok((
        backend.restore_buffer(&zero_cache)?,
        backend.restore_buffer(&zero_cache)?,
    ))
}

fn invalid_position_buffer(backend: &mut CudaBackend) -> TestResult<CudaBuffer> {
    invalid_position_buffer_value(backend, 32)
}

fn invalid_position_buffer_value(
    backend: &mut CudaBackend,
    position: usize,
) -> TestResult<CudaBuffer> {
    Ok(backend.upload(
        BufferLayout::u32(1)?,
        &u32::try_from(position)?.to_le_bytes(),
    )?)
}

#[test]
#[ignore = "requires an SM89 CUDA GPU"]
fn fused_f16_span_verifier_matches_fp64_qk_rope_and_attention_oracles() -> TestResult {
    for head_dim in HEAD_DIMS {
        let mut case = build_fused_case(head_dim)?;
        check_fused_qk(&mut case)?;
        check_fused_cache(&mut case)?;
        check_fused_attention(&mut case)?;
        reject_wrong_qk_shape(&mut case)?;
    }
    Ok(())
}

fn build_fused_case(head_dim: usize) -> TestResult<FusedCase> {
    let (shape, query_shape, key_shape) = fused_shapes(head_dim)?;
    let start_position = 2;
    let positions = 4;
    let inputs = fused_inputs(shape, query_shape, key_shape, positions);
    let expected_query = rms_rope_rows(
        &inputs.query,
        &inputs.query_weight,
        query_shape,
        start_position,
    );
    let expected_key = rms_rope_rows(&inputs.key, &inputs.key_weight, key_shape, start_position);
    let expected_key_cache = fused_cache_values(&expected_key, key_shape, start_position, shape);
    let expected_value_cache =
        fused_cache_values_f32(&inputs.value, key_shape, start_position, shape);
    let mut backend = CudaBackend::new(0)?;
    backend.configure_rope(shape.head_dim(), THETA, None, RopePairing::HalfSplit)?;
    let input_buffers = upload_fused_inputs(&mut backend, &inputs)?;
    let output_buffers =
        allocate_fused_outputs(&mut backend, shape, query_shape, key_shape, positions)?;
    let mut case = FusedCase {
        backend,
        shape,
        query_shape,
        key_shape,
        start_position,
        positions,
        query: input_buffers.0,
        query_weight: input_buffers.1,
        key: input_buffers.2,
        key_weight: input_buffers.3,
        value: input_buffers.4,
        query_output: output_buffers.0,
        key_output: output_buffers.1,
        key_cache: output_buffers.2,
        value_cache: output_buffers.3,
        expected_query,
        expected_key,
        expected_key_cache,
        expected_value_cache,
    };
    append_fused(&mut case)?;
    case.backend.synchronize()?;
    Ok(case)
}

fn fused_shapes(head_dim: usize) -> TestResult<(AttentionShape, VectorShape, VectorShape)> {
    let shape = attention_shape(head_dim)?;
    let query_shape = VectorShape::new(shape.n_head(), shape.head_dim())?;
    let key_shape = VectorShape::new(shape.n_head_kv(), shape.head_dim())?;
    Ok((shape, query_shape, key_shape))
}

fn fused_inputs(
    shape: AttentionShape,
    query_shape: VectorShape,
    key_shape: VectorShape,
    positions: usize,
) -> FusedInputs {
    FusedInputs {
        query: canonical_values(positions * vector_elements(query_shape), 173),
        key: canonical_values(positions * vector_elements(key_shape), 197),
        value: canonical_values(positions * vector_elements(key_shape), 223),
        query_weight: canonical_weights(shape.head_dim(), 251),
        key_weight: canonical_weights(shape.head_dim(), 277),
    }
}

fn upload_fused_inputs(
    backend: &mut CudaBackend,
    inputs: &FusedInputs,
) -> TestResult<(CudaBuffer, CudaBuffer, CudaBuffer, CudaBuffer, CudaBuffer)> {
    Ok((
        upload_f32(backend, &inputs.query)?,
        upload_f32(backend, &inputs.query_weight)?,
        upload_f32(backend, &inputs.key)?,
        upload_f32(backend, &inputs.key_weight)?,
        upload_f32(backend, &inputs.value)?,
    ))
}

fn allocate_fused_outputs(
    backend: &mut CudaBackend,
    shape: AttentionShape,
    query_shape: VectorShape,
    key_shape: VectorShape,
    positions: usize,
) -> TestResult<(CudaBuffer, CudaBuffer, CudaBuffer, CudaBuffer)> {
    let query_output = zero_buffer(
        backend,
        BufferLayout::f32(vector_elements(query_shape) * positions)?,
    )?;
    let key_output = zero_buffer(
        backend,
        BufferLayout::f32(vector_elements(key_shape) * positions)?,
    )?;
    let cache_layout = BufferLayout::f16(cache_elements(shape))?;
    let key_cache = zero_buffer(backend, cache_layout)?;
    let value_cache = zero_buffer(backend, cache_layout)?;
    Ok((query_output, key_output, key_cache, value_cache))
}

fn append_fused(case: &mut FusedCase) -> TestResult {
    let target = KvWriteSpan::new(
        &mut case.key_cache,
        &mut case.value_cache,
        0,
        case.shape.max_context(),
    )?;
    case.backend.verify_qk_norm_rope_kv_append_span(
        &case.query,
        &case.query_weight,
        &mut case.query_output,
        case.query_shape,
        &case.key,
        &case.key_weight,
        &mut case.key_output,
        case.key_shape,
        &case.value,
        target,
        case.shape,
        case.start_position,
        case.positions,
        EPSILON,
        THETA,
    )?;
    Ok(())
}

fn check_fused_qk(case: &mut FusedCase) -> TestResult {
    let actual_query = snapshot_f32(&mut case.backend, &case.query_output)?;
    let actual_key = snapshot_f32(&mut case.backend, &case.key_output)?;
    assert_close(
        "fused verifier query",
        &actual_query,
        &case.expected_query,
        5e-5,
        3e-4,
    );
    assert_close(
        "fused verifier key",
        &actual_key,
        &case.expected_key,
        5e-5,
        3e-4,
    );
    Ok(())
}

fn check_fused_cache(case: &mut FusedCase) -> TestResult {
    let actual_key = f16_bytes_values(&download_bytes(&mut case.backend, &case.key_cache)?);
    let expected_key = case
        .expected_key_cache
        .iter()
        .map(|value| f64::from(f16::from_f32(*value).to_f32()))
        .collect::<Vec<_>>();
    assert_close(
        "fused verifier key cache",
        &actual_key,
        &expected_key,
        2e-3,
        2e-3,
    );
    assert_eq!(
        case.backend.download_buffer(&case.value_cache)?.bytes(),
        f16_cache_bytes(&case.expected_value_cache, case.shape),
        "fused verifier value cache"
    );
    Ok(())
}

fn check_fused_attention(case: &mut FusedCase) -> TestResult {
    let mapped_tokens = case.start_position + case.positions;
    let span = KvReadSpan::new(
        &case.key_cache,
        &case.value_cache,
        0,
        mapped_tokens,
        case.shape.max_context(),
    )?;
    let spans = [span];
    let view = KvReadView::new(&spans)?;
    let mut output = case.backend.allocate(case.query_output_layout()?)?;
    case.backend.verify_attention_spans(
        &case.query_output,
        view,
        &mut output,
        case.shape,
        case.start_position,
        case.positions,
    )?;
    case.backend.synchronize()?;
    let actual = snapshot_f32(&mut case.backend, &output)?;
    let expected_query = expected_query_as_f32(&case.expected_query);
    let keys = logical_f16_values(&case.expected_key_cache, case.shape);
    let values = logical_f16_values(&case.expected_value_cache, case.shape);
    let expected = causal_oracle(
        &expected_query,
        &keys,
        &values,
        case.shape,
        case.start_position,
        case.positions,
        false,
    );
    assert_close("fused verifier attention", &actual, &expected, 7e-4, 5e-3);
    Ok(())
}

impl FusedCase {
    fn query_output_layout(&self) -> Result<BufferLayout, leone::BackendError> {
        BufferLayout::f32(vector_elements(self.query_shape) * self.positions)
    }
}

fn append_spans(
    backend: &mut CudaBackend,
    kind: CacheKind,
    shape: AttentionShape,
    keys: &[f32],
    values: &[f32],
) -> TestResult<SpanStorage> {
    let sources = upload_span_sources(backend, shape, keys, values)?;
    let mut storage = allocate_span_storage(backend, kind, shape)?;
    append_span_chunks(backend, shape, &sources, &mut storage)?;
    backend.synchronize()?;
    check_span_chunks(backend, kind, shape, keys, values, &storage)?;
    Ok(storage)
}

fn upload_span_sources(
    backend: &mut CudaBackend,
    shape: AttentionShape,
    keys: &[f32],
    values: &[f32],
) -> TestResult<SpanSources> {
    let projected = shape.projected_kv_elements()?;
    let mut key_inputs = Vec::new();
    let mut value_inputs = Vec::new();
    for spec in SPANS {
        let (start, end) = span_source_range(spec, projected);
        key_inputs.push(backend.upload(
            BufferLayout::f32(end - start)?,
            &f32_bytes(&keys[start..end]),
        )?);
        value_inputs.push(backend.upload(
            BufferLayout::f32(end - start)?,
            &f32_bytes(&values[start..end]),
        )?);
    }
    Ok(SpanSources {
        keys: key_inputs,
        values: value_inputs,
    })
}

fn allocate_span_storage(
    backend: &mut CudaBackend,
    kind: CacheKind,
    shape: AttentionShape,
) -> TestResult<SpanStorage> {
    let projected = shape.projected_kv_elements()?;
    let mut keys = Vec::new();
    let mut values = Vec::new();
    for spec in SPANS {
        let layout = cache_layout(kind, spec.capacity * projected)?;
        let initial = BufferSnapshot::new(layout, initial_span_bytes(kind, shape, spec.capacity))?;
        keys.push(backend.restore_buffer(&initial)?);
        values.push(backend.restore_buffer(&initial)?);
    }
    Ok(SpanStorage { keys, values })
}

fn append_span_chunks(
    backend: &mut CudaBackend,
    shape: AttentionShape,
    sources: &SpanSources,
    storage: &mut SpanStorage,
) -> TestResult {
    for (index, spec) in SPANS.iter().enumerate() {
        let target = KvWriteSpan::new(
            &mut storage.keys[index],
            &mut storage.values[index],
            spec.logical_start,
            spec.capacity,
        )?;
        backend.kv_append_chunk_span(
            &sources.keys[index],
            &sources.values[index],
            target,
            shape,
            spec.logical_start,
            spec.tokens,
        )?;
    }
    Ok(())
}

fn check_span_chunks(
    backend: &mut CudaBackend,
    kind: CacheKind,
    shape: AttentionShape,
    keys: &[f32],
    values: &[f32],
    storage: &SpanStorage,
) -> TestResult {
    let projected = shape.projected_kv_elements()?;
    for (index, spec) in SPANS.iter().enumerate() {
        let (start, end) = span_source_range(*spec, projected);
        let expected_keys =
            expected_span_bytes(kind, &keys[start..end], shape, spec.tokens, spec.capacity);
        let expected_values =
            expected_span_bytes(kind, &values[start..end], shape, spec.tokens, spec.capacity);
        assert_eq!(
            backend.download_buffer(&storage.keys[index])?.bytes(),
            expected_keys,
            "span key append {index}"
        );
        assert_eq!(
            backend.download_buffer(&storage.values[index])?.bytes(),
            expected_values,
            "span value append {index}"
        );
    }
    Ok(())
}

fn span_source_range(spec: SpanSpec, projected: usize) -> (usize, usize) {
    let start = spec.logical_start * projected;
    (start, start + spec.tokens * projected)
}

fn run_decode(
    backend: &mut CudaBackend,
    storage: &SpanStorage,
    shape: AttentionShape,
    query: &[f32],
) -> TestResult<Vec<f32>> {
    let query_buffer = backend.upload(BufferLayout::f32(query.len())?, &f32_bytes(query))?;
    let mut output = backend.allocate(BufferLayout::f32(query_elements(shape))?)?;
    let spans = read_spans(storage)?;
    let view = KvReadView::new(&spans)?;
    backend.attention_decode_spans(&query_buffer, view, &mut output, shape, Position::Host(8))?;
    backend.synchronize()?;
    snapshot_f32(backend, &output)
}

fn run_prefill(
    backend: &mut CudaBackend,
    storage: &SpanStorage,
    shape: AttentionShape,
    query: &[f32],
    start_position: usize,
    tokens: usize,
) -> TestResult<Vec<f32>> {
    prepare_prefill(backend, shape, tokens)?;
    run_prefill_operation(backend, storage, shape, query, start_position, tokens)
}

fn prepare_prefill(backend: &mut CudaBackend, shape: AttentionShape, tokens: usize) -> TestResult {
    let plan = PrefillPlan::new(
        tokens,
        shape.max_context(),
        shape.n_head(),
        shape.n_head_kv(),
        shape.head_dim(),
        query_elements(shape),
        query_elements(shape),
        tokens,
    )?;
    backend.prepare_prefill(plan)?;
    Ok(())
}

fn run_prefill_operation(
    backend: &mut CudaBackend,
    storage: &SpanStorage,
    shape: AttentionShape,
    query: &[f32],
    start_position: usize,
    tokens: usize,
) -> TestResult<Vec<f32>> {
    let query_buffer = backend.upload(BufferLayout::f32(query.len())?, &f32_bytes(query))?;
    let mut output = backend.allocate(BufferLayout::f32(query.len())?)?;
    let spans = read_spans(storage)?;
    let view = KvReadView::new(&spans)?;
    backend.attention_prefill_spans(
        &query_buffer,
        view,
        &mut output,
        shape,
        start_position,
        tokens,
    )?;
    backend.synchronize()?;
    snapshot_f32(backend, &output)
}

fn run_verify_attention(
    backend: &mut CudaBackend,
    storage: &SpanStorage,
    shape: AttentionShape,
    query: &[f32],
    start_position: usize,
    positions: usize,
) -> TestResult<Vec<f32>> {
    let query_buffer = backend.upload(BufferLayout::f32(query.len())?, &f32_bytes(query))?;
    let mut output = backend.allocate(BufferLayout::f32(query.len())?)?;
    let spans = read_spans(storage)?;
    let view = KvReadView::new(&spans)?;
    backend.verify_attention_spans(
        &query_buffer,
        view,
        &mut output,
        shape,
        start_position,
        positions,
    )?;
    backend.synchronize()?;
    snapshot_f32(backend, &output)
}

fn read_spans<'a>(storage: &'a SpanStorage) -> TestResult<Vec<KvReadSpan<'a, CudaBuffer>>> {
    SPANS
        .iter()
        .enumerate()
        .map(|(index, spec)| {
            Ok(KvReadSpan::new(
                &storage.keys[index],
                &storage.values[index],
                spec.logical_start,
                spec.tokens,
                spec.capacity,
            )?)
        })
        .collect()
}

fn reject_wrong_qk_shape(case: &mut FusedCase) -> TestResult {
    let mut bad = bad_qk_buffers(case)?;
    let before = bad_qk_snapshots(&mut case.backend, &case.key_cache, &case.value_cache, &bad)?;
    let result = launch_bad_qk(case, &mut bad);
    assert!(result.is_err(), "wrong QK columns must be rejected");
    case.backend.synchronize()?;
    assert_bad_qk_unchanged(
        &mut case.backend,
        &case.key_cache,
        &case.value_cache,
        &bad,
        before,
    )?;
    Ok(())
}

fn bad_qk_buffers(case: &mut FusedCase) -> TestResult<BadQkBuffers> {
    let key_shape = VectorShape::new(case.shape.n_head_kv(), case.shape.head_dim() / 2)?;
    let key_values = canonical_values(case.positions * vector_elements(key_shape), 307);
    let value_values = canonical_values(case.positions * vector_elements(key_shape), 331);
    let weight_values = canonical_weights(key_shape.columns(), 353);
    let key = upload_f32(&mut case.backend, &key_values)?;
    let value = upload_f32(&mut case.backend, &value_values)?;
    let weight = upload_f32(&mut case.backend, &weight_values)?;
    let query_output = zero_f32(
        &mut case.backend,
        vector_elements(case.query_shape) * case.positions,
    )?;
    let key_output = zero_f32(
        &mut case.backend,
        vector_elements(key_shape) * case.positions,
    )?;
    Ok(BadQkBuffers {
        key_shape,
        key,
        weight,
        value,
        query_output,
        key_output,
    })
}

fn launch_bad_qk(case: &mut FusedCase, bad: &mut BadQkBuffers) -> Result<(), leone::BackendError> {
    let target = KvWriteSpan::new(
        &mut case.key_cache,
        &mut case.value_cache,
        0,
        case.shape.max_context(),
    )?;
    case.backend.verify_qk_norm_rope_kv_append_span(
        &case.query,
        &case.query_weight,
        &mut bad.query_output,
        case.query_shape,
        &bad.key,
        &bad.weight,
        &mut bad.key_output,
        bad.key_shape,
        &bad.value,
        target,
        case.shape,
        case.start_position,
        case.positions,
        EPSILON,
        THETA,
    )
}

fn bad_qk_snapshots(
    backend: &mut CudaBackend,
    key_cache: &CudaBuffer,
    value_cache: &CudaBuffer,
    bad: &BadQkBuffers,
) -> TestResult<BadQkSnapshots> {
    Ok(BadQkSnapshots {
        key: download_bytes(backend, key_cache)?,
        value: download_bytes(backend, value_cache)?,
        query_output: download_bytes(backend, &bad.query_output)?,
        key_output: download_bytes(backend, &bad.key_output)?,
    })
}

fn assert_bad_qk_unchanged(
    backend: &mut CudaBackend,
    key_cache: &CudaBuffer,
    value_cache: &CudaBuffer,
    bad: &BadQkBuffers,
    before: BadQkSnapshots,
) -> TestResult {
    assert_eq!(before.key, download_bytes(backend, key_cache)?);
    assert_eq!(before.value, download_bytes(backend, value_cache)?);
    assert_eq!(
        before.query_output,
        download_bytes(backend, &bad.query_output)?
    );
    assert_eq!(before.key_output, download_bytes(backend, &bad.key_output)?);
    Ok(())
}

fn attention_shape(head_dim: usize) -> TestResult<AttentionShape> {
    Ok(AttentionShape::new(4, 2, head_dim, SHAPE_MAX_CONTEXT)?)
}

fn query_elements(shape: AttentionShape) -> usize {
    shape.query_elements().expect("valid attention shape")
}

fn cache_elements(shape: AttentionShape) -> usize {
    shape.cache_elements().expect("valid attention shape")
}

fn vector_elements(shape: VectorShape) -> usize {
    shape.elements().expect("valid vector shape")
}

fn cache_layout(kind: CacheKind, elements: usize) -> TestResult<BufferLayout> {
    Ok(match kind {
        CacheKind::F16 => BufferLayout::f16(elements)?,
        CacheKind::Q8 => BufferLayout::q8_kv(elements)?,
    })
}

fn expected_span_bytes(
    kind: CacheKind,
    values: &[f32],
    shape: AttentionShape,
    tokens: usize,
    capacity: usize,
) -> Vec<u8> {
    let projected = shape.n_head_kv() * shape.head_dim();
    let row_bytes = cache_row_bytes(kind, shape.head_dim());
    let mut output = initial_span_bytes(kind, shape, capacity);
    for head in 0..shape.n_head_kv() {
        for token in 0..tokens {
            let source = (token * projected + head * shape.head_dim())
                ..(token * projected + (head + 1) * shape.head_dim());
            let destination = match kind {
                CacheKind::F16 | CacheKind::Q8 => (head * capacity + token) * row_bytes,
            };
            let row = &values[source];
            match kind {
                CacheKind::F16 => write_f16_row(&mut output[destination..], row),
                CacheKind::Q8 => {
                    output[destination..destination + row_bytes].copy_from_slice(&q8_row(row))
                }
            }
        }
    }
    output
}

fn initial_span_bytes(kind: CacheKind, shape: AttentionShape, capacity: usize) -> Vec<u8> {
    let bytes = capacity * shape.n_head_kv() * cache_row_bytes(kind, shape.head_dim());
    vec![0xa5_u8; bytes]
}

fn cache_row_bytes(kind: CacheKind, head_dim: usize) -> usize {
    match kind {
        CacheKind::F16 => head_dim * 2,
        CacheKind::Q8 => head_dim / Q8_BLOCK_ELEMENTS * Q8_BLOCK_BYTES,
    }
}

fn logical_cache(kind: CacheKind, values: &[f32], shape: AttentionShape) -> Vec<f32> {
    let projected = shape.n_head_kv() * shape.head_dim();
    let mut output = vec![0.0; cache_elements(shape)];
    for position in 0..values.len() / projected {
        for head in 0..shape.n_head_kv() {
            let source = position * projected + head * shape.head_dim();
            let destination = (head * shape.max_context() + position) * shape.head_dim();
            let row = &values[source..source + shape.head_dim()];
            let decoded = cache_row(kind, row);
            output[destination..destination + shape.head_dim()].copy_from_slice(&decoded);
        }
    }
    output
}

fn cache_row(kind: CacheKind, row: &[f32]) -> Vec<f32> {
    match kind {
        CacheKind::F16 => row
            .iter()
            .map(|value| f16::from_f32(*value).to_f32())
            .collect(),
        CacheKind::Q8 => q8_row(row)
            .chunks_exact(Q8_BLOCK_BYTES)
            .flat_map(q8_decode_row)
            .collect(),
    }
}

fn q8_row(values: &[f32]) -> Vec<u8> {
    values
        .chunks_exact(Q8_BLOCK_ELEMENTS)
        .flat_map(q8_block)
        .collect()
}

fn q8_block(values: &[f32]) -> Vec<u8> {
    let maximum = values
        .iter()
        .fold(0.0_f32, |current, value| current.max(value.abs()));
    let scale = f16::from_f32(maximum / 127.0);
    let stored_scale = scale.to_f32();
    let mut output = vec![0_u8; Q8_BLOCK_BYTES];
    output[..2].copy_from_slice(&scale.to_bits().to_le_bytes());
    for (code, value) in output[2..].iter_mut().zip(values) {
        let quantized = if stored_scale == 0.0 {
            0
        } else {
            (*value / stored_scale)
                .round_ties_even()
                .clamp(-127.0, 127.0) as i8
        };
        *code = quantized as u8;
    }
    output
}

fn q8_decode_row(encoded: &[u8]) -> Vec<f32> {
    let scale = f16::from_bits(u16::from_le_bytes([encoded[0], encoded[1]])).to_f32();
    encoded[2..]
        .iter()
        .map(|code| scale * f32::from(*code as i8))
        .collect()
}

fn f16_cache_bytes(values: &[f32], shape: AttentionShape) -> Vec<u8> {
    let mut output = vec![0_u8; cache_elements(shape) * 2];
    for (index, value) in values.iter().enumerate() {
        let start = index * 2;
        output[start..start + 2].copy_from_slice(&f16::from_f32(*value).to_bits().to_le_bytes());
    }
    output
}

fn fused_cache_values(
    rows: &[f64],
    row_shape: VectorShape,
    start_position: usize,
    shape: AttentionShape,
) -> Vec<f32> {
    let mut output = vec![0.0; cache_elements(shape)];
    let row_elements = vector_elements(row_shape);
    for token in 0..rows.len() / row_elements {
        for head in 0..row_shape.rows() {
            let source = token * row_elements + head * row_shape.columns();
            let destination =
                (head * shape.max_context() + start_position + token) * shape.head_dim();
            for dimension in 0..row_shape.columns() {
                output[destination + dimension] = rows[source + dimension] as f32;
            }
        }
    }
    output
}

fn fused_cache_values_f32(
    rows: &[f32],
    row_shape: VectorShape,
    start_position: usize,
    shape: AttentionShape,
) -> Vec<f32> {
    let rows = rows
        .iter()
        .map(|value| f64::from(*value))
        .collect::<Vec<_>>();
    fused_cache_values(&rows, row_shape, start_position, shape)
}

fn logical_f16_values(values: &[f32], shape: AttentionShape) -> Vec<f32> {
    values
        .iter()
        .map(|value| f16::from_f32(*value).to_f32())
        .take(cache_elements(shape))
        .collect()
}

fn f16_bytes_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
        .collect()
}

fn rms_rope_rows(
    input: &[f32],
    weight: &[f32],
    shape: VectorShape,
    start_position: usize,
) -> Vec<f64> {
    let elements = vector_elements(shape);
    let tokens = input.len() / elements;
    let mut output = vec![0.0; input.len()];
    let half = shape.columns() / 2;
    for token in 0..tokens {
        let input_base = token * elements;
        let normalized = rms_rows(&input[input_base..input_base + elements], weight, shape);
        for head in 0..shape.rows() {
            let base = input_base + head * shape.columns();
            for pair in 0..half {
                let angle = (start_position + token) as f64
                    * f64::from(THETA).powf(-2.0 * pair as f64 / shape.columns() as f64);
                let (sine, cosine) = angle.sin_cos();
                let normalized_base = head * shape.columns();
                let first = normalized[normalized_base + pair];
                let second = normalized[normalized_base + pair + half];
                output[base + pair] = first * cosine - second * sine;
                output[base + pair + half] = first * sine + second * cosine;
            }
        }
    }
    output
}

fn rms_rows(input: &[f32], weight: &[f32], shape: VectorShape) -> Vec<f64> {
    let mut output = vec![0.0; vector_elements(shape)];
    for row in 0..shape.rows() {
        let base = row * shape.columns();
        let square_sum = (0..shape.columns())
            .map(|column| {
                let value = f64::from(input[base + column]);
                value * value
            })
            .sum::<f64>();
        let inverse = 1.0 / (square_sum / shape.columns() as f64 + f64::from(EPSILON)).sqrt();
        for column in 0..shape.columns() {
            output[base + column] =
                f64::from(input[base + column]) * f64::from(weight[column]) * inverse;
        }
    }
    output
}

fn expected_query_as_f32(values: &[f64]) -> Vec<f32> {
    values.iter().map(|value| *value as f32).collect()
}

fn causal_oracle(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    shape: AttentionShape,
    start_position: usize,
    tokens: usize,
    round_query: bool,
) -> Vec<f64> {
    let query_elements = query_elements(shape);
    let mut output = Vec::with_capacity(tokens * query_elements);
    for token in 0..tokens {
        let base = token * query_elements;
        let row = &query[base..base + query_elements];
        let row = if round_query {
            row.iter()
                .map(|value| f16::from_f32(*value).to_f32())
                .collect()
        } else {
            row.to_vec()
        };
        output.extend(attention_oracle(
            &row,
            keys,
            values,
            shape,
            start_position + token + 1,
        ));
    }
    output
}

fn attention_oracle(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    shape: AttentionShape,
    context_length: usize,
) -> Vec<f64> {
    let mut output = vec![0.0; query_elements(shape)];
    let group_size = shape.n_head() / shape.n_head_kv();
    let scale = 1.0 / (shape.head_dim() as f64).sqrt();
    for query_head in 0..shape.n_head() {
        let kv_head = query_head / group_size;
        let scores = (0..context_length)
            .map(|position| {
                let base = (kv_head * shape.max_context() + position) * shape.head_dim();
                query[query_head * shape.head_dim()..(query_head + 1) * shape.head_dim()]
                    .iter()
                    .zip(&keys[base..base + shape.head_dim()])
                    .map(|(left, right)| f64::from(*left) * f64::from(*right))
                    .sum::<f64>()
                    * scale
            })
            .collect::<Vec<_>>();
        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights = scores
            .iter()
            .map(|score| (score - maximum).exp())
            .collect::<Vec<_>>();
        let denominator = weights.iter().sum::<f64>();
        for dimension in 0..shape.head_dim() {
            output[query_head * shape.head_dim() + dimension] = weights
                .iter()
                .enumerate()
                .map(|(position, weight)| {
                    let index =
                        (kv_head * shape.max_context() + position) * shape.head_dim() + dimension;
                    weight * f64::from(values[index])
                })
                .sum::<f64>()
                / denominator;
        }
    }
    output
}

fn canonical_values(elements: usize, seed: usize) -> Vec<f32> {
    (0..elements)
        .map(|index| {
            let value = ((index.wrapping_mul(37).wrapping_add(seed)) % 101) as f32;
            value * 0.013 - 0.65
        })
        .collect()
}

fn canonical_weights(elements: usize, seed: usize) -> Vec<f32> {
    (0..elements)
        .map(|index| 0.7 + ((index.wrapping_mul(19).wrapping_add(seed)) % 31) as f32 * 0.017)
        .collect()
}

fn zero_buffer(backend: &mut CudaBackend, layout: BufferLayout) -> TestResult<CudaBuffer> {
    Ok(backend.restore_buffer(&BufferSnapshot::new(layout, vec![0_u8; layout.bytes()])?)?)
}

fn upload_f32(backend: &mut CudaBackend, values: &[f32]) -> TestResult<CudaBuffer> {
    Ok(backend.upload(BufferLayout::f32(values.len())?, &f32_bytes(values))?)
}

fn zero_f32(backend: &mut CudaBackend, elements: usize) -> TestResult<CudaBuffer> {
    zero_buffer(backend, BufferLayout::f32(elements)?)
}

fn write_f16_row(destination: &mut [u8], values: &[f32]) {
    for (index, value) in values.iter().enumerate() {
        let start = index * 2;
        destination[start..start + 2]
            .copy_from_slice(&f16::from_f32(*value).to_bits().to_le_bytes());
    }
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn snapshot_f32(backend: &mut CudaBackend, buffer: &CudaBuffer) -> TestResult<Vec<f32>> {
    Ok(backend
        .download_buffer(buffer)?
        .bytes()
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect())
}

fn download_bytes(backend: &mut CudaBackend, buffer: &CudaBuffer) -> TestResult<Vec<u8>> {
    Ok(backend.download_buffer(buffer)?.bytes().to_vec())
}

fn assert_close(name: &str, actual: &[f32], expected: &[f64], absolute: f64, relative: f64) {
    assert_eq!(actual.len(), expected.len(), "{name} length");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let error = (f64::from(*actual) - expected).abs();
        let bound = absolute + relative * expected.abs().max(1.0);
        assert!(
            error <= bound,
            "{name} index {index}: actual={actual}, expected={expected}, error={error}, bound={bound}"
        );
    }
}
