#![cfg(target_os = "macos")]

use half::f16;
use leone::{
    AttentionDecodeRow, AttentionShape, Backend, BufferLayout, KvReadSpan, KvReadView, Position,
};
use leone_metal::{MetalAttentionPath, MetalBackend};

const ABSOLUTE_TOLERANCE: f32 = 5e-3;
const RELATIVE_TOLERANCE: f32 = 5e-3;

#[test]
#[ignore = "requires a Metal device"]
fn compiled_shader_matches_build_identity() {
    let backend = MetalBackend::new().expect("Metal backend initializes");
    let metadata = backend.device_metadata().expect("Metal device metadata");
    assert_eq!(
        metadata.shader_source_hash,
        leone_metal::shader_identity().sha256
    );
}

#[test]
#[ignore = "requires a Metal device"]
fn four_paths_match_fp64_oracle_and_fixed_path_repeats() {
    let shape = AttentionShape::new(4, 2, 8, 37).expect("valid shape");
    let positions = [36, 31, 7];
    let queries = make_queries(positions.len(), shape.query_elements().unwrap());
    let keys = make_cache(shape, 0x91);
    let values = make_cache(shape, 0x37);
    let oracle = queries
        .iter()
        .zip(positions)
        .map(|(query, position)| fp64_oracle(query, &keys, &values, shape, position))
        .collect::<Vec<_>>();

    for path in [
        MetalAttentionPath::PerRow,
        MetalAttentionPath::FixedTilePerRow,
        MetalAttentionPath::SharedReadUnconstrained,
        MetalAttentionPath::SharedReadFixedReduction,
    ] {
        let actual = run_path(path, shape, &queries, &keys, &values, &positions)
            .expect("Metal path completes");
        assert_matches_oracle(path, &actual, &oracle);
    }

    let first = run_path(
        MetalAttentionPath::SharedReadFixedReduction,
        shape,
        &queries,
        &keys,
        &values,
        &positions,
    )
    .expect("first fixed reduction run");
    let second = run_path(
        MetalAttentionPath::SharedReadFixedReduction,
        shape,
        &queries,
        &keys,
        &values,
        &positions,
    )
    .expect("second fixed reduction run");
    assert_eq!(flatten_bits(&first), flatten_bits(&second));
}

#[test]
#[ignore = "requires a Metal device"]
fn four_paths_group_heterogeneous_context_shapes() {
    let shape_a = AttentionShape::new(4, 2, 8, 37).expect("valid first shape");
    let shape_b = AttentionShape::new(4, 2, 8, 41).expect("valid second shape");
    let positions = [36, 8];
    let queries = make_queries(positions.len(), shape_a.query_elements().unwrap());
    let oracle = [
        fp64_oracle(
            &queries[0],
            &make_cache(shape_a, 0x43),
            &make_cache(shape_a, 0x77),
            shape_a,
            positions[0],
        ),
        fp64_oracle(
            &queries[1],
            &make_cache(shape_b, 0x43),
            &make_cache(shape_b, 0x77),
            shape_b,
            positions[1],
        ),
    ];
    for path in [
        MetalAttentionPath::PerRow,
        MetalAttentionPath::FixedTilePerRow,
        MetalAttentionPath::SharedReadUnconstrained,
        MetalAttentionPath::SharedReadFixedReduction,
    ] {
        let actual = run_heterogeneous_path(path, shape_a, shape_b, &queries, &positions)
            .expect("Metal heterogeneous path completes");
        assert_matches_oracle(path, &actual, &oracle);
    }
}

#[test]
#[ignore = "requires a Metal device"]
fn four_paths_share_prefix_before_divergent_tails() {
    let shape = AttentionShape::new(4, 2, 8, 37).expect("valid shape");
    let positions = [36, 34];
    let queries = make_queries(positions.len(), shape.query_elements().unwrap());
    let shared_keys = make_segment_cache(shape, 32, 0x11);
    let shared_values = make_segment_cache(shape, 32, 0x31);
    let tail_a_keys = make_segment_cache(shape, 8, 0x51);
    let tail_a_values = make_segment_cache(shape, 8, 0x71);
    let tail_b_keys = make_segment_cache(shape, 8, 0x91);
    let tail_b_values = make_segment_cache(shape, 8, 0xb1);
    let full_a_keys = expand_branch_cache(shape, &shared_keys, &tail_a_keys, 5);
    let full_a_values = expand_branch_cache(shape, &shared_values, &tail_a_values, 5);
    let full_b_keys = expand_branch_cache(shape, &shared_keys, &tail_b_keys, 3);
    let full_b_values = expand_branch_cache(shape, &shared_values, &tail_b_values, 3);
    let oracle = vec![
        fp64_oracle(
            &queries[0],
            &full_a_keys,
            &full_a_values,
            shape,
            positions[0],
        ),
        fp64_oracle(
            &queries[1],
            &full_b_keys,
            &full_b_values,
            shape,
            positions[1],
        ),
    ];
    for path in [
        MetalAttentionPath::PerRow,
        MetalAttentionPath::FixedTilePerRow,
        MetalAttentionPath::SharedReadUnconstrained,
        MetalAttentionPath::SharedReadFixedReduction,
    ] {
        let actual = run_divergent_path(
            path,
            shape,
            &queries,
            &shared_keys,
            &shared_values,
            &tail_a_keys,
            &tail_a_values,
            &tail_b_keys,
            &tail_b_values,
            &positions,
        )
        .expect("Metal divergent-tail path completes");
        assert_matches_oracle(path, &actual, &oracle);
    }
}

#[test]
#[ignore = "requires a Metal device"]
fn failed_decode_retirement_clears_prepared_layer_plans() {
    let shape = AttentionShape::new(2, 1, 8, 33).expect("valid shape");
    let queries = make_queries(1, shape.query_elements().expect("valid query shape"));
    let mut backend = MetalBackend::new().expect("Metal initialization");
    let (key, value) = upload_generated_cache(&mut backend, shape).expect("cache upload");
    let span = KvReadSpan::new(&key, &value, 0, shape.max_context(), shape.max_context())
        .expect("valid cache span");
    let cache = KvReadView::new(std::slice::from_ref(&span)).expect("valid cache view");
    let (query_buffers, mut output_buffers) =
        make_buffers(&mut backend, &queries).expect("query buffers");
    {
        let mut rows = make_rows(&query_buffers, &mut output_buffers, cache, shape, &[32]);
        backend
            .prepare_attention_decode_batch_spans(&rows)
            .expect("first layer preparation");
        backend
            .prepare_attention_decode_batch_spans(&rows)
            .expect("second layer preparation");
        backend
            .attention_decode_batch_spans(&mut rows)
            .expect("first layer dispatch");
        let mut bad_output = backend
            .allocate(
                BufferLayout::f32(shape.query_elements().expect("query size"))
                    .expect("valid output layout"),
            )
            .expect("failure probe output");
        let error = backend
            .copy_f32_row(
                &query_buffers[0],
                1,
                shape.query_elements().expect("query size"),
                &mut bad_output,
            )
            .expect_err("failed backend work must be retired");
        assert!(matches!(error, leone::BackendError::RowOutOfBounds { .. }));
    }
    backend.drop_decode_graph().expect("retire failed decode");

    let (fresh_key, fresh_value) =
        upload_generated_cache(&mut backend, shape).expect("fresh cache upload");
    let fresh_span = KvReadSpan::new(
        &fresh_key,
        &fresh_value,
        0,
        shape.max_context(),
        shape.max_context(),
    )
    .expect("fresh cache span");
    let fresh_cache = KvReadView::new(std::slice::from_ref(&fresh_span)).expect("fresh cache view");
    let (fresh_queries, mut fresh_outputs) =
        make_buffers(&mut backend, &queries).expect("fresh query buffers");
    let mut fresh_rows = make_rows(
        &fresh_queries,
        &mut fresh_outputs,
        fresh_cache,
        shape,
        &[32],
    );
    execute_rows(&mut backend, &mut fresh_rows).expect("fresh batch after retirement");
}

fn run_path(
    path: MetalAttentionPath,
    shape: AttentionShape,
    queries: &[Vec<f32>],
    keys: &[f16],
    values: &[f16],
    positions: &[usize],
) -> Result<Vec<Vec<f32>>, leone::BackendError> {
    let mut backend = MetalBackend::new()?;
    backend.set_attention_path(path);
    let (key, value) = upload_cache(&mut backend, keys, values)?;
    let span = KvReadSpan::new(&key, &value, 0, shape.max_context(), shape.max_context())?;
    let cache = KvReadView::new(std::slice::from_ref(&span))?;
    let (query_buffers, mut output_buffers) = make_buffers(&mut backend, queries)?;
    let mut rows = make_rows(&query_buffers, &mut output_buffers, cache, shape, positions);
    execute_rows(&mut backend, &mut rows)
}

fn run_heterogeneous_path(
    path: MetalAttentionPath,
    shape_a: AttentionShape,
    shape_b: AttentionShape,
    queries: &[Vec<f32>],
    positions: &[usize],
) -> Result<Vec<Vec<f32>>, leone::BackendError> {
    let mut backend = MetalBackend::new()?;
    backend.set_attention_path(path);
    let (key_a, value_a) = upload_generated_cache(&mut backend, shape_a)?;
    let (key_b, value_b) = upload_generated_cache(&mut backend, shape_b)?;
    let span_a = KvReadSpan::new(
        &key_a,
        &value_a,
        0,
        shape_a.max_context(),
        shape_a.max_context(),
    )?;
    let span_b = KvReadSpan::new(
        &key_b,
        &value_b,
        0,
        shape_b.max_context(),
        shape_b.max_context(),
    )?;
    let cache_a = KvReadView::new(std::slice::from_ref(&span_a))?;
    let cache_b = KvReadView::new(std::slice::from_ref(&span_b))?;
    let (query_buffers, mut output_buffers) = make_buffers(&mut backend, queries)?;
    let (first_output, remaining_outputs) = output_buffers.split_at_mut(1);
    let second_output = &mut remaining_outputs[0];
    let mut rows = vec![
        AttentionDecodeRow {
            query: &query_buffers[0],
            cache: cache_a,
            output: &mut first_output[0],
            shape: shape_a,
            position: Position::Host(positions[0]),
        },
        AttentionDecodeRow {
            query: &query_buffers[1],
            cache: cache_b,
            output: second_output,
            shape: shape_b,
            position: Position::Host(positions[1]),
        },
    ];
    execute_rows(&mut backend, &mut rows)
}

#[allow(clippy::too_many_arguments)]
fn run_divergent_path(
    path: MetalAttentionPath,
    shape: AttentionShape,
    queries: &[Vec<f32>],
    shared_keys: &[f16],
    shared_values: &[f16],
    tail_a_keys: &[f16],
    tail_a_values: &[f16],
    tail_b_keys: &[f16],
    tail_b_values: &[f16],
    positions: &[usize],
) -> Result<Vec<Vec<f32>>, leone::BackendError> {
    let mut backend = MetalBackend::new()?;
    backend.set_attention_path(path);
    let caches = upload_divergent_caches(
        &mut backend,
        shared_keys,
        shared_values,
        tail_a_keys,
        tail_a_values,
        tail_b_keys,
        tail_b_values,
    )?;
    execute_divergent_rows(&mut backend, queries, shape, positions, caches)
}

#[allow(clippy::too_many_arguments)]
fn upload_divergent_caches(
    backend: &mut MetalBackend,
    shared_keys: &[f16],
    shared_values: &[f16],
    tail_a_keys: &[f16],
    tail_a_values: &[f16],
    tail_b_keys: &[f16],
    tail_b_values: &[f16],
) -> Result<
    (
        leone_metal::MetalBuffer,
        leone_metal::MetalBuffer,
        leone_metal::MetalBuffer,
        leone_metal::MetalBuffer,
        leone_metal::MetalBuffer,
        leone_metal::MetalBuffer,
    ),
    leone::BackendError,
> {
    let (shared_key, shared_value) = upload_cache(backend, shared_keys, shared_values)?;
    let (tail_a_key, tail_a_value) = upload_cache(backend, tail_a_keys, tail_a_values)?;
    let (tail_b_key, tail_b_value) = upload_cache(backend, tail_b_keys, tail_b_values)?;
    Ok((
        shared_key,
        shared_value,
        tail_a_key,
        tail_a_value,
        tail_b_key,
        tail_b_value,
    ))
}

#[allow(clippy::too_many_arguments)]
fn execute_divergent_rows(
    backend: &mut MetalBackend,
    queries: &[Vec<f32>],
    shape: AttentionShape,
    positions: &[usize],
    caches: (
        leone_metal::MetalBuffer,
        leone_metal::MetalBuffer,
        leone_metal::MetalBuffer,
        leone_metal::MetalBuffer,
        leone_metal::MetalBuffer,
        leone_metal::MetalBuffer,
    ),
) -> Result<Vec<Vec<f32>>, leone::BackendError> {
    let (shared_key, shared_value, tail_a_key, tail_a_value, tail_b_key, tail_b_value) = caches;
    let shared_span = KvReadSpan::new(&shared_key, &shared_value, 0, 32, 32)?;
    let tail_a_span = KvReadSpan::new(&tail_a_key, &tail_a_value, 32, 5, 8)?;
    let tail_b_span = KvReadSpan::new(&tail_b_key, &tail_b_value, 32, 3, 8)?;
    let spans_a = [shared_span, tail_a_span];
    let spans_b = [shared_span, tail_b_span];
    let cache_a = KvReadView::new(&spans_a)?;
    let cache_b = KvReadView::new(&spans_b)?;
    let (query_buffers, mut output_buffers) = make_buffers(backend, queries)?;
    let (first_output, remaining_outputs) = output_buffers.split_at_mut(1);
    let second_output = &mut remaining_outputs[0];
    let mut rows = vec![
        AttentionDecodeRow {
            query: &query_buffers[0],
            cache: cache_a,
            output: &mut first_output[0],
            shape,
            position: Position::Host(positions[0]),
        },
        AttentionDecodeRow {
            query: &query_buffers[1],
            cache: cache_b,
            output: second_output,
            shape,
            position: Position::Host(positions[1]),
        },
    ];
    execute_rows(backend, &mut rows)
}

fn execute_rows(
    backend: &mut MetalBackend,
    rows: &mut [AttentionDecodeRow<'_, leone_metal::MetalBuffer>],
) -> Result<Vec<Vec<f32>>, leone::BackendError> {
    backend.prepare_attention_decode_batch_spans(rows)?;
    backend.attention_decode_batch_spans(rows)?;
    rows.iter()
        .map(|row| read_f32(backend, row.output))
        .collect()
}

fn upload_cache(
    backend: &mut MetalBackend,
    keys: &[f16],
    values: &[f16],
) -> Result<(leone_metal::MetalBuffer, leone_metal::MetalBuffer), leone::BackendError> {
    let key = backend.upload(BufferLayout::f16(keys.len())?, &f16_bytes(keys))?;
    let value = backend.upload(BufferLayout::f16(values.len())?, &f16_bytes(values))?;
    Ok((key, value))
}

fn upload_generated_cache(
    backend: &mut MetalBackend,
    shape: AttentionShape,
) -> Result<(leone_metal::MetalBuffer, leone_metal::MetalBuffer), leone::BackendError> {
    let keys = make_cache(shape, 0x43);
    let values = make_cache(shape, 0x77);
    upload_cache(backend, &keys, &values)
}

fn make_buffers(
    backend: &mut MetalBackend,
    queries: &[Vec<f32>],
) -> Result<(Vec<leone_metal::MetalBuffer>, Vec<leone_metal::MetalBuffer>), leone::BackendError> {
    let query_buffers = queries
        .iter()
        .map(|query| backend.upload(BufferLayout::f32(query.len())?, &f32_bytes(query)))
        .collect::<Result<Vec<_>, _>>()?;
    let output_buffers = queries
        .iter()
        .map(|query| backend.allocate(BufferLayout::f32(query.len()).expect("query layout")))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((query_buffers, output_buffers))
}

fn make_rows<'a>(
    query_buffers: &'a [leone_metal::MetalBuffer],
    output_buffers: &'a mut [leone_metal::MetalBuffer],
    cache: KvReadView<'a, leone_metal::MetalBuffer>,
    shape: AttentionShape,
    positions: &[usize],
) -> Vec<AttentionDecodeRow<'a, leone_metal::MetalBuffer>> {
    let mut rows = Vec::with_capacity(query_buffers.len());
    for ((query, output), position) in query_buffers
        .iter()
        .zip(output_buffers.iter_mut())
        .zip(positions.iter().copied())
    {
        rows.push(AttentionDecodeRow {
            query,
            cache,
            output,
            shape,
            position: Position::Host(position),
        });
    }
    rows
}

fn assert_matches_oracle(path: MetalAttentionPath, actual: &[Vec<f32>], oracle: &[Vec<f64>]) {
    for (row, (values, expected)) in actual.iter().zip(oracle).enumerate() {
        for (column, (&value, &reference)) in values.iter().zip(expected).enumerate() {
            let reference = reference as f32;
            let bound = ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * reference.abs();
            assert!(
                (value - reference).abs() <= bound,
                "{path:?} row {row} column {column}: {value} versus {reference}"
            );
        }
    }
}

fn fp64_oracle(
    query: &[f32],
    keys: &[f16],
    values: &[f16],
    shape: AttentionShape,
    position: usize,
) -> Vec<f64> {
    let mut output = vec![0.0; shape.query_elements().expect("valid query shape")];
    let scale = 1.0 / (shape.head_dim() as f64).sqrt();
    for query_head in 0..shape.n_head() {
        let kv_head = query_head / (shape.n_head() / shape.n_head_kv());
        let query_offset = query_head * shape.head_dim();
        let mut scores = Vec::with_capacity(position + 1);
        for token in 0..=position {
            let key_offset = (kv_head * shape.max_context() + token) * shape.head_dim();
            let dot = (0..shape.head_dim())
                .map(|index| {
                    query[query_offset + index] as f64
                        * f64::from(f32::from(keys[key_offset + index]))
                })
                .sum::<f64>();
            scores.push(dot * scale);
        }
        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights = scores
            .iter()
            .map(|score| (score - maximum).exp())
            .collect::<Vec<_>>();
        let denominator = weights.iter().sum::<f64>();
        for column in 0..shape.head_dim() {
            let value = weights
                .iter()
                .enumerate()
                .map(|(token, weight)| {
                    let offset = (kv_head * shape.max_context() + token) * shape.head_dim();
                    weight * f64::from(f32::from(values[offset + column]))
                })
                .sum::<f64>();
            output[query_offset + column] = value / denominator;
        }
    }
    output
}

fn make_queries(rows: usize, elements: usize) -> Vec<Vec<f32>> {
    (0..rows)
        .map(|row| {
            (0..elements)
                .map(|index| ((row * elements + index) % 29) as f32 / 17.0 - 0.8)
                .collect()
        })
        .collect()
}

fn make_cache(shape: AttentionShape, seed: usize) -> Vec<f16> {
    (0..shape.n_head_kv() * shape.max_context() * shape.head_dim())
        .map(|index| f16::from_f32(((index + seed) % 23) as f32 / 19.0 - 0.55))
        .collect()
}

fn make_segment_cache(shape: AttentionShape, tokens: usize, seed: usize) -> Vec<f16> {
    (0..shape.n_head_kv() * tokens * shape.head_dim())
        .map(|index| f16::from_f32(((index + seed) % 23) as f32 / 19.0 - 0.55))
        .collect()
}

fn expand_branch_cache(
    shape: AttentionShape,
    shared: &[f16],
    tail: &[f16],
    tail_tokens: usize,
) -> Vec<f16> {
    let mut expanded = vec![f16::from_f32(0.0); shape.cache_elements().unwrap()];
    let shared_tokens = 32;
    for head in 0..shape.n_head_kv() {
        let destination = head * shape.max_context() * shape.head_dim();
        let shared_source = head * shared_tokens * shape.head_dim();
        let tail_source = head * 8 * shape.head_dim();
        let shared_end = destination + shared_tokens * shape.head_dim();
        expanded[destination..shared_end].copy_from_slice(
            &shared[shared_source..shared_source + shared_tokens * shape.head_dim()],
        );
        let tail_end = destination + (shared_tokens + tail_tokens) * shape.head_dim();
        expanded[shared_end..tail_end]
            .copy_from_slice(&tail[tail_source..tail_source + tail_tokens * shape.head_dim()]);
    }
    expanded
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_bits().to_le_bytes())
        .collect()
}

fn f16_bytes(values: &[f16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_bits().to_le_bytes())
        .collect()
}

fn read_f32(
    backend: &mut MetalBackend,
    buffer: &leone_metal::MetalBuffer,
) -> Result<Vec<f32>, leone::BackendError> {
    let snapshot = backend.download_buffer(buffer)?;
    Ok(snapshot
        .bytes()
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("f32 bytes")))
        .collect())
}

fn flatten_bits(values: &[Vec<f32>]) -> Vec<u32> {
    values
        .iter()
        .flat_map(|row| row.iter().map(|value| value.to_bits()))
        .collect()
}
