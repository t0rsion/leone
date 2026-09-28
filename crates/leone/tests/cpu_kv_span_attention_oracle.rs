use half::f16;
use leone::{
    AttentionShape, Backend, BackendError, BufferLayout, BufferSnapshot, CpuBackend, CpuBuffer,
    KvReadSpan, KvReadView, Position,
};

const Q8_SCALE: f32 = 0.125;
const OUTPUT_TOLERANCE: f64 = 2.0e-5;

#[derive(Clone, Copy, Debug)]
enum CacheStorage {
    F32,
    F16,
    Q8,
}

impl CacheStorage {
    fn layout(self, shape: AttentionShape, capacity: usize) -> Result<BufferLayout, BackendError> {
        let elements = shape.n_head_kv() * capacity * shape.head_dim();
        match self {
            Self::F32 => BufferLayout::f32(elements),
            Self::F16 => BufferLayout::f16(elements),
            Self::Q8 => BufferLayout::q8_kv(elements),
        }
    }
}

struct SpanSpec {
    logical_start: usize,
    tokens: usize,
    capacity: usize,
}

struct EncodedSpan {
    key: CpuBuffer,
    value: CpuBuffer,
    key_bytes: Vec<u8>,
    value_bytes: Vec<u8>,
    storage: CacheStorage,
    logical_start: usize,
    tokens: usize,
    capacity: usize,
}

#[test]
fn cpu_span_attention_matches_independent_fp64_oracle() -> Result<(), BackendError> {
    let shape = AttentionShape::new(4, 2, 32, 12)?;
    let specs = [
        SpanSpec {
            logical_start: 0,
            tokens: 2,
            capacity: 4,
        },
        SpanSpec {
            logical_start: 2,
            tokens: 3,
            capacity: 5,
        },
        SpanSpec {
            logical_start: 5,
            tokens: 4,
            capacity: 7,
        },
    ];
    for storage in [CacheStorage::F32, CacheStorage::F16, CacheStorage::Q8] {
        run_storage_case(storage, shape, &specs)?;
    }
    Ok(())
}

#[test]
fn cpu_span_attention_does_not_map_pending_capacity() {
    let shape = AttentionShape::new(4, 2, 32, 12).unwrap();
    let specs = [
        SpanSpec {
            logical_start: 0,
            tokens: 2,
            capacity: 4,
        },
        SpanSpec {
            logical_start: 2,
            tokens: 3,
            capacity: 5,
        },
        SpanSpec {
            logical_start: 5,
            tokens: 4,
            capacity: 7,
        },
    ];
    let mut backend = CpuBackend::new();
    let spans = build_spans(&mut backend, CacheStorage::F32, shape, &specs).unwrap();
    let descriptors = read_spans(&spans).unwrap();
    let view = KvReadView::new(&descriptors).unwrap();
    assert_eq!(view.mapped_tokens(), 9);
    assert!(view.span_for_position(9).is_none());

    let query_bytes = encode_query(&query_values(9, 1, shape));
    let query = backend
        .upload(
            BufferLayout::f32(shape.query_elements().unwrap()).unwrap(),
            &query_bytes,
        )
        .unwrap();
    let mut output = backend
        .allocate(BufferLayout::f32(shape.query_elements().unwrap()).unwrap())
        .unwrap();
    let error = backend
        .attention_decode_spans(&query, view, &mut output, shape, Position::Host(9))
        .expect_err("capacity tail must remain unmapped");
    assert!(matches!(
        error,
        BackendError::PositionOutOfBounds {
            position: 10,
            max_context: 9
        }
    ));
}

fn run_storage_case(
    storage: CacheStorage,
    shape: AttentionShape,
    specs: &[SpanSpec],
) -> Result<(), BackendError> {
    let mut backend = CpuBackend::new();
    let spans = build_spans(&mut backend, storage, shape, specs)?;
    let descriptors = read_spans(&spans)?;
    let view = KvReadView::new(&descriptors)?;
    for &position in &[0, 1, 2, 4, 5, 8] {
        compare_decode(&mut backend, view, &spans, shape, position, storage)?;
    }
    compare_prefill(&mut backend, view, &spans, shape, 1, 8, storage)
}

fn compare_decode(
    backend: &mut CpuBackend,
    view: KvReadView<'_, CpuBuffer>,
    spans: &[EncodedSpan],
    shape: AttentionShape,
    position: usize,
    storage: CacheStorage,
) -> Result<(), BackendError> {
    let query_bytes = encode_query(&query_values(position, 1, shape));
    let query = backend.upload(BufferLayout::f32(shape.query_elements()?)?, &query_bytes)?;
    let mut output = backend.allocate(BufferLayout::f32(shape.query_elements()?)?)?;
    backend.attention_decode_spans(&query, view, &mut output, shape, Position::Host(position))?;
    let mut actual = vec![0.0_f32; shape.query_elements()?];
    backend.read_f32(&output, &mut actual)?;
    let expected = reference_decode(&query_bytes, spans, shape, position, storage);
    assert_output_close("decode", storage, position, &actual, &expected);
    Ok(())
}

fn compare_prefill(
    backend: &mut CpuBackend,
    view: KvReadView<'_, CpuBuffer>,
    spans: &[EncodedSpan],
    shape: AttentionShape,
    start_position: usize,
    tokens: usize,
    storage: CacheStorage,
) -> Result<(), BackendError> {
    let query_bytes = encode_query(&query_values(start_position, tokens, shape));
    let elements = tokens * shape.query_elements()?;
    let query = backend.upload(BufferLayout::f32(elements)?, &query_bytes)?;
    let mut output = backend.allocate(BufferLayout::f32(elements)?)?;
    backend.attention_prefill_spans(&query, view, &mut output, shape, start_position, tokens)?;
    let mut actual = vec![0.0_f32; elements];
    backend.read_f32(&output, &mut actual)?;
    let expected = reference_prefill(&query_bytes, spans, shape, start_position, tokens, storage);
    assert_output_close("prefill", storage, start_position, &actual, &expected);
    Ok(())
}

fn build_spans(
    backend: &mut CpuBackend,
    storage: CacheStorage,
    shape: AttentionShape,
    specs: &[SpanSpec],
) -> Result<Vec<EncodedSpan>, BackendError> {
    specs
        .iter()
        .map(|spec| {
            let key_bytes = encode_cache(storage, shape, spec, false);
            let value_bytes = encode_cache(storage, shape, spec, true);
            let layout = storage.layout(shape, spec.capacity)?;
            let key_snapshot = BufferSnapshot::new(layout, key_bytes.clone())?;
            let value_snapshot = BufferSnapshot::new(layout, value_bytes.clone())?;
            let key = backend.restore_buffer(&key_snapshot)?;
            let value = backend.restore_buffer(&value_snapshot)?;
            Ok(EncodedSpan {
                key,
                value,
                key_bytes,
                value_bytes,
                storage,
                logical_start: spec.logical_start,
                tokens: spec.tokens,
                capacity: spec.capacity,
            })
        })
        .collect()
}

fn read_spans<'a>(
    spans: &'a [EncodedSpan],
) -> Result<Vec<KvReadSpan<'a, CpuBuffer>>, BackendError> {
    spans
        .iter()
        .map(|span| {
            KvReadSpan::new(
                &span.key,
                &span.value,
                span.logical_start,
                span.tokens,
                span.capacity,
            )
        })
        .collect()
}

fn encode_cache(
    storage: CacheStorage,
    shape: AttentionShape,
    spec: &SpanSpec,
    value: bool,
) -> Vec<u8> {
    match storage {
        CacheStorage::F32 | CacheStorage::F16 => encode_dense(storage, shape, spec, value),
        CacheStorage::Q8 => encode_q8(shape, spec, value),
    }
}

fn encode_dense(
    storage: CacheStorage,
    shape: AttentionShape,
    spec: &SpanSpec,
    value: bool,
) -> Vec<u8> {
    let element_count = shape.n_head_kv() * spec.capacity * shape.head_dim();
    let element_bytes = match storage {
        CacheStorage::F32 => 4,
        CacheStorage::F16 => 2,
        CacheStorage::Q8 => unreachable!(),
    };
    let mut bytes = Vec::with_capacity(element_count * element_bytes);
    for head in 0..shape.n_head_kv() {
        for local in 0..spec.capacity {
            let mapped = local < spec.tokens;
            let position = spec.logical_start + local;
            for dimension in 0..shape.head_dim() {
                let sample = if mapped {
                    dense_value(value, head, position, dimension)
                } else {
                    pending_value(value)
                };
                append_dense(&mut bytes, storage, sample);
            }
        }
    }
    bytes
}

fn encode_q8(shape: AttentionShape, spec: &SpanSpec, value: bool) -> Vec<u8> {
    let row_count = shape.n_head_kv() * spec.capacity;
    let mut bytes = Vec::with_capacity(row_count * 34);
    for head in 0..shape.n_head_kv() {
        for local in 0..spec.capacity {
            let mapped = local < spec.tokens;
            let position = spec.logical_start + local;
            bytes.extend_from_slice(&f16::from_f32(Q8_SCALE).to_bits().to_le_bytes());
            for dimension in 0..shape.head_dim() {
                let code = if mapped {
                    q8_code(value, head, position, dimension)
                } else if value {
                    -127
                } else {
                    127
                };
                bytes.push(code as u8);
            }
        }
    }
    bytes
}

fn append_dense(bytes: &mut Vec<u8>, storage: CacheStorage, value: f64) {
    match storage {
        CacheStorage::F32 => bytes.extend_from_slice(&(value as f32).to_le_bytes()),
        CacheStorage::F16 => {
            bytes.extend_from_slice(&f16::from_f32(value as f32).to_bits().to_le_bytes())
        }
        CacheStorage::Q8 => unreachable!(),
    }
}

fn dense_value(value: bool, head: usize, position: usize, dimension: usize) -> f64 {
    let offset = if value { 5 } else { 0 };
    let seed = (head * 13 + position * 7 + dimension * 3 + offset) % 29;
    (seed as f64 - 14.0) * 0.0375
}

fn pending_value(value: bool) -> f64 {
    if value {
        -91.0
    } else {
        91.0
    }
}

fn q8_code(value: bool, head: usize, position: usize, dimension: usize) -> i8 {
    let offset = if value { 11 } else { 0 };
    let seed = (head * 3 + position * 5 + dimension * 7 + offset) % 15;
    let code = seed as i8 - 7;
    if value {
        code
    } else {
        -code
    }
}

fn query_values(start_position: usize, tokens: usize, shape: AttentionShape) -> Vec<f64> {
    let mut values = Vec::with_capacity(tokens * shape.query_elements().unwrap());
    for token in 0..tokens {
        let position = start_position + token;
        for head in 0..shape.n_head() {
            for dimension in 0..shape.head_dim() {
                let seed = (position * 11 + head * 5 + dimension * 3) % 23;
                values.push((seed as f64 - 11.0) * 0.041);
            }
        }
    }
    values
}

fn encode_query(values: &[f64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| (*value as f32).to_le_bytes())
        .collect()
}

fn reference_decode(
    query_bytes: &[u8],
    spans: &[EncodedSpan],
    shape: AttentionShape,
    position: usize,
    storage: CacheStorage,
) -> Vec<f64> {
    let query = decode_f32(query_bytes);
    scalar_attention_row(&query, spans, shape, position, storage)
}

fn reference_prefill(
    query_bytes: &[u8],
    spans: &[EncodedSpan],
    shape: AttentionShape,
    start_position: usize,
    tokens: usize,
    storage: CacheStorage,
) -> Vec<f64> {
    let query = decode_f32(query_bytes);
    let row_elements = shape.query_elements().unwrap();
    let mut output = Vec::with_capacity(tokens * row_elements);
    for token in 0..tokens {
        let begin = token * row_elements;
        let end = begin + row_elements;
        output.extend(scalar_attention_row(
            &query[begin..end],
            spans,
            shape,
            start_position + token,
            storage,
        ));
    }
    output
}

fn scalar_attention_row(
    query: &[f64],
    spans: &[EncodedSpan],
    shape: AttentionShape,
    position: usize,
    storage: CacheStorage,
) -> Vec<f64> {
    let group_size = shape.n_head() / shape.n_head_kv();
    let scale = (shape.head_dim() as f64).sqrt().recip();
    let mut output = vec![0.0; shape.query_elements().unwrap()];
    for query_head in 0..shape.n_head() {
        let query_base = query_head * shape.head_dim();
        let kv_head = query_head / group_size;
        let scores: Vec<f64> = (0..=position)
            .map(|logical_position| {
                let mut dot = 0.0;
                for dimension in 0..shape.head_dim() {
                    dot += query[query_base + dimension]
                        * encoded_cache_value(
                            spans,
                            storage,
                            shape,
                            kv_head,
                            logical_position,
                            dimension,
                            false,
                        );
                }
                dot * scale
            })
            .collect();
        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = scores.iter().map(|score| (score - maximum).exp()).collect();
        let denominator: f64 = weights.iter().sum();
        for dimension in 0..shape.head_dim() {
            let numerator: f64 = weights
                .iter()
                .enumerate()
                .map(|(logical_position, weight)| {
                    weight
                        * encoded_cache_value(
                            spans,
                            storage,
                            shape,
                            kv_head,
                            logical_position,
                            dimension,
                            true,
                        )
                })
                .sum();
            output[query_base + dimension] = numerator / denominator;
        }
    }
    output
}

fn encoded_cache_value(
    spans: &[EncodedSpan],
    storage: CacheStorage,
    shape: AttentionShape,
    kv_head: usize,
    position: usize,
    dimension: usize,
    value: bool,
) -> f64 {
    let span = spans
        .iter()
        .find(|span| position >= span.logical_start && position < span.logical_start + span.tokens)
        .expect("oracle reads only mapped logical KV positions");
    debug_assert_eq!(span.storage as u8, storage as u8);
    let local = position - span.logical_start;
    let element = (kv_head * span.capacity + local) * shape.head_dim() + dimension;
    let bytes = if value {
        &span.value_bytes
    } else {
        &span.key_bytes
    };
    decode_cache_element(bytes, storage, element)
}

fn decode_cache_element(bytes: &[u8], storage: CacheStorage, element: usize) -> f64 {
    match storage {
        CacheStorage::F32 => {
            let offset = element * 4;
            f64::from(f32::from_le_bytes(
                bytes[offset..offset + 4].try_into().unwrap(),
            ))
        }
        CacheStorage::F16 => {
            let offset = element * 2;
            f64::from(
                f16::from_bits(u16::from_le_bytes(
                    bytes[offset..offset + 2].try_into().unwrap(),
                ))
                .to_f32(),
            )
        }
        CacheStorage::Q8 => {
            let block = element / 32;
            let offset = block * 34;
            let scale = f16::from_bits(u16::from_le_bytes(
                bytes[offset..offset + 2].try_into().unwrap(),
            ))
            .to_f32();
            f64::from(bytes[offset + 2 + element % 32] as i8) * f64::from(scale)
        }
    }
}

fn decode_f32(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f64::from(f32::from_le_bytes(chunk.try_into().unwrap())))
        .collect()
}

fn assert_output_close(
    operation: &str,
    storage: CacheStorage,
    position: usize,
    actual: &[f32],
    expected: &[f64],
) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let error = (f64::from(actual) - expected).abs();
        assert!(
            error <= OUTPUT_TOLERANCE,
            "{operation} {storage:?} position {position} index {index}: actual {actual:?}, expected {expected:?}, error {error}"
        );
    }
}
