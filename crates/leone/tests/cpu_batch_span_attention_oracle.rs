use half::f16;
use leone::{
    AttentionDecodeRow, AttentionShape, Backend, BufferLayout, BufferSnapshot, CpuBackend,
    CpuBuffer, KvReadSpan, KvReadView, Position,
};

const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 128;
const ROW_ELEMENTS: usize = HEADS * HEAD_DIM;

struct Chunk {
    key: CpuBuffer,
    value: CpuBuffer,
    host_key: Vec<f16>,
    host_value: Vec<f16>,
    start: usize,
    capacity: usize,
}

fn upload_f32(backend: &mut CpuBackend, values: &[f32]) -> CpuBuffer {
    let bytes: Vec<_> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    backend
        .upload(BufferLayout::f32(values.len()).unwrap(), &bytes)
        .unwrap()
}

fn upload_f16(backend: &mut CpuBackend, values: &[f16]) -> CpuBuffer {
    let bytes: Vec<_> = values
        .iter()
        .flat_map(|value| value.to_bits().to_le_bytes())
        .collect();
    let snapshot = BufferSnapshot::new(BufferLayout::f16(values.len()).unwrap(), bytes).unwrap();
    backend.restore_buffer(&snapshot).unwrap()
}

fn chunk_values(capacity: usize, initialized: usize, seed: usize) -> Vec<f16> {
    let mut values = vec![f16::NAN; KV_HEADS * capacity * HEAD_DIM];
    for head in 0..KV_HEADS {
        for token in 0..initialized {
            for dimension in 0..HEAD_DIM {
                let index = (head * capacity + token) * HEAD_DIM + dimension;
                let code = (head * 31 + token * 17 + dimension * 7 + seed) % 101;
                values[index] = f16::from_f32((code as f32 - 50.0) / 64.0);
            }
        }
    }
    values
}

fn chunk(
    backend: &mut CpuBackend,
    start: usize,
    capacity: usize,
    initialized: usize,
    seed: usize,
) -> Chunk {
    let host_key = chunk_values(capacity, initialized, seed);
    let host_value = chunk_values(capacity, initialized, seed + 19);
    Chunk {
        key: upload_f16(backend, &host_key),
        value: upload_f16(backend, &host_value),
        host_key,
        host_value,
        start,
        capacity,
    }
}

fn span(chunk: &Chunk, mapped: usize) -> KvReadSpan<'_, CpuBuffer> {
    KvReadSpan::new(
        &chunk.key,
        &chunk.value,
        chunk.start,
        mapped,
        chunk.capacity,
    )
    .unwrap()
}

fn cache_value(
    prefix: &Chunk,
    tail: &Chunk,
    head: usize,
    token: usize,
    dimension: usize,
    value: bool,
) -> f64 {
    let chunk = if token < tail.start { prefix } else { tail };
    let index = (head * chunk.capacity + token - chunk.start) * HEAD_DIM + dimension;
    let values = if value {
        &chunk.host_value
    } else {
        &chunk.host_key
    };
    f64::from(values[index].to_f32())
}

fn oracle(query: &[f32], prefix: &Chunk, tail: &Chunk, position: usize) -> Vec<f64> {
    let mut output = vec![0.0; ROW_ELEMENTS];
    for head in 0..HEADS {
        let kv_head = head / (HEADS / KV_HEADS);
        let scores: Vec<f64> = (0..=position)
            .map(|token| {
                let dot: f64 = (0..HEAD_DIM)
                    .map(|dimension| {
                        f64::from(query[head * HEAD_DIM + dimension])
                            * cache_value(prefix, tail, kv_head, token, dimension, false)
                    })
                    .sum();
                dot / (HEAD_DIM as f64).sqrt()
            })
            .collect();
        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<_> = scores.iter().map(|score| (score - maximum).exp()).collect();
        let denominator: f64 = weights.iter().sum();
        for dimension in 0..HEAD_DIM {
            let sum: f64 = weights
                .iter()
                .enumerate()
                .map(|(token, weight)| {
                    weight * cache_value(prefix, tail, kv_head, token, dimension, true)
                })
                .sum();
            output[head * HEAD_DIM + dimension] = sum / denominator;
        }
    }
    output
}

fn read_output(backend: &mut CpuBackend, output: &CpuBuffer) -> Vec<f32> {
    let mut values = vec![0.0; ROW_ELEMENTS];
    backend.read_f32(output, &mut values).unwrap();
    values
}

fn assert_oracle(actual: &[f32], expected: &[f64]) {
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(actual.is_finite());
        assert!(
            (f64::from(*actual) - expected).abs() <= 2.0e-5,
            "actual {actual}, expected {expected}"
        );
    }
}

#[test]
fn shared_prefix_batch_matches_fp64_and_row_partitions() {
    let mut backend = CpuBackend::new();
    let shared = chunk(&mut backend, 0, 7, 3, 1);
    let unrelated = chunk(&mut backend, 0, 5, 3, 41);
    let tails = [
        chunk(&mut backend, 3, 8, 3, 13),
        chunk(&mut backend, 3, 9, 2, 29),
        chunk(&mut backend, 3, 6, 3, 59),
    ];
    let prefixes = [&shared, &shared, &unrelated];
    let descriptors: Vec<_> = prefixes
        .iter()
        .zip(&tails)
        .map(|(prefix, tail)| [span(prefix, 3), span(tail, tail.capacity)])
        .collect();
    let host_queries: Vec<Vec<f32>> = (0..3)
        .map(|row| {
            (0..ROW_ELEMENTS)
                .map(|index| ((index * 11 + row * 23) % 61) as f32 / 64.0 - 0.5)
                .collect()
        })
        .collect();
    let queries: Vec<_> = host_queries
        .iter()
        .map(|values| upload_f32(&mut backend, values))
        .collect();
    let mut outputs: Vec<_> = (0..3)
        .map(|_| upload_f32(&mut backend, &vec![37.0; ROW_ELEMENTS]))
        .collect();
    let mut device_position = backend
        .upload(BufferLayout::u32(1).unwrap(), &u32::MAX.to_le_bytes())
        .unwrap();
    prepare_rows(
        &mut backend,
        &queries,
        &descriptors,
        &mut outputs,
        &device_position,
    );
    for output in &outputs {
        assert!(read_output(&mut backend, output)
            .iter()
            .all(|value| *value == 37.0));
    }
    backend.write_u32(&mut device_position, &[4]).unwrap();
    run_row_partitions(
        &mut backend,
        &queries,
        &descriptors,
        &mut outputs,
        &device_position,
        std::slice::from_ref(&(0..3)),
    );
    let first: Vec<_> = outputs
        .iter()
        .map(|output| read_output(&mut backend, output))
        .collect();
    for row in 0..3 {
        assert_oracle(
            &first[row],
            &oracle(
                &host_queries[row],
                prefixes[row],
                &tails[row],
                [5, 4, 5][row],
            ),
        );
    }
    run_row_partitions(
        &mut backend,
        &queries,
        &descriptors,
        &mut outputs,
        &device_position,
        &[2..3, 0..2],
    );
    for (row, output) in outputs.iter().enumerate() {
        let repeated = read_output(&mut backend, output);
        assert_eq!(
            repeated
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            first[row]
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }
}

fn rows<'a>(
    queries: &'a [CpuBuffer],
    descriptors: &'a [[KvReadSpan<'a, CpuBuffer>; 2]],
    outputs: &'a mut [CpuBuffer],
    device_position: &'a CpuBuffer,
) -> Vec<AttentionDecodeRow<'a, CpuBuffer>> {
    outputs
        .iter_mut()
        .enumerate()
        .map(|(row, output)| AttentionDecodeRow {
            query: &queries[row],
            cache: KvReadView::new(&descriptors[row]).unwrap(),
            output,
            shape: AttentionShape::new(HEADS, KV_HEADS, HEAD_DIM, [32, 64, 128][row]).unwrap(),
            position: if row == 1 {
                Position::Device(device_position)
            } else {
                Position::Host(5)
            },
        })
        .collect()
}

fn prepare_rows(
    backend: &mut CpuBackend,
    queries: &[CpuBuffer],
    descriptors: &[[KvReadSpan<'_, CpuBuffer>; 2]],
    outputs: &mut [CpuBuffer],
    device_position: &CpuBuffer,
) {
    let descriptors = rows(queries, descriptors, outputs, device_position);
    backend
        .prepare_attention_decode_batch_spans(&descriptors)
        .unwrap();
}

fn run_row_partitions(
    backend: &mut CpuBackend,
    queries: &[CpuBuffer],
    descriptors: &[[KvReadSpan<'_, CpuBuffer>; 2]],
    outputs: &mut [CpuBuffer],
    device_position: &CpuBuffer,
    partitions: &[std::ops::Range<usize>],
) {
    let mut rows = rows(queries, descriptors, outputs, device_position);
    for partition in partitions {
        backend
            .attention_decode_batch_spans(&mut rows[partition.clone()])
            .unwrap();
    }
}
