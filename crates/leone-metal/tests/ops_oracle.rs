mod oracle_support;

use leone::{
    AttentionShape, Backend, BackendError, BufferLayout, CpuBackend, Position, VectorShape,
};

use oracle_support::{
    assert_close, metal_backend, read_f16, read_f32, seeded_values, upload_f32, upload_u32,
};

#[test]
fn fused_vector_ops_match_cpu_oracle() {
    let Some(mut metal) = metal_backend() else {
        return;
    };
    let mut cpu = CpuBackend::new();
    let shape = VectorShape::new(3, 8).expect("vector shape");
    let left = seeded_values(shape.elements().expect("vector elements"), 0x4c45_4654);
    let right = seeded_values(shape.elements().expect("vector elements"), 0x5249_4748);
    let weight = seeded_values(shape.columns(), 0x5745_4947)
        .into_iter()
        .map(|value| value.abs() + 0.25)
        .collect::<Vec<_>>();
    let cpu_left = upload_f32(&mut cpu, &left);
    let cpu_right = upload_f32(&mut cpu, &right);
    let cpu_weight = upload_f32(&mut cpu, &weight);
    let metal_left = upload_f32(&mut metal, &left);
    let metal_right = upload_f32(&mut metal, &right);
    let metal_weight = upload_f32(&mut metal, &weight);

    let mut cpu_norm = cpu
        .allocate(BufferLayout::f32(left.len()).expect("CPU norm layout"))
        .expect("CPU norm output");
    let mut metal_norm = metal
        .allocate(BufferLayout::f32(left.len()).expect("Metal norm layout"))
        .expect("Metal norm output");
    cpu.rms_norm(&cpu_left, &cpu_weight, &mut cpu_norm, shape, 1e-5)
        .expect("CPU RMSNorm");
    metal
        .rms_norm(&metal_left, &metal_weight, &mut metal_norm, shape, 1e-5)
        .expect("Metal RMSNorm");
    let cpu_norm_values = read_f32(&mut cpu, &cpu_norm, left.len());
    let metal_norm_values = read_f32(&mut metal, &metal_norm, left.len());
    assert_close("RMSNorm", &metal_norm_values, &cpu_norm_values, 3e-3, 2e-3);

    let mut cpu_residual_norm = cpu
        .allocate(BufferLayout::f32(left.len()).expect("CPU residual norm layout"))
        .expect("CPU residual norm output");
    let mut metal_residual_norm = metal
        .allocate(BufferLayout::f32(left.len()).expect("Metal residual norm layout"))
        .expect("Metal residual norm output");
    cpu.rms_norm_residual(
        &cpu_left,
        &cpu_right,
        &cpu_weight,
        &mut cpu_residual_norm,
        shape,
        1e-5,
    )
    .expect("CPU residual RMSNorm");
    metal
        .rms_norm_residual(
            &metal_left,
            &metal_right,
            &metal_weight,
            &mut metal_residual_norm,
            shape,
            1e-5,
        )
        .expect("Metal residual RMSNorm");
    let cpu_residual_norm_values = read_f32(&mut cpu, &cpu_residual_norm, left.len());
    let metal_residual_norm_values = read_f32(&mut metal, &metal_residual_norm, left.len());
    assert_close(
        "residual RMSNorm",
        &metal_residual_norm_values,
        &cpu_residual_norm_values,
        3e-3,
        2e-3,
    );

    let mut cpu_residual_store = cpu
        .allocate(BufferLayout::f32(left.len()).expect("CPU residual store layout"))
        .expect("CPU residual store");
    let mut metal_residual_store = metal
        .allocate(BufferLayout::f32(left.len()).expect("Metal residual store layout"))
        .expect("Metal residual store");
    let mut cpu_store_norm = cpu
        .allocate(BufferLayout::f32(left.len()).expect("CPU store norm layout"))
        .expect("CPU store norm output");
    let mut metal_store_norm = metal
        .allocate(BufferLayout::f32(left.len()).expect("Metal store norm layout"))
        .expect("Metal store norm output");
    cpu.rms_norm_residual_store(
        &cpu_left,
        &cpu_right,
        &cpu_weight,
        &mut cpu_residual_store,
        &mut cpu_store_norm,
        shape,
        1e-5,
    )
    .expect("CPU residual store RMSNorm");
    metal
        .rms_norm_residual_store(
            &metal_left,
            &metal_right,
            &metal_weight,
            &mut metal_residual_store,
            &mut metal_store_norm,
            shape,
            1e-5,
        )
        .expect("Metal residual store RMSNorm");
    let cpu_store_values = read_f32(&mut cpu, &cpu_store_norm, left.len());
    let metal_store_values = read_f32(&mut metal, &metal_store_norm, left.len());
    let cpu_residual_values = read_f32(&mut cpu, &cpu_residual_store, left.len());
    let metal_residual_values = read_f32(&mut metal, &metal_residual_store, left.len());
    assert_close(
        "stored residual RMSNorm",
        &metal_store_values,
        &cpu_store_values,
        3e-3,
        2e-3,
    );
    assert_close(
        "stored residual",
        &metal_residual_values,
        &cpu_residual_values,
        1e-6,
        1e-6,
    );

    let gate = seeded_values(left.len(), 0x4741_5445)
        .into_iter()
        .map(|value| value * 4.0)
        .collect::<Vec<_>>();
    let up = seeded_values(left.len(), 0x5550_4441);
    let cpu_gate = upload_f32(&mut cpu, &gate);
    let cpu_up = upload_f32(&mut cpu, &up);
    let metal_gate = upload_f32(&mut metal, &gate);
    let metal_up = upload_f32(&mut metal, &up);
    let mut cpu_swiglu = cpu
        .allocate(BufferLayout::f32(left.len()).expect("CPU SwiGLU layout"))
        .expect("CPU SwiGLU output");
    let mut metal_swiglu = metal
        .allocate(BufferLayout::f32(left.len()).expect("Metal SwiGLU layout"))
        .expect("Metal SwiGLU output");
    cpu.swiglu(&cpu_gate, &cpu_up, &mut cpu_swiglu)
        .expect("CPU SwiGLU");
    metal
        .swiglu(&metal_gate, &metal_up, &mut metal_swiglu)
        .expect("Metal SwiGLU");
    let cpu_swiglu_values = read_f32(&mut cpu, &cpu_swiglu, left.len());
    let metal_swiglu_values = read_f32(&mut metal, &metal_swiglu, left.len());
    assert_close(
        "SwiGLU",
        &metal_swiglu_values,
        &cpu_swiglu_values,
        2e-5,
        2e-5,
    );

    let mut cpu_added = cpu
        .allocate(BufferLayout::f32(left.len()).expect("CPU residual add layout"))
        .expect("CPU residual add output");
    let mut metal_added = metal
        .allocate(BufferLayout::f32(left.len()).expect("Metal residual add layout"))
        .expect("Metal residual add output");
    cpu.residual_add(&cpu_left, &cpu_right, &mut cpu_added)
        .expect("CPU residual add");
    metal
        .residual_add(&metal_left, &metal_right, &mut metal_added)
        .expect("Metal residual add");
    let cpu_added_values = read_f32(&mut cpu, &cpu_added, left.len());
    let metal_added_values = read_f32(&mut metal, &metal_added, left.len());
    assert_close(
        "residual add",
        &metal_added_values,
        &cpu_added_values,
        1e-6,
        1e-6,
    );
}

#[test]
fn f16_gqa_attention_decode_and_prefill_match_scalar_oracle() {
    let Some(mut metal) = metal_backend() else {
        return;
    };
    let mut cpu = CpuBackend::new();
    let shape = AttentionShape::new(4, 2, 4, 8).expect("GQA shape");
    let tokens = 6;
    let projected = shape
        .projected_kv_elements()
        .expect("projected KV elements");
    let keys = seeded_values(tokens * projected, 0x4b45_5953);
    let values = seeded_values(tokens * projected, 0x5641_4c53);
    let (cpu_key_cache, cpu_value_cache) =
        append_f16_cache(&mut cpu, shape, &keys, &values, tokens);
    let (metal_key_cache, metal_value_cache) =
        append_f16_cache(&mut metal, shape, &keys, &values, tokens);
    let cache_elements = shape.cache_elements().expect("cache elements");
    let cpu_keys = read_f16(&mut cpu, &cpu_key_cache, cache_elements);
    let metal_keys = read_f16(&mut metal, &metal_key_cache, cache_elements);
    let cpu_values = read_f16(&mut cpu, &cpu_value_cache, cache_elements);
    let metal_values = read_f16(&mut metal, &metal_value_cache, cache_elements);
    assert_close("F16 key cache", &metal_keys, &cpu_keys, 0.0, 0.0);
    assert_close("F16 value cache", &metal_values, &cpu_values, 0.0, 0.0);

    let query = seeded_values(shape.query_elements().expect("query elements"), 0x4445_434f);
    let cpu_query = upload_f32(&mut cpu, &query);
    let metal_query = upload_f32(&mut metal, &query);
    let mut cpu_decode = cpu
        .allocate(
            BufferLayout::f32(shape.query_elements().expect("query elements"))
                .expect("CPU decode layout"),
        )
        .expect("CPU decode output");
    let mut metal_decode = metal
        .allocate(
            BufferLayout::f32(shape.query_elements().expect("query elements"))
                .expect("Metal decode layout"),
        )
        .expect("Metal decode output");
    cpu.attention_decode(
        &cpu_query,
        &cpu_key_cache,
        &cpu_value_cache,
        &mut cpu_decode,
        shape,
        Position::Host(5),
    )
    .expect("CPU decode attention");
    metal
        .attention_decode(
            &metal_query,
            &metal_key_cache,
            &metal_value_cache,
            &mut metal_decode,
            shape,
            Position::Host(5),
        )
        .expect("Metal decode attention");
    let cpu_decode_values = read_f32(&mut cpu, &cpu_decode, shape.query_elements().unwrap());
    let metal_decode_values = read_f32(&mut metal, &metal_decode, shape.query_elements().unwrap());
    let expected_decode = attention_oracle(&query, &cpu_keys, &cpu_values, shape, 5, 1);
    assert_close(
        "CPU decode attention oracle",
        &cpu_decode_values,
        &expected_decode,
        3e-3,
        3e-3,
    );
    assert_close(
        "Metal decode attention oracle",
        &metal_decode_values,
        &expected_decode,
        4e-3,
        4e-3,
    );

    let prefill_tokens = 3;
    let prefill_query = seeded_values(
        prefill_tokens * shape.query_elements().expect("query elements"),
        0x5052_4546,
    );
    let cpu_prefill_query = upload_f32(&mut cpu, &prefill_query);
    let metal_prefill_query = upload_f32(&mut metal, &prefill_query);
    let mut cpu_prefill = cpu
        .allocate(BufferLayout::f32(prefill_query.len()).expect("CPU prefill output layout"))
        .expect("CPU prefill output");
    let mut metal_prefill = metal
        .allocate(BufferLayout::f32(prefill_query.len()).expect("Metal prefill output layout"))
        .expect("Metal prefill output");
    cpu.attention_prefill(
        &cpu_prefill_query,
        &cpu_key_cache,
        &cpu_value_cache,
        &mut cpu_prefill,
        shape,
        1,
        prefill_tokens,
    )
    .expect("CPU prefill attention");
    metal
        .attention_prefill(
            &metal_prefill_query,
            &metal_key_cache,
            &metal_value_cache,
            &mut metal_prefill,
            shape,
            1,
            prefill_tokens,
        )
        .expect("Metal prefill attention");
    let cpu_prefill_values = read_f32(&mut cpu, &cpu_prefill, prefill_query.len());
    let metal_prefill_values = read_f32(&mut metal, &metal_prefill, prefill_query.len());
    let expected_prefill = attention_oracle(
        &prefill_query,
        &cpu_keys,
        &cpu_values,
        shape,
        1,
        prefill_tokens,
    );
    assert_close(
        "CPU prefill attention oracle",
        &cpu_prefill_values,
        &expected_prefill,
        3e-3,
        3e-3,
    );
    assert_close(
        "Metal prefill attention oracle",
        &metal_prefill_values,
        &expected_prefill,
        4e-3,
        4e-3,
    );
}

#[test]
fn f16_gqa_attention_head_dim_boundaries_match_f64_oracle() {
    let Some(mut metal) = metal_backend() else {
        return;
    };
    let mut cpu = CpuBackend::new();
    for head_dim in [255, 256, 257] {
        let shape = AttentionShape::new(4, 2, head_dim, 6).expect("boundary GQA shape");
        let projected = shape
            .projected_kv_elements()
            .expect("boundary projected KV elements");
        let keys = seeded_values(6 * projected, 0x424f_554e ^ head_dim as u32);
        let values = seeded_values(6 * projected, 0x56414c55 ^ head_dim as u32);
        let decode_query = seeded_values(
            shape.query_elements().expect("boundary query elements"),
            0x4445434f ^ head_dim as u32,
        );
        let prefill_query = seeded_values(
            3 * shape.query_elements().expect("boundary query elements"),
            0x50524546 ^ head_dim as u32,
        );
        let cpu_result = run_contiguous_attention_case(
            &mut cpu,
            shape,
            &keys,
            &values,
            &decode_query,
            &prefill_query,
        );
        let metal_result = run_contiguous_attention_case(
            &mut metal,
            shape,
            &keys,
            &values,
            &decode_query,
            &prefill_query,
        );
        assert_close(
            "boundary key cache",
            &metal_result.keys,
            &cpu_result.keys,
            0.0,
            0.0,
        );
        assert_close(
            "boundary value cache",
            &metal_result.values,
            &cpu_result.values,
            0.0,
            0.0,
        );
        let expected_decode = attention_oracle(
            &decode_query,
            &cpu_result.keys,
            &cpu_result.values,
            shape,
            5,
            1,
        );
        let expected_prefill = attention_oracle(
            &prefill_query,
            &cpu_result.keys,
            &cpu_result.values,
            shape,
            2,
            3,
        );
        assert_close(
            "boundary CPU decode",
            &cpu_result.decode,
            &expected_decode,
            3e-3,
            3e-3,
        );
        assert_close(
            "boundary Metal decode",
            &metal_result.decode,
            &expected_decode,
            4e-3,
            4e-3,
        );
        assert_close(
            "boundary CPU prefill",
            &cpu_result.prefill,
            &expected_prefill,
            3e-3,
            3e-3,
        );
        assert_close(
            "boundary Metal prefill",
            &metal_result.prefill,
            &expected_prefill,
            4e-3,
            4e-3,
        );
    }
}

#[test]
fn operation_contracts_match_typed_errors() {
    let Some(mut metal) = metal_backend() else {
        return;
    };
    let shape = leone::QuantMatrix::new(1, 256, leone::QuantFormat::Q4K).expect("table shape");
    let table = metal
        .upload(
            shape.layout().expect("table layout"),
            &vec![0_u8; shape.layout().unwrap().bytes()],
        )
        .expect("table upload");
    let invalid_row = upload_u32(&mut metal, &[1]);
    let mut embedding = metal
        .allocate(BufferLayout::f32(256).expect("embedding layout"))
        .expect("embedding output");
    let error = metal
        .embed_gather(&table, &invalid_row, &mut embedding, shape)
        .expect_err("embedding row must be checked");
    assert_eq!(error, BackendError::RowOutOfBounds { row: 1, rows: 1 });

    let invalid_rows = upload_u32(&mut metal, &[0, 1]);
    let mut batch = metal
        .allocate(BufferLayout::f32(2 * 256).expect("batch embedding layout"))
        .expect("batch embedding output");
    let error = metal
        .embed_gather_batch(&table, &invalid_rows, &mut batch, shape, 2)
        .expect_err("batch embedding rows must be checked");
    assert_eq!(error, BackendError::RowOutOfBounds { row: 1, rows: 1 });

    let attention_shape = AttentionShape::new(2, 1, 4, 4).expect("attention shape");
    let key = upload_f32(&mut metal, &[0.0; 4]);
    let value = upload_f32(&mut metal, &[0.0; 4]);
    let mut key_cache = metal
        .allocate(BufferLayout::f16(attention_shape.cache_elements().unwrap()).unwrap())
        .expect("key cache");
    let mut value_cache = metal
        .allocate(BufferLayout::f16(attention_shape.cache_elements().unwrap()).unwrap())
        .expect("value cache");
    let error = metal
        .kv_append_chunk(
            &key,
            &value,
            &mut key_cache,
            &mut value_cache,
            attention_shape,
            0,
            0,
        )
        .expect_err("zero KV chunk must be rejected");
    assert_eq!(
        error,
        BackendError::Zero {
            field: "KV chunk tokens"
        }
    );

    let query = upload_f32(&mut metal, &[0.0; 8]);
    let mut output = metal
        .allocate(BufferLayout::f32(8).expect("attention output layout"))
        .expect("attention output");
    let error = metal
        .attention_decode(
            &query,
            &key_cache,
            &value_cache,
            &mut output,
            attention_shape,
            Position::Host(4),
        )
        .expect_err("decode position at capacity must be rejected");
    assert_eq!(
        error,
        BackendError::PositionOutOfBounds {
            position: 4,
            max_context: 4
        }
    );

    let wrong_cache = metal
        .allocate(BufferLayout::f32(attention_shape.cache_elements().unwrap()).unwrap())
        .expect("wrong storage cache");
    let error = metal
        .attention_decode(
            &query,
            &wrong_cache,
            &value_cache,
            &mut output,
            attention_shape,
            Position::Host(0),
        )
        .expect_err("wrong KV storage must be rejected");
    assert!(matches!(error, BackendError::Operation { .. }));
}

fn append_f16_cache<B: Backend>(
    backend: &mut B,
    shape: AttentionShape,
    keys: &[f32],
    values: &[f32],
    tokens: usize,
) -> (B::Buffer, B::Buffer) {
    let key = upload_f32(backend, keys);
    let value = upload_f32(backend, values);
    let layout = BufferLayout::f16(shape.cache_elements().expect("cache elements"))
        .expect("F16 cache layout");
    let mut key_cache = backend.allocate(layout).expect("key cache");
    let mut value_cache = backend.allocate(layout).expect("value cache");
    backend
        .kv_append_chunk(
            &key,
            &value,
            &mut key_cache,
            &mut value_cache,
            shape,
            0,
            tokens,
        )
        .expect("append F16 KV cache");
    (key_cache, value_cache)
}

struct AttentionCaseResult {
    decode: Vec<f32>,
    prefill: Vec<f32>,
    keys: Vec<f32>,
    values: Vec<f32>,
}

fn run_contiguous_attention_case<B: Backend>(
    backend: &mut B,
    shape: AttentionShape,
    keys: &[f32],
    values: &[f32],
    decode_query: &[f32],
    prefill_query: &[f32],
) -> AttentionCaseResult {
    let (key_cache, value_cache) =
        append_f16_cache(backend, shape, keys, values, shape.max_context());
    let cache_elements = shape.cache_elements().expect("boundary cache elements");
    let cache_keys = read_f16(backend, &key_cache, cache_elements);
    let cache_values = read_f16(backend, &value_cache, cache_elements);
    let query = upload_f32(backend, decode_query);
    let mut decode = backend
        .allocate(
            BufferLayout::f32(shape.query_elements().expect("boundary query elements"))
                .expect("boundary decode layout"),
        )
        .expect("boundary decode output");
    backend
        .attention_decode(
            &query,
            &key_cache,
            &value_cache,
            &mut decode,
            shape,
            Position::Host(5),
        )
        .expect("boundary decode");
    let prefill_input = upload_f32(backend, prefill_query);
    let mut prefill = backend
        .allocate(BufferLayout::f32(prefill_query.len()).expect("boundary prefill layout"))
        .expect("boundary prefill output");
    backend
        .attention_prefill(
            &prefill_input,
            &key_cache,
            &value_cache,
            &mut prefill,
            shape,
            2,
            3,
        )
        .expect("boundary prefill");
    AttentionCaseResult {
        decode: read_f32(
            backend,
            &decode,
            shape.query_elements().expect("boundary query elements"),
        ),
        prefill: read_f32(backend, &prefill, prefill_query.len()),
        keys: cache_keys,
        values: cache_values,
    }
}

fn attention_oracle(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    shape: AttentionShape,
    start_position: usize,
    tokens: usize,
) -> Vec<f32> {
    let query_elements = shape.query_elements().expect("query elements");
    let mut output = Vec::with_capacity(tokens * query_elements);
    for token in 0..tokens {
        let context = start_position + token + 1;
        for head in 0..shape.n_head() {
            let kv_head = head / (shape.n_head() / shape.n_head_kv());
            let query_start = token * query_elements + head * shape.head_dim();
            let scores = (0..context)
                .map(|position| score(query, keys, shape, query_start, kv_head, position))
                .collect::<Vec<_>>();
            let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights = scores
                .iter()
                .map(|score| (*score - maximum).exp())
                .collect::<Vec<_>>();
            let denominator = weights.iter().sum::<f64>();
            for column in 0..shape.head_dim() {
                let value = (0..context)
                    .map(|position| {
                        let index =
                            (kv_head * shape.max_context() + position) * shape.head_dim() + column;
                        weights[position] * f64::from(values[index])
                    })
                    .sum::<f64>()
                    / denominator;
                output.push(value as f32);
            }
        }
    }
    output
}

fn score(
    query: &[f32],
    keys: &[f32],
    shape: AttentionShape,
    query_start: usize,
    kv_head: usize,
    position: usize,
) -> f64 {
    let key_start = (kv_head * shape.max_context() + position) * shape.head_dim();
    let dot = (0..shape.head_dim())
        .map(|column| f64::from(query[query_start + column]) * f64::from(keys[key_start + column]))
        .sum::<f64>();
    dot / f64::from(shape.head_dim() as u32).sqrt()
}
