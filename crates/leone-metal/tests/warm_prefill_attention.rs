mod oracle_support;
mod warm_prefill_support;

use leone::{AttentionShape, Backend, BufferLayout, KvReadSpan, KvReadView, Position};
use leone_metal::{MetalBackend, MetalBuffer};
use oracle_support::{assert_close, f32_bytes, metal_backend, read_f16, read_f32, upload_f32};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use warm_prefill_support::{
    allocate_f16_cache, allocate_f16_cache_with_value, attention_stress_query_values,
    attention_stress_values, attention_values, f32_bits, f64_attention_oracle,
    prepare_backend_preferred, prepare_decode_equivalent,
};

struct AttentionArtifactWriter {
    root: Option<PathBuf>,
}

impl AttentionArtifactWriter {
    fn from_env() -> Self {
        Self {
            root: std::env::var_os("LEONE_METAL_ATTENTION_ARTIFACT_DIR").map(PathBuf::from),
        }
    }

    fn write(&self, label: &str, values: &[f32]) {
        let Some(root) = self.root.as_deref() else {
            return;
        };
        write_attention_artifact(root, label, values);
    }
}

fn write_attention_artifact(root: &Path, label: &str, values: &[f32]) {
    std::fs::create_dir_all(root).expect("create attention artifact directory");
    let bytes = f32_bytes(values);
    let digest = Sha256::digest(&bytes);
    std::fs::write(root.join(format!("{label}.f32le")), &bytes).expect("write attention artifact");
    std::fs::write(
        root.join(format!("{label}.sha256")),
        format!("{digest:x}\n"),
    )
    .expect("write attention artifact hash");
}

fn assert_context_diversity(values: &[f32], per_token: usize, label: &str) {
    let fingerprints = values
        .chunks_exact(per_token)
        .take(8)
        .map(|row| row.iter().map(|value| value.to_bits()).collect::<Vec<_>>())
        .collect::<BTreeSet<_>>();
    assert!(
        fingerprints.len() >= 4,
        "{label} has insufficient token diversity: {} distinct rows",
        fingerprints.len()
    );
}

fn assert_output_diversity(values: &[f32], per_token: usize, label: &str) {
    assert_ne!(
        &values[..per_token],
        &values[per_token..2 * per_token],
        "{label} did not respond to the token query"
    );
}

struct SpanAttentionOutputs {
    prefill: Vec<f32>,
    repeated: Vec<f32>,
}

#[test]
fn decode_equivalent_attention_prefill_matches_repeated_decode_contiguous() {
    let Some(mut metal) = metal_backend() else {
        return;
    };
    let artifacts = AttentionArtifactWriter::from_env();
    assert!(metal.decode_equivalent_prefill_supported());
    let shape = AttentionShape::new(2, 1, 32, 16).expect("contiguous attention shape");
    let start_position = 4;
    let tokens = 3;
    let projected = shape
        .projected_kv_elements()
        .expect("projected KV elements");
    let query_elements = shape.query_elements().expect("query elements");
    let key_values = attention_values(16 * projected, 0x4b45_5931);
    let value_values = attention_values(16 * projected, 0x5641_4c31);
    let query_values = attention_values(tokens * query_elements, 0x5152_5931);
    let key = upload_f32(&mut metal, &key_values);
    let value = upload_f32(&mut metal, &value_values);
    let mut key_cache = allocate_f16_cache(&mut metal, shape.cache_elements().unwrap());
    let mut value_cache = allocate_f16_cache(&mut metal, shape.cache_elements().unwrap());
    metal
        .kv_append_chunk(&key, &value, &mut key_cache, &mut value_cache, shape, 0, 16)
        .expect("contiguous KV append");
    let query = upload_f32(&mut metal, &query_values);
    let mut batched = metal
        .allocate(BufferLayout::f32(tokens * query_elements).expect("prefill output layout"))
        .expect("prefill output");
    prepare_decode_equivalent(
        &mut metal,
        tokens,
        shape.max_context(),
        shape.n_head(),
        shape.n_head_kv(),
        shape.head_dim(),
        query_elements,
        query_elements,
        query_elements,
    );
    metal
        .attention_prefill(
            &query,
            &key_cache,
            &value_cache,
            &mut batched,
            shape,
            start_position,
            tokens,
        )
        .expect("contiguous prefill attention");
    let actual = read_f32(&mut metal, &batched, tokens * query_elements);
    let mut repeated = Vec::with_capacity(actual.len());
    for token in 0..tokens {
        let token_query = upload_f32(
            &mut metal,
            &query_values[token * query_elements..(token + 1) * query_elements],
        );
        let mut token_output = metal
            .allocate(BufferLayout::f32(query_elements).expect("decode output layout"))
            .expect("decode output");
        metal
            .attention_decode(
                &token_query,
                &key_cache,
                &value_cache,
                &mut token_output,
                shape,
                Position::Host(start_position + token),
            )
            .expect("repeated decode attention");
        repeated.extend(read_f32(&mut metal, &token_output, query_elements));
    }
    assert_eq!(
        f32_bits(&actual),
        f32_bits(&repeated),
        "contiguous attention prefill differs from repeated decode"
    );
    artifacts.write("contiguous_prefill", &actual);
    artifacts.write("contiguous_repeated_decode", &repeated);
}

#[test]
fn decode_equivalent_attention_prefill_matches_decode_across_direct_and_gathered_spans() {
    let Some(mut metal) = metal_backend() else {
        return;
    };
    let artifacts = AttentionArtifactWriter::from_env();
    assert!(metal.decode_equivalent_prefill_supported());
    let capacities = [3, 4, 5, 6];
    let total_tokens = capacities.iter().sum();
    let shape = AttentionShape::new(2, 1, 32, total_tokens).expect("span shape");
    let projected = shape.projected_kv_elements().expect("span projected KV");
    let key_values = attention_values(total_tokens * projected, 0x4b45_5900);
    let value_values = attention_values(total_tokens * projected, 0x5641_4c00);
    let direct = run_span_case(&mut metal, &capacities, &key_values, &value_values, shape);
    let gathered = run_span_case(
        &mut metal,
        &[2, 4, 3, 5, 4],
        &key_values,
        &value_values,
        shape,
    );
    assert_eq!(
        f32_bits(&direct.prefill),
        f32_bits(&direct.repeated),
        "four-span prefill differs from repeated decode"
    );
    assert_eq!(
        f32_bits(&gathered.prefill),
        f32_bits(&gathered.repeated),
        "five-span prefill differs from repeated decode"
    );
    assert_eq!(
        f32_bits(&direct.prefill),
        f32_bits(&gathered.prefill),
        "direct and gathered prefill differ"
    );
    assert_eq!(
        f32_bits(&direct.repeated),
        f32_bits(&gathered.repeated),
        "direct and gathered decode differ"
    );
    artifacts.write("span_direct_prefill", &direct.prefill);
    artifacts.write("span_direct_repeated_decode", &direct.repeated);
    artifacts.write("span_gathered_prefill", &gathered.prefill);
    artifacts.write("span_gathered_repeated_decode", &gathered.repeated);
}

#[test]
fn tiled_attention_prefill_matches_independent_oracle_across_reduction_ranges() {
    let Some(mut metal) = metal_backend() else {
        return;
    };
    let artifacts = AttentionArtifactWriter::from_env();
    assert!(metal.decode_equivalent_prefill_supported());
    for &(head_dim, context, seed) in &[
        (31, 512, 0x3100_0001),
        (32, 512, 0x3200_0001),
        (33, 512, 0x3300_0001),
        (63, 981, 0x6300_0002),
        (64, 981, 0x6400_0002),
        (65, 981, 0x6500_0002),
        (127, 512, 0x1270_0003),
        (128, 512, 0x1280_0003),
        (129, 512, 0x1290_0003),
        (255, 512, 0x2550_0004),
        (256, 981, 0x2560_0004),
        (257, 512, 0x2570_0005),
    ] {
        run_contiguous_oracle_case(&mut metal, head_dim, context, seed, &artifacts);
    }
}

#[test]
fn tiled_span_attention_prefill_matches_independent_oracle_with_padding() {
    let Some(mut metal) = metal_backend() else {
        return;
    };
    let artifacts = AttentionArtifactWriter::from_env();
    assert!(metal.decode_equivalent_prefill_supported());
    let direct = run_span_oracle_case(
        &mut metal,
        64,
        &[(127, 131), (89, 97), (173, 181), (123, 137)],
        0x5a00_0001,
        "span_oracle_direct_head_dim_64_context_512",
        &artifacts,
    );
    let gathered = run_span_oracle_case(
        &mut metal,
        64,
        &[(73, 79), (211, 223), (101, 113), (67, 71), (60, 67)],
        0x5a00_0001,
        "span_oracle_gathered_head_dim_64_context_512",
        &artifacts,
    );
    assert_eq!(
        f32_bits(&direct),
        f32_bits(&gathered),
        "direct and gathered stress attention differ"
    );
    run_span_oracle_case(
        &mut metal,
        128,
        &[(127, 131), (241, 251), (173, 181), (211, 223), (229, 241)],
        0x5a00_0002,
        "span_oracle_gathered_head_dim_128_context_981",
        &artifacts,
    );
}

fn run_contiguous_oracle_case(
    metal: &mut MetalBackend,
    head_dim: usize,
    context: usize,
    seed: u32,
    artifacts: &AttentionArtifactWriter,
) {
    let shape = AttentionShape::new(4, 2, head_dim, context).expect("oracle attention shape");
    let tokens = 2;
    let start_position = context - tokens;
    let projected = shape.projected_kv_elements().expect("oracle projected KV");
    let query_elements = shape.query_elements().expect("oracle query elements");
    let key_values = attention_stress_values(context * projected, seed);
    let value_values = attention_stress_values(context * projected, seed ^ 0x55aa_00ff);
    assert_context_diversity(&key_values, projected, "contiguous stress keys");
    assert_context_diversity(&value_values, projected, "contiguous stress values");
    let query_values = attention_stress_query_values(tokens * query_elements, seed ^ 0xaa55_ff00);
    let key = upload_f32(metal, &key_values);
    let value = upload_f32(metal, &value_values);
    let mut key_cache = allocate_f16_cache(metal, shape.cache_elements().expect("oracle cache"));
    let mut value_cache = allocate_f16_cache(metal, shape.cache_elements().expect("oracle cache"));
    metal
        .kv_append_chunk(
            &key,
            &value,
            &mut key_cache,
            &mut value_cache,
            shape,
            0,
            context,
        )
        .expect("oracle KV append");
    let cache_keys = read_f16(
        metal,
        &key_cache,
        shape.cache_elements().expect("oracle cache"),
    );
    let cache_values = read_f16(
        metal,
        &value_cache,
        shape.cache_elements().expect("oracle cache"),
    );
    let query = upload_f32(metal, &query_values);
    let mut output = metal
        .allocate(BufferLayout::f32(tokens * query_elements).expect("oracle output layout"))
        .expect("oracle output");
    if head_dim <= 128 && head_dim.is_multiple_of(2) {
        prepare_decode_equivalent(
            metal,
            tokens,
            shape.max_context(),
            shape.n_head(),
            shape.n_head_kv(),
            shape.head_dim(),
            query_elements,
            query_elements,
            query_elements,
        );
    } else {
        prepare_backend_preferred(
            metal,
            tokens,
            shape.max_context(),
            shape.n_head(),
            shape.n_head_kv(),
            shape.head_dim(),
            query_elements,
            query_elements,
            query_elements,
        );
    }
    metal
        .attention_prefill(
            &query,
            &key_cache,
            &value_cache,
            &mut output,
            shape,
            start_position,
            tokens,
        )
        .expect("oracle prefill attention");
    let actual = read_f32(metal, &output, tokens * query_elements);
    let expected = f64_attention_oracle(
        &query_values,
        &cache_keys,
        &cache_values,
        shape,
        start_position,
        tokens,
    );
    assert_close(
        "independent contiguous attention oracle",
        &actual,
        &expected,
        8e-3,
        8e-3,
    );
    assert_output_diversity(&expected, query_elements, "contiguous stress oracle");
    artifacts.write(
        &format!("contiguous_oracle_head_dim_{head_dim}_context_{context}"),
        &actual,
    );
}

struct SpanOracleCache {
    key: MetalBuffer,
    value: MetalBuffer,
    logical_start: usize,
    tokens: usize,
    capacity: usize,
}

fn run_span_oracle_case(
    metal: &mut MetalBackend,
    head_dim: usize,
    layout: &[(usize, usize)],
    seed: u32,
    label: &str,
    artifacts: &AttentionArtifactWriter,
) -> Vec<f32> {
    let context = layout.iter().map(|(tokens, _)| tokens).sum();
    let shape = AttentionShape::new(4, 2, head_dim, context).expect("span oracle shape");
    let tokens = 2;
    let start_position = context - tokens;
    let projected = shape
        .projected_kv_elements()
        .expect("span oracle projected KV");
    let query_elements = shape.query_elements().expect("span oracle query elements");
    let key_values = attention_stress_values(context * projected, seed);
    let value_values = attention_stress_values(context * projected, seed ^ 0x55aa_00ff);
    assert_context_diversity(&key_values, projected, "span stress keys");
    assert_context_diversity(&value_values, projected, "span stress values");
    let query_values = attention_stress_query_values(tokens * query_elements, seed ^ 0xaa55_ff00);
    let caches = build_span_oracle_caches(metal, shape, layout, &key_values, &value_values);
    let spans = caches
        .iter()
        .map(|cache| {
            KvReadSpan::new(
                &cache.key,
                &cache.value,
                cache.logical_start,
                cache.tokens,
                cache.capacity,
            )
            .expect("span oracle descriptor")
        })
        .collect::<Vec<_>>();
    let query = upload_f32(metal, &query_values);
    let mut output = metal
        .allocate(BufferLayout::f32(tokens * query_elements).expect("span oracle output layout"))
        .expect("span oracle output");
    prepare_decode_equivalent(
        metal,
        tokens,
        shape.max_context(),
        shape.n_head(),
        shape.n_head_kv(),
        shape.head_dim(),
        query_elements,
        query_elements,
        query_elements,
    );
    let actual = {
        let view = KvReadView::new(&spans).expect("span oracle view");
        metal
            .attention_prefill_spans(&query, view, &mut output, shape, start_position, tokens)
            .expect("span oracle prefill attention");
        read_f32(metal, &output, tokens * query_elements)
    };
    let (cache_keys, cache_values) = flatten_span_oracle(metal, shape, &caches);
    let expected = f64_attention_oracle(
        &query_values,
        &cache_keys,
        &cache_values,
        shape,
        start_position,
        tokens,
    );
    assert_close(
        "independent span attention oracle",
        &actual,
        &expected,
        8e-3,
        8e-3,
    );
    assert_output_diversity(&expected, query_elements, "span stress oracle");
    artifacts.write(label, &actual);
    actual
}

fn build_span_oracle_caches(
    metal: &mut MetalBackend,
    shape: AttentionShape,
    layout: &[(usize, usize)],
    key_values: &[f32],
    value_values: &[f32],
) -> Vec<SpanOracleCache> {
    let projected = shape.projected_kv_elements().expect("span projected KV");
    let mut logical_start = 0;
    let mut caches = Vec::with_capacity(layout.len());
    for &(tokens, capacity) in layout {
        let physical_shape = AttentionShape::new(
            shape.n_head(),
            shape.n_head_kv(),
            shape.head_dim(),
            capacity,
        )
        .expect("physical span shape");
        let source_start = logical_start * projected;
        let source_end = (logical_start + tokens) * projected;
        let key = upload_f32(metal, &key_values[source_start..source_end]);
        let value = upload_f32(metal, &value_values[source_start..source_end]);
        let mut key_cache = allocate_f16_cache_with_value(
            metal,
            physical_shape
                .cache_elements()
                .expect("physical span cache"),
            65_504.0,
        );
        let mut value_cache = allocate_f16_cache_with_value(
            metal,
            physical_shape
                .cache_elements()
                .expect("physical span cache"),
            -65_504.0,
        );
        metal
            .kv_append_chunk(
                &key,
                &value,
                &mut key_cache,
                &mut value_cache,
                physical_shape,
                0,
                tokens,
            )
            .expect("physical span KV append");
        caches.push(SpanOracleCache {
            key: key_cache,
            value: value_cache,
            logical_start,
            tokens,
            capacity,
        });
        logical_start += tokens;
    }
    assert_eq!(logical_start, shape.max_context());
    caches
}

fn flatten_span_oracle(
    metal: &mut MetalBackend,
    shape: AttentionShape,
    caches: &[SpanOracleCache],
) -> (Vec<f32>, Vec<f32>) {
    let head_dim = shape.head_dim();
    let cache_elements = shape.cache_elements().expect("flat oracle cache");
    let mut keys = vec![0.0; cache_elements];
    let mut values = vec![0.0; cache_elements];
    for cache in caches {
        let physical_shape =
            AttentionShape::new(shape.n_head(), shape.n_head_kv(), head_dim, cache.capacity)
                .expect("physical oracle shape");
        let physical_elements = physical_shape
            .cache_elements()
            .expect("physical oracle cache");
        let source_keys = read_f16(metal, &cache.key, physical_elements);
        let source_values = read_f16(metal, &cache.value, physical_elements);
        for kv_head in 0..shape.n_head_kv() {
            for local in 0..cache.tokens {
                let source = (kv_head * cache.capacity + local) * head_dim;
                let destination =
                    (kv_head * shape.max_context() + cache.logical_start + local) * head_dim;
                keys[destination..destination + head_dim]
                    .copy_from_slice(&source_keys[source..source + head_dim]);
                values[destination..destination + head_dim]
                    .copy_from_slice(&source_values[source..source + head_dim]);
            }
        }
    }
    (keys, values)
}

fn run_span_case(
    metal: &mut leone_metal::MetalBackend,
    capacities: &[usize],
    key_values: &[f32],
    value_values: &[f32],
    shape: AttentionShape,
) -> SpanAttentionOutputs {
    let start_position = 7;
    let tokens = 3;
    let projected = shape.projected_kv_elements().expect("span projected KV");
    let query_elements = shape.query_elements().expect("span query elements");
    let mut caches = Vec::with_capacity(capacities.len());
    let mut logical_start = 0;
    for &capacity in capacities {
        let physical_shape = AttentionShape::new(2, 1, 32, capacity).expect("physical span shape");
        let start = logical_start * projected;
        let end = (logical_start + capacity) * projected;
        let key = upload_f32(metal, &key_values[start..end]);
        let value = upload_f32(metal, &value_values[start..end]);
        let mut key_cache = allocate_f16_cache(metal, physical_shape.cache_elements().unwrap());
        let mut value_cache = allocate_f16_cache(metal, physical_shape.cache_elements().unwrap());
        metal
            .kv_append_chunk(
                &key,
                &value,
                &mut key_cache,
                &mut value_cache,
                physical_shape,
                0,
                capacity,
            )
            .expect("span KV append");
        caches.push((key_cache, value_cache, logical_start, capacity));
        logical_start += capacity;
    }
    let spans = caches
        .iter()
        .map(|(key, value, logical_start, capacity)| {
            KvReadSpan::new(key, value, *logical_start, *capacity, *capacity).expect("read span")
        })
        .collect::<Vec<_>>();
    let view = KvReadView::new(&spans).expect("span view");
    let query_values = attention_values(tokens * query_elements, 0x5152_5900);
    let query = upload_f32(metal, &query_values);
    let mut batched = metal
        .allocate(BufferLayout::f32(tokens * query_elements).expect("span prefill layout"))
        .expect("span prefill output");
    prepare_decode_equivalent(
        metal,
        tokens,
        shape.max_context(),
        shape.n_head(),
        shape.n_head_kv(),
        shape.head_dim(),
        query_elements,
        query_elements,
        query_elements,
    );
    metal
        .attention_prefill_spans(&query, view, &mut batched, shape, start_position, tokens)
        .expect("span prefill attention");
    let actual = read_f32(metal, &batched, tokens * query_elements);
    let mut repeated = Vec::with_capacity(actual.len());
    for token in 0..tokens {
        let token_query = upload_f32(
            metal,
            &query_values[token * query_elements..(token + 1) * query_elements],
        );
        let mut token_output = metal
            .allocate(BufferLayout::f32(query_elements).expect("span decode layout"))
            .expect("span decode output");
        metal
            .attention_decode_spans(
                &token_query,
                KvReadView::new(&spans).expect("decode span view"),
                &mut token_output,
                shape,
                Position::Host(start_position + token),
            )
            .expect("span decode attention");
        repeated.extend(read_f32(metal, &token_output, query_elements));
    }
    SpanAttentionOutputs {
        prefill: actual,
        repeated,
    }
}
