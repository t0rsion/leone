const DIMENSION: usize = 256;
const Q4_BYTES: usize = 36_864;

fn put_string(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn put_metadata_string(bytes: &mut Vec<u8>, key: &str, value: &str) {
    put_string(bytes, key);
    bytes.extend_from_slice(&8_u32.to_le_bytes());
    put_string(bytes, value);
}

fn put_metadata_u32(bytes: &mut Vec<u8>, key: &str, value: u32) {
    put_string(bytes, key);
    bytes.extend_from_slice(&4_u32.to_le_bytes());
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn put_metadata_f32(bytes: &mut Vec<u8>, key: &str, value: f32) {
    put_string(bytes, key);
    bytes.extend_from_slice(&6_u32.to_le_bytes());
    bytes.extend_from_slice(&value.to_bits().to_le_bytes());
}

fn put_metadata_strings(bytes: &mut Vec<u8>, key: &str, values: &[String]) {
    put_string(bytes, key);
    bytes.extend_from_slice(&9_u32.to_le_bytes());
    bytes.extend_from_slice(&8_u32.to_le_bytes());
    bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for value in values {
        put_string(bytes, value);
    }
}

fn put_metadata_i32s(bytes: &mut Vec<u8>, key: &str, values: &[i32]) {
    put_string(bytes, key);
    bytes.extend_from_slice(&9_u32.to_le_bytes());
    bytes.extend_from_slice(&5_u32.to_le_bytes());
    bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
}

fn tensors() -> Vec<(&'static str, Vec<u64>, u32, usize)> {
    let dense = DIMENSION * 4;
    let quant = Q4_BYTES;
    vec![
        ("token_embd.weight", vec![256, 256], 12, quant),
        ("output_norm.weight", vec![256], 0, dense),
        ("blk.0.attn_norm.weight", vec![256], 0, dense),
        ("blk.0.attn_q.weight", vec![256, 256], 12, quant),
        ("blk.0.attn_k.weight", vec![256, 256], 12, quant),
        ("blk.0.attn_v.weight", vec![256, 256], 12, quant),
        ("blk.0.attn_q_norm.weight", vec![256], 0, dense),
        ("blk.0.attn_k_norm.weight", vec![256], 0, dense),
        ("blk.0.attn_output.weight", vec![256, 256], 12, quant),
        ("blk.0.ffn_norm.weight", vec![256], 0, dense),
        ("blk.0.ffn_gate.weight", vec![256, 256], 12, quant),
        ("blk.0.ffn_up.weight", vec![256, 256], 12, quant),
        ("blk.0.ffn_down.weight", vec![256, 256], 12, quant),
    ]
}

/// Returns a one-layer Qwen3 GGUF with zeroed weights for CPU lifecycle tests.
pub fn bytes() -> Vec<u8> {
    let tokens = vec!["a".to_owned(); DIMENSION];
    let token_types = vec![1_i32; DIMENSION];
    let tensors = tensors();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3_u32.to_le_bytes());
    bytes.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&14_u64.to_le_bytes());
    put_metadata_string(&mut bytes, "general.architecture", "qwen3");
    put_metadata_u32(&mut bytes, "qwen3.block_count", 1);
    put_metadata_u32(&mut bytes, "qwen3.attention.head_count", 1);
    put_metadata_u32(&mut bytes, "qwen3.attention.head_count_kv", 1);
    put_metadata_u32(&mut bytes, "qwen3.embedding_length", 256);
    put_metadata_u32(&mut bytes, "qwen3.feed_forward_length", 256);
    put_metadata_f32(&mut bytes, "qwen3.rope.freq_base", 10_000.0);
    put_metadata_f32(&mut bytes, "qwen3.attention.layer_norm_rms_epsilon", 1e-6);
    put_metadata_u32(&mut bytes, "qwen3.context_length", 64);
    put_metadata_string(&mut bytes, "tokenizer.ggml.model", "gpt2");
    put_metadata_string(&mut bytes, "tokenizer.ggml.pre", "qwen2");
    put_metadata_strings(&mut bytes, "tokenizer.ggml.tokens", &tokens);
    put_metadata_i32s(&mut bytes, "tokenizer.ggml.token_type", &token_types);
    put_metadata_strings(&mut bytes, "tokenizer.ggml.merges", &[]);
    let mut offset = 0_u64;
    for (name, shape, dtype, size) in &tensors {
        put_string(&mut bytes, name);
        bytes.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        for dimension in shape {
            bytes.extend_from_slice(&dimension.to_le_bytes());
        }
        bytes.extend_from_slice(&dtype.to_le_bytes());
        bytes.extend_from_slice(&offset.to_le_bytes());
        offset += *size as u64;
    }
    let tensor_data_start = (bytes.len() + 31) & !31;
    bytes.resize(tensor_data_start, 0);
    bytes.resize(tensor_data_start + offset as usize, 0);
    bytes
}
