#![deny(unsafe_code)]

//! Measures native Metal operations and a short model-backed prefill and decode run.
//!
//! The operation fixture is synthetic and deterministic. A model path adds a real
//! GGUF run through Runtime. Each invocation writes linked runtime and quality
//! receipts. Quality stays unverified until an independent oracle is supplied.

use chrono::{SecondsFormat, Utc};
use half::f16;
use leone::{
    AttentionShape, Backend, BufferLayout, DecodeExecution, MemoryAccounting, MemoryBudget,
    Position, QuantFormat, QuantMatrix, Runtime, VectorShape,
};
use leone_gguf::{model::ModelConfig as GgufModelConfig, GgmlType, Gguf};
use leone_metal::{MetalBackend, MetalBuffer, MetalDeviceInfo};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::env;
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
#[cfg(unix)]
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const DEFAULT_PROMPT: &str = "Explain one bounded memory optimization for a consumer GPU.";
const DEFAULT_REPETITIONS: usize = 5;
const DEFAULT_DECODE_TOKENS: usize = 8;
const DEFAULT_PREFILL_CHUNK: usize = 128;
const DEFAULT_PREFILL_TOKENS: usize = 32;
const MAX_REPETITIONS: usize = 32;
const MAX_DECODE_TOKENS: usize = 64;
const MAX_PREFILL_CHUNK: usize = 256;
const MAX_PREFILL_TOKENS: usize = 256;
const MAX_PROMPT_BYTES: usize = 4096;
const MAX_FIXTURE_HOST_BYTES: usize = 128 * 1024 * 1024;
const MAX_SYNTHETIC_CONTEXT: usize = 256;
const DEFAULT_FIXTURE: FixtureConfig = FixtureConfig {
    embedding: 4096,
    feed_forward: 12288,
    heads: 32,
    heads_kv: 8,
    head_dim: 128,
    context: MAX_SYNTHETIC_CONTEXT,
    model_derived: false,
    primary_format: QuantFormat::Q4K,
    secondary_format: None,
};

#[derive(Debug)]
struct Arguments {
    model: Option<PathBuf>,
    prompt: String,
    repetitions: usize,
    decode_tokens: usize,
    prefill_tokens: usize,
    prefill_chunk: usize,
    memory_budget_bytes: Option<u64>,
    receipt_dir: PathBuf,
    synthetic: bool,
}

#[derive(Debug, Serialize)]
struct ProfileReceipt {
    schema_version: &'static str,
    receipt_id: String,
    created_utc: String,
    runtime_receipt_id: String,
    quality_receipt_id: String,
    input_sha256: String,
    source: SourceIdentity,
    build: BuildIdentity,
    invocation: Invocation,
    backend: &'static str,
    device: DeviceReceipt,
    fixture: FixtureReceipt,
    memory: MemoryEnvelope,
    operations: Vec<OperationReceipt>,
    model_run: Option<ModelRunReceipt>,
    quality: QualityLink,
    notes: Vec<String>,
}

#[derive(Debug, Serialize)]
struct QualityReceipt {
    schema_version: &'static str,
    receipt_id: String,
    created_utc: String,
    runtime_receipt_id: String,
    status: &'static str,
    oracle: Option<String>,
    model_sha256: Option<String>,
    input_sha256: String,
    reason: &'static str,
}

#[derive(Debug, Serialize)]
struct SourceIdentity {
    git_commit: String,
    source_tree_sha256: String,
    source_tree_dirty: bool,
    source_paths: Vec<String>,
}

#[derive(Debug, Serialize)]
struct BuildIdentity {
    profile: &'static str,
    target: String,
    host: &'static str,
    package_version: &'static str,
    executable_sha256: String,
    rustc: &'static str,
    rustc_sha256: &'static str,
    toolchain_sha256: &'static str,
    tool_versions: &'static str,
    features: &'static str,
    build_flags: &'static str,
    build_flags_sha256: &'static str,
    build_flags_raw_sha256: &'static str,
    build_flags_redacted: bool,
    build_flags_redaction_count: usize,
    profile_inputs: &'static str,
    profile_inputs_sha256: &'static str,
    linker_inputs: &'static str,
    linker_inputs_sha256: &'static str,
    native_provenance: &'static str,
    native_tools_sha256: &'static str,
    native_tool_versions: &'static str,
    build_config_sha256: &'static str,
    provenance_unknown: bool,
}

#[derive(Debug, Serialize)]
struct Invocation {
    argv_sha256: String,
    argument_count: usize,
}

#[derive(Debug, Serialize)]
struct DeviceReceipt {
    max_buffer_length: u64,
    recommended_working_set: u64,
    driver_current_allocated_at_init: u64,
    registry_id: u64,
    device_name: String,
    architecture_name: String,
    os_version: String,
    compiler_version: Option<String>,
    shader_source_hash: String,
    fast_math_enabled: bool,
    capacity: String,
}

#[derive(Debug)]
struct ActiveDeviceMetadata {
    registry_id: u64,
    device_name: String,
    architecture_name: String,
    os_version: String,
    compiler_version: Option<String>,
    shader_source_hash: String,
    fast_math_enabled: bool,
}

#[derive(Debug, Serialize)]
struct FixtureReceipt {
    kind: &'static str,
    scopes: Vec<&'static str>,
    model_derived: bool,
    seed: u64,
    independent_validation: &'static str,
}

#[derive(Debug, Serialize)]
struct MemoryEnvelope {
    before: MemoryReceipt,
    after: MemoryReceipt,
    peak: MemoryReceipt,
    scope: &'static str,
}

#[derive(Debug, Serialize, Clone)]
struct MemoryReceipt {
    scope: &'static str,
    live_bytes: u64,
    reserved_bytes: u64,
    peak_live_bytes: u64,
    peak_owned_and_reserved_bytes: u64,
    live_allocations: u64,
    peak_live_allocations: u64,
    allocations: u64,
    frees: u64,
    untracked_objects: u64,
    budget: String,
    classes: BTreeMap<String, MemoryClassReceipt>,
}

#[derive(Debug, Serialize, Clone)]
struct MemoryClassReceipt {
    live_bytes: u64,
    peak_live_bytes: u64,
    live_allocations: u64,
    peak_live_allocations: u64,
    allocations: u64,
    frees: u64,
}

#[derive(Debug, Serialize)]
struct OperationReceipt {
    name: &'static str,
    phase: &'static str,
    fixture: &'static str,
    shape: ShapeReceipt,
    repetitions: usize,
    setup_wall_time_ns: u64,
    warmup_wall_time_ns: u64,
    wall_time_ns: Vec<u64>,
    wall_time_ns_median: u64,
    wall_time_ns_min: u64,
    wall_time_ns_max: u64,
    wall_cost: &'static str,
    dispatches_per_sample: usize,
    estimated_bytes: ByteEstimate,
    measured_device_bytes: Option<u64>,
    memory_before_setup: MemoryReceipt,
    memory_before: MemoryReceipt,
    memory_after: MemoryReceipt,
}

struct OperationTiming {
    repetitions: usize,
    setup_wall_time_ns: u64,
    warmup_wall_time_ns: u64,
    wall_time_ns: Vec<u64>,
    memory_before_setup: MemoryReceipt,
    memory_before: MemoryReceipt,
    memory_after: MemoryReceipt,
    dispatches_per_sample: usize,
}

struct SampleTiming {
    warmup_wall_time_ns: u64,
    wall_time_ns: Vec<u64>,
}

#[derive(Debug, Serialize)]
struct ShapeReceipt {
    rows: Option<usize>,
    columns: Option<usize>,
    tokens: Option<usize>,
    n_head: Option<usize>,
    n_head_kv: Option<usize>,
    head_dim: Option<usize>,
    context_tokens: Option<usize>,
    format: Option<&'static str>,
}

#[derive(Debug, Serialize)]
struct ByteEstimate {
    weight_bytes: u64,
    input_read_bytes: u64,
    auxiliary_read_bytes: u64,
    output_write_bytes: u64,
    total_bytes: u64,
    provenance: &'static str,
}

#[derive(Debug, Serialize)]
struct ModelRunReceipt {
    sha256: String,
    file_bytes: u64,
    architecture: String,
    config: ModelConfigReceipt,
    prompt: InputReceipt,
    repetitions: usize,
    decode_tokens: usize,
    prefill_chunk_tokens: usize,
    wall_cost: &'static str,
    warmup_prefill_wall_time_ns: u64,
    warmup_decode_wall_time_ns: u64,
    prefill_total_wall_time_ns: Vec<u64>,
    prefill_setup_wall_time_ns: Vec<u64>,
    prefill_wall_time_ns: Vec<u64>,
    prefill_ttft_ns: Vec<u64>,
    prefill_tok_s: Vec<f64>,
    decode_total_wall_time_ns: Vec<u64>,
    decode_setup_wall_time_ns: Vec<u64>,
    decode_wall_time_ns: Vec<u64>,
    decode_ttft_ns: Vec<u64>,
    decode_tok_s: Vec<f64>,
    prefill_methods: Vec<String>,
    decode_transcript_sha256: Vec<String>,
    transcript_consistent: bool,
    memory_before_load: MemoryReceipt,
    memory_while_runtime_live: MemoryReceipt,
}

#[derive(Debug, Clone, Copy)]
struct FixtureConfig {
    embedding: usize,
    feed_forward: usize,
    heads: usize,
    heads_kv: usize,
    head_dim: usize,
    context: usize,
    model_derived: bool,
    primary_format: QuantFormat,
    secondary_format: Option<QuantFormat>,
}

struct ModelSamples {
    prefill_total_wall_time_ns: Vec<u64>,
    prefill_setup_wall_time_ns: Vec<u64>,
    prefill_wall_time_ns: Vec<u64>,
    prefill_ttft_ns: Vec<u64>,
    prefill_tok_s: Vec<f64>,
    decode_total_wall_time_ns: Vec<u64>,
    decode_setup_wall_time_ns: Vec<u64>,
    decode_wall_time_ns: Vec<u64>,
    decode_ttft_ns: Vec<u64>,
    decode_tok_s: Vec<f64>,
    prefill_methods: Vec<String>,
    transcripts: Vec<String>,
}

struct AttentionInputs<'a> {
    shape: AttentionShape,
    context: usize,
    key_cache: &'a MetalBuffer,
    value_cache: &'a MetalBuffer,
}

struct QkvBuffers {
    query_shape: QuantMatrix,
    key_shape: QuantMatrix,
    value_shape: QuantMatrix,
    query_weights: MetalBuffer,
    key_weights: MetalBuffer,
    value_weights: MetalBuffer,
    input: MetalBuffer,
    query: MetalBuffer,
    key: MetalBuffer,
    value: MetalBuffer,
}

struct QkvShapes {
    query_rows: usize,
    kv_rows: usize,
    query: QuantMatrix,
    key: QuantMatrix,
    value: QuantMatrix,
}

#[derive(Debug, Serialize)]
struct ModelConfigReceipt {
    layers: usize,
    heads: usize,
    heads_kv: usize,
    embedding: usize,
    feed_forward: usize,
    head_dim: usize,
    vocabulary: usize,
    context: usize,
}

#[derive(Debug, Serialize)]
struct InputReceipt {
    prompt_sha256: String,
    token_sha256: String,
    token_count: usize,
    artifact: Option<String>,
    #[serde(skip)]
    token_ids: Vec<u32>,
}

#[derive(Debug, Serialize)]
struct InputArtifact {
    schema_version: &'static str,
    model_sha256: String,
    prompt_sha256: String,
    token_sha256: String,
    token_count: usize,
    token_ids: Vec<u32>,
}

#[derive(Debug, Serialize)]
struct QualityLink {
    status: &'static str,
    receipt: String,
}

#[derive(Debug, Eq, PartialEq)]
struct ModelIdentity {
    sha256: String,
    file_bytes: u64,
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
}

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = parse_arguments(env::args().skip(1).collect())?;
    let (source, build) = validated_provenance()?;
    let started = Utc::now();
    let receipt_id = format!(
        "metal-profile-{}-{}",
        started.format("%Y%m%dT%H%M%SZ"),
        std::process::id()
    );
    let quality_receipt_id = format!("{receipt_id}-quality");
    let runtime_name = format!("{receipt_id}-runtime.json");
    let quality_name = format!("{quality_receipt_id}.json");
    let runtime_path = arguments.receipt_dir.join(&runtime_name);
    let quality_path = arguments.receipt_dir.join(&quality_name);
    let mut profile = collect_profile(
        &arguments,
        &receipt_id,
        &quality_receipt_id,
        &quality_name,
        source,
        build,
    )?;
    let input_artifact = write_input_artifact(&arguments.receipt_dir, &profile)?;
    attach_input_artifact(&mut profile, input_artifact);
    let quality = QualityReceipt {
        schema_version: "leone.metal-quality.v1",
        receipt_id: quality_receipt_id,
        created_utc: started.to_rfc3339_opts(SecondsFormat::Nanos, true),
        runtime_receipt_id: profile.runtime_receipt_id.clone(),
        status: "unverified",
        oracle: None,
        model_sha256: profile.model_run.as_ref().map(|run| run.sha256.clone()),
        input_sha256: profile.input_sha256.clone(),
        reason: "No independent BF16 or scalar quality oracle was supplied.",
    };
    let runtime_json = serialize_json(&profile)?;
    let quality_json = serialize_json(&quality)?;
    write_json(&quality_path, &quality_json)?;
    write_json(&runtime_path, &runtime_json)?;
    println!("runtime receipt: {}", runtime_path.display());
    println!("quality receipt: {}", quality_path.display());
    Ok(())
}

fn attach_input_artifact(profile: &mut ProfileReceipt, artifact: Option<String>) {
    if let Some(artifact) = artifact {
        if let Some(model_run) = profile.model_run.as_mut() {
            model_run.prompt.artifact = Some(artifact);
        }
    }
}

fn collect_profile(
    arguments: &Arguments,
    receipt_id: &str,
    quality_receipt_id: &str,
    quality_name: &str,
    source: SourceIdentity,
    build: BuildIdentity,
) -> Result<ProfileReceipt, Box<dyn Error>> {
    let initial_model_identity = arguments
        .model
        .as_deref()
        .map(read_model_identity)
        .transpose()?;
    let fixture = fixture_config(arguments.model.as_deref())?;
    validate_fixture_tokens(arguments, fixture)?;
    let (mut backend, device, device_metadata, before) = prepare_backend(arguments)?;
    let operations = run_requested_operations(&mut backend, arguments, fixture)?;
    let (model_run, after) = finish_workload(backend, arguments, initial_model_identity.as_ref())?;
    let peak = peak_memory(&before, &after);
    let created_utc = Utc::now().to_rfc3339_opts(SecondsFormat::Nanos, true);
    Ok(ProfileReceipt {
        schema_version: "leone.metal-profile.v1",
        receipt_id: receipt_id.to_owned(),
        created_utc,
        runtime_receipt_id: receipt_id.to_owned(),
        quality_receipt_id: quality_receipt_id.to_owned(),
        input_sha256: input_identity(arguments, model_run.as_ref()),
        source,
        build,
        invocation: Invocation {
            argv_sha256: invocation_hash(),
            argument_count: env::args().count(),
        },
        backend: "metal",
        device: device_receipt(device, device_metadata),
        fixture: FixtureReceipt {
            kind: if arguments.synthetic {
                "synthetic_deterministic"
            } else {
                "gguf_model"
            },
            scopes: fixture_scopes(arguments),
            model_derived: fixture.model_derived,
            seed: 0x4c45_4f4e_455f_5034,
            independent_validation:
                "crates/leone-metal/tests/backend.rs, ops_oracle.rs, and rms_oracle.rs; quality remains unverified",
        },
        memory: MemoryEnvelope {
            before,
            after,
            peak,
            scope: "Leone-owned tracked allocations; after includes live runtime resources",
        },
        operations,
        model_run,
        quality: QualityLink {
            status: "unverified",
            receipt: quality_name.to_owned(),
        },
        notes: vec![
            "Operation wall time includes synchronous dispatch and backend wait.".to_owned(),
            "The command sequence measures four synchronous GEMV dispatch and backend wait pairs per sample."
                .to_owned(),
            "Estimated operation bytes are logical reads and writes, not measured device traffic."
                .to_owned(),
            "Metal does not expose measured device traffic through this harness.".to_owned(),
            "Metal shader_source_hash identifies source compiled at runtime by MTLDevice::newLibraryWithSource; it is not an offline metal compiler or metallib hash."
                .to_owned(),
            "Memory peaks are Leone-owned tracked high-water marks, including admitted reservations, not physical Metal peaks."
                .to_owned(),
            "Model prefill records the benchmark's typed method; quality remains unverified."
                .to_owned(),
            "Model setup fields are residual host preparation and allocation time around each benchmark."
                .to_owned(),
            "Model-shaped operation buffers use deterministic fixture bytes, not GGUF tensor payloads."
                .to_owned(),
            "Quality remains unverified until an independent oracle is linked.".to_owned(),
        ],
    })
}

fn validate_fixture_tokens(
    arguments: &Arguments,
    fixture: FixtureConfig,
) -> Result<(), Box<dyn Error>> {
    if arguments.synthetic && arguments.prefill_tokens > fixture.context {
        return Err(format!(
            "prefill tokens {} exceed fixture context {}",
            arguments.prefill_tokens, fixture.context
        )
        .into());
    }
    if arguments.synthetic {
        validate_fixture_host_bytes(fixture, arguments.prefill_tokens)?;
    }
    Ok(())
}

fn prepare_backend(
    arguments: &Arguments,
) -> Result<
    (
        MetalBackend,
        MetalDeviceInfo,
        ActiveDeviceMetadata,
        MemoryReceipt,
    ),
    Box<dyn Error>,
> {
    let mut backend = MetalBackend::new()?;
    let device = backend.device_info();
    let budget = memory_budget(
        arguments.memory_budget_bytes,
        device.recommended_working_set,
    )?;
    backend.set_memory_budget(budget)?;
    let device_metadata = active_device_metadata(&backend)?;
    let before = memory_receipt(&backend.memory_accounting());
    Ok((backend, device, device_metadata, before))
}

fn run_requested_operations(
    backend: &mut MetalBackend,
    arguments: &Arguments,
    fixture: FixtureConfig,
) -> Result<Vec<OperationReceipt>, Box<dyn Error>> {
    if arguments.synthetic {
        run_synthetic_operations(
            backend,
            fixture,
            arguments.prefill_tokens,
            arguments.repetitions,
        )
    } else {
        Ok(Vec::new())
    }
}

fn finish_workload(
    mut backend: MetalBackend,
    arguments: &Arguments,
    initial_model_identity: Option<&ModelIdentity>,
) -> Result<(Option<ModelRunReceipt>, MemoryReceipt), Box<dyn Error>> {
    match &arguments.model {
        Some(path) => {
            let expected_identity = initial_model_identity
                .ok_or("model identity was not captured before fixture setup")?;
            let (run, memory) = run_model(backend, path, arguments, expected_identity)?;
            Ok((Some(run), memory))
        }
        None => {
            backend.synchronize()?;
            Ok((None, memory_receipt(&backend.memory_accounting())))
        }
    }
}

fn run_synthetic_operations(
    backend: &mut MetalBackend,
    fixture: FixtureConfig,
    prefill_tokens: usize,
    repetitions: usize,
) -> Result<Vec<OperationReceipt>, Box<dyn Error>> {
    let mut operations = Vec::new();
    for format in fixture_formats(fixture) {
        for rows in [1, fixture.embedding, fixture.feed_forward] {
            operations.push(run_gemv(
                backend,
                rows,
                fixture.embedding,
                format,
                repetitions,
            )?);
        }
        operations.push(run_qkv_gemv(backend, fixture, format, repetitions)?);
        operations.push(run_gemv_sequence(
            backend,
            fixture.embedding,
            format,
            4,
            repetitions,
        )?);
        operations.push(run_prefill_gemm(
            backend,
            fixture.embedding,
            fixture.embedding,
            format,
            prefill_tokens,
            repetitions,
        )?);
    }
    for rows in [1, 8, 128] {
        operations.push(run_rms_norm(backend, rows, fixture.embedding, repetitions)?);
    }
    operations.extend(run_attention(
        backend,
        fixture,
        prefill_tokens,
        repetitions,
    )?);
    Ok(operations)
}

fn run_gemv(
    backend: &mut MetalBackend,
    rows: usize,
    columns: usize,
    format: QuantFormat,
    repetitions: usize,
) -> Result<OperationReceipt, Box<dyn Error>> {
    let memory_before_setup = memory_receipt(&backend.memory_accounting());
    let setup_started = Instant::now();
    let (shape, weights, input, mut output) = gemv_buffers(backend, rows, columns, format)?;
    let setup_wall_time_ns = duration_ns(setup_started.elapsed());
    let before = memory_receipt(&backend.memory_accounting());
    let timing = measure_repetitions(backend, repetitions, |backend| {
        backend.gemv(&weights, &input, &mut output, shape)
    })?;
    let after = memory_receipt(&backend.memory_accounting());
    let weight_bytes = shape.layout()?.bytes() as u64;
    let input_bytes = (columns * 4) as u64;
    let output_bytes = (rows * 4) as u64;
    Ok(operation_receipt(
        "gemv",
        "decode",
        ShapeReceipt {
            rows: Some(rows),
            columns: Some(columns),
            tokens: Some(1),
            n_head: None,
            n_head_kv: None,
            head_dim: None,
            context_tokens: None,
            format: Some(quant_format_name(format)),
        },
        ByteEstimate {
            weight_bytes,
            input_read_bytes: input_bytes,
            auxiliary_read_bytes: 0,
            output_write_bytes: output_bytes,
            total_bytes: weight_bytes + input_bytes + output_bytes,
            provenance: "logical operation estimate",
        },
        OperationTiming {
            repetitions,
            setup_wall_time_ns,
            warmup_wall_time_ns: timing.warmup_wall_time_ns,
            wall_time_ns: timing.wall_time_ns,
            memory_before_setup,
            memory_before: before,
            memory_after: after,
            dispatches_per_sample: 1,
        },
    ))
}

fn run_qkv_gemv(
    backend: &mut MetalBackend,
    fixture: FixtureConfig,
    format: QuantFormat,
    repetitions: usize,
) -> Result<OperationReceipt, Box<dyn Error>> {
    let memory_before_setup = memory_receipt(&backend.memory_accounting());
    let setup_started = Instant::now();
    let mut buffers = qkv_buffers(backend, fixture, format)?;
    let setup_wall_time_ns = duration_ns(setup_started.elapsed());
    let before = memory_receipt(&backend.memory_accounting());
    let timing = measure_repetitions(backend, repetitions, |backend| {
        backend.qkv_gemv(
            &buffers.query_weights,
            buffers.query_shape,
            &buffers.key_weights,
            buffers.key_shape,
            &buffers.value_weights,
            buffers.value_shape,
            &buffers.input,
            &mut buffers.query,
            &mut buffers.key,
            &mut buffers.value,
        )
    })?;
    let after = memory_receipt(&backend.memory_accounting());
    Ok(operation_receipt(
        "qkv_gemv",
        "decode",
        ShapeReceipt {
            rows: Some(buffers.query_shape.rows()),
            columns: Some(buffers.query_shape.columns()),
            tokens: Some(1),
            n_head: Some(fixture.heads),
            n_head_kv: Some(fixture.heads_kv),
            head_dim: Some(fixture.head_dim),
            context_tokens: None,
            format: Some(quant_format_name(format)),
        },
        qkv_estimate(&buffers)?,
        OperationTiming {
            repetitions,
            setup_wall_time_ns,
            warmup_wall_time_ns: timing.warmup_wall_time_ns,
            wall_time_ns: timing.wall_time_ns,
            memory_before_setup,
            memory_before: before,
            memory_after: after,
            dispatches_per_sample: 3,
        },
    ))
}

fn qkv_buffers(
    backend: &mut MetalBackend,
    fixture: FixtureConfig,
    format: QuantFormat,
) -> Result<QkvBuffers, Box<dyn Error>> {
    let shapes = qkv_shapes(fixture, format)?;
    let (query_weights, key_weights, value_weights) =
        qkv_weights(backend, format, shapes.query, shapes.key, shapes.value)?;
    let input = backend.upload(
        BufferLayout::f32(fixture.embedding)?,
        &f32_bytes(fixture.embedding, 0x514b_5649),
    )?;
    let (query, key, value) = qkv_outputs(backend, shapes.query_rows, shapes.kv_rows)?;
    Ok(QkvBuffers {
        query_shape: shapes.query,
        key_shape: shapes.key,
        value_shape: shapes.value,
        query_weights,
        key_weights,
        value_weights,
        input,
        query,
        key,
        value,
    })
}

fn qkv_shapes(fixture: FixtureConfig, format: QuantFormat) -> Result<QkvShapes, Box<dyn Error>> {
    let query_rows = fixture
        .heads
        .checked_mul(fixture.head_dim)
        .ok_or("QKV query rows overflow")?;
    let kv_rows = fixture
        .heads_kv
        .checked_mul(fixture.head_dim)
        .ok_or("QKV KV rows overflow")?;
    let query_shape = QuantMatrix::new(query_rows, fixture.embedding, format)?;
    let key_shape = QuantMatrix::new(kv_rows, fixture.embedding, format)?;
    let value_shape = QuantMatrix::new(kv_rows, fixture.embedding, format)?;
    Ok(QkvShapes {
        query_rows,
        kv_rows,
        query: query_shape,
        key: key_shape,
        value: value_shape,
    })
}

fn qkv_weights(
    backend: &mut MetalBackend,
    format: QuantFormat,
    query_shape: QuantMatrix,
    key_shape: QuantMatrix,
    value_shape: QuantMatrix,
) -> Result<(MetalBuffer, MetalBuffer, MetalBuffer), Box<dyn Error>> {
    let query = upload_qkv_weight(backend, query_shape, format, 0x514b_5651)?;
    let key = upload_qkv_weight(backend, key_shape, format, 0x514b_564b)?;
    let value = upload_qkv_weight(backend, value_shape, format, 0x514b_5656)?;
    Ok((query, key, value))
}

fn upload_qkv_weight(
    backend: &mut MetalBackend,
    shape: QuantMatrix,
    format: QuantFormat,
    seed: u64,
) -> Result<MetalBuffer, Box<dyn Error>> {
    let layout = shape.layout()?;
    Ok(backend.upload(layout, &quantized_bytes(layout.bytes(), format, seed))?)
}

fn qkv_outputs(
    backend: &mut MetalBackend,
    query_rows: usize,
    kv_rows: usize,
) -> Result<(MetalBuffer, MetalBuffer, MetalBuffer), Box<dyn Error>> {
    let query = backend.allocate(BufferLayout::f32(query_rows)?)?;
    let key = backend.allocate(BufferLayout::f32(kv_rows)?)?;
    let value = backend.allocate(BufferLayout::f32(kv_rows)?)?;
    Ok((query, key, value))
}

fn qkv_estimate(buffers: &QkvBuffers) -> Result<ByteEstimate, Box<dyn Error>> {
    let weight_bytes = qkv_weight_bytes(buffers)?;
    let input_bytes = qkv_input_bytes(buffers)?;
    let output_bytes = qkv_output_bytes(buffers)?;
    let input_reads = input_bytes
        .checked_mul(3)
        .ok_or("QKV input read estimate overflow")?;
    let total_bytes = weight_bytes
        .checked_add(input_reads)
        .and_then(|bytes| bytes.checked_add(output_bytes))
        .ok_or("QKV byte estimate overflow")?;
    Ok(ByteEstimate {
        weight_bytes,
        input_read_bytes: input_reads,
        auxiliary_read_bytes: 0,
        output_write_bytes: output_bytes,
        total_bytes,
        provenance: "logical operation estimate",
    })
}

fn qkv_weight_bytes(buffers: &QkvBuffers) -> Result<u64, Box<dyn Error>> {
    let query = buffers.query_shape.layout()?.bytes();
    let key = buffers.key_shape.layout()?.bytes();
    let value = buffers.value_shape.layout()?.bytes();
    Ok(query
        .checked_add(key)
        .and_then(|bytes| bytes.checked_add(value))
        .ok_or("QKV weight byte estimate overflow")? as u64)
}

fn qkv_input_bytes(buffers: &QkvBuffers) -> Result<u64, Box<dyn Error>> {
    Ok(buffers
        .query_shape
        .columns()
        .checked_mul(4)
        .ok_or("QKV input byte estimate overflow")? as u64)
}

fn qkv_output_bytes(buffers: &QkvBuffers) -> Result<u64, Box<dyn Error>> {
    let rows = buffers
        .query_shape
        .rows()
        .checked_add(buffers.key_shape.rows())
        .and_then(|rows| rows.checked_add(buffers.value_shape.rows()))
        .ok_or("QKV output rows overflow")?;
    Ok(rows
        .checked_mul(4)
        .ok_or("QKV output byte estimate overflow")? as u64)
}

fn run_gemv_sequence(
    backend: &mut MetalBackend,
    columns: usize,
    format: QuantFormat,
    dispatches: usize,
    repetitions: usize,
) -> Result<OperationReceipt, Box<dyn Error>> {
    let rows = 1;
    let memory_before_setup = memory_receipt(&backend.memory_accounting());
    let setup_started = Instant::now();
    let (shape, weights, input, mut output) = gemv_buffers(backend, rows, columns, format)?;
    let setup_wall_time_ns = duration_ns(setup_started.elapsed());
    let before = memory_receipt(&backend.memory_accounting());
    let timing = measure_repetitions(backend, repetitions, |backend| {
        for _ in 0..dispatches {
            backend.gemv(&weights, &input, &mut output, shape)?;
        }
        Ok(())
    })?;
    let after = memory_receipt(&backend.memory_accounting());
    let single = gemv_estimate(shape, rows, columns)?;
    Ok(operation_receipt(
        "gemv_command_sequence",
        "decode",
        ShapeReceipt {
            rows: Some(rows),
            columns: Some(columns),
            tokens: Some(dispatches),
            n_head: None,
            n_head_kv: None,
            head_dim: None,
            context_tokens: None,
            format: Some(quant_format_name(format)),
        },
        scale_bytes(single, dispatches),
        OperationTiming {
            repetitions,
            setup_wall_time_ns,
            warmup_wall_time_ns: timing.warmup_wall_time_ns,
            wall_time_ns: timing.wall_time_ns,
            memory_before_setup,
            memory_before: before,
            memory_after: after,
            dispatches_per_sample: dispatches,
        },
    ))
}

fn scale_bytes(bytes: ByteEstimate, factor: usize) -> ByteEstimate {
    let factor = factor as u64;
    ByteEstimate {
        weight_bytes: bytes.weight_bytes * factor,
        input_read_bytes: bytes.input_read_bytes * factor,
        auxiliary_read_bytes: bytes.auxiliary_read_bytes * factor,
        output_write_bytes: bytes.output_write_bytes * factor,
        total_bytes: bytes.total_bytes * factor,
        provenance: bytes.provenance,
    }
}

fn quant_format_name(format: QuantFormat) -> &'static str {
    match format {
        QuantFormat::Q4K => "q4_k",
        QuantFormat::Q6K => "q6_k",
    }
}

fn gemv_estimate(
    shape: QuantMatrix,
    rows: usize,
    columns: usize,
) -> Result<ByteEstimate, Box<dyn Error>> {
    let weight_bytes = shape.layout()?.bytes() as u64;
    let input_bytes = (columns * 4) as u64;
    let output_bytes = (rows * 4) as u64;
    Ok(ByteEstimate {
        weight_bytes,
        input_read_bytes: input_bytes,
        auxiliary_read_bytes: 0,
        output_write_bytes: output_bytes,
        total_bytes: weight_bytes + input_bytes + output_bytes,
        provenance: "logical operation estimate",
    })
}

fn gemv_buffers(
    backend: &mut MetalBackend,
    rows: usize,
    columns: usize,
    format: QuantFormat,
) -> Result<(QuantMatrix, MetalBuffer, MetalBuffer, MetalBuffer), Box<dyn Error>> {
    let shape = QuantMatrix::new(rows, columns, format)?;
    let weights = backend.upload(
        shape.layout()?,
        &quantized_bytes(shape.layout()?.bytes(), format, 0x4745_4d56),
    )?;
    let input = backend.upload(
        BufferLayout::f32(columns)?,
        &f32_bytes(columns, 0x494e_5054),
    )?;
    let output = backend.allocate(BufferLayout::f32(rows)?)?;
    Ok((shape, weights, input, output))
}

fn run_prefill_gemm(
    backend: &mut MetalBackend,
    rows: usize,
    columns: usize,
    format: QuantFormat,
    tokens: usize,
    repetitions: usize,
) -> Result<OperationReceipt, Box<dyn Error>> {
    let memory_before_setup = memory_receipt(&backend.memory_accounting());
    let setup_started = Instant::now();
    let (shape, weights, input, mut output) =
        prefill_buffers(backend, rows, columns, format, tokens)?;
    let setup_wall_time_ns = duration_ns(setup_started.elapsed());
    let before = memory_receipt(&backend.memory_accounting());
    let timing = measure_repetitions(backend, repetitions, |backend| {
        backend.prefill_gemm(&weights, &input, &mut output, shape, tokens)
    })?;
    let after = memory_receipt(&backend.memory_accounting());
    let weight_bytes = shape.layout()?.bytes() as u64;
    let input_bytes = (tokens * columns * 4) as u64;
    let output_bytes = (tokens * rows * 4) as u64;
    Ok(operation_receipt(
        "prefill_gemm",
        "prefill",
        ShapeReceipt {
            rows: Some(rows),
            columns: Some(columns),
            tokens: Some(tokens),
            n_head: None,
            n_head_kv: None,
            head_dim: None,
            context_tokens: None,
            format: Some(quant_format_name(format)),
        },
        ByteEstimate {
            weight_bytes,
            input_read_bytes: input_bytes,
            auxiliary_read_bytes: 0,
            output_write_bytes: output_bytes,
            total_bytes: weight_bytes + input_bytes + output_bytes,
            provenance: "logical operation estimate",
        },
        OperationTiming {
            repetitions,
            setup_wall_time_ns,
            warmup_wall_time_ns: timing.warmup_wall_time_ns,
            wall_time_ns: timing.wall_time_ns,
            memory_before_setup,
            memory_before: before,
            memory_after: after,
            dispatches_per_sample: 1,
        },
    ))
}

fn prefill_buffers(
    backend: &mut MetalBackend,
    rows: usize,
    columns: usize,
    format: QuantFormat,
    tokens: usize,
) -> Result<(QuantMatrix, MetalBuffer, MetalBuffer, MetalBuffer), Box<dyn Error>> {
    let shape = QuantMatrix::new(rows, columns, format)?;
    let weights = backend.upload(
        shape.layout()?,
        &quantized_bytes(shape.layout()?.bytes(), format, 0x5052_4546),
    )?;
    let input = backend.upload(
        BufferLayout::f32(tokens * columns)?,
        &f32_bytes(tokens * columns, 0x5052_494e),
    )?;
    let output = backend.allocate(BufferLayout::f32(tokens * rows)?)?;
    Ok((shape, weights, input, output))
}

fn run_rms_norm(
    backend: &mut MetalBackend,
    rows: usize,
    columns: usize,
    repetitions: usize,
) -> Result<OperationReceipt, Box<dyn Error>> {
    let memory_before_setup = memory_receipt(&backend.memory_accounting());
    let setup_started = Instant::now();
    let (shape, input, weight, mut output) = rms_buffers(backend, rows, columns)?;
    let setup_wall_time_ns = duration_ns(setup_started.elapsed());
    let before = memory_receipt(&backend.memory_accounting());
    let timing = measure_repetitions(backend, repetitions, |backend| {
        backend.prefill_rms_norm(&input, &weight, &mut output, shape, 1e-5)
    })?;
    let after = memory_receipt(&backend.memory_accounting());
    let elements = shape.elements()? as u64;
    let input_bytes = elements * 4;
    let output_bytes = elements * 4;
    let weight_bytes = columns as u64 * 4;
    Ok(operation_receipt(
        "rms_norm",
        "prefill",
        ShapeReceipt {
            rows: Some(rows),
            columns: Some(columns),
            tokens: Some(rows),
            n_head: None,
            n_head_kv: None,
            head_dim: None,
            context_tokens: None,
            format: Some("f32"),
        },
        ByteEstimate {
            weight_bytes,
            input_read_bytes: input_bytes,
            auxiliary_read_bytes: 0,
            output_write_bytes: output_bytes,
            total_bytes: weight_bytes + input_bytes + output_bytes,
            provenance: "logical operation estimate",
        },
        OperationTiming {
            repetitions,
            setup_wall_time_ns,
            warmup_wall_time_ns: timing.warmup_wall_time_ns,
            wall_time_ns: timing.wall_time_ns,
            memory_before_setup,
            memory_before: before,
            memory_after: after,
            dispatches_per_sample: 1,
        },
    ))
}

fn rms_buffers(
    backend: &mut MetalBackend,
    rows: usize,
    columns: usize,
) -> Result<(VectorShape, MetalBuffer, MetalBuffer, MetalBuffer), Box<dyn Error>> {
    let shape = VectorShape::new(rows, columns)?;
    let input = upload_f32(backend, shape.elements()?, 0x524d_5349)?;
    let weight = upload_f32(backend, columns, 0x524d_5357)?;
    let output = backend.allocate(BufferLayout::f32(shape.elements()?)?)?;
    Ok((shape, input, weight, output))
}

fn upload_f32(
    backend: &mut MetalBackend,
    elements: usize,
    seed: u64,
) -> Result<MetalBuffer, Box<dyn Error>> {
    Ok(backend.upload(BufferLayout::f32(elements)?, &f32_bytes(elements, seed))?)
}

fn run_attention(
    backend: &mut MetalBackend,
    fixture: FixtureConfig,
    prefill_tokens: usize,
    repetitions: usize,
) -> Result<Vec<OperationReceipt>, Box<dyn Error>> {
    let shape = AttentionShape::new(
        fixture.heads,
        fixture.heads_kv,
        fixture.head_dim,
        fixture.context,
    )?;
    let decode_memory_before_setup = memory_receipt(&backend.memory_accounting());
    let cache_setup_started = Instant::now();
    let (key_cache, value_cache) = attention_cache(backend, shape)?;
    let cache_setup_wall_time_ns = duration_ns(cache_setup_started.elapsed());
    let inputs = AttentionInputs {
        shape,
        context: fixture.context,
        key_cache: &key_cache,
        value_cache: &value_cache,
    };
    let decode = run_attention_decode(
        backend,
        &inputs,
        repetitions,
        decode_memory_before_setup,
        cache_setup_wall_time_ns,
    )?;
    let prefill_memory_before_setup = memory_receipt(&backend.memory_accounting());
    let prefill = run_attention_prefill(
        backend,
        &inputs,
        prefill_tokens,
        repetitions,
        prefill_memory_before_setup,
    )?;
    Ok(vec![decode, prefill])
}

fn attention_cache(
    backend: &mut MetalBackend,
    shape: AttentionShape,
) -> Result<(MetalBuffer, MetalBuffer), Box<dyn Error>> {
    let cache_elements = shape.cache_elements()?;
    let key_cache = backend.upload(
        BufferLayout::f16(cache_elements)?,
        &f16_bytes(cache_elements, 0x4154_544b),
    )?;
    let value_cache = backend.upload(
        BufferLayout::f16(cache_elements)?,
        &f16_bytes(cache_elements, 0x4154_5456),
    )?;
    Ok((key_cache, value_cache))
}

fn run_attention_decode(
    backend: &mut MetalBackend,
    inputs: &AttentionInputs<'_>,
    repetitions: usize,
    memory_before_setup: MemoryReceipt,
    cache_setup_wall_time_ns: u64,
) -> Result<OperationReceipt, Box<dyn Error>> {
    let query_elements = inputs.shape.query_elements()?;
    let setup_started = Instant::now();
    let query = backend.upload(
        BufferLayout::f32(query_elements)?,
        &f32_bytes(query_elements, 0x4154_5451),
    )?;
    let mut output = backend.allocate(BufferLayout::f32(query_elements)?)?;
    let setup_wall_time_ns =
        cache_setup_wall_time_ns.saturating_add(duration_ns(setup_started.elapsed()));
    let before = memory_receipt(&backend.memory_accounting());
    let timing = measure_repetitions(backend, repetitions, |backend| {
        backend.attention_decode(
            &query,
            inputs.key_cache,
            inputs.value_cache,
            &mut output,
            inputs.shape,
            Position::Host(inputs.context - 1),
        )
    })?;
    let after = memory_receipt(&backend.memory_accounting());
    Ok(operation_receipt(
        "attention",
        "decode",
        attention_shape_receipt(inputs.shape, 1, inputs.context),
        attention_bytes(inputs.shape, 1, inputs.context)?,
        OperationTiming {
            repetitions,
            setup_wall_time_ns,
            warmup_wall_time_ns: timing.warmup_wall_time_ns,
            wall_time_ns: timing.wall_time_ns,
            memory_before_setup,
            memory_before: before,
            memory_after: after,
            dispatches_per_sample: 1,
        },
    ))
}

fn run_attention_prefill(
    backend: &mut MetalBackend,
    inputs: &AttentionInputs<'_>,
    tokens: usize,
    repetitions: usize,
    memory_before_setup: MemoryReceipt,
) -> Result<OperationReceipt, Box<dyn Error>> {
    let query_elements = inputs.shape.query_elements()?;
    let setup_started = Instant::now();
    let prefill_query = backend.upload(
        BufferLayout::f32(tokens * query_elements)?,
        &f32_bytes(tokens * query_elements, 0x4154_5051),
    )?;
    let mut output = backend.allocate(BufferLayout::f32(tokens * query_elements)?)?;
    let setup_wall_time_ns = duration_ns(setup_started.elapsed());
    let before = memory_receipt(&backend.memory_accounting());
    let timing = measure_repetitions(backend, repetitions, |backend| {
        backend.attention_prefill(
            &prefill_query,
            inputs.key_cache,
            inputs.value_cache,
            &mut output,
            inputs.shape,
            0,
            tokens,
        )
    })?;
    let after = memory_receipt(&backend.memory_accounting());
    Ok(operation_receipt(
        "attention",
        "prefill",
        attention_shape_receipt(inputs.shape, tokens, inputs.context),
        attention_bytes(inputs.shape, tokens, tokens)?,
        OperationTiming {
            repetitions,
            setup_wall_time_ns,
            warmup_wall_time_ns: timing.warmup_wall_time_ns,
            wall_time_ns: timing.wall_time_ns,
            memory_before_setup,
            memory_before: before,
            memory_after: after,
            dispatches_per_sample: 1,
        },
    ))
}

fn attention_shape_receipt(shape: AttentionShape, tokens: usize, context: usize) -> ShapeReceipt {
    ShapeReceipt {
        rows: None,
        columns: None,
        tokens: Some(tokens),
        n_head: Some(shape.n_head()),
        n_head_kv: Some(shape.n_head_kv()),
        head_dim: Some(shape.head_dim()),
        context_tokens: Some(context),
        format: Some("q_f32_kv_f16"),
    }
}

fn attention_bytes(
    shape: AttentionShape,
    tokens: usize,
    context: usize,
) -> Result<ByteEstimate, Box<dyn Error>> {
    let query_bytes = (tokens * shape.query_elements()? * 4) as u64;
    let output_bytes = query_bytes;
    let kv_per_token = (shape.projected_kv_elements()? * 2 * 2) as u64;
    let kv_tokens = if tokens == 1 {
        context as u64
    } else {
        (tokens * (tokens + 1) / 2) as u64
    };
    let auxiliary_read_bytes = kv_per_token * kv_tokens;
    Ok(ByteEstimate {
        weight_bytes: 0,
        input_read_bytes: query_bytes,
        auxiliary_read_bytes,
        output_write_bytes: output_bytes,
        total_bytes: query_bytes + auxiliary_read_bytes + output_bytes,
        provenance: "logical operation estimate",
    })
}

fn operation_receipt(
    name: &'static str,
    phase: &'static str,
    shape: ShapeReceipt,
    estimated_bytes: ByteEstimate,
    timing: OperationTiming,
) -> OperationReceipt {
    let mut sorted = timing.wall_time_ns.clone();
    sorted.sort_unstable();
    OperationReceipt {
        name,
        phase,
        fixture: "synthetic_deterministic",
        shape,
        repetitions: timing.repetitions,
        setup_wall_time_ns: timing.setup_wall_time_ns,
        warmup_wall_time_ns: timing.warmup_wall_time_ns,
        wall_time_ns_median: median(&sorted),
        wall_time_ns_min: sorted[0],
        wall_time_ns_max: sorted[sorted.len() - 1],
        wall_time_ns: timing.wall_time_ns,
        wall_cost: "host dispatch plus backend wait",
        dispatches_per_sample: timing.dispatches_per_sample,
        estimated_bytes,
        measured_device_bytes: None,
        memory_before_setup: timing.memory_before_setup,
        memory_before: timing.memory_before,
        memory_after: timing.memory_after,
    }
}

fn measure_repetitions<F>(
    backend: &mut MetalBackend,
    repetitions: usize,
    mut operation: F,
) -> Result<SampleTiming, Box<dyn Error>>
where
    F: FnMut(&mut MetalBackend) -> Result<(), leone::BackendError>,
{
    if repetitions == 0 {
        return Err("repetitions must be positive".into());
    }
    let warmup_started = Instant::now();
    operation(backend)?;
    backend.synchronize()?;
    let warmup_wall_time_ns = duration_ns(warmup_started.elapsed());
    let mut samples = Vec::with_capacity(repetitions);
    for _ in 0..repetitions {
        let started = Instant::now();
        operation(backend)?;
        backend.synchronize()?;
        samples.push(duration_ns(started.elapsed()));
    }
    Ok(SampleTiming {
        warmup_wall_time_ns,
        wall_time_ns: samples,
    })
}

fn median(samples: &[u64]) -> u64 {
    let middle = samples.len() / 2;
    if samples.len().is_multiple_of(2) {
        ((u128::from(samples[middle - 1]) + u128::from(samples[middle])) / 2) as u64
    } else {
        samples[middle]
    }
}

fn run_model(
    backend: MetalBackend,
    path: &Path,
    arguments: &Arguments,
    expected_identity: &ModelIdentity,
) -> Result<(ModelRunReceipt, MemoryReceipt), Box<dyn Error>> {
    require_model_identity(path, expected_identity, "fixture setup")?;
    let memory_before_load = memory_receipt(&backend.memory_accounting());
    let mut runtime = Runtime::load(backend, path)?;
    let config = runtime.model().config().clone();
    let (prompt_tokens, prompt) = model_prompt(&runtime, &arguments.prompt)?;
    let (warmup_prefill_wall_time_ns, warmup_decode_wall_time_ns) =
        warmup_model(&mut runtime, &prompt_tokens, arguments)?;
    let samples = measure_model_samples(&mut runtime, &prompt_tokens, arguments)?;
    require_model_identity(path, expected_identity, "profiling")?;
    let memory_after_run = memory_receipt(&runtime.backend().memory_accounting());
    let transcript_consistent = samples
        .transcripts
        .windows(2)
        .all(|pair| pair[0] == pair[1]);
    let result = ModelRunReceipt {
        sha256: expected_identity.sha256.clone(),
        file_bytes: expected_identity.file_bytes,
        architecture: config.architecture.name().to_owned(),
        config: ModelConfigReceipt {
            layers: config.n_layer,
            heads: config.n_head,
            heads_kv: config.n_head_kv,
            embedding: config.n_embd,
            feed_forward: config.n_ff,
            head_dim: config.head_dim,
            vocabulary: config.vocab_size,
            context: config.context_length,
        },
        prompt,
        repetitions: arguments.repetitions,
        decode_tokens: arguments.decode_tokens,
        prefill_chunk_tokens: arguments.prefill_chunk,
        wall_cost: "runtime wall time includes backend dispatch, synchronization, and host reads",
        warmup_prefill_wall_time_ns,
        warmup_decode_wall_time_ns,
        prefill_total_wall_time_ns: samples.prefill_total_wall_time_ns,
        prefill_setup_wall_time_ns: samples.prefill_setup_wall_time_ns,
        prefill_wall_time_ns: samples.prefill_wall_time_ns,
        prefill_ttft_ns: samples.prefill_ttft_ns,
        prefill_tok_s: samples.prefill_tok_s,
        decode_total_wall_time_ns: samples.decode_total_wall_time_ns,
        decode_setup_wall_time_ns: samples.decode_setup_wall_time_ns,
        decode_wall_time_ns: samples.decode_wall_time_ns,
        decode_ttft_ns: samples.decode_ttft_ns,
        decode_tok_s: samples.decode_tok_s,
        prefill_methods: samples.prefill_methods,
        decode_transcript_sha256: samples.transcripts,
        transcript_consistent,
        memory_before_load,
        memory_while_runtime_live: memory_after_run.clone(),
    };
    drop(runtime);
    Ok((result, memory_after_run))
}

fn require_model_identity(
    path: &Path,
    expected: &ModelIdentity,
    stage: &str,
) -> Result<(), Box<dyn Error>> {
    if read_model_identity(path)? != *expected {
        return Err(format!("model identity changed during {stage}").into());
    }
    Ok(())
}

fn model_prompt(
    runtime: &Runtime<MetalBackend>,
    prompt_text: &str,
) -> Result<(Vec<u32>, InputReceipt), Box<dyn Error>> {
    let prompt_tokens = runtime.model().tokenizer().encode(prompt_text)?;
    if prompt_tokens.is_empty() {
        return Err("the model prompt produced no tokens".into());
    }
    let prompt = InputReceipt {
        prompt_sha256: input_sha256(prompt_text),
        token_sha256: hash_u32s(&prompt_tokens),
        token_count: prompt_tokens.len(),
        artifact: None,
        token_ids: prompt_tokens.clone(),
    };
    Ok((prompt_tokens, prompt))
}

fn warmup_model(
    runtime: &mut Runtime<MetalBackend>,
    prompt_tokens: &[u32],
    arguments: &Arguments,
) -> Result<(u64, u64), Box<dyn Error>> {
    let prefill_started = Instant::now();
    runtime.benchmark_prefill(
        prompt_tokens,
        leone::KvCacheDtype::F16,
        arguments.prefill_chunk,
    )?;
    let prefill_wall_time_ns = duration_ns(prefill_started.elapsed());
    let decode_started = Instant::now();
    runtime.benchmark_decode_with_prefill_chunk(
        prompt_tokens,
        arguments.decode_tokens,
        leone::KvCacheDtype::F16,
        DecodeExecution::Eager,
        arguments.prefill_chunk,
    )?;
    Ok((prefill_wall_time_ns, duration_ns(decode_started.elapsed())))
}

fn measure_model_samples(
    runtime: &mut Runtime<MetalBackend>,
    prompt_tokens: &[u32],
    arguments: &Arguments,
) -> Result<ModelSamples, Box<dyn Error>> {
    let mut samples = ModelSamples::with_capacity(arguments.repetitions);
    for _ in 0..arguments.repetitions {
        samples.record(runtime, prompt_tokens, arguments)?;
    }
    Ok(samples)
}

impl ModelSamples {
    fn with_capacity(repetitions: usize) -> Self {
        Self {
            prefill_total_wall_time_ns: Vec::with_capacity(repetitions),
            prefill_setup_wall_time_ns: Vec::with_capacity(repetitions),
            prefill_wall_time_ns: Vec::with_capacity(repetitions),
            prefill_ttft_ns: Vec::with_capacity(repetitions),
            prefill_tok_s: Vec::with_capacity(repetitions),
            decode_total_wall_time_ns: Vec::with_capacity(repetitions),
            decode_setup_wall_time_ns: Vec::with_capacity(repetitions),
            decode_wall_time_ns: Vec::with_capacity(repetitions),
            decode_ttft_ns: Vec::with_capacity(repetitions),
            decode_tok_s: Vec::with_capacity(repetitions),
            prefill_methods: Vec::with_capacity(repetitions),
            transcripts: Vec::with_capacity(repetitions),
        }
    }

    fn record(
        &mut self,
        runtime: &mut Runtime<MetalBackend>,
        prompt_tokens: &[u32],
        arguments: &Arguments,
    ) -> Result<(), Box<dyn Error>> {
        let prefill_started = Instant::now();
        let prefill = runtime.benchmark_prefill(
            prompt_tokens,
            leone::KvCacheDtype::F16,
            arguments.prefill_chunk,
        )?;
        let prefill_total = duration_ns(prefill_started.elapsed());
        let prefill_execution = duration_ns(prefill.prefill_duration);
        let prefill_ttft = duration_ns(prefill.ttft_duration);
        self.prefill_total_wall_time_ns.push(prefill_total);
        self.prefill_setup_wall_time_ns
            .push(prefill_total.saturating_sub(prefill_ttft));
        self.prefill_wall_time_ns.push(prefill_execution);
        self.prefill_ttft_ns.push(prefill_ttft);
        self.prefill_tok_s
            .push(rate(prompt_tokens.len(), prefill.prefill_duration));
        self.prefill_methods
            .push(prefill.prefill_method.name().to_owned());
        self.record_decode(runtime, prompt_tokens, arguments)
    }

    fn record_decode(
        &mut self,
        runtime: &mut Runtime<MetalBackend>,
        prompt_tokens: &[u32],
        arguments: &Arguments,
    ) -> Result<(), Box<dyn Error>> {
        let decode_started = Instant::now();
        let decode = runtime.benchmark_decode_with_prefill_chunk(
            prompt_tokens,
            arguments.decode_tokens,
            leone::KvCacheDtype::F16,
            DecodeExecution::Eager,
            arguments.prefill_chunk,
        )?;
        let decode_total = duration_ns(decode_started.elapsed());
        let decode_prefill = duration_ns(decode.ttft_duration);
        let decode_execution = duration_ns(decode.decode_duration);
        self.decode_total_wall_time_ns.push(decode_total);
        self.decode_setup_wall_time_ns
            .push(decode_total.saturating_sub(decode_prefill.saturating_add(decode_execution)));
        self.decode_wall_time_ns.push(decode_execution);
        self.decode_ttft_ns.push(decode_prefill);
        self.decode_tok_s
            .push(rate(arguments.decode_tokens, decode.decode_duration));
        self.transcripts.push(decode.transcript_sha256);
        Ok(())
    }
}

fn parse_arguments(values: Vec<String>) -> Result<Arguments, Box<dyn Error>> {
    let mut arguments = Arguments {
        model: None,
        prompt: DEFAULT_PROMPT.to_owned(),
        repetitions: DEFAULT_REPETITIONS,
        decode_tokens: DEFAULT_DECODE_TOKENS,
        prefill_tokens: DEFAULT_PREFILL_TOKENS,
        prefill_chunk: DEFAULT_PREFILL_CHUNK,
        memory_budget_bytes: None,
        receipt_dir: PathBuf::from("receipts"),
        synthetic: true,
    };
    let mut index = 0;
    while index < values.len() {
        let flag = values[index].as_str();
        if parse_path_flag(&mut arguments, flag, &values, &mut index)?
            || parse_measure_flag(&mut arguments, flag, &values, &mut index)?
            || parse_mode_flag(&mut arguments, flag)?
        {
            index += 1;
            continue;
        }
        return Err(format!("unknown argument {flag:?}").into());
    }
    if arguments.model.is_none() && !arguments.synthetic {
        return Err("--model-only requires --model".into());
    }
    Ok(arguments)
}

fn parse_path_flag(
    arguments: &mut Arguments,
    flag: &str,
    values: &[String],
    index: &mut usize,
) -> Result<bool, Box<dyn Error>> {
    if flag == "--model" {
        arguments.model = Some(PathBuf::from(value(values, index)?));
        return Ok(true);
    }
    if flag == "--prompt" {
        set_prompt(arguments, value(values, index)?)?;
        return Ok(true);
    }
    parse_output_flag(arguments, flag, values, index)
}

fn set_prompt(arguments: &mut Arguments, prompt: &str) -> Result<(), Box<dyn Error>> {
    if prompt.len() > MAX_PROMPT_BYTES {
        return Err(format!("prompt must be at most {MAX_PROMPT_BYTES} bytes").into());
    }
    arguments.prompt = prompt.to_owned();
    Ok(())
}

fn parse_output_flag(
    arguments: &mut Arguments,
    flag: &str,
    values: &[String],
    index: &mut usize,
) -> Result<bool, Box<dyn Error>> {
    match flag {
        "--memory-budget-bytes" => {
            arguments.memory_budget_bytes =
                Some(positive_u64(value(values, index)?, "memory budget bytes")?);
            Ok(true)
        }
        "--receipt-dir" => {
            arguments.receipt_dir = PathBuf::from(value(values, index)?);
            Ok(true)
        }
        _ => Ok(false),
    }
}

fn parse_measure_flag(
    arguments: &mut Arguments,
    flag: &str,
    values: &[String],
    index: &mut usize,
) -> Result<bool, Box<dyn Error>> {
    let (field, label, maximum) = match flag {
        "--repetitions" => (&mut arguments.repetitions, "repetitions", MAX_REPETITIONS),
        "--decode-tokens" => (
            &mut arguments.decode_tokens,
            "decode tokens",
            MAX_DECODE_TOKENS,
        ),
        "--prefill-tokens" => (
            &mut arguments.prefill_tokens,
            "prefill tokens",
            MAX_PREFILL_TOKENS,
        ),
        "--prefill-chunk" => (
            &mut arguments.prefill_chunk,
            "prefill chunk",
            MAX_PREFILL_CHUNK,
        ),
        _ => return Ok(false),
    };
    *field = bounded(value(values, index)?, label, maximum)?;
    Ok(true)
}

fn parse_mode_flag(arguments: &mut Arguments, flag: &str) -> Result<bool, Box<dyn Error>> {
    match flag {
        "--model-only" => {
            arguments.synthetic = false;
            Ok(true)
        }
        "--help" => {
            print_help();
            std::process::exit(0);
        }
        _ => Ok(false),
    }
}

fn print_help() {
    println!(
        "Usage: metal_profile [--model PATH] [--prompt TEXT] [--repetitions N] [--decode-tokens N] [--prefill-tokens N] [--prefill-chunk N] [--memory-budget-bytes N] [--receipt-dir PATH] [--model-only]"
    );
    println!("The default run profiles a bounded synthetic fixture.");
    println!("With --model, the run also measures a real GGUF prefill and decode path.");
}

fn value<'a>(values: &'a [String], index: &mut usize) -> Result<&'a str, Box<dyn Error>> {
    *index += 1;
    values
        .get(*index)
        .map(String::as_str)
        .ok_or_else(|| "argument is missing its value".into())
}

fn bounded(value: &str, field: &str, maximum: usize) -> Result<usize, Box<dyn Error>> {
    let value = value
        .parse::<usize>()
        .ok()
        .filter(|number| *number > 0)
        .ok_or_else(|| format!("{field} must be a positive integer"))?;
    if value > maximum {
        return Err(format!("{field} must be at most {maximum}").into());
    }
    Ok(value)
}

fn positive_u64(value: &str, field: &str) -> Result<u64, Box<dyn Error>> {
    value
        .parse::<u64>()
        .ok()
        .filter(|number| *number > 0)
        .ok_or_else(|| format!("{field} must be a positive integer").into())
}

#[cfg(target_os = "macos")]
fn active_device_metadata(backend: &MetalBackend) -> Result<ActiveDeviceMetadata, Box<dyn Error>> {
    let metadata = backend.device_metadata()?;
    Ok(ActiveDeviceMetadata {
        registry_id: metadata.registry_id,
        device_name: metadata.device_name,
        architecture_name: metadata.architecture_name,
        os_version: metadata.os_version,
        compiler_version: metadata.compiler_version,
        shader_source_hash: metadata.shader_source_hash,
        fast_math_enabled: metadata.fast_math_enabled,
    })
}

#[cfg(not(target_os = "macos"))]
fn active_device_metadata(_backend: &MetalBackend) -> Result<ActiveDeviceMetadata, Box<dyn Error>> {
    Err("active Metal device metadata requires macOS".into())
}

fn device_receipt(device: MetalDeviceInfo, metadata: ActiveDeviceMetadata) -> DeviceReceipt {
    DeviceReceipt {
        max_buffer_length: device.max_buffer_length,
        recommended_working_set: device.recommended_working_set,
        driver_current_allocated_at_init: device.current_allocated,
        registry_id: metadata.registry_id,
        device_name: metadata.device_name,
        architecture_name: metadata.architecture_name,
        os_version: metadata.os_version,
        compiler_version: metadata.compiler_version,
        shader_source_hash: metadata.shader_source_hash,
        fast_math_enabled: metadata.fast_math_enabled,
        capacity: "MTLDevice limits and initialization currentAllocatedSize".to_owned(),
    }
}

fn memory_budget(requested: Option<u64>, recommended: u64) -> Result<MemoryBudget, Box<dyn Error>> {
    let limit = requested
        .or((recommended > 0).then_some(recommended))
        .ok_or("Metal did not report a recommended working set; pass --memory-budget-bytes")?;
    if recommended > 0 && limit > recommended {
        return Err(format!(
            "memory budget {limit} exceeds Metal recommended working set {recommended}"
        )
        .into());
    }
    Ok(MemoryBudget::limited(limit)?)
}

fn fixture_config(model: Option<&Path>) -> Result<FixtureConfig, Box<dyn Error>> {
    let Some(path) = model else {
        return Ok(DEFAULT_FIXTURE);
    };
    let gguf = Gguf::open(path)?;
    let config = GgufModelConfig::from_metadata(gguf.metadata())?;
    let (primary_format, secondary_format) = fixture_quant_formats(&gguf)?;
    let head_dim = config.head_dim.unwrap_or(config.n_embd / config.n_head);
    let context = usize_value(config.context_length, "context length")?.min(MAX_SYNTHETIC_CONTEXT);
    Ok(FixtureConfig {
        embedding: usize_value(config.n_embd, "embedding length")?,
        feed_forward: usize_value(config.n_ff, "feed-forward length")?,
        heads: usize_value(config.n_head, "attention heads")?,
        heads_kv: usize_value(config.n_head_kv, "KV attention heads")?,
        head_dim: usize_value(head_dim, "head dimension")?,
        context,
        model_derived: true,
        primary_format,
        secondary_format,
    })
}

fn fixture_quant_formats(
    gguf: &Gguf,
) -> Result<(QuantFormat, Option<QuantFormat>), Box<dyn Error>> {
    let has_q4 = gguf
        .tensors()
        .iter()
        .any(|tensor| tensor.dtype == GgmlType::Q4_K);
    let has_q6 = gguf
        .tensors()
        .iter()
        .any(|tensor| tensor.dtype == GgmlType::Q6_K);
    let primary_format = if has_q4 {
        QuantFormat::Q4K
    } else if has_q6 {
        QuantFormat::Q6K
    } else {
        return Err("model has no Q4_K or Q6_K tensor for the fixture".into());
    };
    let secondary_format = match (has_q4, has_q6) {
        (true, true) => Some(if primary_format == QuantFormat::Q4K {
            QuantFormat::Q6K
        } else {
            QuantFormat::Q4K
        }),
        _ => None,
    };
    Ok((primary_format, secondary_format))
}

fn validate_fixture_host_bytes(
    fixture: FixtureConfig,
    prefill_tokens: usize,
) -> Result<(), Box<dyn Error>> {
    let peak_bytes = validate_quantized_host_bytes(fixture)?
        .max(validate_activation_host_bytes(fixture, prefill_tokens)?);
    let attention_bytes = validate_attention_host_bytes(fixture, prefill_tokens)?;
    let peak_bytes = peak_bytes.max(attention_bytes);
    if peak_bytes > MAX_FIXTURE_HOST_BYTES {
        return Err(format!(
            "synthetic fixture host temporary bytes {peak_bytes} exceed limit {MAX_FIXTURE_HOST_BYTES}"
        )
        .into());
    }
    Ok(())
}

fn validate_quantized_host_bytes(fixture: FixtureConfig) -> Result<usize, Box<dyn Error>> {
    let mut peak_bytes = 0;
    for format in fixture_formats(fixture) {
        for rows in [1, fixture.embedding, fixture.feed_forward] {
            peak_bytes = peak_bytes.max(quantized_fixture_bytes(rows, fixture.embedding, format)?);
        }
    }
    Ok(peak_bytes)
}

fn validate_activation_host_bytes(
    fixture: FixtureConfig,
    prefill_tokens: usize,
) -> Result<usize, Box<dyn Error>> {
    let prefill_elements = checked_product(prefill_tokens, fixture.embedding, "prefill elements")?;
    Ok(temporary_f32_bytes(fixture.embedding, "GEMV input")?
        .max(temporary_f32_bytes(prefill_elements, "prefill input")?))
}

fn validate_attention_host_bytes(
    fixture: FixtureConfig,
    prefill_tokens: usize,
) -> Result<usize, Box<dyn Error>> {
    let attention = AttentionShape::new(
        fixture.heads,
        fixture.heads_kv,
        fixture.head_dim,
        fixture.context,
    )?;
    let prefill_query_elements = checked_product(
        prefill_tokens,
        attention.query_elements()?,
        "attention prefill query elements",
    )?;
    Ok(
        temporary_f16_bytes(attention.cache_elements()?, "attention cache")?
            .max(temporary_f32_bytes(
                attention.query_elements()?,
                "attention query",
            )?)
            .max(temporary_f32_bytes(
                prefill_query_elements,
                "attention prefill query",
            )?),
    )
}

fn quantized_fixture_bytes(
    rows: usize,
    columns: usize,
    format: QuantFormat,
) -> Result<usize, Box<dyn Error>> {
    Ok(QuantMatrix::new(rows, columns, format)?.layout()?.bytes())
}

fn temporary_f32_bytes(elements: usize, field: &str) -> Result<usize, Box<dyn Error>> {
    temporary_element_bytes(elements, 4, field)
}

fn temporary_f16_bytes(elements: usize, field: &str) -> Result<usize, Box<dyn Error>> {
    temporary_element_bytes(elements, 2, field)
}

fn temporary_element_bytes(
    elements: usize,
    bytes_per_element: usize,
    field: &str,
) -> Result<usize, Box<dyn Error>> {
    let bytes = checked_product(elements, bytes_per_element, field)?;
    checked_product(bytes, 2, &format!("{field} temporary bytes"))
}

fn checked_product(left: usize, right: usize, field: &str) -> Result<usize, Box<dyn Error>> {
    left.checked_mul(right)
        .ok_or_else(|| format!("{field} exceeds host size").into())
}

fn fixture_formats(fixture: FixtureConfig) -> Vec<QuantFormat> {
    let mut formats = vec![fixture.primary_format];
    if let Some(format) = fixture.secondary_format {
        formats.push(format);
    }
    formats
}

fn usize_value(value: u64, field: &str) -> Result<usize, Box<dyn Error>> {
    usize::try_from(value).map_err(|_| format!("{field} does not fit the host size").into())
}

fn fixture_scopes(arguments: &Arguments) -> Vec<&'static str> {
    let mut scopes = Vec::with_capacity(2);
    if arguments.synthetic {
        scopes.push("synthetic_deterministic");
    }
    if arguments.model.is_some() {
        scopes.push("gguf_model");
    }
    scopes
}

fn validated_provenance() -> Result<(SourceIdentity, BuildIdentity), Box<dyn Error>> {
    let source = source_identity();
    let build = build_identity()?;
    if !source_identity_complete(&source, &build) {
        return Err("profiling receipts require a clean, identified source build".into());
    }
    if !build_identity_complete(&build) {
        return Err("profiling receipts require complete build identity".into());
    }
    Ok((source, build))
}

fn source_identity_complete(source: &SourceIdentity, build: &BuildIdentity) -> bool {
    !source.source_tree_dirty
        && source.git_commit != "unknown"
        && source.source_tree_sha256 != "unknown"
        && !build.provenance_unknown
}

fn build_identity_complete(build: &BuildIdentity) -> bool {
    base_build_identity_complete(build)
        && toolchain_identity_complete(build)
        && native_identity_complete(build)
}

fn base_build_identity_complete(build: &BuildIdentity) -> bool {
    build.target != "unknown"
        && build.host != "unknown"
        && build.rustc != "unknown"
        && build.build_config_sha256 != "unknown"
        && build.build_flags_raw_sha256 != "unknown"
}

fn toolchain_identity_complete(build: &BuildIdentity) -> bool {
    build.rustc_sha256 != "unknown"
        && build.toolchain_sha256 != "unknown"
        && build.profile_inputs_sha256 != "unknown"
        && build.linker_inputs_sha256 != "unknown"
}

fn native_identity_complete(build: &BuildIdentity) -> bool {
    build.native_provenance != "incomplete" && build.native_tools_sha256 != "unknown"
}

fn memory_receipt(memory: &MemoryAccounting) -> MemoryReceipt {
    let classes = memory
        .classes
        .iter()
        .map(|(class, stats)| {
            (
                class.name().to_owned(),
                MemoryClassReceipt {
                    live_bytes: stats.live_bytes,
                    peak_live_bytes: stats.peak_live_bytes,
                    live_allocations: stats.live_allocations,
                    peak_live_allocations: stats.peak_live_allocations,
                    allocations: stats.allocations,
                    frees: stats.frees,
                },
            )
        })
        .collect();
    MemoryReceipt {
        scope: "leone_owned_tracked_allocations",
        live_bytes: memory.live_bytes,
        reserved_bytes: memory.reserved_bytes,
        peak_live_bytes: memory.peak_live_bytes,
        peak_owned_and_reserved_bytes: memory.peak_owned_and_reserved_bytes,
        live_allocations: memory.live_allocations,
        peak_live_allocations: memory.peak_live_allocations,
        allocations: memory.allocations,
        frees: memory.frees,
        untracked_objects: memory.untracked.object_count(),
        budget: memory_budget_name(memory.budget),
        classes,
    }
}

fn memory_budget_name(budget: MemoryBudget) -> String {
    match budget {
        MemoryBudget::Unlimited => "unlimited".to_owned(),
        MemoryBudget::Bytes(bytes) => format!("bytes:{bytes}"),
    }
}

fn peak_memory(before: &MemoryReceipt, after: &MemoryReceipt) -> MemoryReceipt {
    let mut peak = after.clone();
    peak.peak_live_bytes = before.peak_live_bytes.max(after.peak_live_bytes);
    peak.peak_owned_and_reserved_bytes = before
        .peak_owned_and_reserved_bytes
        .max(after.peak_owned_and_reserved_bytes);
    peak.peak_live_allocations = before
        .peak_live_allocations
        .max(after.peak_live_allocations);
    peak.live_bytes = before.live_bytes.max(after.live_bytes);
    peak.live_allocations = before.live_allocations.max(after.live_allocations);
    peak
}

fn deterministic_bytes(length: usize, seed: u64) -> Vec<u8> {
    let mut state = seed;
    (0..length)
        .map(|_| {
            state ^= state << 7;
            state ^= state >> 9;
            state ^= state << 8;
            state as u8
        })
        .collect()
}

fn quantized_bytes(length: usize, format: QuantFormat, seed: u64) -> Vec<u8> {
    let mut bytes = deterministic_bytes(length, seed);
    match format {
        QuantFormat::Q4K => {
            for block in bytes.chunks_exact_mut(format.block_bytes()) {
                block[0] = 0;
                block[1] = 0x3c;
                block[2] = 0;
                block[3] = 0x3c;
            }
        }
        QuantFormat::Q6K => {
            for block in bytes.chunks_exact_mut(format.block_bytes()) {
                block[192..208].fill(1);
                block[208] = 0;
                block[209] = 0x3c;
            }
        }
    }
    bytes
}

fn f32_bytes(elements: usize, seed: u64) -> Vec<u8> {
    deterministic_bytes(elements * 4, seed)
        .chunks_exact(4)
        .flat_map(|bytes| {
            let raw = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            let value = ((raw % 2048) as f32 - 1024.0) / 1024.0;
            value.to_bits().to_le_bytes()
        })
        .collect()
}

fn f16_bytes(elements: usize, seed: u64) -> Vec<u8> {
    deterministic_bytes(elements * 2, seed)
        .chunks_exact(2)
        .flat_map(|bytes| {
            let value = (u16::from_le_bytes([bytes[0], bytes[1]]) % 1024) as f32 / 1024.0;
            f16::from_f32(value).to_bits().to_le_bytes()
        })
        .collect()
}

fn duration_ns(duration: Duration) -> u64 {
    duration.as_nanos().try_into().unwrap_or(u64::MAX)
}

fn rate(tokens: usize, duration: Duration) -> f64 {
    tokens as f64 / duration.as_secs_f64()
}

fn input_sha256(prompt: &str) -> String {
    hash_bytes(prompt.as_bytes())
}

fn input_identity(arguments: &Arguments, model_run: Option<&ModelRunReceipt>) -> String {
    let mut hasher = Sha256::new();
    hasher.update(arguments.prompt.as_bytes());
    hasher.update(arguments.repetitions.to_le_bytes());
    hasher.update(arguments.decode_tokens.to_le_bytes());
    hasher.update(arguments.prefill_tokens.to_le_bytes());
    hasher.update(arguments.prefill_chunk.to_le_bytes());
    hasher.update(arguments.memory_budget_bytes.unwrap_or(0).to_le_bytes());
    hasher.update([u8::from(arguments.synthetic)]);
    if let Some(model_run) = model_run {
        hasher.update(model_run.sha256.as_bytes());
    }
    hex_hash(hasher.finalize())
}

fn hash_u32s(values: &[u32]) -> String {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    hex_hash(hasher.finalize())
}

fn hash_file(path: &Path) -> Result<String, Box<dyn Error>> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1 << 20];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(hex_hash(hasher.finalize()))
}

fn read_model_identity(path: &Path) -> Result<ModelIdentity, Box<dyn Error>> {
    let metadata = fs::metadata(path)?;
    Ok(ModelIdentity {
        sha256: hash_file(path)?,
        file_bytes: metadata.len(),
        #[cfg(unix)]
        device: metadata.dev(),
        #[cfg(unix)]
        inode: metadata.ino(),
    })
}

fn hash_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex_hash(hasher.finalize())
}

fn hex_hash(hash: impl AsRef<[u8]>) -> String {
    hash.as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn source_identity() -> SourceIdentity {
    SourceIdentity {
        git_commit: env!("LEONE_SOURCE_COMMIT").to_owned(),
        source_tree_sha256: env!("LEONE_SOURCE_TREE_SHA256").to_owned(),
        source_tree_dirty: env!("LEONE_SOURCE_DIRTY") == "true",
        source_paths: env!("LEONE_SOURCE_PATHS")
            .split(',')
            .map(str::to_owned)
            .collect(),
    }
}

fn build_identity() -> Result<BuildIdentity, Box<dyn Error>> {
    let executable_path = env::current_exe()?;
    Ok(BuildIdentity {
        profile: env!("LEONE_BUILD_PROFILE"),
        target: env!("LEONE_BUILD_TARGET").to_owned(),
        host: env!("LEONE_BUILD_HOST"),
        package_version: env!("CARGO_PKG_VERSION"),
        executable_sha256: hash_file(&executable_path)?,
        rustc: env!("LEONE_BUILD_RUSTC"),
        rustc_sha256: env!("LEONE_BUILD_RUSTC_SHA256"),
        toolchain_sha256: env!("LEONE_BUILD_TOOLCHAIN_SHA256"),
        tool_versions: env!("LEONE_BUILD_TOOL_VERSIONS"),
        features: env!("LEONE_BUILD_FEATURES"),
        build_flags: env!("LEONE_BUILD_FLAGS"),
        build_flags_sha256: env!("LEONE_BUILD_FLAGS_SHA256"),
        build_flags_raw_sha256: env!("LEONE_BUILD_FLAGS_RAW_SHA256"),
        build_flags_redacted: env!("LEONE_BUILD_FLAGS_REDACTED") == "true",
        build_flags_redaction_count: env!("LEONE_BUILD_FLAGS_REDACTION_COUNT")
            .parse()
            .map_err(|_| "build provenance redaction count is invalid")?,
        profile_inputs: env!("LEONE_BUILD_PROFILE_INPUTS"),
        profile_inputs_sha256: env!("LEONE_BUILD_PROFILE_INPUTS_SHA256"),
        linker_inputs: env!("LEONE_BUILD_LINKER_INPUTS"),
        linker_inputs_sha256: env!("LEONE_BUILD_LINKER_INPUTS_SHA256"),
        native_provenance: env!("LEONE_BUILD_NATIVE_PROVENANCE"),
        native_tools_sha256: env!("LEONE_BUILD_NATIVE_TOOLS_SHA256"),
        native_tool_versions: env!("LEONE_BUILD_NATIVE_TOOL_VERSIONS"),
        build_config_sha256: env!("LEONE_BUILD_CONFIG_SHA256"),
        provenance_unknown: env!("LEONE_BUILD_PROVENANCE_UNKNOWN") == "true",
    })
}

fn invocation_hash() -> String {
    let mut hasher = Sha256::new();
    for argument in env::args() {
        hasher.update((argument.len() as u64).to_le_bytes());
        hasher.update(argument.as_bytes());
    }
    hex_hash(hasher.finalize())
}

fn serialize_json(value: &impl Serialize) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn write_input_artifact(
    receipt_dir: &Path,
    profile: &ProfileReceipt,
) -> Result<Option<String>, Box<dyn Error>> {
    let Some(model_run) = profile.model_run.as_ref() else {
        return Ok(None);
    };
    let artifact_name = format!(
        "metal-profile-input-{}-{}-{}.json",
        model_run.sha256, model_run.prompt.prompt_sha256, model_run.prompt.token_sha256
    );
    let bytes = serialize_json(&input_artifact(model_run))?;
    let path = receipt_dir.join(&artifact_name);
    write_content_addressed(&path, &bytes)?;
    Ok(Some(artifact_name))
}

fn input_artifact(model_run: &ModelRunReceipt) -> InputArtifact {
    InputArtifact {
        schema_version: "leone.metal-input.v1",
        model_sha256: model_run.sha256.clone(),
        prompt_sha256: model_run.prompt.prompt_sha256.clone(),
        token_sha256: model_run.prompt.token_sha256.clone(),
        token_count: model_run.prompt.token_count,
        token_ids: model_run.prompt.token_ids.clone(),
    }
}

fn write_content_addressed(path: &Path, bytes: &[u8]) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    match OpenOptions::new().write(true).create_new(true).open(path) {
        Ok(mut file) => file.write_all(bytes)?,
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            if fs::read(path)? != bytes {
                return Err("content-addressed input artifact differs".into());
            }
        }
        Err(error) => return Err(error.into()),
    }
    Ok(())
}

fn write_json(path: &Path, contents: &[u8]) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(contents)?;
    Ok(())
}
