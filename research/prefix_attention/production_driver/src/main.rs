use half::f16;
use leone::PrefillPlan;
use leone_cuda::{
    attention_prefill_f16, AttentionShape, Context, CublasLt, DeviceBuffer, Event, PrefillScratch,
    Stream,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
struct Manifest {
    calibration: Vec<CaseSpec>,
    evaluation: Vec<CaseSpec>,
    input_generator: String,
    input_digest: String,
    oracle_id: String,
    warmup_runs: u32,
    measured_runs: u32,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct CaseSpec {
    name: String,
    tokens: usize,
    query_rows: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    tile_tokens: usize,
    group_rows: usize,
    seed: u64,
    shared_prefix: bool,
}

#[derive(Debug)]
struct Problem {
    spec: CaseSpec,
    queries: Vec<f32>,
    keys: Vec<u16>,
    values: Vec<u16>,
    oracle: Vec<f64>,
}

#[derive(Debug)]
struct Options {
    phase: String,
    output: PathBuf,
}

struct ManifestSelection {
    cases: Vec<CaseSpec>,
    metadata: ReceiptMetadata,
}

struct ReceiptMetadata {
    phase: String,
    input_generator: String,
    input_digest_algorithm: String,
    oracle_id: String,
    warmups: u32,
    repetitions: u32,
}

struct AttentionExecution<'a> {
    context: &'a Context,
    handle: &'a CublasLt,
    stream: &'a Stream,
    query: &'a DeviceBuffer<f32>,
    keys: &'a DeviceBuffer<u16>,
    values: &'a DeviceBuffer<u16>,
    output: &'a mut DeviceBuffer<f32>,
    shape: AttentionShape,
    start_position: usize,
    query_rows: usize,
    scratch: &'a mut PrefillScratch,
}

impl AttentionExecution<'_> {
    fn execute(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        attention_prefill_f16(
            self.handle,
            self.stream,
            self.query,
            self.keys,
            self.values,
            self.output,
            self.shape,
            self.start_position,
            self.query_rows,
            self.scratch,
        )?;
        self.stream.synchronize()?;
        Ok(())
    }
}

#[derive(Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
        value ^ (value >> 31)
    }

    fn unit_float(&mut self) -> f32 {
        let bits = self.next() >> 40;
        bits as f32 / 16_777_215.0 * 2.0 - 1.0
    }
}

fn parse_options() -> Options {
    let mut phase = String::new();
    let mut output = None;
    let mut arguments = env::args().skip(1);
    while let Some(argument) = arguments.next() {
        let value = arguments.next().unwrap_or_default();
        match argument.as_str() {
            "--phase" => phase = value,
            "--output" => output = Some(PathBuf::from(value)),
            _ => panic!("unknown option: {argument}"),
        }
    }
    if phase != "calibration" && phase != "evaluation" {
        panic!("--phase must be calibration or evaluation");
    }
    Options {
        phase,
        output: output.expect("--output is required"),
    }
}

fn make_problem(spec: CaseSpec) -> Problem {
    let mut generator = SplitMix64 { state: spec.seed };
    let query_count = spec.query_rows * spec.query_heads * spec.head_dim;
    let kv_count = spec.kv_heads * spec.tokens * spec.head_dim;
    let queries = (0..query_count).map(|_| generator.unit_float()).collect();
    let mut keys = Vec::with_capacity(kv_count);
    let mut values = Vec::with_capacity(kv_count);
    for _ in 0..kv_count {
        keys.push(f16::from_f32(generator.unit_float()).to_bits());
        values.push(f16::from_f32(generator.unit_float()).to_bits());
    }
    let mut problem = Problem {
        spec,
        queries,
        keys,
        values,
        oracle: Vec::new(),
    };
    problem.oracle = oracle(&problem);
    problem
}

fn cache_base(spec: &CaseSpec, row: usize, kv_head: usize) -> usize {
    let _ = row;
    kv_head * spec.tokens * spec.head_dim
}

fn oracle(problem: &Problem) -> Vec<f64> {
    let spec = &problem.spec;
    let query_stride = spec.query_heads * spec.head_dim;
    let mut output = vec![0.0; spec.query_rows * query_stride];
    let start_position = spec.tokens - spec.query_rows;
    for row in 0..spec.query_rows {
        for query_head in 0..spec.query_heads {
            let kv_head = query_head * spec.kv_heads / spec.query_heads;
            let query_base = row * query_stride + query_head * spec.head_dim;
            let base = cache_base(spec, row, kv_head);
            let context_length = start_position + row + 1;
            let mut scores = Vec::with_capacity(context_length);
            for token in 0..context_length {
                let key_base = base + token * spec.head_dim;
                let dot = (0..spec.head_dim)
                    .map(|index| {
                        f64::from(f16::from_f32(problem.queries[query_base + index]).to_f32())
                            * f64::from(f16::from_bits(problem.keys[key_base + index]).to_f32())
                    })
                    .sum::<f64>();
                scores.push(dot / (spec.head_dim as f64).sqrt());
            }
            let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights = scores
                .iter()
                .map(|score| (score - maximum).exp())
                .collect::<Vec<_>>();
            let normalizer = weights.iter().sum::<f64>();
            for index in 0..spec.head_dim {
                output[query_base + index] = weights
                    .iter()
                    .enumerate()
                    .map(|(token, weight)| {
                        *weight
                            * f64::from(
                                f16::from_bits(
                                    problem.values[base + token * spec.head_dim + index],
                                )
                                .to_f32(),
                            )
                    })
                    .sum::<f64>()
                    / normalizer;
            }
        }
    }
    output
}

fn input_digest(problem: &Problem) -> String {
    let mut hash = 1_469_598_103_934_665_603_u64;
    for value in &problem.queries {
        update_u32(&mut hash, value.to_bits());
    }
    for value in problem.keys.iter().chain(&problem.values) {
        update_u16(&mut hash, *value);
    }
    format!("{hash:016x}")
}

fn oracle_digest(values: &[f64]) -> String {
    let mut hash = 1_469_598_103_934_665_603_u64;
    for value in values {
        for byte in value.to_bits().to_le_bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(1_099_511_628_211);
        }
    }
    format!("{hash:016x}")
}

fn update_u32(hash: &mut u64, value: u32) {
    for byte in value.to_le_bytes() {
        *hash = (*hash ^ u64::from(byte)).wrapping_mul(1_099_511_628_211);
    }
}

fn update_u16(hash: &mut u64, value: u16) {
    for byte in value.to_le_bytes() {
        *hash = (*hash ^ u64::from(byte)).wrapping_mul(1_099_511_628_211);
    }
}

fn output_digest(values: &[f32]) -> String {
    let mut hash = 1_469_598_103_934_665_603_u64;
    for value in values {
        update_u32(&mut hash, value.to_bits());
    }
    format!("{hash:016x}")
}

fn errors(actual: &[f32], expected: &[f64]) -> (f64, f64) {
    actual
        .iter()
        .zip(expected)
        .fold((0.0, 0.0), |(abs_max, rel_max), (actual, expected)| {
            let difference = (f64::from(*actual) - expected).abs();
            (
                abs_max.max(difference),
                rel_max.max(difference / expected.abs().max(1.0e-12)),
            )
        })
}

fn median(samples: &[f32]) -> f32 {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f32::total_cmp);
    sorted[sorted.len() / 2]
}

fn run_warmups(
    execution: &mut AttentionExecution<'_>,
    warmups: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    for _ in 0..warmups {
        execution.execute()?;
    }
    Ok(())
}

fn measure_repetitions(
    execution: &mut AttentionExecution<'_>,
    repetitions: u32,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut samples = Vec::with_capacity(repetitions as usize);
    for _ in 0..repetitions {
        let context = execution.context;
        let stream = execution.stream;
        let mut start = Event::new(context)?;
        let mut end = Event::new(context)?;
        start.record(stream)?;
        execution.execute()?;
        end.record(stream)?;
        end.synchronize()?;
        samples.push(Event::elapsed_ms(&start, &end)?);
    }
    Ok(samples)
}

fn copy_output(
    output: &DeviceBuffer<f32>,
    elements: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut values = vec![0.0; elements];
    output.copy_to(&mut values)?;
    Ok(values)
}

fn run_path(
    problem: &Problem,
    context: &Context,
    stream: &Stream,
    handle: &CublasLt,
    scratch: &mut PrefillScratch,
    warmups: u32,
    repetitions: u32,
) -> Result<(Vec<f32>, Vec<f32>), Box<dyn std::error::Error>> {
    let spec = &problem.spec;
    let shape = AttentionShape::new(spec.query_heads, spec.kv_heads, spec.head_dim, spec.tokens)?;
    let d_query = context.copy_to_device(&problem.queries)?;
    let d_keys = context.copy_to_device(&problem.keys)?;
    let d_values = context.copy_to_device(&problem.values)?;
    let mut d_output: DeviceBuffer<f32> =
        context.alloc(spec.query_rows * shape.query_elements())?;
    let start_position = spec.tokens - spec.query_rows;
    let output_elements = spec.query_rows * shape.query_elements();
    let mut execution = AttentionExecution {
        context,
        handle,
        stream,
        query: &d_query,
        keys: &d_keys,
        values: &d_values,
        output: &mut d_output,
        shape,
        start_position,
        query_rows: spec.query_rows,
        scratch,
    };
    run_warmups(&mut execution, warmups)?;
    let samples = measure_repetitions(&mut execution, repetitions)?;
    let output = copy_output(execution.output, output_elements)?;
    Ok((output, samples))
}

fn run_case(
    spec: CaseSpec,
    context: &Context,
    stream: &Stream,
    handle: &CublasLt,
    warmups: u32,
    repetitions: u32,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let problem = make_problem(spec.clone());
    let plan = PrefillPlan::new(
        spec.query_rows,
        spec.tokens,
        spec.query_heads,
        spec.kv_heads,
        spec.head_dim,
        spec.query_heads * spec.head_dim,
        spec.query_heads * spec.head_dim,
        spec.query_heads * spec.head_dim,
    )?;
    let mut scratch = PrefillScratch::new(context, plan)?;
    let (output, samples) = run_path(
        &problem,
        context,
        stream,
        handle,
        &mut scratch,
        warmups,
        repetitions,
    )?;
    let (max_abs, max_rel) = errors(&output, &problem.oracle);
    let output_hash = output_digest(&output);
    Ok(json!({
        "name": spec.name,
        "input_digest": input_digest(&problem),
        "oracle_digest": oracle_digest(&problem.oracle),
        "spec": spec,
        "samples_ms": samples,
        "median_ms": median(&samples),
        "quality_max_abs": max_abs,
        "quality_max_rel": max_rel,
        "output_values": output,
        "output_digest": output_hash,
        "production_query_rows": spec.query_rows,
        "layout_scope": "causal_single_cache",
    }))
}

fn select_manifest_cases(manifest: Manifest, phase: &str) -> ManifestSelection {
    let cases = if phase == "calibration" {
        manifest.calibration
    } else {
        manifest.evaluation
    };
    ManifestSelection {
        cases,
        metadata: ReceiptMetadata {
            phase: phase.to_owned(),
            input_generator: manifest.input_generator,
            input_digest_algorithm: manifest.input_digest,
            oracle_id: manifest.oracle_id,
            warmups: manifest.warmup_runs,
            repetitions: manifest.measured_runs,
        },
    }
}

fn run_cases(
    cases: Vec<CaseSpec>,
    context: &Context,
    stream: &Stream,
    handle: &CublasLt,
    warmups: u32,
    repetitions: u32,
) -> Result<Vec<serde_json::Value>, Box<dyn std::error::Error>> {
    let mut records = Vec::with_capacity(cases.len());
    for case in cases {
        records.push(run_case(
            case,
            context,
            stream,
            handle,
            warmups,
            repetitions,
        )?);
    }
    Ok(records)
}

fn write_receipt(
    output: &Path,
    metadata: &ReceiptMetadata,
    records: Vec<serde_json::Value>,
) -> Result<(), Box<dyn std::error::Error>> {
    let receipt = json!({
        "schema": "prefix-attention-production-receipt-v1",
        "operator": "leone-production-prefill-attention-f16",
        "storage": "fp16-kv-fp32-accumulation",
        "backend": "cuda",
        "phase": metadata.phase,
        "input_generator": metadata.input_generator,
        "input_digest_algorithm": metadata.input_digest_algorithm,
        "oracle_id": metadata.oracle_id,
        "warmups": metadata.warmups,
        "repetitions": metadata.repetitions,
        "cases": records,
    });
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(output, serde_json::to_vec_pretty(&receipt)?)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = parse_options();
    let manifest: Manifest = serde_json::from_str(include_str!("../../gpu_manifest.json"))?;
    let ManifestSelection { cases, metadata } = select_manifest_cases(manifest, &options.phase);
    let context = Context::new(0)?;
    let stream = Stream::new(&context)?;
    let handle = CublasLt::new(&context)?;
    let records = run_cases(
        cases,
        &context,
        &stream,
        &handle,
        metadata.warmups,
        metadata.repetitions,
    )?;
    write_receipt(&options.output, &metadata, records)
}
