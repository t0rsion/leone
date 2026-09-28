mod backend_adapter;
mod build_info;

use backend_adapter::{BatchPath, BatchStats, DriverBackend, DriverRuntime};
use leone::{
    Backend, BatchSession, DecodeExecution, GenerateOptions, GenerationSession, PrefillProgress,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

type DriverResult<T> = Result<T, Box<dyn std::error::Error>>;

const MANIFEST_SCHEMA: &str = "prefix-attention-runtime-manifest-v1";
const RECEIPT_SCHEMA: &str = "prefix-attention-runtime-receipt-v1";

#[derive(Debug, Deserialize)]
struct Manifest {
    schema: String,
    fixture: String,
    source_manifest: Option<String>,
    source_manifest_sha256: Option<String>,
    source_plan_sha256: Option<String>,
    #[serde(skip)]
    source_subject_sha256: Option<String>,
    #[serde(skip)]
    source_phase: Option<String>,
    #[serde(skip)]
    source_vocab: Option<usize>,
    calibration: Vec<CaseSpec>,
    evaluation: Option<Vec<CaseSpec>>,
    #[serde(skip)]
    source_fixture_sha256: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SourceManifest {
    schema: String,
    phase: String,
    model_contract: SourceModelContract,
    token_fixture: SourceFixture,
    slices: BTreeMap<String, SourceSlice>,
    cases: Vec<SourceCase>,
}

#[derive(Debug, Deserialize)]
struct SourceModelContract {
    architecture: String,
    tokenizer: String,
    vocab: usize,
    subject_sha256: String,
}

#[derive(Debug, Deserialize)]
struct SourceFixture {
    path: String,
    encoding: String,
    bytes: usize,
    sha256: String,
}

#[derive(Debug, Deserialize)]
struct SourceSlice {
    offset: usize,
    count: usize,
}

#[derive(Debug, Deserialize)]
struct SourceCase {
    name: String,
    class: String,
    operations: Vec<SourceEvent>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind")]
enum SourceEvent {
    #[serde(rename = "decode")]
    Decode {
        name: String,
        streams: Vec<SourceStream>,
    },
    #[serde(rename = "copy")]
    Copy {
        name: String,
        source: usize,
        target: usize,
        p0: usize,
        p1: isize,
    },
    #[serde(rename = "remove")]
    Remove {
        name: String,
        sequence: usize,
        p0: usize,
        p1: isize,
    },
    #[serde(rename = "decode_stepwise")]
    DecodeStepwise {
        name: String,
        streams: Vec<SourceStream>,
        capture_steps: Vec<usize>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SourceEventKind {
    Decode,
    Copy,
    Remove,
    DecodeStepwise,
}

const SHARED_SOURCE_EVENTS: &[(&str, SourceEventKind)] = &[
    ("prefix", SourceEventKind::Decode),
    ("fork_1", SourceEventKind::Copy),
    ("fork_2", SourceEventKind::Copy),
    ("fork_3", SourceEventKind::Copy),
    ("fork_4", SourceEventKind::Copy),
    ("suffix", SourceEventKind::Decode),
    ("continuation", SourceEventKind::DecodeStepwise),
];
const OWNED_SOURCE_EVENTS: &[(&str, SourceEventKind)] = &[
    ("owned_prefix", SourceEventKind::Decode),
    ("suffix", SourceEventKind::Decode),
    ("continuation", SourceEventKind::DecodeStepwise),
];
const COLD_SOURCE_EVENTS: &[(&str, SourceEventKind)] = &[
    ("full_prompt", SourceEventKind::Decode),
    ("continuation", SourceEventKind::DecodeStepwise),
];
const UNRELATED_SOURCE_EVENTS: &[(&str, SourceEventKind)] = &[
    ("unrelated_prefix", SourceEventKind::Decode),
    ("suffix", SourceEventKind::Decode),
    ("continuation", SourceEventKind::DecodeStepwise),
];
const NESTED_SOURCE_EVENTS: &[(&str, SourceEventKind)] = &[
    ("parent_prefix", SourceEventKind::Decode),
    ("partial_fork", SourceEventKind::Copy),
    ("intermediate_tail", SourceEventKind::Decode),
    ("nested_fork_2", SourceEventKind::Copy),
    ("nested_fork_3", SourceEventKind::Copy),
    ("nested_fork_4", SourceEventKind::Copy),
    ("nested_fork_5", SourceEventKind::Copy),
    ("suffix", SourceEventKind::Decode),
    ("continuation", SourceEventKind::DecodeStepwise),
];
const GROWTH_SOURCE_EVENTS: &[(&str, SourceEventKind)] = &[
    ("prefix", SourceEventKind::Decode),
    ("fork_1", SourceEventKind::Copy),
    ("fork_2", SourceEventKind::Copy),
    ("fork_3", SourceEventKind::Copy),
    ("fork_4", SourceEventKind::Copy),
    ("growth_suffix", SourceEventKind::Decode),
    ("growth_continuation", SourceEventKind::DecodeStepwise),
];
const MEMBERSHIP_SOURCE_EVENTS: &[(&str, SourceEventKind)] = &[
    ("prefix", SourceEventKind::Decode),
    ("fork_1", SourceEventKind::Copy),
    ("fork_2", SourceEventKind::Copy),
    ("fork_3", SourceEventKind::Copy),
    ("suffix", SourceEventKind::Decode),
    ("active_four", SourceEventKind::Decode),
    ("remove_seq_1", SourceEventKind::Remove),
    ("active_three", SourceEventKind::Decode),
    ("remove_seq_2", SourceEventKind::Remove),
    ("active_two", SourceEventKind::Decode),
    ("remove_seq_0", SourceEventKind::Remove),
    ("active_one", SourceEventKind::Decode),
];
const ROOT_SEQUENCES: &[usize] = &[0];
const FOUR_BRANCH_SEQUENCES: &[usize] = &[1, 2, 3, 4];
const NESTED_BRANCH_SEQUENCES: &[usize] = &[2, 3, 4, 5];
const INTERMEDIATE_SEQUENCE: &[usize] = &[1];
const MEMBERSHIP_FOUR_SEQUENCES: &[usize] = &[0, 1, 2, 3];
const MEMBERSHIP_THREE_SEQUENCES: &[usize] = &[0, 2, 3];
const MEMBERSHIP_TWO_SEQUENCES: &[usize] = &[0, 3];
const MEMBERSHIP_ONE_SEQUENCE: &[usize] = &[3];
const STANDARD_CAPTURE_STEPS: &[usize] = &[0, 1, 2, 3];
const GROWTH_CAPTURE_STEPS: &[usize] = &[0, 31, 32, 39];

const DECODE_SEQUENCE_BINDINGS: &[(&str, &str, &[usize])] = &[
    ("shared_prefix", "prefix", ROOT_SEQUENCES),
    ("shared_prefix", "suffix", FOUR_BRANCH_SEQUENCES),
    ("shared_prefix", "continuation", FOUR_BRANCH_SEQUENCES),
    ("short_prefix_control", "prefix", ROOT_SEQUENCES),
    ("short_prefix_control", "suffix", FOUR_BRANCH_SEQUENCES),
    (
        "short_prefix_control",
        "continuation",
        FOUR_BRANCH_SEQUENCES,
    ),
    ("private_tail_growth", "prefix", ROOT_SEQUENCES),
    (
        "private_tail_growth",
        "growth_suffix",
        FOUR_BRANCH_SEQUENCES,
    ),
    (
        "private_tail_growth",
        "growth_continuation",
        FOUR_BRANCH_SEQUENCES,
    ),
    ("owned_equal_prefix", "owned_prefix", FOUR_BRANCH_SEQUENCES),
    ("owned_equal_prefix", "suffix", FOUR_BRANCH_SEQUENCES),
    ("owned_equal_prefix", "continuation", FOUR_BRANCH_SEQUENCES),
    ("cold_full_history", "full_prompt", FOUR_BRANCH_SEQUENCES),
    ("cold_full_history", "continuation", FOUR_BRANCH_SEQUENCES),
    (
        "unrelated_control",
        "unrelated_prefix",
        FOUR_BRANCH_SEQUENCES,
    ),
    ("unrelated_control", "suffix", FOUR_BRANCH_SEQUENCES),
    ("unrelated_control", "continuation", FOUR_BRANCH_SEQUENCES),
    ("nested_shared_prefix", "parent_prefix", ROOT_SEQUENCES),
    (
        "nested_shared_prefix",
        "intermediate_tail",
        INTERMEDIATE_SEQUENCE,
    ),
    ("nested_shared_prefix", "suffix", NESTED_BRANCH_SEQUENCES),
    (
        "nested_shared_prefix",
        "continuation",
        NESTED_BRANCH_SEQUENCES,
    ),
    ("batch_membership_change", "prefix", ROOT_SEQUENCES),
    (
        "batch_membership_change",
        "suffix",
        MEMBERSHIP_FOUR_SEQUENCES,
    ),
    (
        "batch_membership_change",
        "active_four",
        MEMBERSHIP_FOUR_SEQUENCES,
    ),
    (
        "batch_membership_change",
        "active_three",
        MEMBERSHIP_THREE_SEQUENCES,
    ),
    (
        "batch_membership_change",
        "active_two",
        MEMBERSHIP_TWO_SEQUENCES,
    ),
    (
        "batch_membership_change",
        "active_one",
        MEMBERSHIP_ONE_SEQUENCE,
    ),
];

#[derive(Debug, Deserialize)]
struct SourceStream {
    seq: usize,
    position: usize,
    tokens: Vec<String>,
    #[serde(default)]
    outputs: Option<serde_json::Value>,
}

type MembershipSchedule = (
    Vec<Vec<usize>>,
    Vec<Vec<usize>>,
    Vec<Vec<usize>>,
    Vec<usize>,
    Vec<String>,
    Vec<Vec<usize>>,
);
type SourceStepSchedule = (Vec<Vec<usize>>, Vec<Vec<usize>>, usize);
type CopyBinding = (usize, usize, usize, isize);

#[derive(Debug, Deserialize, Clone, PartialEq, Eq)]
struct CaseSpec {
    source_case: String,
    name: String,
    topology: String,
    #[serde(default)]
    branch_sequences: Vec<usize>,
    #[serde(default)]
    retained_sequences: Vec<usize>,
    prefix_tokens: Vec<usize>,
    prefix_offsets: Vec<usize>,
    tail_offsets: Vec<usize>,
    tail_tokens: Vec<usize>,
    common_tail_offset: Option<usize>,
    common_tail_tokens: Option<usize>,
    partial_prefix_tokens: Option<usize>,
    teacher_offsets: Vec<Vec<usize>>,
    decode_steps: usize,
    capture_steps: Vec<usize>,
    #[serde(default)]
    setup_events: Vec<SetupEventSpec>,
    #[serde(default)]
    step_events: Vec<String>,
    #[serde(default)]
    step_input_positions: Vec<Vec<usize>>,
    #[serde(default)]
    active_sequences: Option<Vec<Vec<usize>>>,
    #[serde(default)]
    removed_sequences: Option<Vec<Vec<usize>>>,
}

#[derive(Debug, Deserialize, Clone, PartialEq, Eq)]
struct SetupEventSpec {
    name: String,
    sequences: Vec<usize>,
    positions: Vec<usize>,
}

#[derive(Debug, Serialize)]
struct Receipt {
    schema: &'static str,
    phase: String,
    path: String,
    device: serde_json::Value,
    quality: &'static str,
    claim_scope: &'static str,
    model_name: String,
    model_sha256: String,
    fixture_sha256: String,
    source_manifest_sha256: Option<String>,
    source_plan_sha256: Option<String>,
    source_subject_sha256: Option<String>,
    source_phase: Option<String>,
    memory_after_load: MemoryRecord,
    cases: Vec<CaseRecord>,
}

#[derive(Debug, Serialize)]
struct CaseRecord {
    name: String,
    source_case: String,
    topology: String,
    input_digest: String,
    decode_steps: usize,
    branch_count: usize,
    retained_branch_count: usize,
    raw_logit_vocab: usize,
    raw_logit_rows: usize,
    sidecar_logit_rows: usize,
    logit_bindings: Vec<LogitBinding>,
    memory_before: Vec<MemoryRecord>,
    memory_after_setup: Vec<MemoryRecord>,
    memory_after_decode: Vec<MemoryRecord>,
    memory_after_retire: Vec<MemoryRecord>,
    setup_ms: Vec<f64>,
    decode_ms: Vec<f64>,
    logits_digest: Vec<String>,
    raw_logit_position_digest: Vec<String>,
    token_digest: Vec<String>,
    logits_sidecars: Vec<SidecarRecord>,
    setup_stats: Vec<StatsRecord>,
    stats: Vec<StatsRecord>,
    batch_schedule: Vec<usize>,
    setup_median_ms: f64,
    decode_median_ms: f64,
}

#[derive(Debug, Serialize, Clone, PartialEq, Eq)]
struct LogitBinding {
    sequence: usize,
    event: String,
    token_index: usize,
    input_position: usize,
    predicted_position: usize,
    vocab: usize,
}

#[derive(Debug, Serialize, Clone, Copy, PartialEq, Eq)]
struct MemoryRecord {
    live_bytes: u64,
    reserved_bytes: u64,
    peak_live_bytes: u64,
    peak_owned_and_reserved_bytes: u64,
    live_allocations: u64,
    peak_live_allocations: u64,
    allocations: u64,
    frees: u64,
}

#[derive(Debug, Serialize)]
struct SidecarRecord {
    file: String,
    sha256: String,
}

#[derive(Debug, Serialize)]
struct StatsRecord {
    dispatches: usize,
    rows: usize,
    groups: usize,
    multi_row_groups: usize,
}

struct RunResult {
    setup_ms: f64,
    decode_ms: f64,
    logits_digest: String,
    position_digest: String,
    token_digest: String,
    setup_stats: StatsRecord,
    stats: StatsRecord,
    batch_schedule: Vec<usize>,
    raw_rows: usize,
    sidecar_rows: usize,
    bindings: Vec<LogitBinding>,
    memory_before: MemoryRecord,
    memory_after_setup: MemoryRecord,
    memory_after_decode: MemoryRecord,
    memory_after_retire: MemoryRecord,
}

struct RepetitionResult {
    run: RunResult,
    sidecar: SidecarRecord,
}

struct CaseSamples {
    setup_ms: Vec<f64>,
    decode_ms: Vec<f64>,
    logits_digest: Vec<String>,
    position_digest: Vec<String>,
    token_digest: Vec<String>,
    setup_stats: Vec<StatsRecord>,
    stats: Vec<StatsRecord>,
    sidecars: Vec<SidecarRecord>,
    raw_rows: Vec<usize>,
    sidecar_rows: Vec<usize>,
    memory_before: Vec<MemoryRecord>,
    memory_after_setup: Vec<MemoryRecord>,
    memory_after_decode: Vec<MemoryRecord>,
    memory_after_retire: Vec<MemoryRecord>,
    bindings: Option<Vec<LogitBinding>>,
    batch_schedule: Option<Vec<usize>>,
}

impl CaseSamples {
    fn with_capacity(repetitions: usize) -> Self {
        Self {
            setup_ms: Vec::with_capacity(repetitions),
            decode_ms: Vec::with_capacity(repetitions),
            logits_digest: Vec::with_capacity(repetitions),
            position_digest: Vec::with_capacity(repetitions),
            token_digest: Vec::with_capacity(repetitions),
            setup_stats: Vec::with_capacity(repetitions),
            stats: Vec::with_capacity(repetitions),
            sidecars: Vec::with_capacity(repetitions),
            raw_rows: Vec::with_capacity(repetitions),
            sidecar_rows: Vec::with_capacity(repetitions),
            memory_before: Vec::with_capacity(repetitions),
            memory_after_setup: Vec::with_capacity(repetitions),
            memory_after_decode: Vec::with_capacity(repetitions),
            memory_after_retire: Vec::with_capacity(repetitions),
            bindings: None,
            batch_schedule: None,
        }
    }

    fn push(&mut self, result: RepetitionResult, case_name: &str) -> DriverResult<()> {
        let RepetitionResult { run, sidecar } = result;
        let RunResult {
            setup_ms,
            decode_ms,
            logits_digest,
            position_digest,
            token_digest,
            setup_stats,
            stats,
            batch_schedule,
            raw_rows,
            sidecar_rows,
            bindings,
            memory_before,
            memory_after_setup,
            memory_after_decode,
            memory_after_retire,
        } = run;
        record_consistent(&mut self.bindings, bindings, "logit bindings", case_name)?;
        record_consistent(
            &mut self.batch_schedule,
            batch_schedule,
            "batch schedule",
            case_name,
        )?;
        self.setup_ms.push(setup_ms);
        self.decode_ms.push(decode_ms);
        self.logits_digest.push(logits_digest);
        self.position_digest.push(position_digest);
        self.token_digest.push(token_digest);
        self.setup_stats.push(setup_stats);
        self.stats.push(stats);
        self.sidecars.push(sidecar);
        self.raw_rows.push(raw_rows);
        self.sidecar_rows.push(sidecar_rows);
        self.memory_before.push(memory_before);
        self.memory_after_setup.push(memory_after_setup);
        self.memory_after_decode.push(memory_after_decode);
        self.memory_after_retire.push(memory_after_retire);
        Ok(())
    }
}

#[derive(Debug)]
struct Options {
    phase: String,
    path: BatchPath,
    manifest: PathBuf,
    model: PathBuf,
    output: PathBuf,
    repetitions: usize,
    warmups: usize,
}

#[derive(Default)]
struct RawOptions {
    phase: Option<String>,
    path: Option<BatchPath>,
    manifest: Option<PathBuf>,
    model: Option<PathBuf>,
    output: Option<PathBuf>,
    repetitions: usize,
    warmups: usize,
}

#[derive(Debug)]
struct BranchState {
    session: GenerationSession<DriverBackend>,
    transcript: Vec<u32>,
}

struct Sidecar {
    file: File,
    digest: Sha256,
}

#[derive(Default)]
struct OutputTrace {
    raw_rows: usize,
    sidecar_rows: usize,
    bindings: Vec<LogitBinding>,
    batch_schedule: Vec<usize>,
}

struct StepRun<'a> {
    fixture: &'a [u32],
    options: &'a GenerateOptions,
    trace: &'a mut OutputTrace,
}

impl Sidecar {
    fn create(path: &Path) -> DriverResult<Self> {
        let file = File::options().write(true).create_new(true).open(path)?;
        Ok(Self {
            file,
            digest: Sha256::new(),
        })
    }

    fn write_logits(
        &mut self,
        sequence: usize,
        token_index: usize,
        position: usize,
        values: &[f32],
    ) -> DriverResult<()> {
        let mut bytes = self.logit_header(sequence, token_index, position, values.len())?;
        for value in values {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        self.write(&bytes)
    }

    fn logit_header(
        &self,
        sequence: usize,
        token_index: usize,
        position: usize,
        vocab: usize,
    ) -> DriverResult<Vec<u8>> {
        let mut bytes = Vec::with_capacity(24 + vocab.saturating_mul(4));
        bytes.extend_from_slice(&u32::try_from(sequence)?.to_le_bytes());
        bytes.extend_from_slice(&u32::try_from(token_index)?.to_le_bytes());
        bytes.extend_from_slice(&u64::try_from(position)?.to_le_bytes());
        bytes.extend_from_slice(&u32::try_from(vocab)?.to_le_bytes());
        Ok(bytes)
    }

    fn write(&mut self, bytes: &[u8]) -> DriverResult<()> {
        self.file.write_all(bytes)?;
        self.digest.update(bytes);
        Ok(())
    }

    fn finish(mut self) -> DriverResult<String> {
        self.file.flush()?;
        self.file.sync_all()?;
        Ok(hex(&self.digest.finalize()))
    }
}

impl OutputTrace {
    #[allow(clippy::too_many_arguments)]
    fn record(
        &mut self,
        sequence: usize,
        event: &str,
        token_index: usize,
        input_position: usize,
        predicted_position: usize,
        vocab: usize,
        values: &[f32],
        sidecar: Option<&mut Sidecar>,
    ) -> DriverResult<()> {
        self.raw_rows = self
            .raw_rows
            .checked_add(1)
            .ok_or("raw logit row count overflow")?;
        if let Some(sidecar) = sidecar {
            sidecar.write_logits(sequence, token_index, predicted_position, values)?;
            self.sidecar_rows = self
                .sidecar_rows
                .checked_add(1)
                .ok_or("sidecar logit row count overflow")?;
        }
        self.bindings.push(LogitBinding {
            sequence,
            event: event.to_owned(),
            token_index,
            input_position,
            predicted_position,
            vocab,
        });
        Ok(())
    }
}

fn parse_options() -> DriverResult<Options> {
    let mut values = env::args().skip(1);
    let mut raw = RawOptions {
        repetitions: 1,
        ..RawOptions::default()
    };
    while let Some(flag) = values.next() {
        let value = values
            .next()
            .ok_or_else(|| format!("{flag} requires a value"))?;
        apply_option(&mut raw, &flag, value)?;
    }
    finish_options(raw)
}

fn finish_options(raw: RawOptions) -> DriverResult<Options> {
    let phase = raw.phase.ok_or("--phase is required")?;
    if phase != "calibration" && phase != "evaluation" {
        return Err("--phase must be calibration or evaluation".into());
    }
    if raw.repetitions == 0 {
        return Err("--repetitions must be nonzero".into());
    }
    Ok(Options {
        phase,
        path: raw.path.ok_or("--path is required")?,
        manifest: raw.manifest.ok_or("--manifest is required")?,
        model: raw.model.ok_or("--model is required")?,
        output: raw.output.ok_or("--output is required")?,
        repetitions: raw.repetitions,
        warmups: raw.warmups,
    })
}

fn apply_option(raw: &mut RawOptions, flag: &str, value: String) -> DriverResult<()> {
    match flag {
        "--phase" => raw.phase = Some(value),
        "--path" => raw.path = Some(parse_path(&value)?),
        _ => apply_secondary_option(raw, flag, value)?,
    }
    Ok(())
}

fn apply_secondary_option(raw: &mut RawOptions, flag: &str, value: String) -> DriverResult<()> {
    match flag {
        "--manifest" => raw.manifest = Some(PathBuf::from(value)),
        "--model" => raw.model = Some(PathBuf::from(value)),
        "--output" => raw.output = Some(PathBuf::from(value)),
        "--repetitions" => raw.repetitions = value.parse()?,
        "--warmups" => raw.warmups = value.parse()?,
        _ => return Err(format!("unknown option {flag}").into()),
    }
    Ok(())
}

fn parse_path(value: &str) -> DriverResult<BatchPath> {
    backend_adapter::parse_path(value).map_err(Into::into)
}

fn path_name(path: BatchPath) -> &'static str {
    backend_adapter::path_name(path)
}

fn load_manifest(path: &Path) -> DriverResult<Manifest> {
    let mut manifest: Manifest = serde_json::from_slice(&fs::read(path)?)?;
    let source = load_and_validate_source(path, &manifest)?;
    finalize_manifest(&mut manifest, source.as_ref())?;
    Ok(manifest)
}

fn load_and_validate_source(
    path: &Path,
    manifest: &Manifest,
) -> DriverResult<Option<SourceManifest>> {
    validate_manifest_schema(manifest)?;
    let source = load_source_manifest(path, manifest)?;
    validate_loaded_source(path, manifest, source.as_ref())?;
    Ok(source)
}

fn finalize_manifest(manifest: &mut Manifest, source: Option<&SourceManifest>) -> DriverResult<()> {
    fill_evaluation_cases(manifest, source)?;
    manifest.source_fixture_sha256 = source.map(|source| source.token_fixture.sha256.clone());
    manifest.source_subject_sha256 =
        source.map(|source| source.model_contract.subject_sha256.clone());
    manifest.source_phase = source.map(|source| source.phase.clone());
    manifest.source_vocab = source.map(|source| source.model_contract.vocab);
    validate_manifest_cases(manifest, source)
}

fn validate_loaded_source(
    manifest_path: &Path,
    manifest: &Manifest,
    source: Option<&SourceManifest>,
) -> DriverResult<()> {
    if let Some(source) = source {
        validate_source_plan(manifest_path, manifest, source)?;
    }
    Ok(())
}

fn validate_source_plan(
    manifest_path: &Path,
    manifest: &Manifest,
    source: &SourceManifest,
) -> DriverResult<()> {
    let expected = manifest
        .source_plan_sha256
        .as_deref()
        .ok_or("source plan digest is missing")?;
    let source_name = manifest
        .source_manifest
        .as_deref()
        .ok_or("source manifest path is missing")?;
    let source_path = manifest_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(source_name);
    let fixture_path = resolve_source_path(&source_path, &source.token_fixture.path);
    let fixture = read_tokens(&fixture_path)?;
    validate_source_positions(source, &fixture)?;
    let actual = source_plan_digest(source, &fixture)?;
    if actual != expected {
        return Err(
            format!("source plan digest differs: expected {expected}, got {actual}").into(),
        );
    }
    Ok(())
}

fn validate_manifest_schema(manifest: &Manifest) -> DriverResult<()> {
    if manifest.schema == MANIFEST_SCHEMA {
        Ok(())
    } else {
        Err(format!("unsupported manifest schema {}", manifest.schema).into())
    }
}

fn fill_evaluation_cases(
    manifest: &mut Manifest,
    source: Option<&SourceManifest>,
) -> DriverResult<()> {
    let Some(source) = source else {
        if manifest.evaluation.is_none() {
            return Err("evaluation cases are missing".into());
        }
        return Ok(());
    };
    let expected = source_cases(source)?;
    if let Some(actual) = manifest.evaluation.as_ref() {
        if actual != &expected {
            return Err("evaluation cases differ from the pinned source workload".into());
        }
    } else {
        manifest.evaluation = Some(expected);
    }
    Ok(())
}

fn validate_manifest_cases(
    manifest: &Manifest,
    source: Option<&SourceManifest>,
) -> DriverResult<()> {
    let evaluation = manifest
        .evaluation
        .as_ref()
        .ok_or("evaluation cases are missing")?;
    validate_cases(&manifest.calibration)?;
    validate_cases(evaluation)?;
    if manifest.calibration.is_empty() || evaluation.is_empty() {
        return Err("both manifest phases require cases".into());
    }
    if let Some(source) = source {
        validate_source_cases(source, evaluation)?;
    }
    Ok(())
}

fn load_source_manifest(path: &Path, manifest: &Manifest) -> DriverResult<Option<SourceManifest>> {
    let (Some(source), Some(expected)) =
        (&manifest.source_manifest, &manifest.source_manifest_sha256)
    else {
        return Ok(None);
    };
    let source_path = path.parent().unwrap_or_else(|| Path::new(".")).join(source);
    validate_source_digest(&source_path, expected)?;
    let source_manifest = read_source_manifest(&source_path)?;
    validate_source_manifest(&source_path, &source_manifest)?;
    Ok(Some(source_manifest))
}

fn validate_source_digest(path: &Path, expected: &str) -> DriverResult<()> {
    if sha256_file(path)? != expected {
        return Err(format!("source workload digest differs for {}", path.display()).into());
    }
    Ok(())
}

fn read_source_manifest(path: &Path) -> DriverResult<SourceManifest> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn validate_source_manifest(path: &Path, source: &SourceManifest) -> DriverResult<()> {
    if source.schema != "leone.llama-cached-workload.v2"
        || source.phase != "exploratory_calibration"
    {
        return Err(format!("unsupported source workload {}", source.schema).into());
    }
    validate_source_model_contract(&source.model_contract)?;
    validate_source_fixture(&source.token_fixture)?;
    validate_source_fixture_file(path, &source.token_fixture)?;
    validate_source_events(&source.cases)
}

fn validate_source_model_contract(contract: &SourceModelContract) -> DriverResult<()> {
    if contract.architecture != "qwen3"
        || contract.tokenizer != "gpt2"
        || contract.vocab == 0
        || contract.subject_sha256.len() != 64
        || !contract
            .subject_sha256
            .chars()
            .all(|character| character.is_ascii_digit() || ('a'..='f').contains(&character))
    {
        return Err("invalid source model contract".into());
    }
    Ok(())
}

fn validate_source_fixture(fixture: &SourceFixture) -> DriverResult<()> {
    if fixture.encoding != "u32le" || fixture.bytes == 0 || fixture.sha256.len() != 64 {
        return Err("invalid source token fixture identity".into());
    }
    if fixture.path.is_empty() {
        return Err("source token fixture path is empty".into());
    }
    Ok(())
}

fn validate_source_fixture_file(
    source_manifest_path: &Path,
    fixture: &SourceFixture,
) -> DriverResult<()> {
    let fixture_path = resolve_source_path(source_manifest_path, &fixture.path);
    let bytes = fs::metadata(&fixture_path)?.len();
    if bytes != u64::try_from(fixture.bytes)? {
        return Err(format!(
            "source token fixture size differs for {}",
            fixture_path.display()
        )
        .into());
    }
    if sha256_file(&fixture_path)? != fixture.sha256 {
        return Err(format!(
            "source token fixture digest differs for {}",
            fixture_path.display()
        )
        .into());
    }
    Ok(())
}

fn resolve_source_path(source_manifest_path: &Path, declared: &str) -> PathBuf {
    let declared = Path::new(declared);
    if declared.is_absolute() {
        return declared.to_owned();
    }
    source_manifest_path
        .ancestors()
        .map(|ancestor| ancestor.join(declared))
        .find(|candidate| candidate.is_file())
        .unwrap_or_else(|| declared.to_owned())
}

fn validate_source_events(cases: &[SourceCase]) -> DriverResult<()> {
    if cases.is_empty() {
        return Err("source workload has no cases".into());
    }
    for case in cases {
        if case.name.is_empty() || case.operations.is_empty() {
            return Err(format!("source case {} is empty", case.name).into());
        }
        for event in &case.operations {
            validate_source_event(event)?;
        }
        validate_source_order(case)?;
    }
    Ok(())
}

fn validate_source_positions(source: &SourceManifest, fixture: &[u32]) -> DriverResult<()> {
    for case in &source.cases {
        let mut state = vec![BTreeSet::new(); 8];
        for event in &case.operations {
            validate_source_position_event(source, fixture, event, &mut state, &case.name)?;
        }
    }
    Ok(())
}

fn validate_source_position_event(
    source: &SourceManifest,
    fixture: &[u32],
    event: &SourceEvent,
    state: &mut [BTreeSet<usize>],
    case_name: &str,
) -> DriverResult<()> {
    match event {
        SourceEvent::Decode { streams, .. } => {
            validate_source_decode(source, fixture, streams, None, state, case_name)
        }
        SourceEvent::DecodeStepwise { streams, .. } => {
            validate_source_stepwise(source, fixture, streams, state, case_name)
        }
        SourceEvent::Copy {
            source: source_sequence,
            target,
            p0,
            p1,
            ..
        } => validate_source_copy(state, *source_sequence, *target, *p0, *p1, case_name),
        SourceEvent::Remove {
            sequence, p0, p1, ..
        } => validate_source_remove(state, *sequence, *p0, *p1, case_name),
    }
}

fn validate_source_stepwise(
    source: &SourceManifest,
    fixture: &[u32],
    streams: &[SourceStream],
    state: &mut [BTreeSet<usize>],
    case_name: &str,
) -> DriverResult<()> {
    let steps = source_step_count(source, fixture, streams)?;
    for step in 0..steps {
        validate_source_decode(source, fixture, streams, Some(step), state, case_name)?;
    }
    Ok(())
}

fn validate_source_decode(
    source: &SourceManifest,
    fixture: &[u32],
    streams: &[SourceStream],
    step: Option<usize>,
    state: &mut [BTreeSet<usize>],
    case_name: &str,
) -> DriverResult<()> {
    for stream in streams {
        validate_source_stream_position(source, fixture, stream, step, state, case_name)?;
    }
    Ok(())
}

fn validate_source_stream_position(
    source: &SourceManifest,
    fixture: &[u32],
    stream: &SourceStream,
    step: Option<usize>,
    state: &mut [BTreeSet<usize>],
    case_name: &str,
) -> DriverResult<()> {
    let tokens = source_stream_tokens(source, fixture, stream)?;
    let offset = stream
        .position
        .checked_add(step.unwrap_or(0))
        .ok_or("source stream position overflow")?;
    let count = step.map_or(tokens.len(), |_| 1);
    let end = offset
        .checked_add(count)
        .ok_or("source stream range overflow")?;
    let positions = state
        .get(stream.seq)
        .ok_or("source sequence is out of range")?;
    validate_source_position_range(positions, offset, end, case_name)?;
    state[stream.seq].extend(offset..end);
    Ok(())
}

fn validate_source_position_range(
    positions: &BTreeSet<usize>,
    offset: usize,
    end: usize,
    case_name: &str,
) -> DriverResult<()> {
    if positions
        .iter()
        .any(|position| *position >= offset && *position < end)
    {
        return Err(format!("source decode overlaps sequence in {case_name}").into());
    }
    let expected = positions
        .iter()
        .next_back()
        .and_then(|position| position.checked_add(1))
        .unwrap_or(0);
    if offset != expected {
        return Err(format!(
            "source decode position {offset} differs from {expected} in {case_name}"
        )
        .into());
    }
    Ok(())
}

fn validate_source_copy(
    state: &mut [BTreeSet<usize>],
    source: usize,
    target: usize,
    start: usize,
    end: isize,
    case_name: &str,
) -> DriverResult<()> {
    if source >= state.len()
        || target >= state.len()
        || source == target
        || !state[target].is_empty()
    {
        return Err(format!("invalid source copy in {case_name}").into());
    }
    let upper = if end < 0 {
        usize::MAX
    } else {
        usize::try_from(end)?
    };
    let copied = state[source]
        .iter()
        .filter(|position| **position >= start && **position < upper)
        .copied()
        .collect::<Vec<_>>();
    validate_contiguous_copy(&copied, case_name)?;
    state[target].extend(copied);
    Ok(())
}

fn validate_contiguous_copy(copied: &[usize], case_name: &str) -> DriverResult<()> {
    if copied.is_empty() || copied.windows(2).any(|pair| pair[1] != pair[0] + 1) {
        return Err(format!("source copy is not contiguous in {case_name}").into());
    }
    Ok(())
}

fn validate_source_remove(
    state: &mut [BTreeSet<usize>],
    sequence: usize,
    start: usize,
    end: isize,
    case_name: &str,
) -> DriverResult<()> {
    if sequence >= state.len() {
        return Err(format!("invalid source removal in {case_name}").into());
    }
    let upper = if end < 0 {
        usize::MAX
    } else {
        usize::try_from(end)?
    };
    let removed = state[sequence]
        .iter()
        .filter(|position| **position >= start && **position < upper)
        .copied()
        .collect::<Vec<_>>();
    if removed.is_empty() {
        return Err(format!("source removal is empty in {case_name}").into());
    }
    for position in removed {
        state[sequence].remove(&position);
    }
    Ok(())
}

fn source_stream_tokens(
    source: &SourceManifest,
    fixture: &[u32],
    stream: &SourceStream,
) -> DriverResult<Vec<u32>> {
    let mut tokens = Vec::new();
    for name in &stream.tokens {
        let slice = source
            .slices
            .get(name)
            .ok_or_else(|| format!("unknown source slice {name}"))?;
        tokens.extend_from_slice(checked_slice(fixture, slice.offset, slice.count)?);
    }
    if tokens.is_empty() {
        return Err("source stream has no resolved tokens".into());
    }
    Ok(tokens)
}

fn source_plan_digest(source: &SourceManifest, fixture: &[u32]) -> DriverResult<String> {
    let mut bytes = b"LCMPPLN1".to_vec();
    plan_u32(&mut bytes, 2, "plan version")?;
    plan_u32(&mut bytes, source.cases.len(), "plan case count")?;
    for case in &source.cases {
        plan_string(&mut bytes, &case.name)?;
        plan_u32(
            &mut bytes,
            expanded_event_count(source, fixture, case)?,
            "plan event count",
        )?;
        for event in &case.operations {
            encode_source_event(source, fixture, event, &mut bytes)?;
        }
    }
    Ok(hex(&Sha256::digest(bytes)))
}

fn expanded_event_count(
    source: &SourceManifest,
    fixture: &[u32],
    case: &SourceCase,
) -> DriverResult<usize> {
    case.operations.iter().try_fold(0usize, |count, event| {
        let addition = match event {
            SourceEvent::Decode { .. } | SourceEvent::Copy { .. } | SourceEvent::Remove { .. } => 1,
            SourceEvent::DecodeStepwise { streams, .. } => {
                source_step_count(source, fixture, streams)?
            }
        };
        count
            .checked_add(addition)
            .ok_or_else(|| "plan event count overflow".into())
    })
}

fn source_step_count(
    source: &SourceManifest,
    fixture: &[u32],
    streams: &[SourceStream],
) -> DriverResult<usize> {
    let first = streams.first().ok_or("stepwise event has no streams")?;
    let count = source_stream_tokens(source, fixture, first)?.len();
    for stream in streams.iter().skip(1) {
        if source_stream_tokens(source, fixture, stream)?.len() != count {
            return Err("stepwise source streams have unequal lengths".into());
        }
    }
    Ok(count)
}

fn encode_source_event(
    source: &SourceManifest,
    fixture: &[u32],
    event: &SourceEvent,
    bytes: &mut Vec<u8>,
) -> DriverResult<()> {
    match event {
        SourceEvent::Decode { name, streams } => {
            encode_source_decode(source, fixture, name, streams, None, false, bytes)
        }
        SourceEvent::DecodeStepwise {
            name,
            streams,
            capture_steps,
        } => encode_source_stepwise(source, fixture, name, streams, capture_steps, bytes),
        SourceEvent::Copy {
            name,
            source,
            target,
            p0,
            p1,
        } => encode_source_copy(bytes, name, *source, *target, *p0, *p1),
        SourceEvent::Remove {
            name,
            sequence,
            p0,
            p1,
        } => encode_source_remove(bytes, name, *sequence, *p0, *p1),
    }
}

fn encode_source_stepwise(
    source: &SourceManifest,
    fixture: &[u32],
    name: &str,
    streams: &[SourceStream],
    capture_steps: &[usize],
    bytes: &mut Vec<u8>,
) -> DriverResult<()> {
    let count = source_step_count(source, fixture, streams)?;
    for step in 0..count {
        let step_name = format!("{name}_{step}");
        encode_source_decode(
            source,
            fixture,
            &step_name,
            streams,
            Some(step),
            capture_steps.contains(&step),
            bytes,
        )?;
    }
    Ok(())
}

fn encode_source_copy(
    bytes: &mut Vec<u8>,
    name: &str,
    source: usize,
    target: usize,
    p0: usize,
    p1: isize,
) -> DriverResult<()> {
    bytes.push(2);
    plan_string(bytes, name)?;
    plan_i32(bytes, source as i64)?;
    plan_i32(bytes, target as i64)?;
    plan_i32(bytes, p0 as i64)?;
    plan_i32(bytes, p1 as i64)
}

fn encode_source_remove(
    bytes: &mut Vec<u8>,
    name: &str,
    sequence: usize,
    p0: usize,
    p1: isize,
) -> DriverResult<()> {
    bytes.push(3);
    plan_string(bytes, name)?;
    plan_i32(bytes, sequence as i64)?;
    plan_i32(bytes, p0 as i64)?;
    plan_i32(bytes, p1 as i64)
}

fn encode_source_decode(
    source: &SourceManifest,
    fixture: &[u32],
    name: &str,
    streams: &[SourceStream],
    step: Option<usize>,
    capture: bool,
    bytes: &mut Vec<u8>,
) -> DriverResult<()> {
    bytes.push(1);
    plan_string(bytes, name)?;
    plan_u32(bytes, streams.len(), "plan stream count")?;
    for stream in streams {
        encode_source_stream(source, fixture, stream, step, capture, bytes)?;
    }
    Ok(())
}

fn encode_source_stream(
    source: &SourceManifest,
    fixture: &[u32],
    stream: &SourceStream,
    step: Option<usize>,
    capture: bool,
    bytes: &mut Vec<u8>,
) -> DriverResult<()> {
    let tokens = source_stream_tokens(source, fixture, stream)?;
    let position = stream
        .position
        .checked_add(step.unwrap_or(0))
        .ok_or("stepwise source position overflow")?;
    let count = step.map_or(tokens.len(), |_| 1);
    plan_i32(bytes, stream.seq as i64)?;
    plan_i32(bytes, position as i64)?;
    plan_u32(bytes, count, "plan token count")?;
    encode_source_stream_values(stream, &tokens, step, capture, bytes)
}

fn encode_source_stream_values(
    stream: &SourceStream,
    tokens: &[u32],
    step: Option<usize>,
    capture: bool,
    bytes: &mut Vec<u8>,
) -> DriverResult<()> {
    let count = step.map_or(tokens.len(), |_| 1);
    for index in 0..count {
        let token = step.map_or(tokens[index], |step| tokens[step]);
        plan_i32(bytes, i64::from(token))?;
        let selected = match step {
            Some(_) => capture,
            None => source_output_selected(stream.outputs.as_ref(), index, tokens.len())?,
        };
        bytes.push(u8::from(selected));
    }
    Ok(())
}

fn source_output_selected(
    outputs: Option<&serde_json::Value>,
    index: usize,
    count: usize,
) -> DriverResult<bool> {
    let Some(outputs) = outputs else {
        return Ok(false);
    };
    match outputs {
        serde_json::Value::String(value) => match value.as_str() {
            "none" => Ok(false),
            "last" => Ok(index + 1 == count),
            "all" => Ok(true),
            _ => Err("invalid source output selector".into()),
        },
        serde_json::Value::Array(values) => Ok(values.iter().any(|value| {
            value
                .as_u64()
                .is_some_and(|selected| selected == index as u64)
        })),
        _ => Err("invalid source output selector".into()),
    }
}

fn plan_u32(bytes: &mut Vec<u8>, value: usize, field: &str) -> DriverResult<()> {
    bytes.extend_from_slice(
        &u32::try_from(value)
            .map_err(|_| format!("{field} is too large"))?
            .to_le_bytes(),
    );
    Ok(())
}

fn plan_i32(bytes: &mut Vec<u8>, value: i64) -> DriverResult<()> {
    bytes.extend_from_slice(&i32::try_from(value)?.to_le_bytes());
    Ok(())
}

fn plan_string(bytes: &mut Vec<u8>, value: &str) -> DriverResult<()> {
    let value = value.as_bytes();
    plan_u32(bytes, value.len(), "plan string")?;
    bytes.extend_from_slice(value);
    Ok(())
}

fn validate_source_order(case: &SourceCase) -> DriverResult<()> {
    let mut live = [false; 8];
    for event in &case.operations {
        validate_source_event_order(event, &mut live, &case.name)?;
    }
    Ok(())
}

fn validate_source_event_order(
    event: &SourceEvent,
    live: &mut [bool; 8],
    case_name: &str,
) -> DriverResult<()> {
    match event {
        SourceEvent::Decode { streams, .. } => activate_streams(streams, live),
        SourceEvent::DecodeStepwise { streams, .. } => require_live_streams(streams, live),
        SourceEvent::Copy { source, target, .. } => {
            if !live[*source] || live[*target] {
                return Err(format!("invalid copy order in {case_name}").into());
            }
            live[*target] = true;
            Ok(())
        }
        SourceEvent::Remove { sequence, .. } => {
            if !live[*sequence] {
                return Err(format!("invalid remove order in {case_name}").into());
            }
            live[*sequence] = false;
            Ok(())
        }
    }
}

fn activate_streams(streams: &[SourceStream], live: &mut [bool; 8]) -> DriverResult<()> {
    for stream in streams {
        if stream.seq >= live.len() {
            return Err("source stream sequence is out of range".into());
        }
        live[stream.seq] = true;
    }
    Ok(())
}

fn require_live_streams(streams: &[SourceStream], live: &[bool; 8]) -> DriverResult<()> {
    if streams
        .iter()
        .any(|stream| stream.seq >= live.len() || !live[stream.seq])
    {
        return Err("stepwise source stream is not live".into());
    }
    Ok(())
}

fn validate_source_event(event: &SourceEvent) -> DriverResult<()> {
    match event {
        SourceEvent::Decode { name, streams } => validate_source_streams(name, streams)?,
        SourceEvent::Copy {
            name,
            source,
            target,
            p0,
            p1,
        } => {
            validate_copy_event(name, *source, *target, *p0, *p1)?;
        }
        SourceEvent::Remove {
            name,
            sequence,
            p0,
            p1,
        } => {
            validate_remove_event(name, *sequence, *p0, *p1)?;
        }
        SourceEvent::DecodeStepwise {
            name,
            streams,
            capture_steps,
        } => {
            validate_stepwise_event(name, streams, capture_steps)?;
        }
    }
    Ok(())
}

fn validate_remove_event(
    name: &str,
    sequence: usize,
    start: usize,
    end: isize,
) -> DriverResult<()> {
    let invalid_range = end < -1 || (end >= 0 && end as usize <= start);
    if name.is_empty() || sequence >= 8 || invalid_range {
        Err(format!("invalid remove event {name}").into())
    } else {
        Ok(())
    }
}

fn validate_copy_event(
    name: &str,
    source: usize,
    target: usize,
    start: usize,
    end: isize,
) -> DriverResult<()> {
    let invalid_range = end < -1 || (end >= 0 && end as usize <= start);
    if name.is_empty() || source >= 8 || target >= 8 || invalid_range {
        Err(format!("invalid copy event {name}").into())
    } else {
        Ok(())
    }
}

fn validate_stepwise_event(
    name: &str,
    streams: &[SourceStream],
    capture_steps: &[usize],
) -> DriverResult<()> {
    validate_source_streams(name, streams)?;
    if capture_steps.is_empty() || capture_steps.windows(2).any(|steps| steps[0] >= steps[1]) {
        Err(format!("invalid capture steps for {name}").into())
    } else {
        Ok(())
    }
}

fn validate_source_streams(name: &str, streams: &[SourceStream]) -> DriverResult<()> {
    if name.is_empty() || streams.is_empty() {
        return Err(format!("source event {name} has no streams").into());
    }
    for stream in streams {
        if stream.tokens.is_empty() || stream.outputs.as_ref().is_some_and(|value| value.is_null())
        {
            return Err(format!("source event {name} has an invalid stream").into());
        }
    }
    Ok(())
}

fn source_cases(source: &SourceManifest) -> DriverResult<Vec<CaseSpec>> {
    source
        .cases
        .iter()
        .map(|case| source_case(source, case))
        .collect()
}

fn validate_source_cases(source: &SourceManifest, evaluation: &[CaseSpec]) -> DriverResult<()> {
    for case in &source.cases {
        validate_source_case_events(case)?;
        let required = required_decode_events(case)?;
        if decode_events(case).len() < required {
            return Err(format!("source case {} lacks setup events", case.name).into());
        }
    }
    for case in evaluation {
        if !source
            .cases
            .iter()
            .any(|source| source.name == case.source_case)
        {
            return Err(
                format!("runtime case {} is absent from source workload", case.name).into(),
            );
        }
    }
    Ok(())
}

fn validate_source_case_events(case: &SourceCase) -> DriverResult<()> {
    let expected = expected_source_events(&case.class)?;
    if case.operations.len() != expected.len() {
        return Err(format!(
            "source case {} has {} events, expected {}",
            case.name,
            case.operations.len(),
            expected.len()
        )
        .into());
    }
    for (event, (name, kind)) in case.operations.iter().zip(expected) {
        if source_event_name(event) != *name || source_event_kind(event) != *kind {
            return Err(format!("source event order differs in {}", case.name).into());
        }
        validate_source_event_binding(case, event)?;
    }
    Ok(())
}

fn expected_source_events(class: &str) -> DriverResult<&'static [(&'static str, SourceEventKind)]> {
    match class {
        "nested_shared_prefix" => Ok(NESTED_SOURCE_EVENTS),
        "batch_membership_change" => Ok(MEMBERSHIP_SOURCE_EVENTS),
        "private_tail_growth" => Ok(GROWTH_SOURCE_EVENTS),
        "owned_equal_prefix" => Ok(OWNED_SOURCE_EVENTS),
        "cold_full_history" => Ok(COLD_SOURCE_EVENTS),
        "unrelated_control" => Ok(UNRELATED_SOURCE_EVENTS),
        "shared_prefix" | "short_prefix_control" => Ok(SHARED_SOURCE_EVENTS),
        _ => Err(format!("unsupported source case class {class}").into()),
    }
}

fn source_event_kind(event: &SourceEvent) -> SourceEventKind {
    match event {
        SourceEvent::Decode { .. } => SourceEventKind::Decode,
        SourceEvent::Copy { .. } => SourceEventKind::Copy,
        SourceEvent::Remove { .. } => SourceEventKind::Remove,
        SourceEvent::DecodeStepwise { .. } => SourceEventKind::DecodeStepwise,
    }
}

fn validate_source_event_binding(case: &SourceCase, event: &SourceEvent) -> DriverResult<()> {
    match event {
        SourceEvent::Decode { name, streams } => {
            validate_decode_binding(case, name, streams, None)?;
        }
        SourceEvent::DecodeStepwise {
            name,
            streams,
            capture_steps,
        } => {
            validate_decode_binding(case, name, streams, Some(capture_steps))?;
        }
        SourceEvent::Copy {
            name,
            source,
            target,
            p0,
            p1,
        } => {
            validate_copy_event_binding(name, *source, *target, *p0, *p1)?;
        }
        SourceEvent::Remove {
            name,
            sequence,
            p0,
            p1,
        } => {
            validate_remove_event_binding(name, *sequence, *p0, *p1)?;
        }
    }
    Ok(())
}

fn validate_copy_event_binding(
    name: &str,
    source: usize,
    target: usize,
    p0: usize,
    p1: isize,
) -> DriverResult<()> {
    let expected = expected_copy_binding(name)?;
    if (source, target, p0, p1) == expected {
        Ok(())
    } else {
        Err(format!("source copy binding differs for {name}").into())
    }
}

fn validate_remove_event_binding(
    name: &str,
    sequence: usize,
    p0: usize,
    p1: isize,
) -> DriverResult<()> {
    let expected = expected_remove_sequence(name)?;
    if sequence == expected && p0 == 0 && p1 == -1 {
        Ok(())
    } else {
        Err(format!("source remove binding differs for {name}").into())
    }
}

fn validate_decode_binding(
    case: &SourceCase,
    name: &str,
    streams: &[SourceStream],
    capture_steps: Option<&[usize]>,
) -> DriverResult<()> {
    let expected = expected_decode_sequences(&case.class, name)?;
    let actual = streams.iter().map(|stream| stream.seq).collect::<Vec<_>>();
    if actual.as_slice() != expected {
        return Err(format!("source decode stream binding differs for {name}").into());
    }
    if !decode_outputs_match(streams, capture_steps) {
        return Err(format!("source output binding differs for {name}").into());
    }
    if let Some(capture_steps) = capture_steps {
        let expected = expected_capture_steps(&case.class)?;
        if capture_steps != expected {
            return Err(format!("source capture binding differs for {name}").into());
        }
    }
    Ok(())
}

fn decode_outputs_match(streams: &[SourceStream], capture_steps: Option<&[usize]>) -> bool {
    streams.iter().all(|stream| match capture_steps {
        Some(_) => stream.outputs.is_none(),
        None => is_last_output(stream.outputs.as_ref()),
    })
}

fn expected_decode_sequences(class: &str, name: &str) -> DriverResult<&'static [usize]> {
    DECODE_SEQUENCE_BINDINGS
        .iter()
        .find(|(candidate_class, candidate_name, _)| {
            *candidate_class == class && *candidate_name == name
        })
        .map(|(_, _, sequences)| *sequences)
        .ok_or_else(|| format!("unsupported decode event {name} in {class}").into())
}

fn expected_capture_steps(class: &str) -> DriverResult<&'static [usize]> {
    match class {
        "private_tail_growth" => Ok(GROWTH_CAPTURE_STEPS),
        "nested_shared_prefix"
        | "owned_equal_prefix"
        | "cold_full_history"
        | "unrelated_control"
        | "shared_prefix"
        | "short_prefix_control" => Ok(STANDARD_CAPTURE_STEPS),
        _ => Err(format!("unsupported stepwise source class {class}").into()),
    }
}

const COPY_BINDINGS: &[(&str, CopyBinding)] = &[
    ("partial_fork", (0, 1, 0, 257)),
    ("nested_fork_2", (1, 2, 0, -1)),
    ("nested_fork_3", (1, 3, 0, -1)),
    ("nested_fork_4", (1, 4, 0, -1)),
    ("nested_fork_5", (1, 5, 0, -1)),
    ("fork_1", (0, 1, 0, -1)),
    ("fork_2", (0, 2, 0, -1)),
    ("fork_3", (0, 3, 0, -1)),
    ("fork_4", (0, 4, 0, -1)),
];

fn expected_copy_binding(name: &str) -> DriverResult<CopyBinding> {
    COPY_BINDINGS
        .iter()
        .find(|(candidate, _)| *candidate == name)
        .map(|(_, binding)| *binding)
        .ok_or_else(|| format!("unsupported copy event {name}").into())
}

fn expected_remove_sequence(name: &str) -> DriverResult<usize> {
    match name {
        "remove_seq_0" => Ok(0),
        "remove_seq_1" => Ok(1),
        "remove_seq_2" => Ok(2),
        _ => Err(format!("unsupported remove event {name}").into()),
    }
}

fn required_decode_events(case: &SourceCase) -> DriverResult<usize> {
    match case.class.as_str() {
        "nested_shared_prefix" => Ok(3),
        "shared_prefix"
        | "short_prefix_control"
        | "private_tail_growth"
        | "owned_equal_prefix"
        | "unrelated_control" => Ok(2),
        "cold_full_history" => Ok(1),
        "batch_membership_change" => Ok(6),
        _ => Err(format!("unsupported source case class {}", case.class).into()),
    }
}

fn source_case(source: &SourceManifest, case: &SourceCase) -> DriverResult<CaseSpec> {
    match case.class.as_str() {
        "shared_prefix" | "short_prefix_control" | "private_tail_growth" => {
            source_shared_case(source, case)
        }
        "owned_equal_prefix" => source_owned_case(source, case),
        "cold_full_history" => source_cold_case(source, case),
        "unrelated_control" => source_unrelated_case(source, case),
        "nested_shared_prefix" => source_nested_case(source, case),
        "batch_membership_change" => source_membership_case(source, case),
        _ => Err(format!("unsupported source case class {}", case.class).into()),
    }
}

fn source_shared_case(source: &SourceManifest, case: &SourceCase) -> DriverResult<CaseSpec> {
    let decodes = decode_events(case);
    let prefix = stream_range(source, stream(decodes[0], 0)?, "prefix")?;
    let suffixes = branch_ranges(source, decodes[1], &[1, 2, 3, 4], "suffix")?;
    build_source_case(
        source,
        case,
        "shared",
        &[1, 2, 3, 4],
        vec![prefix; 4],
        suffixes,
        None,
    )
}

fn source_owned_case(source: &SourceManifest, case: &SourceCase) -> DriverResult<CaseSpec> {
    let decodes = decode_events(case);
    let prefixes = branch_ranges(source, decodes[0], &[1, 2, 3, 4], "prefix")?;
    let suffixes = branch_ranges(source, decodes[1], &[1, 2, 3, 4], "suffix")?;
    build_source_case(
        source,
        case,
        "owned_equal",
        &[1, 2, 3, 4],
        prefixes,
        suffixes,
        None,
    )
}

fn source_cold_case(source: &SourceManifest, case: &SourceCase) -> DriverResult<CaseSpec> {
    let decodes = decode_events(case);
    let streams = [1, 2, 3, 4]
        .into_iter()
        .map(|sequence| stream(decodes[0], sequence))
        .collect::<DriverResult<Vec<_>>>()?;
    let prefixes = streams
        .iter()
        .map(|stream| stream_range_first(source, stream, "prefix"))
        .collect::<DriverResult<Vec<_>>>()?;
    let suffixes = streams
        .iter()
        .map(|stream| stream_range_after_prefix(source, stream, "suffix"))
        .collect::<DriverResult<Vec<_>>>()?;
    build_source_case(
        source,
        case,
        "cold",
        &[1, 2, 3, 4],
        prefixes,
        suffixes,
        None,
    )
}

fn source_unrelated_case(source: &SourceManifest, case: &SourceCase) -> DriverResult<CaseSpec> {
    let decodes = decode_events(case);
    let prefixes = branch_ranges(source, decodes[0], &[1, 2, 3, 4], "prefix")?;
    let suffixes = branch_ranges(source, decodes[1], &[1, 2, 3, 4], "suffix")?;
    build_source_case(
        source,
        case,
        "unrelated",
        &[1, 2, 3, 4],
        prefixes,
        suffixes,
        None,
    )
}

fn source_nested_case(source: &SourceManifest, case: &SourceCase) -> DriverResult<CaseSpec> {
    let decodes = decode_events(case);
    let prefix = stream_range(source, stream(decodes[0], 0)?, "prefix")?;
    let common_tail = stream_range(source, stream(decodes[1], 1)?, "common tail")?;
    let suffixes = branch_ranges(source, decodes[2], &[2, 3, 4, 5], "suffix")?;
    let partial_prefix_tokens = copy_cut(case)?;
    build_source_case(
        source,
        case,
        "fork_of_fork",
        &[2, 3, 4, 5],
        vec![prefix; 4],
        suffixes,
        Some((common_tail, partial_prefix_tokens)),
    )
}

fn source_membership_case(source: &SourceManifest, case: &SourceCase) -> DriverResult<CaseSpec> {
    let decodes = decode_events(case);
    let prefix = stream_range(source, stream(decodes[0], 0)?, "prefix")?;
    let suffixes = branch_ranges(source, decodes[1], &[0, 1, 2, 3], "suffix")?;
    let mut decode_count = 0;
    let membership_events = case
        .operations
        .iter()
        .filter(|event| is_membership_event(event, &mut decode_count));
    let (
        teacher_offsets,
        active_sequences,
        removed_sequences,
        capture_steps,
        step_events,
        step_input_positions,
    ) = membership_schedule(source, membership_events)?;
    let decode_steps = teacher_offsets.first().map(Vec::len).unwrap_or(0);
    let case_spec = CaseSpec {
        source_case: case.name.clone(),
        name: case.name.clone(),
        topology: "shared".to_owned(),
        branch_sequences: vec![0, 1, 2, 3],
        retained_sequences: Vec::new(),
        prefix_tokens: vec![prefix.1; 4],
        prefix_offsets: vec![prefix.0; 4],
        tail_offsets: suffixes.iter().map(|(offset, _)| *offset).collect(),
        tail_tokens: suffixes.iter().map(|(_, count)| *count).collect(),
        common_tail_offset: None,
        common_tail_tokens: None,
        partial_prefix_tokens: None,
        teacher_offsets,
        decode_steps,
        capture_steps,
        setup_events: setup_event_specs(&decodes[..2])?,
        step_events,
        step_input_positions,
        active_sequences: Some(active_sequences),
        removed_sequences: Some(removed_sequences),
    };
    Ok(case_spec)
}

fn is_membership_event(event: &SourceEvent, decode_count: &mut usize) -> bool {
    match event {
        SourceEvent::Decode { .. } => {
            *decode_count += 1;
            *decode_count >= 3
        }
        SourceEvent::Remove { .. } => *decode_count >= 2,
        _ => false,
    }
}

fn membership_schedule<'a, I>(
    source: &SourceManifest,
    events: I,
) -> DriverResult<MembershipSchedule>
where
    I: Iterator<Item = &'a SourceEvent>,
{
    let mut teachers = vec![Vec::new(); 4];
    let mut active = Vec::new();
    let mut removed = Vec::new();
    let mut captures = Vec::new();
    let mut step_events = Vec::new();
    let mut step_input_positions = Vec::new();
    let mut step = 0;
    let mut pending_removals = Vec::new();
    for event in events {
        if let SourceEvent::Remove { sequence, .. } = event {
            pending_removals.push(*sequence);
            continue;
        }
        step = append_membership_decode(
            source,
            event,
            &mut teachers,
            &mut active,
            &mut removed,
            &mut captures,
            &mut step_events,
            &mut step_input_positions,
            &mut pending_removals,
            step,
        )?;
    }
    if !pending_removals.is_empty() {
        return Err("membership removal has no following decode".into());
    }
    Ok((
        teachers,
        active,
        removed,
        captures,
        step_events,
        step_input_positions,
    ))
}

#[allow(clippy::too_many_arguments)]
fn append_membership_decode(
    source: &SourceManifest,
    event: &SourceEvent,
    teachers: &mut [Vec<usize>],
    active: &mut Vec<Vec<usize>>,
    removed: &mut Vec<Vec<usize>>,
    captures: &mut Vec<usize>,
    step_events: &mut Vec<String>,
    step_input_positions: &mut Vec<Vec<usize>>,
    pending_removals: &mut Vec<usize>,
    step: usize,
) -> DriverResult<usize> {
    let streams = decode_streams(event)?;
    let rows = streams.iter().map(|stream| stream.seq).collect::<Vec<_>>();
    let count = event_token_count(source, streams)?;
    append_membership_rows(
        event,
        streams,
        count,
        &rows,
        active,
        removed,
        step_events,
        step_input_positions,
        pending_removals,
    )?;
    append_membership_teachers(source, streams, count, teachers)?;
    let next_step = step
        .checked_add(count)
        .ok_or("membership step count overflow")?;
    captures.push(next_step.saturating_sub(1));
    pending_removals.clear();
    Ok(next_step)
}

#[allow(clippy::too_many_arguments)]
fn append_membership_rows(
    event: &SourceEvent,
    streams: &[SourceStream],
    count: usize,
    rows: &[usize],
    active: &mut Vec<Vec<usize>>,
    removed: &mut Vec<Vec<usize>>,
    step_events: &mut Vec<String>,
    step_input_positions: &mut Vec<Vec<usize>>,
    pending_removals: &[usize],
) -> DriverResult<()> {
    for index in 0..count {
        active.push(rows.to_vec());
        step_events.push(source_event_name(event).to_owned());
        step_input_positions.push(membership_positions(streams, index)?);
        removed.push(if index == 0 {
            pending_removals.to_vec()
        } else {
            Vec::new()
        });
    }
    Ok(())
}

fn membership_positions(streams: &[SourceStream], index: usize) -> DriverResult<Vec<usize>> {
    streams
        .iter()
        .map(|stream| {
            stream
                .position
                .checked_add(index)
                .ok_or_else(|| "membership position overflow".to_owned().into())
        })
        .collect()
}

fn append_membership_teachers(
    source: &SourceManifest,
    streams: &[SourceStream],
    count: usize,
    teachers: &mut [Vec<usize>],
) -> DriverResult<()> {
    for index in 0..count {
        for (branch, teacher) in teachers.iter_mut().enumerate() {
            let offset = streams
                .iter()
                .find(|stream| stream.seq == branch)
                .map(|stream| token_offset(source, stream, index))
                .transpose()?
                .unwrap_or(0);
            teacher.push(offset);
        }
    }
    Ok(())
}

fn decode_streams(event: &SourceEvent) -> DriverResult<&[SourceStream]> {
    match event {
        SourceEvent::Decode { streams, .. } => Ok(streams),
        _ => Err("membership event is not a decode".into()),
    }
}

fn source_event_name(event: &SourceEvent) -> &str {
    match event {
        SourceEvent::Decode { name, .. }
        | SourceEvent::Copy { name, .. }
        | SourceEvent::Remove { name, .. }
        | SourceEvent::DecodeStepwise { name, .. } => name,
    }
}

fn setup_event_specs(events: &[&SourceEvent]) -> DriverResult<Vec<SetupEventSpec>> {
    events
        .iter()
        .filter_map(|event| match event {
            SourceEvent::Decode { streams, .. } => Some((event, streams)),
            _ => None,
        })
        .map(|(event, streams)| {
            if streams
                .iter()
                .any(|stream| !is_last_output(stream.outputs.as_ref()))
            {
                return Err(format!(
                    "setup event {} requests an unsupported output",
                    source_event_name(event)
                )
                .into());
            }
            Ok(SetupEventSpec {
                name: source_event_name(event).to_owned(),
                sequences: streams.iter().map(|stream| stream.seq).collect(),
                positions: streams.iter().map(|stream| stream.position).collect(),
            })
        })
        .collect()
}

fn is_last_output(outputs: Option<&serde_json::Value>) -> bool {
    matches!(
        outputs,
        Some(serde_json::Value::String(value)) if value == "last"
    )
}

fn setup_events_for(case: &SourceCase) -> DriverResult<Vec<SetupEventSpec>> {
    let decodes = decode_events(case);
    let count = match case.class.as_str() {
        "nested_shared_prefix" => 3,
        "cold_full_history" => 1,
        "batch_membership_change"
        | "shared_prefix"
        | "short_prefix_control"
        | "private_tail_growth"
        | "owned_equal_prefix"
        | "unrelated_control" => 2,
        _ => return Err(format!("unsupported source case class {}", case.class).into()),
    };
    if decodes.len() < count {
        return Err(format!("source case {} lacks setup events", case.name).into());
    }
    setup_event_specs(&decodes[..count])
}

fn event_token_count(source: &SourceManifest, streams: &[SourceStream]) -> DriverResult<usize> {
    let first = streams.first().ok_or("membership event has no streams")?;
    let count = source
        .slices
        .get(
            first
                .tokens
                .first()
                .ok_or("membership stream has no tokens")?,
        )
        .ok_or("membership stream names an unknown slice")?
        .count;
    if streams.iter().any(|stream| {
        stream.tokens.len() != 1
            || source
                .slices
                .get(&stream.tokens[0])
                .is_none_or(|slice| slice.count != count)
    }) {
        return Err("membership events use unequal token slices".into());
    }
    Ok(count)
}

fn token_offset(
    source: &SourceManifest,
    stream: &SourceStream,
    index: usize,
) -> DriverResult<usize> {
    let name = stream
        .tokens
        .first()
        .ok_or("membership stream has no tokens")?;
    let slice = source
        .slices
        .get(name)
        .ok_or_else(|| format!("unknown membership slice {name}"))?;
    Ok(slice.offset + index)
}

fn build_source_case(
    source: &SourceManifest,
    case: &SourceCase,
    topology: &str,
    branch_sequences: &[usize],
    prefixes: Vec<(usize, usize)>,
    suffixes: Vec<(usize, usize)>,
    nested: Option<((usize, usize), usize)>,
) -> DriverResult<CaseSpec> {
    let stepwise = stepwise_event(case)?;
    let (teacher_offsets, step_input_positions, decode_steps) =
        source_teacher_schedule(source, stepwise, branch_sequences, case)?;
    let (common_tail_offset, common_tail_tokens, partial_prefix_tokens) = nested_fields(nested);
    let setup_events = setup_events_for(case)?;
    let step_events = vec![source_event_name(stepwise).to_owned(); decode_steps];
    Ok(CaseSpec {
        source_case: case.name.clone(),
        name: case.name.clone(),
        topology: topology.to_owned(),
        branch_sequences: branch_sequences.to_vec(),
        retained_sequences: retained_sequences(topology, branch_sequences),
        prefix_tokens: prefixes.iter().map(|(_, count)| *count).collect(),
        prefix_offsets: prefixes.iter().map(|(offset, _)| *offset).collect(),
        tail_offsets: suffixes.iter().map(|(offset, _)| *offset).collect(),
        tail_tokens: suffixes.iter().map(|(_, count)| *count).collect(),
        common_tail_offset,
        common_tail_tokens,
        partial_prefix_tokens,
        teacher_offsets,
        decode_steps,
        capture_steps: stepwise_capture_steps(stepwise),
        setup_events,
        step_events,
        step_input_positions,
        active_sequences: None,
        removed_sequences: None,
    })
}

fn retained_sequences(topology: &str, branch_sequences: &[usize]) -> Vec<usize> {
    match topology {
        "shared" if !branch_sequences.contains(&0) => vec![0],
        "fork_of_fork" => vec![0, 1],
        _ => Vec::new(),
    }
}

fn source_teacher_schedule(
    source: &SourceManifest,
    stepwise: &SourceEvent,
    branch_sequences: &[usize],
    case: &SourceCase,
) -> DriverResult<SourceStepSchedule> {
    let teacher_offsets = teacher_offsets(source, stepwise, branch_sequences)?;
    let decode_steps = teacher_offsets.first().map(Vec::len).unwrap_or(0);
    if teacher_offsets
        .iter()
        .any(|offsets| offsets.len() != decode_steps)
    {
        return Err(format!("source case {} has unequal teacher lengths", case.name).into());
    }
    let step_input_positions =
        source_step_input_positions(stepwise, branch_sequences, decode_steps)?;
    Ok((teacher_offsets, step_input_positions, decode_steps))
}

fn source_step_input_positions(
    event: &SourceEvent,
    branch_sequences: &[usize],
    decode_steps: usize,
) -> DriverResult<Vec<Vec<usize>>> {
    let SourceEvent::DecodeStepwise { streams, .. } = event else {
        return Err("source step positions require a stepwise event".into());
    };
    (0..decode_steps)
        .map(|step| {
            branch_sequences
                .iter()
                .map(|sequence| {
                    streams
                        .iter()
                        .find(|stream| stream.seq == *sequence)
                        .ok_or_else(|| format!("stepwise decode lacks sequence {sequence}"))?
                        .position
                        .checked_add(step)
                        .ok_or_else(|| "source step position overflow".into())
                })
                .collect()
        })
        .collect()
}

fn nested_fields(
    nested: Option<((usize, usize), usize)>,
) -> (Option<usize>, Option<usize>, Option<usize>) {
    match nested {
        Some(((offset, count), prefix)) => (Some(offset), Some(count), Some(prefix)),
        None => (None, None, None),
    }
}

fn decode_events(case: &SourceCase) -> Vec<&SourceEvent> {
    case.operations
        .iter()
        .filter(|event| matches!(event, SourceEvent::Decode { .. }))
        .collect()
}

fn stream(event: &SourceEvent, sequence: usize) -> DriverResult<&SourceStream> {
    let SourceEvent::Decode { streams, .. } = event else {
        return Err("source event is not a decode".into());
    };
    streams
        .iter()
        .find(|stream| stream.seq == sequence)
        .ok_or_else(|| format!("source decode lacks sequence {sequence}").into())
}

fn branch_ranges(
    source: &SourceManifest,
    event: &SourceEvent,
    sequences: &[usize],
    field: &str,
) -> DriverResult<Vec<(usize, usize)>> {
    sequences
        .iter()
        .map(|sequence| stream_range(source, stream(event, *sequence)?, field))
        .collect()
}

fn stream_range(
    source: &SourceManifest,
    stream: &SourceStream,
    field: &str,
) -> DriverResult<(usize, usize)> {
    if stream.tokens.len() != 1 {
        return Err(format!("{field} stream has multiple token slices").into());
    }
    stream_range_first(source, stream, field)
}

fn stream_range_first(
    source: &SourceManifest,
    stream: &SourceStream,
    _field: &str,
) -> DriverResult<(usize, usize)> {
    let name = stream.tokens.first().ok_or("source stream has no tokens")?;
    let slice = source
        .slices
        .get(name)
        .ok_or_else(|| format!("unknown source slice {name}"))?;
    Ok((slice.offset, slice.count))
}

fn stream_range_after_prefix(
    source: &SourceManifest,
    stream: &SourceStream,
    field: &str,
) -> DriverResult<(usize, usize)> {
    if stream.tokens.len() != 2 {
        return Err(format!("{field} stream has no separate suffix slice").into());
    }
    let name = stream.tokens.get(1).ok_or("source suffix is missing")?;
    let slice = source
        .slices
        .get(name)
        .ok_or_else(|| format!("unknown source slice {name}"))?;
    Ok((slice.offset, slice.count))
}

fn stepwise_event(case: &SourceCase) -> DriverResult<&SourceEvent> {
    case.operations
        .iter()
        .find(|event| matches!(event, SourceEvent::DecodeStepwise { .. }))
        .ok_or_else(|| format!("source case {} lacks stepwise decode", case.name).into())
}

fn stepwise_capture_steps(event: &SourceEvent) -> Vec<usize> {
    match event {
        SourceEvent::DecodeStepwise { capture_steps, .. } => capture_steps.clone(),
        _ => Vec::new(),
    }
}

fn teacher_offsets(
    source: &SourceManifest,
    event: &SourceEvent,
    sequences: &[usize],
) -> DriverResult<Vec<Vec<usize>>> {
    let SourceEvent::DecodeStepwise { streams, .. } = event else {
        return Err("source event is not stepwise".into());
    };
    sequences
        .iter()
        .map(|sequence| {
            let stream = streams
                .iter()
                .find(|stream| stream.seq == *sequence)
                .ok_or_else(|| format!("stepwise decode lacks sequence {sequence}"))?;
            let mut offsets = Vec::new();
            for name in &stream.tokens {
                let slice = source
                    .slices
                    .get(name)
                    .ok_or_else(|| format!("unknown source slice {name}"))?;
                offsets.extend((0..slice.count).map(|index| slice.offset + index));
            }
            Ok(offsets)
        })
        .collect()
}

fn copy_cut(case: &SourceCase) -> DriverResult<usize> {
    case.operations
        .iter()
        .find_map(|event| match event {
            SourceEvent::Copy { p1, .. } if *p1 >= 0 => usize::try_from(*p1).ok(),
            _ => None,
        })
        .ok_or_else(|| format!("source case {} lacks a finite copy cut", case.name).into())
}

fn validate_cases(cases: &[CaseSpec]) -> DriverResult<()> {
    for case in cases {
        validate_case(case)?;
    }
    Ok(())
}

fn validate_case(case: &CaseSpec) -> DriverResult<()> {
    validate_case_name(&case.name)?;
    let branches = case.prefix_offsets.len();
    validate_dimensions(case, branches)?;
    validate_branch_sequences(case, branches)?;
    validate_teacher_schedule(case)?;
    validate_capture_steps(case)?;
    validate_active_sequences(case, branches)?;
    validate_removed_sequences(case, branches)?;
    validate_output_bindings(case)?;
    validate_topology(case)
}

fn validate_dimensions(case: &CaseSpec, branches: usize) -> DriverResult<()> {
    let valid = branches != 0
        && case.prefix_tokens.len() == branches
        && case.tail_offsets.len() == branches
        && case.tail_tokens.len() == branches
        && case.teacher_offsets.len() == branches
        && !case.source_case.is_empty()
        && case.decode_steps != 0;
    if valid {
        Ok(())
    } else {
        Err(format!("invalid branch dimensions for {}", case.name).into())
    }
}

fn validate_branch_sequences(case: &CaseSpec, branches: usize) -> DriverResult<()> {
    if case.branch_sequences.is_empty() {
        return validate_empty_branch_sequences(case);
    }
    validate_active_branch_sequences(case, branches)?;
    validate_retained_branch_sequences(case)?;
    Ok(())
}

fn validate_empty_branch_sequences(case: &CaseSpec) -> DriverResult<()> {
    if case.retained_sequences.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "retained source sequences lack active rows for {}",
            case.name
        )
        .into())
    }
}

fn validate_active_branch_sequences(case: &CaseSpec, branches: usize) -> DriverResult<()> {
    let valid = case.branch_sequences.len() == branches
        && case
            .branch_sequences
            .windows(2)
            .all(|pair| pair[0] < pair[1])
        && case.branch_sequences.iter().all(|sequence| *sequence < 8);
    if valid {
        Ok(())
    } else {
        Err(format!("invalid source branch sequence mapping for {}", case.name).into())
    }
}

fn validate_retained_branch_sequences(case: &CaseSpec) -> DriverResult<()> {
    let valid = case
        .retained_sequences
        .iter()
        .all(|sequence| *sequence < 8 && !case.branch_sequences.contains(sequence));
    if valid {
        Ok(())
    } else {
        Err(format!("invalid retained source sequence mapping for {}", case.name).into())
    }
}

fn validate_output_bindings(case: &CaseSpec) -> DriverResult<()> {
    validate_setup_bindings(case)?;
    validate_step_bindings(case)
}

fn validate_setup_bindings(case: &CaseSpec) -> DriverResult<()> {
    if case.setup_events.iter().any(|event| {
        event.name.is_empty()
            || event.sequences.is_empty()
            || event.positions.len() != event.sequences.len()
            || event.sequences.iter().any(|sequence| *sequence >= 8)
    }) {
        return Err(format!("invalid setup output binding for {}", case.name).into());
    }
    Ok(())
}

fn validate_step_bindings(case: &CaseSpec) -> DriverResult<()> {
    if !case.step_events.is_empty() && case.step_events.len() != case.decode_steps {
        return Err(format!("invalid step output binding for {}", case.name).into());
    }
    if !case.step_input_positions.is_empty() {
        let valid = case.step_input_positions.len() == case.decode_steps
            && case
                .step_input_positions
                .iter()
                .enumerate()
                .all(|(step, positions)| {
                    let rows = case
                        .active_sequences
                        .as_ref()
                        .and_then(|active| active.get(step))
                        .map_or(case.prefix_offsets.len(), Vec::len);
                    positions.len() == rows
                });
        if !valid {
            return Err(format!("invalid step position binding for {}", case.name).into());
        }
    }
    Ok(())
}

fn validate_capture_steps(case: &CaseSpec) -> DriverResult<()> {
    let valid = !case.capture_steps.is_empty()
        && case
            .capture_steps
            .windows(2)
            .all(|steps| steps[0] < steps[1])
        && case
            .capture_steps
            .iter()
            .all(|step| *step < case.decode_steps);
    if valid {
        Ok(())
    } else {
        Err(format!("invalid capture steps for {}", case.name).into())
    }
}

fn validate_active_sequences(case: &CaseSpec, branches: usize) -> DriverResult<()> {
    let Some(active) = &case.active_sequences else {
        return Ok(());
    };
    let valid = active.len() == case.decode_steps
        && active.iter().all(|rows| {
            !rows.is_empty()
                && rows.windows(2).all(|pair| pair[0] < pair[1])
                && rows.iter().all(|row| *row < branches)
        });
    if valid {
        Ok(())
    } else {
        Err(format!("invalid active row schedule for {}", case.name).into())
    }
}

fn validate_removed_sequences(case: &CaseSpec, branches: usize) -> DriverResult<()> {
    let Some(removed) = &case.removed_sequences else {
        return Ok(());
    };
    if removed.len() != case.decode_steps {
        return Err(format!("invalid removal schedule for {}", case.name).into());
    }
    let mut retired = vec![false; branches];
    for (step, rows) in removed.iter().enumerate() {
        for row in rows {
            if *row >= branches || retired[*row] {
                return Err(format!("invalid removal row {} at step {step}", case.name).into());
            }
            retired[*row] = true;
        }
        if let Some(active) = &case.active_sequences {
            if active[step].iter().any(|row| retired[*row]) {
                return Err(format!("removed row remains active in {}", case.name).into());
            }
        }
    }
    Ok(())
}

fn validate_teacher_schedule(case: &CaseSpec) -> DriverResult<()> {
    if case
        .teacher_offsets
        .iter()
        .all(|offsets| offsets.len() == case.decode_steps)
    {
        Ok(())
    } else {
        Err(format!("invalid teacher schedule for {}", case.name).into())
    }
}

fn validate_topology(case: &CaseSpec) -> DriverResult<()> {
    let accepted = matches!(
        case.topology.as_str(),
        "shared" | "owned_equal" | "cold" | "fork_of_fork" | "tail_growth" | "unrelated"
    );
    if !accepted {
        return Err(format!("invalid topology for {}", case.name).into());
    }
    if case.topology == "fork_of_fork"
        && (case.common_tail_offset.is_none()
            || case.common_tail_tokens.is_none()
            || case.partial_prefix_tokens.is_none())
    {
        return Err(format!("fork_of_fork case {} lacks its partial copy", case.name).into());
    }
    Ok(())
}

fn read_tokens(path: &Path) -> DriverResult<Vec<u32>> {
    let bytes = fs::read(path)?;
    let mut chunks = bytes.chunks_exact(4);
    let tokens = chunks
        .by_ref()
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect::<Vec<_>>();
    if !chunks.remainder().is_empty() || tokens.is_empty() {
        return Err(format!(
            "token fixture {} is not a nonempty u32 stream",
            path.display()
        )
        .into());
    }
    Ok(tokens)
}

fn checked_slice(tokens: &[u32], offset: usize, length: usize) -> DriverResult<&[u32]> {
    let end = offset
        .checked_add(length)
        .ok_or("token fixture range overflow")?;
    tokens
        .get(offset..end)
        .ok_or_else(|| format!("token fixture range {offset}..{end} is outside the fixture").into())
}

fn prompt_tokens(case: &CaseSpec, branch: usize, fixture: &[u32]) -> DriverResult<Vec<u32>> {
    let mut prompt = checked_slice(
        fixture,
        case.prefix_offsets[branch],
        case.prefix_tokens[branch],
    )?
    .to_vec();
    prompt.extend_from_slice(checked_slice(
        fixture,
        case.tail_offsets[branch],
        case.tail_tokens[branch],
    )?);
    Ok(prompt)
}

fn generation_options() -> GenerateOptions {
    let mut options = GenerateOptions::greedy(1);
    options.decode_execution = DecodeExecution::Eager;
    options
}

fn prefill(
    runtime: &mut DriverRuntime,
    session: &mut GenerationSession<DriverBackend>,
    prompt: &[u32],
    options: &GenerateOptions,
) -> DriverResult<()> {
    let mut pending = runtime.begin_prefill(session, prompt, options.clone())?;
    loop {
        let budget = pending.minimum_budget();
        match runtime.advance_prefill(pending, budget, || false)? {
            PrefillProgress::Pending(next) => pending = next,
            PrefillProgress::Ready(ready) => {
                runtime.finish_prefill(ready, session)?;
                return Ok(());
            }
            PrefillProgress::Cancelled(_) => return Err("prefill was cancelled".into()),
        }
    }
}

fn make_branch(
    runtime: &mut DriverRuntime,
    prompt: &[u32],
    options: &GenerateOptions,
) -> DriverResult<BranchState> {
    let mut session = GenerationSession::new();
    prefill(runtime, &mut session, prompt, options)?;
    validate_session_history(&session, prompt, "prefill")?;
    Ok(BranchState {
        session,
        transcript: prompt.to_vec(),
    })
}

fn make_branches(
    runtime: &mut DriverRuntime,
    case: &CaseSpec,
    fixture: &[u32],
    options: &GenerateOptions,
    trace: &mut OutputTrace,
    sidecar: &mut Option<&mut Sidecar>,
) -> DriverResult<Vec<BranchState>> {
    if case.topology == "cold" {
        return make_cold_case(runtime, case, fixture, options, trace, sidecar);
    }
    if is_owned_topology(case) {
        return make_owned_case(runtime, case, fixture, options, trace, sidecar);
    }
    if case.topology == "fork_of_fork" {
        return make_nested_branches(runtime, case, fixture, options, trace, sidecar);
    }
    make_shared_case(runtime, case, fixture, options, trace, sidecar)
}

fn is_owned_topology(case: &CaseSpec) -> bool {
    matches!(case.topology.as_str(), "owned_equal" | "unrelated")
}

fn make_owned_case(
    runtime: &mut DriverRuntime,
    case: &CaseSpec,
    fixture: &[u32],
    options: &GenerateOptions,
    trace: &mut OutputTrace,
    sidecar: &mut Option<&mut Sidecar>,
) -> DriverResult<Vec<BranchState>> {
    let prefixes = (0..case.prefix_offsets.len())
        .map(|branch| {
            checked_slice(
                fixture,
                case.prefix_offsets[branch],
                case.prefix_tokens[branch],
            )
            .map(|tokens| tokens.to_vec())
        })
        .collect::<DriverResult<Vec<_>>>()?;
    let mut branches = make_owned_branches(runtime, &prefixes, options)?;
    record_setup_event_if_present(runtime, &branches, case, 0, trace, sidecar)?;
    let tails = tail_vectors(case, fixture)?;
    append_tokens_batched(runtime, &mut branches, &tails, options, trace)?;
    record_setup_event_if_present(runtime, &branches, case, 1, trace, sidecar)?;
    Ok(branches)
}

fn make_cold_case(
    runtime: &mut DriverRuntime,
    case: &CaseSpec,
    fixture: &[u32],
    options: &GenerateOptions,
    trace: &mut OutputTrace,
    sidecar: &mut Option<&mut Sidecar>,
) -> DriverResult<Vec<BranchState>> {
    let prompts = (0..case.prefix_offsets.len())
        .map(|branch| prompt_tokens(case, branch, fixture))
        .collect::<DriverResult<Vec<_>>>()?;
    let branches = make_owned_branches(runtime, &prompts, options)?;
    record_setup_event_if_present(runtime, &branches, case, 0, trace, sidecar)?;
    Ok(branches)
}

fn record_setup_event_if_present(
    runtime: &mut DriverRuntime,
    branches: &[BranchState],
    case: &CaseSpec,
    event_index: usize,
    trace: &mut OutputTrace,
    sidecar: &mut Option<&mut Sidecar>,
) -> DriverResult<()> {
    if case.setup_events.get(event_index).is_some() {
        record_setup_outputs(runtime, branches, case, event_index, trace, sidecar)?;
    }
    Ok(())
}

fn make_shared_case(
    runtime: &mut DriverRuntime,
    case: &CaseSpec,
    fixture: &[u32],
    options: &GenerateOptions,
    trace: &mut OutputTrace,
    sidecar: &mut Option<&mut Sidecar>,
) -> DriverResult<Vec<BranchState>> {
    let mut root = GenerationSession::new();
    let prefix = checked_slice(fixture, case.prefix_offsets[0], case.prefix_tokens[0])?;
    prefill(runtime, &mut root, prefix, options)?;
    validate_session_history(&root, prefix, "shared prefix prefill")?;
    record_shared_prefix_output(runtime, case, &root, trace, sidecar)?;
    let mut branches = make_shared_branches(runtime, case, root, prefix)?;
    let tails = tail_vectors(case, fixture)?;
    append_tokens_batched(runtime, &mut branches, &tails, options, trace)?;
    record_setup_event_if_present(runtime, &branches, case, 1, trace, sidecar)?;
    Ok(branches)
}

fn record_shared_prefix_output(
    runtime: &mut DriverRuntime,
    case: &CaseSpec,
    root: &GenerationSession<DriverBackend>,
    trace: &mut OutputTrace,
    sidecar: &mut Option<&mut Sidecar>,
) -> DriverResult<()> {
    let Some(event) = case.setup_events.first() else {
        return Ok(());
    };
    let sequence = *event
        .sequences
        .first()
        .ok_or("shared setup event has no source sequence")?;
    let token_index = setup_token_index(case, 0, sequence, 0)?;
    record_session_output(
        runtime,
        root,
        sequence,
        &event.name,
        token_index,
        Some(expected_setup_input_position(
            case,
            0,
            sequence,
            token_index,
        )?),
        trace,
        sidecar.as_deref_mut(),
    )
}

fn make_shared_branches(
    runtime: &mut DriverRuntime,
    case: &CaseSpec,
    root: GenerationSession<DriverBackend>,
    prefix: &[u32],
) -> DriverResult<Vec<BranchState>> {
    let branch_count = case.prefix_offsets.len();
    let root_active = case
        .branch_sequences
        .first()
        .is_some_and(|sequence| *sequence == 0);
    if root_active {
        return make_active_root_branches(runtime, root, prefix, branch_count);
    }
    let mut branches = fork_branches(runtime, &root, branch_count)?;
    branches.push(BranchState {
        session: root,
        transcript: prefix.to_vec(),
    });
    Ok(branches)
}

fn make_active_root_branches(
    runtime: &mut DriverRuntime,
    root: GenerationSession<DriverBackend>,
    prefix: &[u32],
    branch_count: usize,
) -> DriverResult<Vec<BranchState>> {
    let mut children = fork_branches(runtime, &root, branch_count.saturating_sub(1))?;
    let root_branch = BranchState {
        session: root,
        transcript: prefix.to_vec(),
    };
    let mut branches = Vec::with_capacity(branch_count);
    branches.push(root_branch);
    branches.append(&mut children);
    Ok(branches)
}

fn make_owned_branches(
    runtime: &mut DriverRuntime,
    prompts: &[Vec<u32>],
    options: &GenerateOptions,
) -> DriverResult<Vec<BranchState>> {
    prompts
        .iter()
        .map(|prompt| make_branch(runtime, prompt, options))
        .collect()
}

fn fork_branches(
    runtime: &mut DriverRuntime,
    source: &GenerationSession<DriverBackend>,
    count: usize,
) -> DriverResult<Vec<BranchState>> {
    let mut branches = Vec::with_capacity(count);
    for _ in 0..count {
        let session = runtime.fork_session(source)?;
        branches.push(BranchState {
            transcript: source.evaluated_tokens().to_vec(),
            session,
        });
    }
    validate_branch_histories(&branches)?;
    Ok(branches)
}

fn make_nested_branches(
    runtime: &mut DriverRuntime,
    case: &CaseSpec,
    fixture: &[u32],
    options: &GenerateOptions,
    trace: &mut OutputTrace,
    sidecar: &mut Option<&mut Sidecar>,
) -> DriverResult<Vec<BranchState>> {
    let parent_prompt = checked_slice(fixture, case.prefix_offsets[0], case.prefix_tokens[0])?;
    let parent = make_nested_parent(runtime, case, parent_prompt, options, trace, sidecar)?;
    let intermediate_prompt = nested_prompt(case, fixture)?;
    let intermediate = make_nested_intermediate(
        runtime,
        case,
        &parent.session,
        &intermediate_prompt,
        options,
        trace,
        sidecar,
    )?;
    let mut branches = make_nested_children(
        runtime,
        case,
        &intermediate.session,
        fixture,
        options,
        trace,
        sidecar,
    )?;
    branches.push(parent);
    branches.push(intermediate);
    validate_branch_histories(&branches)?;
    Ok(branches)
}

fn make_nested_parent(
    runtime: &mut DriverRuntime,
    case: &CaseSpec,
    prompt: &[u32],
    options: &GenerateOptions,
    trace: &mut OutputTrace,
    sidecar: &mut Option<&mut Sidecar>,
) -> DriverResult<BranchState> {
    let parent = make_branch(runtime, prompt, options)?;
    if let Some(event) = case.setup_events.first() {
        let token_index = setup_token_index(case, 0, 0, 0)?;
        record_session_output(
            runtime,
            &parent.session,
            0,
            &event.name,
            token_index,
            Some(expected_setup_input_position(case, 0, 0, token_index)?),
            trace,
            sidecar.as_deref_mut(),
        )?;
    }
    Ok(parent)
}

fn make_nested_intermediate(
    runtime: &mut DriverRuntime,
    case: &CaseSpec,
    parent: &GenerationSession<DriverBackend>,
    prompt: &[u32],
    options: &GenerateOptions,
    trace: &mut OutputTrace,
    sidecar: &mut Option<&mut Sidecar>,
) -> DriverResult<BranchState> {
    let partial_prefix = case.partial_prefix_tokens.ok_or("missing partial prefix")?;
    let mut session = runtime.fork_session_prefix(parent, partial_prefix)?;
    prefill(runtime, &mut session, prompt, options)?;
    validate_session_history(&session, prompt, "nested fork")?;
    let token_index = setup_token_index(case, 1, 1, 0)?;
    if case.setup_events.get(1).is_some() {
        record_session_output(
            runtime,
            &session,
            1,
            &case.setup_events[1].name,
            token_index,
            Some(expected_setup_input_position(case, 1, 1, token_index)?),
            trace,
            sidecar.as_deref_mut(),
        )?;
    }
    Ok(BranchState {
        session,
        transcript: prompt.to_vec(),
    })
}

fn make_nested_children(
    runtime: &mut DriverRuntime,
    case: &CaseSpec,
    intermediate: &GenerationSession<DriverBackend>,
    fixture: &[u32],
    options: &GenerateOptions,
    trace: &mut OutputTrace,
    sidecar: &mut Option<&mut Sidecar>,
) -> DriverResult<Vec<BranchState>> {
    let mut branches = fork_branches(runtime, intermediate, case.prefix_offsets.len())?;
    let tails = tail_vectors(case, fixture)?;
    append_tokens_batched(runtime, &mut branches, &tails, options, trace)?;
    record_setup_event_if_present(runtime, &branches, case, 2, trace, sidecar)?;
    Ok(branches)
}

fn nested_prompt(case: &CaseSpec, fixture: &[u32]) -> DriverResult<Vec<u32>> {
    let prefix_tokens = case.partial_prefix_tokens.ok_or("missing partial prefix")?;
    let common_tail_offset = case.common_tail_offset.ok_or("missing common tail")?;
    let common_tail_tokens = case
        .common_tail_tokens
        .ok_or("missing common tail length")?;
    let prefix = checked_slice(fixture, case.prefix_offsets[0], prefix_tokens)?;
    let mut prompt = prefix.to_vec();
    prompt.extend_from_slice(checked_slice(
        fixture,
        common_tail_offset,
        common_tail_tokens,
    )?);
    Ok(prompt)
}

fn tail_vectors(case: &CaseSpec, fixture: &[u32]) -> DriverResult<Vec<Vec<u32>>> {
    (0..case.prefix_offsets.len())
        .map(|branch| {
            checked_slice(fixture, case.tail_offsets[branch], case.tail_tokens[branch])
                .map(|tokens| tokens.to_vec())
        })
        .collect()
}

fn validate_session_history(
    session: &GenerationSession<DriverBackend>,
    expected: &[u32],
    label: &str,
) -> DriverResult<()> {
    if session.evaluated_tokens() == expected {
        Ok(())
    } else {
        Err(format!("{label} history differs from the declared token sequence").into())
    }
}

fn validate_branch_histories(branches: &[BranchState]) -> DriverResult<()> {
    for branch in branches {
        validate_session_history(&branch.session, &branch.transcript, "branch")?;
    }
    Ok(())
}

fn branch_index_for_sequence(case: &CaseSpec, sequence: usize) -> DriverResult<usize> {
    if let Some(index) = case
        .branch_sequences
        .iter()
        .position(|candidate| *candidate == sequence)
    {
        return Ok(index);
    }
    case.retained_sequences
        .iter()
        .position(|candidate| *candidate == sequence)
        .map(|index| case.branch_sequences.len() + index)
        .ok_or_else(|| format!("source sequence {sequence} is absent from {}", case.name).into())
}

fn setup_token_index(
    case: &CaseSpec,
    event_index: usize,
    sequence: usize,
    branch: usize,
) -> DriverResult<usize> {
    let count = match (case.topology.as_str(), event_index) {
        ("fork_of_fork", 1) => nested_setup_count(case)?,
        ("cold", 0) => cold_setup_count(case, branch)?,
        (_, 0) => prefix_setup_count(case, sequence, branch)?,
        _ => tail_setup_count(case, branch)?,
    };
    count.checked_sub(1).ok_or_else(|| {
        format!("setup event {event_index} has no tokens for sequence {sequence}").into()
    })
}

fn nested_setup_count(case: &CaseSpec) -> DriverResult<usize> {
    case.common_tail_tokens
        .ok_or_else(|| "missing nested tail length".into())
}

fn cold_setup_count(case: &CaseSpec, branch: usize) -> DriverResult<usize> {
    let prefix = case
        .prefix_tokens
        .get(branch)
        .copied()
        .ok_or("cold prefix length is missing")?;
    prefix
        .checked_add(
            case.tail_tokens
                .get(branch)
                .copied()
                .ok_or("cold tail length is missing")?,
        )
        .ok_or_else(|| "cold prompt length overflow".into())
}

fn prefix_setup_count(case: &CaseSpec, sequence: usize, branch: usize) -> DriverResult<usize> {
    let index = if case.topology == "shared" && sequence == 0 {
        0
    } else {
        branch
    };
    case.prefix_tokens
        .get(index)
        .copied()
        .ok_or_else(|| "setup prefix length is missing".into())
}

fn tail_setup_count(case: &CaseSpec, branch: usize) -> DriverResult<usize> {
    case.tail_tokens
        .get(branch)
        .copied()
        .ok_or_else(|| "setup tail length is missing".into())
}

fn expected_setup_input_position(
    case: &CaseSpec,
    event_index: usize,
    sequence: usize,
    token_index: usize,
) -> DriverResult<usize> {
    let event = case
        .setup_events
        .get(event_index)
        .ok_or_else(|| format!("setup event {event_index} is missing from {}", case.name))?;
    let row = event
        .sequences
        .iter()
        .position(|candidate| *candidate == sequence)
        .ok_or_else(|| format!("setup sequence {sequence} is missing from {}", case.name))?;
    event
        .positions
        .get(row)
        .copied()
        .ok_or_else(|| format!("setup position {row} is missing from {}", case.name))?
        .checked_add(token_index)
        .ok_or_else(|| "setup input position overflow".into())
}

#[allow(clippy::too_many_arguments)]
fn record_session_output(
    runtime: &mut DriverRuntime,
    session: &GenerationSession<DriverBackend>,
    sequence: usize,
    event: &str,
    token_index: usize,
    expected_input_position: Option<usize>,
    trace: &mut OutputTrace,
    sidecar: Option<&mut Sidecar>,
) -> DriverResult<()> {
    let mut logits = vec![0.0_f32; runtime.vocab_size()];
    let predicted_position = runtime.read_session_logits(session, &mut logits)?;
    let expected_position = session.evaluated_tokens().len();
    validate_logit_row(
        sequence,
        token_index,
        predicted_position,
        expected_position,
        &logits,
    )?;
    validate_source_position(predicted_position, expected_input_position)?;
    let input_position = predicted_position
        .checked_sub(1)
        .ok_or("logit row has no input position")?;
    trace.record(
        sequence,
        event,
        token_index,
        input_position,
        predicted_position,
        logits.len(),
        &logits,
        sidecar,
    )
}

fn record_setup_outputs(
    runtime: &mut DriverRuntime,
    branches: &[BranchState],
    case: &CaseSpec,
    event_index: usize,
    trace: &mut OutputTrace,
    sidecar: &mut Option<&mut Sidecar>,
) -> DriverResult<()> {
    let event = case
        .setup_events
        .get(event_index)
        .ok_or_else(|| format!("setup event {event_index} is missing from {}", case.name))?;
    for &sequence in &event.sequences {
        let branch = branch_index_for_sequence(case, sequence)?;
        let token_index = setup_token_index(case, event_index, sequence, branch)?;
        let expected_input_position =
            expected_setup_input_position(case, event_index, sequence, token_index)?;
        let state = branches
            .get(branch)
            .ok_or_else(|| format!("setup sequence {sequence} has no branch"))?;
        record_session_output(
            runtime,
            &state.session,
            sequence,
            &event.name,
            token_index,
            Some(expected_input_position),
            trace,
            sidecar.as_deref_mut(),
        )?;
    }
    Ok(())
}

fn append_tokens_batched(
    runtime: &mut DriverRuntime,
    branches: &mut [BranchState],
    tokens: &[Vec<u32>],
    options: &GenerateOptions,
    trace: &mut OutputTrace,
) -> DriverResult<()> {
    if tokens.len() > branches.len() {
        return Err("token rows exceed branch rows".into());
    }
    let max_tokens = tokens.iter().map(Vec::len).max().unwrap_or(0);
    for position in 0..max_tokens {
        let mut active = Vec::new();
        for (branch, state) in branches.iter_mut().take(tokens.len()).enumerate() {
            let Some(token) = tokens[branch].get(position).copied() else {
                continue;
            };
            state.transcript.push(token);
            active.push(branch);
        }
        if !active.is_empty() {
            dispatch_batch(
                runtime,
                branches,
                &active,
                options,
                &mut trace.batch_schedule,
            )?;
            runtime.synchronize()?;
            validate_branch_histories(branches)?;
        }
    }
    Ok(())
}

fn append_teacher_tokens(
    branches: &mut [BranchState],
    case: &CaseSpec,
    fixture: &[u32],
    step: usize,
    active: &[usize],
) -> DriverResult<()> {
    for &branch in active {
        let token = *checked_slice(fixture, case.teacher_offsets[branch][step], 1)?
            .first()
            .ok_or("empty teacher token")?;
        branches[branch].transcript.push(token);
    }
    Ok(())
}

fn dispatch_batch(
    runtime: &mut DriverRuntime,
    branches: &mut [BranchState],
    active: &[usize],
    options: &GenerateOptions,
    batch_schedule: &mut Vec<usize>,
) -> DriverResult<Vec<u32>> {
    let width = runtime.backend().max_batch_size().get();
    let mut tokens = Vec::with_capacity(active.len());
    for rows in active.chunks(width) {
        batch_schedule.push(rows.len());
        let mut inputs = branches
            .iter_mut()
            .enumerate()
            .filter_map(|(index, branch)| {
                rows.contains(&index).then_some(BatchSession {
                    session: &mut branch.session,
                    transcript: &branch.transcript,
                    options,
                })
            })
            .collect::<Vec<_>>();
        let generated = runtime.generate_session_batch_token(&mut inputs)?;
        if generated.len() != rows.len() {
            return Err(format!(
                "backend returned {} tokens for {} active rows",
                generated.len(),
                rows.len()
            )
            .into());
        }
        tokens.extend(generated.into_iter().map(|token| token.id));
    }
    Ok(tokens)
}

fn stats_delta(before: BatchStats, after: BatchStats) -> StatsRecord {
    StatsRecord {
        dispatches: after.dispatches.saturating_sub(before.dispatches),
        rows: after.rows.saturating_sub(before.rows),
        groups: after.groups.saturating_sub(before.groups),
        multi_row_groups: after
            .multi_row_groups
            .saturating_sub(before.multi_row_groups),
    }
}

fn memory_record(runtime: &DriverRuntime) -> MemoryRecord {
    let accounting = runtime.backend().memory_accounting();
    MemoryRecord {
        live_bytes: accounting.live_bytes,
        reserved_bytes: accounting.reserved_bytes,
        peak_live_bytes: accounting.peak_live_bytes,
        peak_owned_and_reserved_bytes: accounting.peak_owned_and_reserved_bytes,
        live_allocations: accounting.live_allocations,
        peak_live_allocations: accounting.peak_live_allocations,
        allocations: accounting.allocations,
        frees: accounting.frees,
    }
}

fn digest_tokens(hasher: &mut Sha256, tokens: &[u32]) {
    for token in tokens {
        hasher.update(token.to_le_bytes());
    }
}

fn input_digest(case: &CaseSpec, fixture: &[u32]) -> DriverResult<String> {
    let mut digest = Sha256::new();
    for branch in 0..case.prefix_offsets.len() {
        let prompt = logical_prompt(case, branch, fixture)?;
        digest_tokens(&mut digest, &prompt);
    }
    for step in 0..case.decode_steps {
        for branch in active_rows(case, step)? {
            digest_tokens(
                &mut digest,
                checked_slice(fixture, case.teacher_offsets[branch][step], 1)?,
            );
        }
    }
    Ok(hex(&digest.finalize()))
}

fn logical_prompt(case: &CaseSpec, branch: usize, fixture: &[u32]) -> DriverResult<Vec<u32>> {
    if case.topology != "fork_of_fork" {
        return prompt_tokens(case, branch, fixture);
    }
    let mut prompt = nested_prompt(case, fixture)?;
    prompt.extend_from_slice(checked_slice(
        fixture,
        case.tail_offsets[branch],
        case.tail_tokens[branch],
    )?);
    Ok(prompt)
}

fn run_steps(
    runtime: &mut DriverRuntime,
    branches: &mut [BranchState],
    case: &CaseSpec,
    fixture: &[u32],
    options: &GenerateOptions,
    mut sidecar: Option<&mut Sidecar>,
    trace: &mut OutputTrace,
) -> DriverResult<(f64, String, String, String)> {
    let mut step_run = StepRun {
        fixture,
        options,
        trace,
    };
    let mut elapsed_ms = 0.0;
    let mut logits_digest = Sha256::new();
    let mut position_digest = Sha256::new();
    let mut token_digest = Sha256::new();
    for step in 0..case.decode_steps {
        retire_removed_sessions(runtime, branches, case, step)?;
        let active = active_rows(case, step)?;
        let (elapsed, tokens) = run_step(runtime, branches, case, step, &active, &mut step_run)?;
        elapsed_ms += elapsed;
        record_step_outputs(
            runtime,
            branches,
            case,
            &tokens,
            step,
            &mut logits_digest,
            &mut position_digest,
            &mut token_digest,
            &active,
            case.capture_steps
                .contains(&step)
                .then_some(sidecar.as_deref_mut())
                .flatten(),
            step_run.trace,
        )?;
    }
    Ok((
        elapsed_ms,
        hex(&logits_digest.finalize()),
        hex(&position_digest.finalize()),
        hex(&token_digest.finalize()),
    ))
}

fn retire_removed_sessions(
    runtime: &mut DriverRuntime,
    branches: &mut [BranchState],
    case: &CaseSpec,
    step: usize,
) -> DriverResult<()> {
    let Some(removed) = case
        .removed_sequences
        .as_ref()
        .and_then(|rows| rows.get(step))
    else {
        return Ok(());
    };
    for branch in removed {
        let branch = branches
            .get_mut(*branch)
            .ok_or_else(|| format!("removed branch {branch} is outside {}", case.name))?;
        runtime.discard_session(&mut branch.session)?;
    }
    Ok(())
}

fn run_step(
    runtime: &mut DriverRuntime,
    branches: &mut [BranchState],
    case: &CaseSpec,
    step: usize,
    active: &[usize],
    step_run: &mut StepRun<'_>,
) -> DriverResult<(f64, Vec<u32>)> {
    append_teacher_tokens(branches, case, step_run.fixture, step, active)?;
    let before = Instant::now();
    let tokens = dispatch_batch(
        runtime,
        branches,
        active,
        step_run.options,
        &mut step_run.trace.batch_schedule,
    )?;
    runtime.synchronize()?;
    Ok((before.elapsed().as_secs_f64() * 1_000.0, tokens))
}

#[allow(clippy::too_many_arguments)]
fn record_step_outputs(
    runtime: &mut DriverRuntime,
    branches: &mut [BranchState],
    case: &CaseSpec,
    tokens: &[u32],
    step: usize,
    logits_digest: &mut Sha256,
    position_digest: &mut Sha256,
    token_digest: &mut Sha256,
    active: &[usize],
    mut sidecar: Option<&mut Sidecar>,
    trace: &mut OutputTrace,
) -> DriverResult<()> {
    if tokens.len() != active.len() {
        return Err(format!(
            "backend returned {} tokens for {} active rows at step {step}",
            tokens.len(),
            active.len()
        )
        .into());
    }
    for (row, (&branch, token)) in active.iter().zip(tokens).enumerate() {
        let sequence = case.branch_sequences.get(branch).copied().unwrap_or(branch);
        let event = case
            .step_events
            .get(step)
            .map(String::as_str)
            .unwrap_or("decode_stepwise");
        let token_index = if case.step_events.is_empty() {
            step
        } else {
            step_token_index(case, step)?
        };
        let expected_input_position = case
            .step_input_positions
            .get(step)
            .and_then(|positions| positions.get(row))
            .copied();
        record_step_output(
            runtime,
            branches,
            branch,
            sequence,
            event,
            *token,
            token_index,
            expected_input_position,
            logits_digest,
            position_digest,
            token_digest,
            sidecar.as_deref_mut(),
            trace,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn record_step_output(
    runtime: &mut DriverRuntime,
    branches: &mut [BranchState],
    branch: usize,
    sequence: usize,
    event: &str,
    token: u32,
    token_index: usize,
    expected_input_position: Option<usize>,
    logits_digest: &mut Sha256,
    position_digest: &mut Sha256,
    token_digest: &mut Sha256,
    sidecar: Option<&mut Sidecar>,
    trace: &mut OutputTrace,
) -> DriverResult<()> {
    token_digest.update(token.to_le_bytes());
    let mut logits = vec![0.0_f32; runtime.vocab_size()];
    let position = runtime.read_session_logits(&branches[branch].session, &mut logits)?;
    let expected_position = branches[branch].session.evaluated_tokens().len();
    validate_logit_row(branch, token_index, position, expected_position, &logits)?;
    validate_source_position(position, expected_input_position)?;
    position_digest.update(u64::try_from(branch)?.to_le_bytes());
    position_digest.update(u64::try_from(token_index)?.to_le_bytes());
    position_digest.update(u64::try_from(position)?.to_le_bytes());
    logits_digest.update(u64::try_from(position)?.to_le_bytes());
    for value in &logits {
        logits_digest.update(value.to_bits().to_le_bytes());
    }
    trace.record(
        sequence,
        event,
        token_index,
        position.saturating_sub(1),
        position,
        logits.len(),
        &logits,
        sidecar,
    )
}

fn validate_logit_row(
    branch: usize,
    step: usize,
    position: usize,
    expected_position: usize,
    logits: &[f32],
) -> DriverResult<()> {
    if position != expected_position {
        return Err(format!(
            "logit position {position} differs from evaluated length {expected_position}"
        )
        .into());
    }
    if logits.iter().any(|value| !value.is_finite()) {
        return Err(format!("nonfinite raw logits for branch {branch} at step {step}").into());
    }
    Ok(())
}

fn validate_source_position(
    predicted_position: usize,
    expected_input_position: Option<usize>,
) -> DriverResult<()> {
    let Some(expected_input_position) = expected_input_position else {
        return Ok(());
    };
    let expected_predicted_position = expected_input_position
        .checked_add(1)
        .ok_or("source predicted position overflow")?;
    if predicted_position != expected_predicted_position {
        return Err(format!(
            "logit position {predicted_position} differs from source position {expected_predicted_position}"
        )
        .into());
    }
    Ok(())
}

fn active_rows(case: &CaseSpec, step: usize) -> DriverResult<Vec<usize>> {
    match &case.active_sequences {
        Some(rows) => rows
            .get(step)
            .cloned()
            .ok_or_else(|| format!("active row schedule ends before step {step}").into()),
        None => Ok((0..case.prefix_offsets.len()).collect()),
    }
}

fn step_token_index(case: &CaseSpec, step: usize) -> DriverResult<usize> {
    let event = case
        .step_events
        .get(step)
        .ok_or_else(|| format!("step event {step} is missing from {}", case.name))?;
    let mut first = step;
    while first > 0 && case.step_events[first - 1] == *event {
        first -= 1;
    }
    Ok(step - first)
}

fn expected_trace_rows(case: &CaseSpec) -> DriverResult<(usize, usize)> {
    let setup_rows =
        case.setup_events
            .iter()
            .try_fold(0_usize, |rows, event| -> DriverResult<usize> {
                rows.checked_add(event.sequences.len())
                    .ok_or_else(|| "setup output row count overflow".to_owned().into())
            })?;
    let mut raw_rows = setup_rows;
    let mut sidecar_rows = setup_rows;
    for step in 0..case.decode_steps {
        let rows = active_rows(case, step)?.len();
        raw_rows = raw_rows
            .checked_add(rows)
            .ok_or("raw output row count overflow")?;
        if case.capture_steps.contains(&step) {
            sidecar_rows = sidecar_rows
                .checked_add(rows)
                .ok_or("sidecar output row count overflow")?;
        }
    }
    Ok((raw_rows, sidecar_rows))
}

fn validate_trace_rows(
    case: &CaseSpec,
    trace: &OutputTrace,
    writes_sidecar: bool,
) -> DriverResult<()> {
    let (expected_raw, expected_sidecar) = expected_trace_rows(case)?;
    if trace.raw_rows != expected_raw {
        return Err(format!(
            "raw output rows for {} are {}, expected {expected_raw}",
            case.name, trace.raw_rows
        )
        .into());
    }
    if writes_sidecar && trace.sidecar_rows != expected_sidecar {
        return Err(format!(
            "sidecar output rows for {} are {}, expected {expected_sidecar}",
            case.name, trace.sidecar_rows
        )
        .into());
    }
    if trace.bindings.len() != expected_raw {
        return Err(format!(
            "logit bindings for {} are {}, expected {expected_raw}",
            case.name,
            trace.bindings.len()
        )
        .into());
    }
    Ok(())
}

fn run_once(
    runtime: &mut DriverRuntime,
    case: &CaseSpec,
    fixture: &[u32],
    options: &GenerateOptions,
    sidecar: Option<&mut Sidecar>,
) -> DriverResult<RunResult> {
    let mut trace = OutputTrace::default();
    let mut sidecar = sidecar;
    let writes_sidecar = sidecar.is_some();
    let memory_before = memory_record(runtime);
    let setup_start = Instant::now();
    let setup_before = backend_adapter::batch_stats(runtime.backend());
    let mut branches = make_branches(runtime, case, fixture, options, &mut trace, &mut sidecar)?;
    let setup_after = backend_adapter::batch_stats(runtime.backend());
    let memory_after_setup = memory_record(runtime);
    let setup_ms = setup_start.elapsed().as_secs_f64() * 1_000.0;
    let before = backend_adapter::batch_stats(runtime.backend());
    let (decode_ms, logits, positions, tokens) = run_steps(
        runtime,
        &mut branches,
        case,
        fixture,
        options,
        sidecar,
        &mut trace,
    )?;
    let after = backend_adapter::batch_stats(runtime.backend());
    let memory_after_decode = memory_record(runtime);
    validate_trace_rows(case, &trace, writes_sidecar)?;
    for branch in &mut branches {
        runtime.discard_session(&mut branch.session)?;
    }
    drop(branches);
    runtime.synchronize()?;
    let memory_after_retire = memory_record(runtime);
    Ok(RunResult {
        setup_ms,
        decode_ms,
        logits_digest: logits,
        position_digest: positions,
        token_digest: tokens,
        setup_stats: stats_delta(setup_before, setup_after),
        stats: stats_delta(before, after),
        batch_schedule: trace.batch_schedule,
        raw_rows: trace.raw_rows,
        sidecar_rows: trace.sidecar_rows,
        bindings: trace.bindings,
        memory_before,
        memory_after_setup,
        memory_after_decode,
        memory_after_retire,
    })
}

fn median(values: &[f64]) -> DriverResult<f64> {
    if values.is_empty() {
        return Err("median needs samples".into());
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    Ok(sorted[sorted.len() / 2])
}

fn record_consistent<T: PartialEq>(
    slot: &mut Option<T>,
    value: T,
    label: &str,
    case_name: &str,
) -> DriverResult<()> {
    if slot.as_ref().is_some_and(|expected| expected != &value) {
        return Err(format!("{label} changed across {case_name}").into());
    }
    if slot.is_none() {
        *slot = Some(value);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn collect_repetitions(
    runtime: &mut DriverRuntime,
    case: &CaseSpec,
    fixture: &[u32],
    options: &GenerateOptions,
    output_dir: &Path,
    path: BatchPath,
    repetitions: usize,
) -> DriverResult<CaseSamples> {
    let mut samples = CaseSamples::with_capacity(repetitions);
    for repetition in 0..repetitions {
        let result = run_repetition(
            runtime, case, fixture, options, output_dir, path, repetition,
        )?;
        samples.push(result, &case.name)?;
    }
    Ok(samples)
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    runtime: &mut DriverRuntime,
    case: &CaseSpec,
    fixture: &[u32],
    options: &GenerateOptions,
    output_dir: &Path,
    path: BatchPath,
    repetitions: usize,
    warmups: usize,
) -> DriverResult<CaseRecord> {
    let case_input_digest = input_digest(case, fixture)?;
    run_warmups(runtime, case, fixture, options, warmups)?;
    let samples = collect_repetitions(
        runtime,
        case,
        fixture,
        options,
        output_dir,
        path,
        repetitions,
    )?;
    let setup_median_ms = median(&samples.setup_ms)?;
    let decode_median_ms = median(&samples.decode_ms)?;
    Ok(CaseRecord {
        name: case.name.clone(),
        source_case: case.source_case.clone(),
        topology: case.topology.clone(),
        input_digest: case_input_digest,
        decode_steps: case.decode_steps,
        branch_count: case.prefix_offsets.len(),
        retained_branch_count: branches_retained_count(case),
        raw_logit_vocab: runtime.vocab_size(),
        raw_logit_rows: samples.raw_rows[0],
        sidecar_logit_rows: samples.sidecar_rows[0],
        logit_bindings: samples.bindings.ok_or("missing logit bindings")?,
        memory_before: samples.memory_before,
        memory_after_setup: samples.memory_after_setup,
        memory_after_decode: samples.memory_after_decode,
        memory_after_retire: samples.memory_after_retire,
        setup_ms: samples.setup_ms,
        decode_ms: samples.decode_ms,
        logits_digest: samples.logits_digest,
        raw_logit_position_digest: samples.position_digest,
        token_digest: samples.token_digest,
        logits_sidecars: samples.sidecars,
        setup_stats: samples.setup_stats,
        stats: samples.stats,
        batch_schedule: samples.batch_schedule.ok_or("missing batch schedule")?,
        setup_median_ms,
        decode_median_ms,
    })
}

fn branches_retained_count(case: &CaseSpec) -> usize {
    if case
        .branch_sequences
        .first()
        .is_some_and(|sequence| *sequence == 0)
    {
        0
    } else if !case.retained_sequences.is_empty() {
        case.retained_sequences.len()
    } else {
        match case.topology.as_str() {
            "shared" => 1,
            "fork_of_fork" => 2,
            _ => 0,
        }
    }
}

fn run_warmups(
    runtime: &mut DriverRuntime,
    case: &CaseSpec,
    fixture: &[u32],
    options: &GenerateOptions,
    warmups: usize,
) -> DriverResult<()> {
    for _ in 0..warmups {
        run_once(runtime, case, fixture, options, None)?;
    }
    Ok(())
}

fn run_repetition(
    runtime: &mut DriverRuntime,
    case: &CaseSpec,
    fixture: &[u32],
    options: &GenerateOptions,
    output_dir: &Path,
    path: BatchPath,
    repetition: usize,
) -> DriverResult<RepetitionResult> {
    let sidecar_path = sidecar_path(output_dir, &case.name, path, repetition)?;
    let mut sidecar = Sidecar::create(&sidecar_path)?;
    let result = run_once(runtime, case, fixture, options, Some(&mut sidecar))?;
    let sidecar_digest = sidecar.finish()?;
    let record = SidecarRecord {
        file: output_dir
            .file_name()
            .map(Path::new)
            .ok_or("sidecar directory name missing")?
            .join(sidecar_path.file_name().ok_or("sidecar name missing")?)
            .to_string_lossy()
            .into_owned(),
        sha256: sidecar_digest,
    };
    Ok(RepetitionResult {
        run: result,
        sidecar: record,
    })
}

/// Accepts one path component of ASCII letters, digits, `_`, and `-`.
fn validate_case_name(name: &str) -> DriverResult<()> {
    let valid = !name.is_empty()
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-');
    if valid {
        Ok(())
    } else {
        Err(format!("case name {name:?} must use only ASCII letters, digits, '_' and '-'").into())
    }
}

fn sidecar_path(
    output_dir: &Path,
    case_name: &str,
    path: BatchPath,
    repetition: usize,
) -> DriverResult<PathBuf> {
    validate_case_name(case_name)?;
    Ok(output_dir.join(format!(
        "{case_name}-{}.{repetition}.logits.bin",
        path_name(path)
    )))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn fixture_path(manifest_path: &Path, fixture: &str) -> PathBuf {
    manifest_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(fixture)
}

fn model_name(path: &Path) -> DriverResult<String> {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .ok_or_else(|| "model path has no file name".into())
}

fn sha256_file(path: &Path) -> DriverResult<String> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(hex(&digest.finalize()))
}

fn selected_cases<'a>(manifest: &'a Manifest, phase: &str) -> &'a [CaseSpec] {
    if phase == "calibration" {
        &manifest.calibration
    } else {
        manifest
            .evaluation
            .as_deref()
            .expect("manifest evaluation cases were validated")
    }
}

struct RunInputs {
    manifest: Manifest,
    fixture_path: PathBuf,
    fixture: Vec<u32>,
    run_dir: PathBuf,
    model_sha256: String,
}

fn prepare_run(options: &Options) -> DriverResult<RunInputs> {
    let manifest = load_manifest(&options.manifest)?;
    let (fixture_path, fixture) = load_run_fixture(&options.manifest, &manifest)?;
    let model_sha256 = sha256_file(&options.model)?;
    validate_model_subject(&manifest, &model_sha256)?;
    let run_dir = reserve_run_directory(&options.output)?;
    Ok(RunInputs {
        manifest,
        fixture_path,
        fixture,
        run_dir,
        model_sha256,
    })
}

fn reserve_run_directory(output: &Path) -> DriverResult<PathBuf> {
    if output.symlink_metadata().is_ok() {
        return Err("refusing to replace an existing runtime receipt".into());
    }
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let mut name = output
        .file_name()
        .ok_or("receipt file name missing")?
        .to_os_string();
    name.push(".sidecars");
    let directory = parent.join(name);
    fs::create_dir(&directory)?;
    Ok(directory)
}

fn execute_run(
    options: &Options,
    inputs: &RunInputs,
) -> DriverResult<(serde_json::Value, MemoryRecord, Vec<CaseRecord>)> {
    let mut runtime = load_checked_runtime(options, &inputs.manifest)?;
    let device = backend_adapter::device_metadata(runtime.backend())?;
    let memory_after_load = memory_record(&runtime);
    let cases = run_cases(
        &mut runtime,
        options,
        &inputs.manifest,
        inputs.fixture.as_slice(),
        &inputs.run_dir,
    )?;
    Ok((device, memory_after_load, cases))
}

fn ensure_model_unchanged(path: &Path, expected: &str) -> DriverResult<()> {
    if sha256_file(path)? != expected {
        return Err("model artifact changed during the run".into());
    }
    Ok(())
}

fn run(options: Options) -> DriverResult<()> {
    let inputs = prepare_run(&options)?;
    let (device, memory_after_load, cases) = execute_run(&options, &inputs)?;
    ensure_model_unchanged(&options.model, &inputs.model_sha256)?;
    let receipt = build_receipt(
        &options,
        &inputs.fixture_path,
        &inputs.manifest,
        inputs.model_sha256,
        device,
        memory_after_load,
        cases,
    )?;
    write_receipt(&options.output, &receipt)?;
    Ok(())
}

fn validate_model_subject(manifest: &Manifest, model_sha256: &str) -> DriverResult<()> {
    if let Some(expected) = &manifest.source_subject_sha256 {
        if model_sha256 != expected {
            return Err("selected model differs from the source workload subject".into());
        }
    }
    Ok(())
}

fn load_checked_runtime(options: &Options, manifest: &Manifest) -> DriverResult<DriverRuntime> {
    let runtime = load_runtime(options)?;
    if let Some(expected) = manifest.source_vocab {
        if runtime.vocab_size() != expected {
            return Err(format!(
                "runtime vocabulary {} differs from source contract {}",
                runtime.vocab_size(),
                expected
            )
            .into());
        }
    }
    Ok(runtime)
}

fn load_run_fixture(
    manifest_path: &Path,
    manifest: &Manifest,
) -> DriverResult<(PathBuf, Vec<u32>)> {
    let path = fixture_path(manifest_path, &manifest.fixture);
    let fixture = read_tokens(&path)?;
    if let Some(expected) = &manifest.source_fixture_sha256 {
        if sha256_file(&path)? != *expected {
            return Err("runtime and source token fixture digests differ".into());
        }
    }
    Ok((path, fixture))
}

fn write_receipt(path: &Path, receipt: &Receipt) -> DriverResult<()> {
    let bytes = serde_json::to_vec_pretty(receipt)?;
    let mut file = File::options().write(true).create_new(true).open(path)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    Ok(())
}

fn load_runtime(options: &Options) -> DriverResult<DriverRuntime> {
    Ok(DriverRuntime::load(
        backend_adapter::load_backend(options.path)?,
        &options.model,
    )?)
}

fn run_cases(
    runtime: &mut DriverRuntime,
    options: &Options,
    manifest: &Manifest,
    fixture: &[u32],
    run_dir: &Path,
) -> DriverResult<Vec<CaseRecord>> {
    let options_for_case = generation_options();
    selected_cases(manifest, &options.phase)
        .iter()
        .map(|case| {
            run_case(
                runtime,
                case,
                fixture,
                &options_for_case,
                run_dir,
                options.path,
                options.repetitions,
                options.warmups,
            )
        })
        .collect()
}

fn build_receipt(
    options: &Options,
    fixture_path: &Path,
    manifest: &Manifest,
    model_sha256: String,
    device: serde_json::Value,
    memory_after_load: MemoryRecord,
    cases: Vec<CaseRecord>,
) -> DriverResult<Receipt> {
    Ok(Receipt {
        schema: RECEIPT_SCHEMA,
        phase: options.phase.clone(),
        path: path_name(options.path).to_owned(),
        device,
        quality: "unverified",
        claim_scope: "opt-in Runtime batch-path differential",
        model_name: model_name(&options.model)?,
        model_sha256,
        fixture_sha256: sha256_file(fixture_path)?,
        source_manifest_sha256: manifest.source_manifest_sha256.clone(),
        source_plan_sha256: manifest.source_plan_sha256.clone(),
        source_subject_sha256: manifest.source_subject_sha256.clone(),
        source_phase: manifest.source_phase.clone(),
        memory_after_load,
        cases,
    })
}

fn main() -> DriverResult<()> {
    if env::args().skip(1).eq(["--build-info"]) {
        println!("{}", serde_json::to_string(&build_info::value())?);
        return Ok(());
    }
    let options = parse_options()?;
    run(options)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_namespaces_and_sidecars_preserve_prior_runs() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock follows epoch")
            .as_nanos();
        let root = env::temp_dir().join(format!("leone-runtime-{}-{nonce}", std::process::id()));
        let first = reserve_run_directory(&root.join("first.json")).expect("reserve first run");
        let path = first.join("case.logits.bin");
        let mut sidecar = Sidecar::create(&path).expect("create first sidecar");
        sidecar.write(b"retained logits").expect("write logits");
        sidecar.finish().expect("close first sidecar");
        assert!(Sidecar::create(&path).is_err());
        assert!(reserve_run_directory(&root.join("first.json")).is_err());
        let second = reserve_run_directory(&root.join("second.json")).expect("reserve second run");
        assert_ne!(first, second);
        Sidecar::create(&second.join("case.logits.bin")).expect("independent sidecar");
        assert_eq!(fs::read(&path).expect("read original"), b"retained logits");
        fs::write(root.join("existing.json"), b"existing receipt").expect("write sentinel");
        assert!(reserve_run_directory(&root.join("existing.json")).is_err());
        assert_eq!(
            fs::read(root.join("existing.json")).unwrap(),
            b"existing receipt"
        );
        fs::remove_dir_all(root).expect("remove test files");
    }

    #[test]
    fn case_names_are_single_safe_components() {
        for name in ["long_shared", "calibration-nested", "A1"] {
            validate_case_name(name).expect("safe name");
        }
        for name in [
            "",
            ".",
            "..",
            "../x",
            "a/b",
            "/abs",
            "C:\\x",
            "a.b",
            "a b",
            "a\0b",
            "caf\u{e9}",
        ] {
            assert!(validate_case_name(name).is_err(), "{name:?}");
        }
    }

    #[test]
    fn sidecar_paths_cannot_leave_the_run_directory() {
        let directory = env::temp_dir().join(format!("leone-runtime-names-{}", std::process::id()));
        fs::create_dir_all(&directory).expect("create directory");
        let path = BatchPath::PerRow;
        for name in ["../escape", "/tmp/escape", "nested/name"] {
            assert!(sidecar_path(&directory, name, path, 0).is_err(), "{name}");
        }
        assert_eq!(fs::read_dir(&directory).expect("list").count(), 0);
        let accepted = sidecar_path(&directory, "long_shared", path, 3).expect("safe name");
        assert_eq!(accepted.parent(), Some(directory.as_path()));
        assert_eq!(
            accepted.file_name().and_then(|name| name.to_str()),
            Some("long_shared-per_row.3.logits.bin")
        );
        fs::remove_dir_all(directory).expect("remove test files");
    }

    #[test]
    fn unsafe_manifest_case_names_fail_case_validation() {
        let mut case = pinned_manifest().calibration.remove(0);
        validate_case(&case).expect("pinned case is valid");
        case.name = "../calibration".to_owned();
        assert!(validate_case(&case).is_err());
    }

    #[test]
    fn pinned_manifest_names_bind_to_the_source_workload() {
        let manifest = pinned_manifest();
        assert!(!manifest.calibration.is_empty());
        assert!(manifest.evaluation.is_some_and(|cases| !cases.is_empty()));
    }

    fn pinned_manifest() -> Manifest {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../runtime_manifest.json");
        load_manifest(&path).expect("pinned manifest loads and validates")
    }

    fn source_stream(seq: usize, outputs: Option<serde_json::Value>) -> SourceStream {
        SourceStream {
            seq,
            position: 0,
            tokens: vec!["test".to_owned()],
            outputs,
        }
    }

    #[test]
    fn pinned_source_events_match_the_runtime_schedule() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../llama_cached_workload.json");
        let source = read_source_manifest(&path).expect("source workload parses");
        for case in &source.cases {
            validate_source_case_events(case).expect("source events are fully bound");
        }
    }

    fn pinned_source_case(name: &str) -> SourceCase {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../llama_cached_workload.json");
        read_source_manifest(&path)
            .expect("source workload parses")
            .cases
            .into_iter()
            .find(|case| case.name == name)
            .expect("pinned source case exists")
    }

    #[test]
    fn dropped_source_events_are_rejected() {
        let case = SourceCase {
            name: "shared".to_owned(),
            class: "shared_prefix".to_owned(),
            operations: Vec::new(),
        };
        assert!(validate_source_case_events(&case).is_err());
    }

    #[test]
    fn unmapped_copy_targets_are_rejected() {
        let event = SourceEvent::Copy {
            name: "fork_4".to_owned(),
            source: 0,
            target: 6,
            p0: 0,
            p1: -1,
        };
        assert!(validate_source_event_binding(
            &SourceCase {
                name: "shared".to_owned(),
                class: "shared_prefix".to_owned(),
                operations: Vec::new(),
            },
            &event,
        )
        .is_err());
    }

    #[test]
    fn omitted_pinned_operation_is_rejected() {
        let mut case = pinned_source_case("long_shared");
        case.operations.remove(1);
        assert!(validate_source_case_events(&case).is_err());
    }

    #[test]
    fn unrepresented_decode_stream_is_rejected() {
        let mut case = pinned_source_case("long_shared");
        let SourceEvent::DecodeStepwise { streams, .. } = case
            .operations
            .last_mut()
            .expect("pinned case has continuation")
        else {
            panic!("pinned case ends with continuation");
        };
        streams.push(source_stream(0, None));
        assert!(validate_source_case_events(&case).is_err());
    }

    #[test]
    fn unsupported_decode_output_selector_is_rejected() {
        let mut case = pinned_source_case("shrinking_membership");
        let event = case
            .operations
            .iter_mut()
            .find(|event| source_event_name(event) == "active_four")
            .expect("pinned case has active_four");
        let SourceEvent::Decode { streams, .. } = event else {
            panic!("active_four is a decode event");
        };
        streams[0].outputs = Some(serde_json::Value::String("all".to_owned()));
        assert!(validate_source_case_events(&case).is_err());
    }
}
