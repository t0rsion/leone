use leone_receipt::{ActiveCompute, QualityReceipt, ResponseReceipt, RuntimeReceipt};
use serde_json::{Map, Value};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn main() -> ExitCode {
    match run(env::args().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}

fn run(arguments: Vec<String>) -> Result<(), String> {
    if matches!(
        arguments.as_slice(),
        [receipt, verify, _] if receipt == "receipt" && verify == "verify-response"
    ) {
        return run_response(&arguments);
    }
    if matches!(
        arguments.first().map(String::as_str),
        Some("runtime" | "quality")
    ) {
        return run_legacy(&arguments);
    }
    let options = ReceiptOptions::parse(&arguments)?;
    options.validate()
}

fn run_response(arguments: &[String]) -> Result<(), String> {
    let path = match arguments {
        [receipt, verify, path] if receipt == "receipt" && verify == "verify-response" => path,
        _ => return Err(usage()),
    };
    let bytes = fs::read(path).map_err(|error| format!("{path}: {error}"))?;
    ResponseReceipt::from_json(&bytes)
        .map(|_| ())
        .map_err(|error| format!("{path}: {error}"))
}

fn run_legacy(arguments: &[String]) -> Result<(), String> {
    if arguments.len() < 2 {
        return Err(usage());
    }
    let kind = ReceiptKind::parse(&arguments[0])?;
    for path in &arguments[1..] {
        validate_receipt_file(kind, Path::new(path))?;
    }
    Ok(())
}

fn validate_receipt_file(kind: ReceiptKind, path: &Path) -> Result<(), String> {
    let bytes = fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    match kind {
        ReceiptKind::Runtime => RuntimeReceipt::from_json(&bytes).map(|_| ()),
        ReceiptKind::Quality => QualityReceipt::from_json(&bytes).map(|_| ()),
    }
    .map_err(|error| format!("{}: {error}", path.display()))
}

fn usage() -> String {
    "usage: leone-receipt-verify receipt verify-response <json> | leone-receipt-verify --offline --root <dir> --manifest <json> --source-manifest <json> --source-commit <sha> --model-sha256 <sha> --record <json> --kind runtime|quality --platform <platform> --target <target> --backend <backend> [--model-family <name>] [--metric-family kld]"
        .to_owned()
}

#[derive(Clone, Copy)]
enum ReceiptKind {
    Runtime,
    Quality,
}

impl ReceiptKind {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "runtime" => Ok(Self::Runtime),
            "quality" => Ok(Self::Quality),
            _ => Err(usage()),
        }
    }

    fn role(self) -> &'static str {
        match self {
            Self::Runtime => "runtime",
            Self::Quality => "quality",
        }
    }

    fn validator(self) -> &'static str {
        match self {
            Self::Runtime => "runtime-receipt-v1",
            Self::Quality => "quality-receipt-v1",
        }
    }
}

struct ReceiptOptions {
    root: PathBuf,
    manifest: PathBuf,
    source_manifest: PathBuf,
    source_commit: String,
    model_sha256: String,
    record: PathBuf,
    kind: ReceiptKind,
    platform: String,
    target: String,
    backend: String,
    model_family: Option<String>,
    metric_family: Option<String>,
}

struct ReceiptFiles {
    record: PathBuf,
    manifest: Value,
    source_manifest: Value,
    record_name: String,
    source_manifest_name: String,
}

struct ManifestBinding {
    model_family: Option<String>,
    metric_family: Option<String>,
}

#[derive(Default)]
struct ParsedOptions {
    offline: bool,
    root: Option<PathBuf>,
    manifest: Option<PathBuf>,
    source_manifest: Option<PathBuf>,
    source_commit: Option<String>,
    model_sha256: Option<String>,
    record: Option<PathBuf>,
    kind: Option<ReceiptKind>,
    platform: Option<String>,
    target: Option<String>,
    backend: Option<String>,
    model_family: Option<String>,
    metric_family: Option<String>,
}

impl ReceiptOptions {
    fn parse(arguments: &[String]) -> Result<Self, String> {
        let mut parsed = ParsedOptions::default();
        let mut index = 0;
        while index < arguments.len() {
            parse_argument(arguments, &mut index, &mut parsed)?;
            index += 1;
        }
        parsed.finish()
    }

    fn validate(&self) -> Result<(), String> {
        validate_package_identity(&self.platform, &self.target, &self.backend)?;
        let root = canonical_directory(&self.root)?;
        let files = self.read_files(&root)?;
        validate_source_manifest(&files.source_manifest, &self.source_commit)?;
        validate_manifest_source_path(&files.manifest, &files.source_manifest_name)?;
        let binding =
            validate_manifest_record(&files.manifest, &files.record_name, self.kind, self)?;
        self.validate_record(&files.record, &binding)
    }

    fn read_files(&self, root: &Path) -> Result<ReceiptFiles, String> {
        let record = checked_file(root, &self.record, "receipt")?;
        let manifest = checked_file(root, &self.manifest, "release manifest")?;
        let source_manifest = checked_file(root, &self.source_manifest, "source manifest")?;
        let record_name = relative_name(root, &record)?;
        let source_manifest_name = relative_name(root, &source_manifest)?;
        Ok(ReceiptFiles {
            record,
            manifest: read_json(&manifest, "release manifest")?,
            source_manifest: read_json(&source_manifest, "source manifest")?,
            record_name,
            source_manifest_name,
        })
    }

    fn validate_record(&self, record: &Path, binding: &ManifestBinding) -> Result<(), String> {
        match self.kind {
            ReceiptKind::Runtime => self.validate_runtime_record(record, binding)?,
            ReceiptKind::Quality => self.validate_quality_record(record, binding)?,
        }
        Ok(())
    }

    fn validate_runtime_record(
        &self,
        record: &Path,
        binding: &ManifestBinding,
    ) -> Result<(), String> {
        let bytes = fs::read(record).map_err(io_error)?;
        let receipt =
            RuntimeReceipt::from_json(&bytes).map_err(|error| format!("receipt: {error}"))?;
        validate_runtime_identity(&receipt, self, binding)
    }

    fn validate_quality_record(
        &self,
        record: &Path,
        binding: &ManifestBinding,
    ) -> Result<(), String> {
        let bytes = fs::read(record).map_err(io_error)?;
        let receipt =
            QualityReceipt::from_json(&bytes).map_err(|error| format!("receipt: {error}"))?;
        validate_quality_identity(&receipt, self, binding)
    }
}

fn parse_argument(
    arguments: &[String],
    index: &mut usize,
    parsed: &mut ParsedOptions,
) -> Result<(), String> {
    let argument = arguments[*index].as_str();
    if argument == "--offline" {
        parsed.offline = true;
        return Ok(());
    }
    if argument == "--kind" {
        let value = next_value(arguments, index, "kind")?;
        return set_once(&mut parsed.kind, ReceiptKind::parse(value)?, "kind");
    }
    if let Some(name) = path_argument_name(argument) {
        return set_path_argument(arguments, index, parsed, name);
    }
    if let Some(name) = string_argument_name(argument) {
        return set_string_argument(arguments, index, parsed, name);
    }
    Err(usage())
}

fn path_argument_name(argument: &str) -> Option<&str> {
    match argument {
        "--root" => Some("root"),
        "--manifest" => Some("manifest"),
        "--source-manifest" => Some("source manifest"),
        "--record" => Some("record"),
        _ => None,
    }
}

fn string_argument_name(argument: &str) -> Option<&str> {
    match argument {
        "--source-commit" => Some("source commit"),
        "--model-sha256" => Some("model SHA-256"),
        "--platform" => Some("platform"),
        "--target" => Some("target"),
        "--backend" => Some("backend"),
        "--model-family" => Some("model family"),
        "--metric-family" => Some("metric family"),
        _ => None,
    }
}

fn set_path_argument(
    arguments: &[String],
    index: &mut usize,
    parsed: &mut ParsedOptions,
    name: &str,
) -> Result<(), String> {
    let value = PathBuf::from(next_value(arguments, index, name)?);
    match name {
        "root" => set_once(&mut parsed.root, value, name),
        "manifest" => set_once(&mut parsed.manifest, value, name),
        "source manifest" => set_once(&mut parsed.source_manifest, value, name),
        "record" => set_once(&mut parsed.record, value, name),
        _ => Err(usage()),
    }
}

fn set_string_argument(
    arguments: &[String],
    index: &mut usize,
    parsed: &mut ParsedOptions,
    name: &str,
) -> Result<(), String> {
    let value = next_value(arguments, index, name)?.to_owned();
    match name {
        "source commit" => set_once(&mut parsed.source_commit, value, name),
        "model SHA-256" => set_once(&mut parsed.model_sha256, value, name),
        "platform" => set_once(&mut parsed.platform, value, name),
        "target" => set_once(&mut parsed.target, value, name),
        "backend" => set_once(&mut parsed.backend, value, name),
        "model family" => set_once(&mut parsed.model_family, value, name),
        "metric family" => set_once(&mut parsed.metric_family, value, name),
        _ => Err(usage()),
    }
}

fn set_once<T>(slot: &mut Option<T>, value: T, name: &str) -> Result<(), String> {
    if slot.is_some() {
        return Err(format!("{name} was specified more than once"));
    }
    *slot = Some(value);
    Ok(())
}

fn next_value<'a>(
    arguments: &'a [String],
    index: &mut usize,
    name: &str,
) -> Result<&'a str, String> {
    *index += 1;
    arguments
        .get(*index)
        .map(String::as_str)
        .filter(|value| !value.starts_with('-'))
        .ok_or_else(|| format!("{name} is missing its value"))
}

impl ParsedOptions {
    fn finish(self) -> Result<ReceiptOptions, String> {
        if !self.offline {
            return Err("receipt verification requires --offline".to_owned());
        }
        let (root, manifest) = required_pair(self.root, "root", self.manifest, "manifest")?;
        let (source_manifest, record) = required_pair(
            self.source_manifest,
            "source manifest",
            self.record,
            "record",
        )?;
        let (platform, target) = required_pair(self.platform, "platform", self.target, "target")?;
        let kind = required(self.kind, "kind")?;
        Ok(ReceiptOptions {
            root,
            manifest,
            source_manifest,
            source_commit: required(self.source_commit, "source commit")?,
            model_sha256: required_digest(self.model_sha256, "model SHA-256")?,
            record,
            kind,
            platform,
            target,
            backend: required(self.backend, "backend")?,
            model_family: self.model_family,
            metric_family: validate_metric_family(kind, self.metric_family)?,
        })
    }
}

fn validate_metric_family(
    kind: ReceiptKind,
    metric_family: Option<String>,
) -> Result<Option<String>, String> {
    match (kind, metric_family) {
        (ReceiptKind::Runtime, Some(_)) => {
            Err("metric family is only valid for quality receipts".to_owned())
        }
        (ReceiptKind::Runtime, None) => Ok(None),
        (ReceiptKind::Quality, Some(value)) if value == "kld" => Ok(Some(value)),
        (ReceiptKind::Quality, Some(_)) => Err("metric family must be kld".to_owned()),
        (ReceiptKind::Quality, None) => {
            Err("metric family is required for quality receipts".to_owned())
        }
    }
}

fn required_pair<T>(
    first: Option<T>,
    first_name: &str,
    second: Option<T>,
    second_name: &str,
) -> Result<(T, T), String> {
    Ok((required(first, first_name)?, required(second, second_name)?))
}

fn required<T>(value: Option<T>, name: &str) -> Result<T, String> {
    value.ok_or_else(|| format!("{name} is required"))
}

fn required_digest(value: Option<String>, name: &str) -> Result<String, String> {
    let value = required(value, name)?;
    if !is_sha256(&value) {
        return Err(format!("{name} must be a SHA-256"));
    }
    Ok(value.to_ascii_lowercase())
}

fn canonical_directory(path: &Path) -> Result<PathBuf, String> {
    let canonical = fs::canonicalize(path).map_err(io_error)?;
    if !canonical.is_dir() {
        return Err(format!("root is not a directory: {}", path.display()));
    }
    Ok(canonical)
}

fn checked_file(root: &Path, path: &Path, name: &str) -> Result<PathBuf, String> {
    let candidate = if path.is_absolute() {
        path.to_owned()
    } else {
        root.join(path)
    };
    let canonical = fs::canonicalize(&candidate)
        .map_err(|error| format!("{name} cannot be read at {}: {error}", candidate.display()))?;
    if !canonical.starts_with(root) {
        return Err(format!("{name} is outside the archive root"));
    }
    if !canonical.is_file() {
        return Err(format!("{name} is not a regular file"));
    }
    Ok(canonical)
}

fn relative_name(root: &Path, path: &Path) -> Result<String, String> {
    let relative = path
        .strip_prefix(root)
        .map_err(|_| "receipt path is outside the archive root".to_owned())?;
    let value = relative
        .to_str()
        .ok_or_else(|| "receipt path is not valid UTF-8".to_owned())?
        .replace('\\', "/");
    if value.is_empty() || value.starts_with('/') || value.split('/').any(|part| part == "..") {
        return Err("receipt path is not normalized".to_owned());
    }
    Ok(value)
}

fn read_json(path: &Path, name: &str) -> Result<Value, String> {
    let bytes = fs::read(path).map_err(io_error)?;
    serde_json::from_slice(&bytes).map_err(|error| format!("{name} is invalid JSON: {error}"))
}

fn validate_source_manifest(value: &Value, source_commit: &str) -> Result<(), String> {
    let object = value
        .as_object()
        .ok_or_else(|| "source manifest is not an object".to_owned())?;
    if object.get("source_commit").and_then(Value::as_str) != Some(source_commit) {
        return Err("source manifest commit does not match receipt provenance".to_owned());
    }
    if !is_full_commit(source_commit) {
        return Err("source commit must be a 40-character hexadecimal hash".to_owned());
    }
    if object
        .get("files")
        .and_then(Value::as_object)
        .is_none_or(|files| files.is_empty())
    {
        return Err("source manifest has no file records".to_owned());
    }
    for (name, record) in object
        .get("files")
        .and_then(Value::as_object)
        .expect("checked above")
    {
        validate_source_record(name, record)?;
    }
    Ok(())
}

fn validate_source_record(name: &str, value: &Value) -> Result<(), String> {
    validate_relative_name(name, "source manifest path")?;
    let record = value
        .as_object()
        .ok_or_else(|| format!("source manifest record is invalid: {name}"))?;
    if !record.get("executable").is_some_and(Value::is_boolean) {
        return Err(format!(
            "source manifest executable mode is invalid: {name}"
        ));
    }
    if !record
        .get("sha256")
        .and_then(Value::as_str)
        .is_some_and(is_sha256)
    {
        return Err(format!("source manifest SHA-256 is invalid: {name}"));
    }
    Ok(())
}

fn validate_relative_name(value: &str, label: &str) -> Result<(), String> {
    let path = std::path::Path::new(value);
    if value.is_empty()
        || path.is_absolute()
        || value.contains('\\')
        || path.components().any(|component| {
            matches!(
                component,
                std::path::Component::ParentDir | std::path::Component::CurDir
            )
        })
    {
        return Err(format!("{label} is unsafe: {value}"));
    }
    Ok(())
}

fn validate_manifest_source_path(value: &Value, source_manifest: &str) -> Result<(), String> {
    if let Some(expected) = value.get("source_manifest").and_then(Value::as_str) {
        if expected != source_manifest {
            return Err("source manifest path does not match release manifest".to_owned());
        }
    }
    Ok(())
}

fn validate_manifest_record(
    value: &Value,
    path: &str,
    kind: ReceiptKind,
    options: &ReceiptOptions,
) -> Result<ManifestBinding, String> {
    let record = find_record_for_options(value, path, options)
        .ok_or_else(|| format!("release manifest has no record for {path}"))?;
    validate_manifest_contract(record, kind)?;
    let model_family = manifest_model_family(record)?;
    let metric_family = manifest_metric_family(record)?;
    validate_manifest_dependencies(record)?;
    Ok(ManifestBinding {
        model_family,
        metric_family,
    })
}

fn find_record_for_options<'a>(
    value: &'a Value,
    path: &str,
    options: &ReceiptOptions,
) -> Option<&'a Map<String, Value>> {
    if value.get("backend_requirements").is_some() {
        find_backend_record(value, path, options)
    } else {
        find_manifest_record(value, path)
    }
}

fn validate_manifest_contract(
    record: &Map<String, Value>,
    kind: ReceiptKind,
) -> Result<(), String> {
    if record.get("role").and_then(Value::as_str) != Some(kind.role())
        || record.get("validator").and_then(Value::as_str) != Some(kind.validator())
    {
        return Err("release manifest record has the wrong receipt validator".to_owned());
    }
    Ok(())
}

fn manifest_model_family(record: &Map<String, Value>) -> Result<Option<String>, String> {
    record
        .get("model_family")
        .map(|value| {
            value
                .as_str()
                .filter(|family| !family.is_empty())
                .map(str::to_owned)
                .ok_or_else(|| "release manifest model_family is invalid".to_owned())
        })
        .transpose()
}

fn manifest_metric_family(record: &Map<String, Value>) -> Result<Option<String>, String> {
    record
        .get("metric_family")
        .map(|value| {
            value
                .as_str()
                .filter(|family| !family.is_empty())
                .map(str::to_owned)
                .ok_or_else(|| "release manifest metric_family is invalid".to_owned())
        })
        .transpose()
}

fn validate_manifest_dependencies(record: &Map<String, Value>) -> Result<(), String> {
    let Some(value) = record.get("dependencies") else {
        return Ok(());
    };
    let dependencies = value
        .as_array()
        .filter(|dependencies| !dependencies.is_empty())
        .ok_or_else(|| "release manifest dependencies are invalid".to_owned())?;
    for dependency in dependencies {
        let dependency = dependency
            .as_str()
            .ok_or_else(|| "release manifest dependency is invalid".to_owned())?;
        validate_relative_name(dependency, "release manifest dependency")?;
    }
    Ok(())
}

fn find_backend_record<'a>(
    value: &'a Value,
    path: &str,
    options: &ReceiptOptions,
) -> Option<&'a Map<String, Value>> {
    let backends = value.get("backend_requirements")?.as_array()?;
    for backend in backends {
        let object = backend.as_object()?;
        if !backend_matches(object, options) {
            continue;
        }
        if let Some(record) = find_record_in_backend(object, path) {
            return Some(record);
        }
    }
    None
}

fn backend_matches(object: &Map<String, Value>, options: &ReceiptOptions) -> bool {
    object.get("platform").and_then(Value::as_str) == Some(options.platform.as_str())
        && object.get("target").and_then(Value::as_str) == Some(options.target.as_str())
        && object.get("backend").and_then(Value::as_str) == Some(options.backend.as_str())
}

fn find_record_in_backend<'a>(
    object: &'a Map<String, Value>,
    path: &str,
) -> Option<&'a Map<String, Value>> {
    let records = object.get("records")?.as_array()?;
    for record in records {
        let record = record.as_object()?;
        if record.get("path").and_then(Value::as_str) == Some(path) {
            return Some(record);
        }
    }
    None
}

fn find_manifest_record<'a>(value: &'a Value, path: &str) -> Option<&'a Map<String, Value>> {
    match value {
        Value::Object(object) => {
            if object.get("path").and_then(Value::as_str) == Some(path) {
                return Some(object);
            }
            object
                .values()
                .find_map(|child| find_manifest_record(child, path))
        }
        Value::Array(values) => values
            .iter()
            .find_map(|child| find_manifest_record(child, path)),
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => None,
    }
}

fn validate_runtime_identity(
    receipt: &RuntimeReceipt,
    options: &ReceiptOptions,
    binding: &ManifestBinding,
) -> Result<(), String> {
    if receipt.schema_version < 10 {
        return Err("canonical v0.4 runtime requires schema v10 provenance".to_owned());
    }
    validate_runtime_backend(receipt, &options.backend)?;
    if receipt.workload.engine.name != "leone" {
        return Err("runtime engine name is not leone".to_owned());
    }
    if receipt.workload.engine.git_commit != options.source_commit {
        return Err("runtime engine commit does not match the source manifest".to_owned());
    }
    validate_model_binding(binding, options, &receipt.workload.model_artifact.sha256)?;
    validate_build_flags(receipt, options)
}

fn validate_quality_identity(
    receipt: &QualityReceipt,
    options: &ReceiptOptions,
    binding: &ManifestBinding,
) -> Result<(), String> {
    if options.metric_family.as_deref() != Some("kld") {
        return Err("canonical quality requires the trusted kld metric family".to_owned());
    }
    if let Some(metric_family) = binding.metric_family.as_deref() {
        if Some(metric_family) != options.metric_family.as_deref() {
            return Err("quality metric family does not match the evidence record".to_owned());
        }
    }
    let execution = receipt
        .execution
        .as_ref()
        .ok_or_else(|| "canonical v0.4 quality requires execution provenance".to_owned())?;
    validate_quality_execution_identity(execution, options)?;
    validate_quality_metric_family(receipt)?;
    validate_quality_subject_identity(receipt, options)?;
    validate_model_binding(binding, options, &receipt.subject.model_artifact.sha256)
}

fn validate_quality_execution_identity(
    execution: &leone_receipt::QualityExecution,
    options: &ReceiptOptions,
) -> Result<(), String> {
    if execution.source_tree_dirty
        || execution.backend != options.backend
        || execution.platform != options.platform
        || execution.target != options.target
        || execution.profile != "release"
        || execution.source_commit != options.source_commit
    {
        return Err("quality execution identity does not match the evidence record".to_owned());
    }
    Ok(())
}

fn validate_quality_metric_family(receipt: &QualityReceipt) -> Result<(), String> {
    if receipt.metrics.is_none() || receipt.batch_invariance.is_some() {
        return Err("canonical quality requires KLD metrics".to_owned());
    }
    Ok(())
}

fn validate_quality_subject_identity(
    receipt: &QualityReceipt,
    options: &ReceiptOptions,
) -> Result<(), String> {
    let expected_engine = format!("leone-{}-eval", options.backend);
    if receipt.subject.engine.name != expected_engine {
        return Err("quality subject engine does not match the evidence backend".to_owned());
    }
    if receipt.subject.engine.git_commit != options.source_commit {
        return Err("quality engine commit does not match the source manifest".to_owned());
    }
    Ok(())
}

fn validate_runtime_backend(receipt: &RuntimeReceipt, expected: &str) -> Result<(), String> {
    let Some(active_compute) = receipt.machine.active_compute.as_ref() else {
        return Err("runtime active backend is unavailable".to_owned());
    };
    let actual = match active_compute {
        ActiveCompute::Cpu { .. } => "cpu",
        ActiveCompute::Cuda { .. } => "cuda",
        ActiveCompute::Metal { .. } => "metal",
    };
    if actual == expected {
        Ok(())
    } else {
        Err("runtime active backend does not match the evidence record".to_owned())
    }
}

fn validate_build_flags(receipt: &RuntimeReceipt, options: &ReceiptOptions) -> Result<(), String> {
    let flags = &receipt.workload.engine.build_flags;
    validate_unique_build_flag(
        flags,
        "target",
        Some(&options.target),
        receipt.schema_version,
    )?;
    validate_unique_build_flag(flags, "profile", Some("release"), receipt.schema_version)?;
    validate_unique_build_flag(
        flags,
        "source_tree_dirty",
        Some("false"),
        receipt.schema_version,
    )?;
    validate_unique_build_flag(
        flags,
        "backend",
        Some(&options.backend),
        receipt.schema_version,
    )?;
    if receipt.schema_version >= 10 && receipt.workload.engine.name == "leone" {
        for name in ["target", "profile", "source_tree_dirty", "backend"] {
            if build_flag_values(flags, name).is_empty() {
                return Err(format!(
                    "runtime build flag {name} is required in schema v10"
                ));
            }
        }
    }
    Ok(())
}

fn validate_unique_build_flag(
    flags: &[String],
    name: &str,
    expected: Option<&str>,
    schema_version: u32,
) -> Result<(), String> {
    let values = build_flag_values(flags, name);
    if values.len() > 1 {
        return Err(format!("runtime build flag {name} is duplicated"));
    }
    if let (Some(value), Some(expected)) = (values.first(), expected) {
        if *value != expected {
            return Err(format!(
                "runtime build flag {name} does not match the evidence record"
            ));
        }
    }
    if schema_version >= 10 && values.iter().any(|value| value.is_empty()) {
        return Err(format!("runtime build flag {name} is malformed"));
    }
    Ok(())
}

fn build_flag_values<'a>(flags: &'a [String], name: &str) -> Vec<&'a str> {
    flags
        .iter()
        .filter_map(|flag| flag.strip_prefix(name)?.strip_prefix('='))
        .collect()
}

fn validate_model_binding(
    binding: &ManifestBinding,
    options: &ReceiptOptions,
    actual_sha256: &str,
) -> Result<(), String> {
    if let (Some(record_family), Some(option_family)) = (
        binding.model_family.as_deref(),
        options.model_family.as_deref(),
    ) {
        if record_family != option_family {
            return Err("model family does not match the evidence record".to_owned());
        }
    }
    if !is_sha256(actual_sha256) || !actual_sha256.eq_ignore_ascii_case(&options.model_sha256) {
        return Err("receipt model SHA-256 does not match the trusted model digest".to_owned());
    }
    Ok(())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn validate_package_identity(platform: &str, target: &str, backend: &str) -> Result<(), String> {
    let expected = match backend {
        "cuda" => ("linux-x86_64", "x86_64-unknown-linux-gnu"),
        "metal" => ("darwin-arm64", "aarch64-apple-darwin"),
        "cpu" if platform == "linux-x86_64" => (platform, "x86_64-unknown-linux-gnu"),
        "cpu" if platform == "darwin-arm64" => (platform, "aarch64-apple-darwin"),
        _ => return Err(format!("unsupported evidence backend: {backend}")),
    };
    if (platform, target) == expected {
        Ok(())
    } else {
        Err("evidence platform, target, and backend do not agree".to_owned())
    }
}

fn is_full_commit(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn io_error(error: std::io::Error) -> String {
    error.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn schema9_cuda_has_no_canonical_backend_binding() {
        let receipt = RuntimeReceipt::from_json(include_bytes!(
            "../../leone-receipt/fixtures/runtime-v9-v030.json"
        ))
        .unwrap();
        assert!(validate_runtime_backend(&receipt, "cuda").is_err());
        assert!(validate_runtime_backend(&receipt, "cpu").is_err());
    }

    #[test]
    fn schema10_duplicate_build_flags_are_rejected() {
        let flags = vec![
            "target=x86_64-unknown-linux-gnu".to_owned(),
            "target=wrong-target".to_owned(),
        ];
        assert!(
            validate_unique_build_flag(&flags, "target", Some("x86_64-unknown-linux-gnu"), 10)
                .is_err()
        );
    }

    #[test]
    fn schema10_build_flags_require_target_profile_and_clean_tree() {
        let mut receipt = RuntimeReceipt::from_json(include_bytes!(
            "../../leone-receipt/fixtures/runtime-v9-v030.json"
        ))
        .unwrap();
        receipt.schema_version = 10;
        receipt.workload.engine.name = "leone".to_owned();
        let options = ReceiptOptions {
            root: PathBuf::new(),
            manifest: PathBuf::new(),
            source_manifest: PathBuf::new(),
            source_commit: "0".repeat(40),
            model_sha256: "a".repeat(64),
            record: PathBuf::new(),
            kind: ReceiptKind::Runtime,
            platform: "linux-x86_64".to_owned(),
            target: "x86_64-unknown-linux-gnu".to_owned(),
            backend: "cuda".to_owned(),
            model_family: Some("qwen3".to_owned()),
            metric_family: None,
        };
        assert!(validate_build_flags(&receipt, &options).is_err());
    }

    #[test]
    fn source_manifest_requires_actual_file_records() {
        let value = json!({
            "source_commit": "a".repeat(40),
            "files": {"crates/main.rs": {"executable": false}}
        });
        assert!(validate_source_manifest(&value, &"a".repeat(40)).is_err());
    }

    #[test]
    fn cpu_package_identity_rejects_unlisted_tuples() {
        assert!(validate_package_identity("invented-platform", "invented-target", "cpu").is_err());
        assert!(
            validate_package_identity("linux-x86_64", "x86_64-unknown-linux-gnu", "cpu").is_ok()
        );
    }

    #[test]
    fn canonical_metric_family_is_required_for_quality() {
        assert_eq!(
            validate_metric_family(ReceiptKind::Quality, Some("kld".to_owned())).unwrap(),
            Some("kld".to_owned())
        );
        assert!(validate_metric_family(ReceiptKind::Quality, None).is_err());
        assert!(validate_metric_family(ReceiptKind::Quality, Some("top1".to_owned())).is_err());
        assert!(validate_metric_family(ReceiptKind::Runtime, Some("kld".to_owned())).is_err());
    }

    #[test]
    fn model_digest_comes_from_the_trusted_release_binding() {
        let digest = "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785";
        let options = ReceiptOptions {
            root: PathBuf::new(),
            manifest: PathBuf::new(),
            source_manifest: PathBuf::new(),
            source_commit: "a".repeat(40),
            model_sha256: digest.to_owned(),
            record: PathBuf::new(),
            kind: ReceiptKind::Runtime,
            platform: "linux-x86_64".to_owned(),
            target: "x86_64-unknown-linux-gnu".to_owned(),
            backend: "cuda".to_owned(),
            model_family: Some("qwen3".to_owned()),
            metric_family: None,
        };
        let binding = ManifestBinding {
            model_family: Some("qwen3".to_owned()),
            metric_family: None,
        };
        assert!(validate_model_binding(&binding, &options, digest).is_ok());
        assert!(validate_model_binding(&binding, &options, &"0".repeat(64)).is_err());
    }

    #[test]
    fn schema3_quality_is_not_canonical_execution_provenance() {
        let receipt: QualityReceipt =
            serde_json::from_slice(include_bytes!("../../../receipts/quality-qwen3-8b.json"))
                .unwrap();
        let options = ReceiptOptions {
            root: PathBuf::new(),
            manifest: PathBuf::new(),
            source_manifest: PathBuf::new(),
            source_commit: receipt.subject.engine.git_commit.clone(),
            model_sha256: receipt.subject.model_artifact.sha256.clone(),
            record: PathBuf::new(),
            kind: ReceiptKind::Quality,
            platform: "linux-x86_64".to_owned(),
            target: "x86_64-unknown-linux-gnu".to_owned(),
            backend: "cuda".to_owned(),
            model_family: Some("qwen3".to_owned()),
            metric_family: Some("kld".to_owned()),
        };
        let binding = ManifestBinding {
            model_family: Some("qwen3".to_owned()),
            metric_family: Some("kld".to_owned()),
        };
        assert!(validate_quality_identity(&receipt, &options, &binding).is_err());
    }

    #[test]
    fn canonical_quality_rejects_dirty_execution() {
        let mut receipt: QualityReceipt =
            serde_json::from_slice(include_bytes!("../../../receipts/quality-qwen3-8b.json"))
                .unwrap();
        receipt.schema_version = 4;
        receipt.subject.engine.name = "leone-cuda-eval".to_owned();
        receipt.subject.engine.git_commit = "0".repeat(40);
        receipt.execution = Some(leone_receipt::QualityExecution {
            backend: "cuda".to_owned(),
            platform: "linux-x86_64".to_owned(),
            target: "x86_64-unknown-linux-gnu".to_owned(),
            profile: "release".to_owned(),
            source_tree_dirty: false,
            source_commit: "0".repeat(40),
        });
        let options = ReceiptOptions {
            root: PathBuf::new(),
            manifest: PathBuf::new(),
            source_manifest: PathBuf::new(),
            source_commit: "0".repeat(40),
            model_sha256: receipt.subject.model_artifact.sha256.clone(),
            record: PathBuf::new(),
            kind: ReceiptKind::Quality,
            platform: "linux-x86_64".to_owned(),
            target: "x86_64-unknown-linux-gnu".to_owned(),
            backend: "cuda".to_owned(),
            model_family: Some("qwen3".to_owned()),
            metric_family: Some("kld".to_owned()),
        };
        let mut binding = ManifestBinding {
            model_family: Some("qwen3".to_owned()),
            metric_family: Some("kld".to_owned()),
        };
        assert!(validate_quality_identity(&receipt, &options, &binding).is_ok());
        binding.metric_family = Some("top1".to_owned());
        assert!(validate_quality_identity(&receipt, &options, &binding).is_err());
        binding.metric_family = Some("kld".to_owned());
        receipt.execution.as_mut().unwrap().source_commit = "1".repeat(40);
        assert!(validate_quality_identity(&receipt, &options, &binding).is_err());
        receipt.execution.as_mut().unwrap().source_commit = "0".repeat(40);
        receipt.execution.as_mut().unwrap().source_tree_dirty = true;
        assert!(validate_quality_identity(&receipt, &options, &binding).is_err());
    }
}
