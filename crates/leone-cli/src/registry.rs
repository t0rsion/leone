use leone_receipt::{sha256_bytes, sha256_file};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::error::Error;
use std::ffi::OsStr;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const BUILTIN_REGISTRY: &str = include_str!("../assets/models.toml");

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Registry {
    schema_version: u32,
    models: Vec<ModelEntry>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelEntry {
    id: String,
    aliases: Vec<String>,
    artifact: String,
    url: String,
    sha256: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct PartialIdentity {
    url_sha256: String,
    sha256: String,
}

struct PullLock {
    _file: File,
}

impl Drop for PullLock {
    fn drop(&mut self) {
        // A concurrent child can retain the file description between fork and exec.
        let _ = self._file.unlock();
    }
}

enum LockMode {
    Shared,
    Exclusive,
}

pub(crate) struct VerifiedArtifact {
    path: PathBuf,
    sha256: String,
    _lease: PullLock,
}

impl VerifiedArtifact {
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    fn cli_path(&self) -> Result<&str, io::Error> {
        self.path
            .to_str()
            .ok_or_else(|| invalid_data("model cache path must contain valid UTF-8 for run"))
    }

    pub(crate) fn sha256(&self) -> &str {
        &self.sha256
    }
}

pub fn pull(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let (name, registry_path, rest) = parse_target(arguments)?;
    if !rest.is_empty() {
        return Err(invalid_data(format!("pull argument is invalid: {}", rest[0])).into());
    }
    let registry = load_registry(registry_path.as_deref())?;
    let model = registry.resolve(&name)?;
    let artifact = ensure_model(model)?;
    println!("ready: {}", artifact.path().display());
    Ok(())
}

pub fn run(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let (name, registry_path, mut rest) = parse_target(arguments)?;
    let serve = if let Some(index) = rest.iter().position(|argument| argument == "--serve") {
        rest.remove(index);
        true
    } else {
        false
    };
    reject_model_override(&rest)?;
    let registry = load_registry(registry_path.as_deref())?;
    let model = registry.resolve(&name)?;
    preflight_run(&rest, serve)?;
    let artifact = ensure_model(model)?;
    let mut resolved = vec!["-m".to_owned(), artifact.cli_path()?.to_owned()];
    resolved.extend(rest);
    if serve {
        crate::server::run_verified(&resolved, &artifact)
    } else {
        crate::chat::run(&resolved)
    }
}

fn preflight_run(arguments: &[String], serve: bool) -> Result<(), Box<dyn Error>> {
    let mut resolved = vec!["-m".to_owned(), "leone-run-preflight-model.gguf".to_owned()];
    resolved.extend(arguments.iter().cloned());
    if serve {
        crate::server::preflight(&resolved)
    } else {
        crate::chat::preflight(&resolved)
    }
}

pub fn list(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let registry_path = parse_registry_only(arguments)?;
    let registry = load_registry(registry_path.as_deref())?;
    for model in registry.models {
        println!("{}\t{}\t{}", model.id, model.artifact, model.sha256);
    }
    Ok(())
}

impl Registry {
    fn parse(source: &str) -> Result<Self, io::Error> {
        let registry: Self = toml::from_str(source)
            .map_err(|error| invalid_data(format!("model registry is invalid: {error}")))?;
        registry.validate()?;
        Ok(registry)
    }

    fn validate(&self) -> Result<(), io::Error> {
        validate_registry_header(self)?;
        let mut names = BTreeSet::new();
        self.models
            .iter()
            .try_for_each(|model| validate_model(model, &mut names))
    }

    fn resolve(&self, name: &str) -> Result<&ModelEntry, io::Error> {
        self.models
            .iter()
            .find(|model| model.id == name || model.aliases.iter().any(|alias| alias == name))
            .ok_or_else(|| invalid_data(format!("model {name:?} is not in the selected registry")))
    }
}

fn validate_registry_header(registry: &Registry) -> Result<(), io::Error> {
    if registry.schema_version != 1 {
        return Err(invalid_data(format!(
            "model registry schema {} is unsupported",
            registry.schema_version
        )));
    }
    if registry.models.is_empty() {
        return Err(invalid_data(
            "model registry must contain at least one model",
        ));
    }
    Ok(())
}

fn validate_model(model: &ModelEntry, names: &mut BTreeSet<String>) -> Result<(), io::Error> {
    if model.id.is_empty() || !names.insert(model.id.clone()) {
        return Err(invalid_data(
            "model registry identifiers must be nonempty and unique",
        ));
    }
    for alias in &model.aliases {
        if alias.is_empty() || !names.insert(alias.clone()) {
            return Err(invalid_data(
                "model registry aliases must be nonempty and unique",
            ));
        }
    }
    validate_model_artifact(model)?;
    validate_model_url(model)
}

fn validate_model_artifact(model: &ModelEntry) -> Result<(), io::Error> {
    let artifact = Path::new(&model.artifact);
    if artifact.file_name() != Some(OsStr::new(&model.artifact)) {
        return Err(invalid_data("model artifact must be one file name"));
    }
    if !valid_sha256(&model.sha256) {
        return Err(invalid_data(
            "model SHA-256 must be 64 lowercase hexadecimal digits",
        ));
    }
    Ok(())
}

fn validate_model_url(model: &ModelEntry) -> Result<(), io::Error> {
    if !model.url.starts_with("https://")
        && !model.url.starts_with("http://")
        && !model.url.starts_with("file://")
    {
        return Err(invalid_data("model URL must use https, http, or file"));
    }
    Ok(())
}

fn load_registry(path: Option<&Path>) -> Result<Registry, io::Error> {
    match path {
        Some(path) => Registry::parse(&fs::read_to_string(path)?),
        None => Registry::parse(BUILTIN_REGISTRY),
    }
}

fn ensure_model(model: &ModelEntry) -> Result<VerifiedArtifact, Box<dyn Error>> {
    let paths = prepare_model_paths(model)?;
    ensure_locked_model_at_lock(model, &paths.destination, &paths.lock)
}

struct ModelPaths {
    destination: PathBuf,
    lock: PathBuf,
}

fn prepare_model_paths(model: &ModelEntry) -> Result<ModelPaths, io::Error> {
    let root = data_home()?;
    fs::create_dir_all(&root)?;
    let namespace = root.join("models-v04");
    ensure_directory(&namespace, "model cache namespace")?;
    let lock_namespace = namespace.join(".locks");
    ensure_directory(&lock_namespace, "model lock namespace")?;
    let directory = namespace.join(&model.sha256);
    ensure_directory(&directory, "model cache digest directory")?;
    Ok(ModelPaths {
        destination: model_cache_path(&root, model),
        lock: digest_lock_path(&namespace, model),
    })
}

fn model_cache_path(root: &Path, model: &ModelEntry) -> PathBuf {
    root.join("models-v04")
        .join(&model.sha256)
        .join(&model.artifact)
}

fn digest_lock_path(namespace: &Path, model: &ModelEntry) -> PathBuf {
    namespace
        .join(".locks")
        .join(format!("{}.lock", model.sha256))
}

fn ensure_directory(path: &Path, label: &str) -> Result<(), io::Error> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            Err(invalid_data(format!("{label} must not be a symlink")))
        }
        Ok(metadata) if !metadata.is_dir() => {
            Err(invalid_data(format!("{label} must be a directory")))
        }
        Ok(_) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            fs::create_dir(path).map_err(|error| {
                invalid_data(format!("cannot create {label} {}: {error}", path.display()))
            })
        }
        Err(error) => Err(error),
    }
}

#[cfg(test)]
fn ensure_locked_model(
    model: &ModelEntry,
    destination: &Path,
) -> Result<VerifiedArtifact, Box<dyn Error>> {
    let namespace = destination
        .parent()
        .ok_or_else(|| invalid_data("model destination has no parent"))?;
    let lock_namespace = namespace.join(".locks");
    ensure_directory(&lock_namespace, "model lock namespace")?;
    let lock_path = digest_lock_path(namespace, model);
    ensure_locked_model_at_lock(model, destination, &lock_path)
}

fn ensure_locked_model_at_lock(
    model: &ModelEntry,
    destination: &Path,
    lock_path: &Path,
) -> Result<VerifiedArtifact, Box<dyn Error>> {
    if let Some(artifact) = verified_destination_with_shared_lock(model, destination, lock_path)? {
        return Ok(artifact);
    }
    ensure_model_with_exclusive_lock(model, destination, lock_path)
}

fn ensure_model_with_exclusive_lock(
    model: &ModelEntry,
    destination: &Path,
    lock_path: &Path,
) -> Result<VerifiedArtifact, Box<dyn Error>> {
    let lock = acquire_lock(lock_path, LockMode::Exclusive)?;
    if use_verified_destination(model, destination)?.is_some() {
        drop(lock);
        return verified_destination_with_shared_lock(model, destination, lock_path)?
            .ok_or_else(|| invalid_data("model artifact changed during verification").into());
    }
    let partial = destination.with_extension(format!("{}.part", extension(destination)));
    ensure_partial(model, &partial)?;
    install_verified_partial(model, &partial, destination)?;
    drop(lock);
    verified_destination_with_shared_lock(model, destination, lock_path)?
        .ok_or_else(|| invalid_data("model artifact changed after installation").into())
}

fn verified_destination_with_shared_lock(
    model: &ModelEntry,
    destination: &Path,
    lock_path: &Path,
) -> Result<Option<VerifiedArtifact>, Box<dyn Error>> {
    let lock = acquire_lock(lock_path, LockMode::Shared)?;
    let Some(path) = verified_destination(model, destination)? else {
        return Ok(None);
    };
    print_verified(model, &path);
    Ok(Some(VerifiedArtifact {
        path,
        sha256: model.sha256.clone(),
        _lease: lock,
    }))
}

fn verified_destination(
    model: &ModelEntry,
    destination: &Path,
) -> Result<Option<PathBuf>, Box<dyn Error>> {
    if !cache_file_state(destination, "model artifact")? {
        return Ok(None);
    }
    let actual = sha256_file(destination)?;
    if actual == model.sha256 {
        return Ok(Some(destination.to_owned()));
    }
    Ok(None)
}

fn use_verified_destination(
    model: &ModelEntry,
    destination: &Path,
) -> Result<Option<PathBuf>, Box<dyn Error>> {
    if !cache_file_state(destination, "model artifact")? {
        return Ok(None);
    }
    let actual = sha256_file(destination)?;
    if actual == model.sha256 {
        return Ok(Some(destination.to_owned()));
    }
    let quarantine = quarantine_path(destination, &actual);
    fs::rename(destination, &quarantine)?;
    eprintln!(
        "quarantined: {} (found sha256 {})",
        quarantine.display(),
        actual
    );
    Ok(None)
}

fn is_verified(model: &ModelEntry, path: &Path) -> Result<bool, Box<dyn Error>> {
    Ok(sha256_file(path)? == model.sha256)
}

fn ensure_partial(model: &ModelEntry, partial: &Path) -> Result<(), Box<dyn Error>> {
    let partial_exists = cache_file_state(partial, "partial download")?;
    if partial_exists && partial_identity_matches(model, partial)? && is_verified(model, partial)? {
        return Ok(());
    }
    prepare_partial(model, partial)?;
    download_model(model, partial)?;
    verify_download(model, partial)
}

fn prepare_partial(model: &ModelEntry, partial: &Path) -> Result<(), io::Error> {
    let identity = partial_identity_path(partial);
    let kept = prepare_partial_file(model, partial)?;
    if !kept {
        remove_if_present(&identity)?;
    }
    let value = PartialIdentity {
        url_sha256: sha256_bytes(model.url.as_bytes()),
        sha256: model.sha256.clone(),
    };
    let bytes = serde_json::to_vec(&value)
        .map_err(|error| invalid_data(format!("cannot record partial identity: {error}")))?;
    write_private_file(&identity, &bytes, "partial identity")
}

fn prepare_partial_file(model: &ModelEntry, partial: &Path) -> Result<bool, io::Error> {
    if !cache_file_state(partial, "partial download")? {
        create_private_file(partial, "partial download")?;
        return Ok(false);
    }
    if partial_identity_matches(model, partial)? {
        return Ok(true);
    }
    remove_if_present(partial)?;
    create_private_file(partial, "partial download")?;
    Ok(false)
}

fn partial_identity_matches(model: &ModelEntry, partial: &Path) -> Result<bool, io::Error> {
    let identity = partial_identity_path(partial);
    if !cache_file_state(&identity, "partial identity")? {
        return Ok(false);
    }
    let bytes = fs::read(identity)?;
    let Ok(value) = serde_json::from_slice::<PartialIdentity>(&bytes) else {
        return Ok(false);
    };
    Ok(value.url_sha256 == sha256_bytes(model.url.as_bytes()) && value.sha256 == model.sha256)
}

fn partial_identity_path(partial: &Path) -> PathBuf {
    partial.with_extension(format!("{}.meta", extension(partial)))
}

fn remove_if_present(path: &Path) -> Result<(), io::Error> {
    if !cache_file_state(path, "cache entry")? {
        return Ok(());
    }
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

fn cache_file_state(path: &Path, label: &str) -> Result<bool, io::Error> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            Err(invalid_data(format!("{label} must not be a symlink")))
        }
        Ok(metadata) if !metadata.is_file() => {
            Err(invalid_data(format!("{label} must be a regular file")))
        }
        Ok(_) => Ok(true),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error),
    }
}

fn create_private_file(path: &Path, label: &str) -> Result<(), io::Error> {
    let mut options = OpenOptions::new();
    options.read(true).write(true).create_new(true);
    #[cfg(unix)]
    options.mode(0o600);
    options
        .open(path)
        .map(|_| ())
        .map_err(|error| invalid_data(format!("cannot create {label} {}: {error}", path.display())))
}

fn write_private_file(path: &Path, bytes: &[u8], label: &str) -> Result<(), io::Error> {
    if cache_file_state(path, label)? {
        let mut options = OpenOptions::new();
        options.write(true).truncate(true);
        #[cfg(unix)]
        options.mode(0o600);
        let mut file = options.open(path)?;
        file.write_all(bytes)
    } else {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        options.mode(0o600);
        let mut file = options.open(path)?;
        file.write_all(bytes)
    }
}

fn download_model(model: &ModelEntry, partial: &Path) -> Result<(), Box<dyn Error>> {
    cache_file_state(partial, "partial download")?;
    if let Some(source) = model.url.strip_prefix("file://") {
        report_download_start(&model.url, 0);
        let mut input = File::open(source)?;
        let mut output = OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(partial)?;
        io::copy(&mut input, &mut output)?;
        return Ok(());
    }
    let offset = fs::metadata(partial)?.len();
    let resumed = offset > 0;
    report_download_start(&model.url, offset);
    handle_resume_result(model, partial, resumed, run_curl(&model.url, partial))
}

fn report_download_start(url: &str, offset: u64) {
    let action = if offset == 0 {
        "downloading model"
    } else {
        "resuming model download"
    };
    eprintln!(
        "{action}: source {}, starting at {offset} bytes",
        redacted_url(url)
    );
}

fn handle_resume_result(
    model: &ModelEntry,
    partial: &Path,
    resumed: bool,
    result: Result<(), Box<dyn Error>>,
) -> Result<(), Box<dyn Error>> {
    if !resumed {
        return result;
    }
    match result {
        Ok(()) if is_verified(model, partial)? => Ok(()),
        Ok(()) => retry_empty_download(model, partial),
        Err(error) => retry_empty_download(model, partial).map_err(|retry| {
            let details = format!("{error}; retry: {retry}");
            invalid_data(format!(
                "download failed after resume retry: {}",
                bound_download_diagnostic(&details)
            ))
            .into()
        }),
    }
}

fn retry_empty_download(model: &ModelEntry, partial: &Path) -> Result<(), Box<dyn Error>> {
    remove_if_present(partial)?;
    create_private_file(partial, "partial download")?;
    report_download_start(&model.url, 0);
    run_curl(&model.url, partial)
}

fn run_curl(url: &str, partial: &Path) -> Result<(), Box<dyn Error>> {
    let output = Command::new("curl")
        .args([
            "--fail",
            "--location",
            "--retry",
            "3",
            "--silent",
            "--continue-at",
            "-",
            "--output",
        ])
        .arg(partial)
        .args(["--write-out", "%{http_code}"])
        .arg(url)
        .stderr(Stdio::null())
        .output()
        .map_err(|error| {
            let detail = bound_download_diagnostic(&error.to_string());
            invalid_data(format!(
                "download failed: source {}; could not start curl: {}",
                redacted_url(url),
                detail
            ))
        })?;
    if !output.status.success() {
        let diagnostic = curl_failure_diagnostic(output.status.code(), &output.stdout);
        return Err(invalid_data(format!(
            "download failed: source {}; curl {}; {diagnostic}",
            redacted_url(url),
            output.status
        ))
        .into());
    }
    Ok(())
}

const MAX_DOWNLOAD_DIAGNOSTIC_BYTES: usize = 512;

fn bound_download_diagnostic(text: &str) -> String {
    let mut diagnostic = String::new();
    for character in text.chars() {
        if diagnostic.len() + character.len_utf8() > MAX_DOWNLOAD_DIAGNOSTIC_BYTES {
            diagnostic.push_str("...");
            break;
        }
        diagnostic.push(character);
    }
    diagnostic
}

fn curl_failure_diagnostic(exit_code: Option<i32>, stdout: &[u8]) -> String {
    let category = curl_failure_category(exit_code);
    match parse_http_status(stdout) {
        Some(status) => format!("{category} with HTTP status {status}"),
        None => category,
    }
}

fn curl_failure_category(exit_code: Option<i32>) -> String {
    match exit_code {
        Some(6) => "could not resolve the source host",
        Some(7) => "could not connect to the source",
        Some(22) => "the HTTP request failed",
        Some(23 | 26) => "could not read or write the model file",
        Some(28) => "the download timed out",
        Some(35 | 51 | 60) => "the TLS connection failed",
        Some(code) => return format!("curl exit code {code}"),
        None => "curl terminated without an exit code",
    }
    .to_owned()
}

fn parse_http_status(stdout: &[u8]) -> Option<u16> {
    std::str::from_utf8(stdout)
        .ok()?
        .trim()
        .parse::<u16>()
        .ok()
        .filter(|status| (100..=599).contains(status))
}

fn verify_download(model: &ModelEntry, partial: &Path) -> Result<(), Box<dyn Error>> {
    let actual = sha256_file(partial)?;
    if actual == model.sha256 {
        return Ok(());
    }
    let quarantine = quarantine_path(partial, &actual);
    fs::rename(partial, &quarantine)?;
    remove_if_present(&partial_identity_path(partial))?;
    Err(invalid_data(format!(
        "downloaded model has sha256 {actual}, expected {}. The invalid file is {}",
        model.sha256,
        quarantine.display()
    ))
    .into())
}

fn install_verified_partial(
    model: &ModelEntry,
    partial: &Path,
    destination: &Path,
) -> Result<(), Box<dyn Error>> {
    let rename_error = match fs::rename(partial, destination) {
        Ok(()) => return remove_partial_identity(partial),
        Err(error) => error,
    };
    recover_install_error(model, partial, destination, rename_error)
}

fn recover_install_error(
    model: &ModelEntry,
    partial: &Path,
    destination: &Path,
    rename_error: io::Error,
) -> Result<(), Box<dyn Error>> {
    if !cache_file_state(destination, "model artifact")? {
        return Err(rename_error.into());
    }
    let actual = sha256_file(destination)?;
    if actual == model.sha256 {
        return remove_partial_files(partial);
    }
    let quarantine = quarantine_path(destination, &actual);
    fs::rename(destination, &quarantine)?;
    fs::rename(partial, destination).map_err(|error| {
        invalid_data(format!(
            "cannot install verified model at {} after rename failed ({rename_error}): {error}",
            destination.display()
        ))
    })?;
    remove_partial_identity(partial)
}

fn remove_partial_identity(partial: &Path) -> Result<(), Box<dyn Error>> {
    remove_if_present(&partial_identity_path(partial))?;
    Ok(())
}

fn remove_partial_files(partial: &Path) -> Result<(), Box<dyn Error>> {
    remove_if_present(partial)?;
    remove_partial_identity(partial)
}

fn print_verified(model: &ModelEntry, path: &Path) {
    println!("model: {}", model.id);
    println!("artifact: {}", path.display());
    println!("sha256: {}", model.sha256);
    println!("source: {}", redacted_url(&model.url));
}

fn redacted_url(url: &str) -> String {
    let without_fragment = url.split('#').next().unwrap_or(url);
    let without_query = without_fragment
        .split('?')
        .next()
        .unwrap_or(without_fragment);
    let Some(scheme_end) = without_query.find("://") else {
        return without_query.to_owned();
    };
    let authority_start = scheme_end + 3;
    let rest = &without_query[authority_start..];
    let path_start = rest.find('/').unwrap_or(rest.len());
    let authority = &rest[..path_start];
    let host = authority
        .rsplit_once('@')
        .map_or(authority, |(_, host)| host);
    format!(
        "{}://{}{}",
        &without_query[..scheme_end],
        host,
        &rest[path_start..]
    )
}

fn acquire_lock(path: &Path, mode: LockMode) -> Result<PullLock, io::Error> {
    cache_file_state(path, "lock file")?;
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)?;
    let result = match mode {
        LockMode::Shared => file.try_lock_shared(),
        LockMode::Exclusive => file.try_lock(),
    };
    result.map_err(|error| match error {
        std::fs::TryLockError::WouldBlock => {
            invalid_data(format!("another pull holds {}", path.display()))
        }
        std::fs::TryLockError::Error(error) => error,
    })?;
    Ok(PullLock { _file: file })
}

fn data_home() -> Result<PathBuf, io::Error> {
    if let Some(path) = std::env::var_os("LEONE_HOME") {
        return nonempty_path(path, "LEONE_HOME");
    }
    if let Some(path) = std::env::var_os("XDG_DATA_HOME") {
        return Ok(nonempty_path(path, "XDG_DATA_HOME")?.join("leone"));
    }
    let home = std::env::var_os("HOME").ok_or_else(|| {
        invalid_data("HOME is not set; set LEONE_HOME to an explicit data directory")
    })?;
    Ok(nonempty_path(home, "HOME")?.join(".local/share/leone"))
}

fn nonempty_path(value: std::ffi::OsString, name: &str) -> Result<PathBuf, io::Error> {
    if value.is_empty() {
        return Err(invalid_data(format!("{name} must not be empty")));
    }
    Ok(PathBuf::from(value))
}

fn parse_target(arguments: &[String]) -> Result<(String, Option<PathBuf>, Vec<String>), io::Error> {
    let name = arguments
        .first()
        .filter(|value| !value.starts_with('-'))
        .ok_or_else(|| invalid_data("a model name is required"))?
        .clone();
    let mut registry = None;
    let mut rest = Vec::new();
    let mut index = 1;
    while index < arguments.len() {
        if arguments[index] == "--registry" {
            index += 1;
            let path = arguments
                .get(index)
                .ok_or_else(|| invalid_data("--registry needs a TOML path"))?;
            registry = Some(PathBuf::from(path));
        } else {
            rest.push(arguments[index].clone());
        }
        index += 1;
    }
    Ok((name, registry, rest))
}

fn parse_registry_only(arguments: &[String]) -> Result<Option<PathBuf>, io::Error> {
    if arguments.is_empty() {
        return Ok(None);
    }
    if arguments.len() == 2 && arguments[0] == "--registry" {
        return Ok(Some(PathBuf::from(&arguments[1])));
    }
    Err(invalid_data("models accepts only --registry <toml>"))
}

fn reject_model_override(arguments: &[String]) -> Result<(), io::Error> {
    if arguments
        .iter()
        .any(|argument| matches!(argument.as_str(), "-m" | "--model"))
    {
        return Err(invalid_data("run supplies the verified model path"));
    }
    Ok(())
}

fn quarantine_path(path: &Path, digest: &str) -> PathBuf {
    let file = path.file_name().and_then(OsStr::to_str).unwrap_or("model");
    let prefix = digest.get(..12).unwrap_or(digest);
    let mut candidate = path.with_file_name(format!("{file}.invalid-{prefix}"));
    let mut counter = 1_u64;
    while candidate.exists() {
        candidate = path.with_file_name(format!("{file}.invalid-{prefix}-{counter}"));
        counter = counter.saturating_add(1);
    }
    candidate
}

fn extension(path: &Path) -> &str {
    path.extension().and_then(OsStr::to_str).unwrap_or("model")
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use leone_receipt::sha256_bytes;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    #[cfg(unix)]
    use std::os::unix::fs::symlink;
    use std::process::{Command, Stdio};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;
    use tempfile::tempdir;

    #[test]
    fn builtin_registry_resolves_the_release_alias() {
        let registry = Registry::parse(BUILTIN_REGISTRY).unwrap();
        let model = registry.resolve("qwen3:8b").unwrap();
        assert_eq!(model.artifact, "Qwen3-8B-Q4_K_M.gguf");
        assert_eq!(
            model.sha256,
            "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"
        );
    }

    #[test]
    fn duplicate_aliases_are_rejected() {
        let source = r#"
schema_version = 1
[[models]]
id = "one"
aliases = ["same"]
artifact = "one.gguf"
url = "file:///one"
sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
[[models]]
id = "two"
aliases = ["same"]
artifact = "two.gguf"
url = "file:///two"
sha256 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
"#;
        assert!(Registry::parse(source).is_err());
    }

    #[test]
    fn run_rejects_a_model_path_override() {
        assert!(reject_model_override(&["--model".to_owned(), "other.gguf".to_owned()]).is_err());
        assert!(reject_model_override(&["--backend".to_owned(), "cpu".to_owned()]).is_ok());
    }

    #[test]
    fn run_rejects_invalid_options_before_model_download() {
        let directory = tempdir().unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let address = listener.local_addr().unwrap();
        let request_count = Arc::new(AtomicUsize::new(0));
        let stop_server = Arc::new(AtomicBool::new(false));
        let server_requests = Arc::clone(&request_count);
        let server_stop = Arc::clone(&stop_server);
        let server = thread::spawn(move || {
            while !server_stop.load(Ordering::Acquire) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        server_requests.fetch_add(1, Ordering::Relaxed);
                        let _ = stream.set_read_timeout(Some(Duration::from_millis(100)));
                        let mut request = [0_u8; 4096];
                        let _ = stream.read(&mut request);
                        let _ = stream.write_all(
                            b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                        );
                    }
                    Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(1));
                    }
                    Err(_) => break,
                }
            }
        });
        let registry = directory.path().join("registry.toml");
        fs::write(
            &registry,
            format!(
                "schema_version = 1\n[[models]]\nid = \"test\"\naliases = []\nartifact = \"model.gguf\"\nurl = \"http://{address}/model.gguf?token=private\"\nsha256 = \"{}\"\n",
                "a".repeat(64)
            ),
        )
        .unwrap();
        let registry = registry.to_string_lossy().into_owned();
        let mut cases = vec![
            (
                vec![
                    "test".to_owned(),
                    "--registry".to_owned(),
                    registry.clone(),
                    "--tokens".to_owned(),
                    "bad".to_owned(),
                ],
                "chat token count is invalid",
            ),
            (
                vec![
                    "test".to_owned(),
                    "--registry".to_owned(),
                    registry.clone(),
                    "--backend".to_owned(),
                    "unknown".to_owned(),
                ],
                "backend is invalid",
            ),
            (
                vec![
                    "test".to_owned(),
                    "--registry".to_owned(),
                    registry.clone(),
                    "--serve".to_owned(),
                    "--bind".to_owned(),
                    "bad".to_owned(),
                ],
                "bind address is invalid",
            ),
        ];
        let (missing_backend, expected) = if cfg!(feature = "metal") {
            ("cuda", "the cuda backend is unavailable")
        } else {
            ("metal", "the metal backend is unavailable")
        };
        if !cfg!(all(feature = "cuda", feature = "metal")) {
            cases.push((
                vec![
                    "test".to_owned(),
                    "--registry".to_owned(),
                    registry.clone(),
                    "--backend".to_owned(),
                    missing_backend.to_owned(),
                ],
                expected,
            ));
        }

        let payload = serde_json::to_string(&cases).unwrap();
        let child = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "registry::tests::preflight_child", "--nocapture"])
            .env("LEONE_HOME", directory.path())
            .env("LEONE_PREFLIGHT_CASES", payload)
            .output()
            .unwrap();
        stop_server.store(true, Ordering::Release);
        server.join().unwrap();
        assert!(
            child.status.success(),
            "{}",
            String::from_utf8_lossy(&child.stderr)
        );
        assert_eq!(request_count.load(Ordering::Acquire), 0);
    }

    #[test]
    fn preflight_child() {
        let Some(payload) = std::env::var_os("LEONE_PREFLIGHT_CASES") else {
            return;
        };
        let cases: Vec<(Vec<String>, String)> =
            serde_json::from_str(payload.to_string_lossy().as_ref()).unwrap();
        for (arguments, expected) in cases {
            let error = run(&arguments).unwrap_err().to_string();
            assert!(error.contains(&expected), "{error}");
        }
    }

    #[test]
    fn download_diagnostic_keeps_category_and_is_bounded() {
        let diagnostic = curl_failure_diagnostic(Some(22), b"404");
        assert_eq!(diagnostic, "the HTTP request failed with HTTP status 404");
        let bounded = bound_download_diagnostic(&"x".repeat(800));
        assert!(bounded.len() <= MAX_DOWNLOAD_DIAGNOSTIC_BYTES + 3);
        assert!(bounded.ends_with("..."));
    }

    #[test]
    fn curl_failure_keeps_status_without_source_or_cache_path() {
        let directory = tempdir().unwrap();
        let partial = directory.path().join("model.gguf.part");
        fs::write(&partial, b"").unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 4096];
            let _ = stream.read(&mut request).unwrap();
            stream
                .write_all(
                    b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                )
                .unwrap();
        });
        let source =
            format!("http://user%3Asecret@{address}/model%2Egguf?token=private%26other#fragment");

        let error = run_curl(&source, &partial).unwrap_err().to_string();
        server.join().unwrap();

        assert!(error.contains("curl"), "{error}");
        assert!(error.contains("22"), "{error}");
        assert!(!error.contains("user%3Asecret"));
        assert!(!error.contains("token=private"));
        assert!(!error.contains("private%26other"));
        assert!(!error.contains("#fragment"));
        assert!(!error.contains(directory.path().to_str().unwrap()));
    }

    #[test]
    fn verified_server_rejects_a_different_model_path() {
        let directory = tempdir().unwrap();
        let source = directory.path().join("source.gguf");
        let destination = directory.path().join("model.gguf");
        let payload = b"server model bytes";
        fs::write(&source, payload).unwrap();
        let model = test_model("server", &source, payload);
        let artifact = ensure_locked_model(&model, &destination).unwrap();
        let arguments = vec![
            "-m".to_owned(),
            directory
                .path()
                .join("other.gguf")
                .to_string_lossy()
                .into_owned(),
        ];

        let error = crate::server::run_verified(&arguments, &artifact)
            .unwrap_err()
            .to_string();
        assert!(error.contains("verified model path does not match server model"));
    }

    #[test]
    fn advisory_lock_recovers_from_existing_lock_file() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("model.gguf.lock");
        fs::write(&path, b"lock left by an exited process").unwrap();

        let first = acquire_lock(&path, LockMode::Exclusive).unwrap();
        assert!(acquire_lock(&path, LockMode::Exclusive).is_err());
        drop(first);
        assert!(acquire_lock(&path, LockMode::Exclusive).is_ok());
        assert!(path.exists());
    }

    #[test]
    fn advisory_lock_release_does_not_wait_for_a_duplicated_descriptor() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("model.gguf.lock");
        let lease = acquire_lock(&path, LockMode::Exclusive).unwrap();
        let duplicate = lease._file.try_clone().unwrap();
        drop(lease);
        let next = acquire_lock(&path, LockMode::Exclusive).unwrap();
        drop(duplicate);
        assert!(acquire_lock(&path, LockMode::Exclusive).is_err());
        drop(next);
        assert!(acquire_lock(&path, LockMode::Exclusive).is_ok());
    }

    #[test]
    fn lock_is_released_after_the_owner_process_is_killed() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("model.gguf.lock");
        let executable = std::env::current_exe().unwrap();
        let mut child = Command::new(executable)
            .args(["--exact", "registry::tests::lock_child", "--nocapture"])
            .env("LEONE_LOCK_CHILD", &path)
            .stdout(Stdio::null())
            .spawn()
            .unwrap();
        let mut ready = false;
        let ready_path = path.with_extension("ready");
        for _ in 0..100 {
            if ready_path.exists() {
                ready = true;
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
        let held = acquire_lock(&path, LockMode::Exclusive).is_err();
        let _ = child.kill();
        let _ = child.wait();

        assert!(ready);
        assert!(held);
        assert!(acquire_lock(&path, LockMode::Exclusive).is_ok());
    }

    #[test]
    fn lock_child() {
        let Some(path) = std::env::var_os("LEONE_LOCK_CHILD") else {
            return;
        };
        let _lock = acquire_lock(Path::new(&path), LockMode::Exclusive).unwrap();
        fs::write(Path::new(&path).with_extension("ready"), b"ready").unwrap();
        loop {
            thread::sleep(Duration::from_secs(1));
        }
    }

    #[test]
    fn verified_partial_is_installed_without_source_access() {
        let directory = tempdir().unwrap();
        let payload = b"verified model payload";
        let destination = directory.path().join("model.gguf");
        let partial = destination.with_extension("gguf.part");
        let model = ModelEntry {
            id: "test".to_owned(),
            aliases: Vec::new(),
            artifact: "model.gguf".to_owned(),
            url: "file:///source-that-is-not-present".to_owned(),
            sha256: sha256_bytes(payload),
        };

        prepare_partial(&model, &partial).unwrap();
        fs::write(&partial, payload).unwrap();
        ensure_locked_model(&model, &destination).unwrap();

        assert_eq!(fs::read(&destination).unwrap(), payload);
        assert!(!partial.exists());
        assert!(!partial_identity_path(&partial).exists());
    }

    #[test]
    fn file_source_copy_does_not_alias_the_source() {
        let directory = tempdir().unwrap();
        let source = directory.path().join("source.gguf");
        let destination = directory.path().join("model.gguf");
        let original = b"original source bytes";
        let model = ModelEntry {
            id: "test".to_owned(),
            aliases: Vec::new(),
            artifact: "model.gguf".to_owned(),
            url: format!("file://{}", source.display()),
            sha256: sha256_bytes(original),
        };
        fs::write(&source, original).unwrap();

        let artifact = ensure_locked_model(&model, &destination).unwrap();
        fs::write(&source, b"mutated source bytes").unwrap();

        assert_eq!(fs::read(artifact.path()).unwrap(), original);
        assert_eq!(sha256_file(artifact.path()).unwrap(), model.sha256);
    }

    #[cfg(unix)]
    #[test]
    fn file_source_permissions_do_not_make_cached_model_public() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempdir().unwrap();
        let source = directory.path().join("source.gguf");
        let destination = directory.path().join("model.gguf");
        let payload = b"private cached model";
        fs::write(&source, payload).unwrap();
        fs::set_permissions(&source, fs::Permissions::from_mode(0o666)).unwrap();
        let model = test_model("private", &source, payload);

        let artifact = ensure_locked_model(&model, &destination).unwrap();
        let mode = fs::metadata(artifact.path()).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
        assert_eq!(fs::read(artifact.path()).unwrap(), payload);
    }

    #[cfg(unix)]
    #[test]
    fn non_utf8_verified_path_is_not_replaced_for_run() {
        use std::os::unix::ffi::OsStrExt;

        let directory = tempdir().unwrap();
        let source = directory.path().join("source.gguf");
        let destination = directory.path().join(OsStr::from_bytes(b"model-\xff.gguf"));
        let payload = b"exact path identity";
        fs::write(&source, payload).unwrap();
        let model = test_model("exact", &source, payload);

        let result = ensure_locked_model(&model, &destination);
        #[cfg(target_os = "macos")]
        if let Err(error) = &result {
            // APFS rejects a name that is not valid UTF-8 with EILSEQ (os error 92).
            let error = error.downcast_ref::<io::Error>().expect("filesystem error");
            assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
            let message = error.to_string();
            assert!(message.starts_with("cannot create partial download "));
            assert!(message.ends_with("(os error 92)"));
            let mut names: Vec<_> = fs::read_dir(directory.path())
                .unwrap()
                .map(|entry| entry.unwrap().file_name())
                .collect();
            names.sort();
            assert_eq!(names, [".locks", "source.gguf"]);
            return;
        }
        let artifact = result.unwrap();
        assert_eq!(artifact.path(), destination);
        assert!(artifact.cli_path().is_err());
    }

    #[test]
    fn verified_readers_share_the_cache_lock() {
        let directory = tempdir().unwrap();
        let source = directory.path().join("source.gguf");
        let destination = directory.path().join("model.gguf");
        let bytes = b"shared model bytes";
        fs::write(&source, bytes).unwrap();
        let model = test_model("test", &source, bytes);

        let first = ensure_locked_model(&model, &destination).unwrap();
        let second = ensure_locked_model(&model, &destination).unwrap();

        assert_eq!(first.path(), second.path());
        drop(second);
        drop(first);
    }

    #[test]
    fn repair_is_blocked_while_a_verified_reader_holds_the_shared_lock() {
        let directory = tempdir().unwrap();
        let source_a = directory.path().join("source-a.gguf");
        let source_b = directory.path().join("source-b.gguf");
        let destination = directory.path().join("model.gguf");
        let bytes_a = b"first model bytes";
        let bytes_b = b"second model bytes";
        fs::write(&source_a, bytes_a).unwrap();
        fs::write(&source_b, bytes_b).unwrap();
        let model_a = test_model("a", &source_a, bytes_a);
        let model_b = test_model("a", &source_b, bytes_b);
        let lock_path = directory.path().join("digest.lock");

        let artifact_a = ensure_locked_model_at_lock(&model_a, &destination, &lock_path).unwrap();
        assert!(ensure_locked_model_at_lock(&model_b, &destination, &lock_path).is_err());
        drop(artifact_a);
        let artifact_b = ensure_locked_model_at_lock(&model_b, &destination, &lock_path).unwrap();
        assert_eq!(fs::read(artifact_b.path()).unwrap(), bytes_b);
    }

    #[test]
    fn content_addressed_cache_paths_do_not_merge_ids() {
        let root = Path::new("cache");
        let source = Path::new("source");
        let first = test_model("a:b", source, b"first");
        let second = test_model("a/b", source, b"second");

        assert_ne!(
            model_cache_path(root, &first),
            model_cache_path(root, &second)
        );
        assert_eq!(
            model_cache_path(root, &first).parent().unwrap().file_name(),
            Some(OsStr::new(&first.sha256))
        );
    }

    #[test]
    fn digest_lock_does_not_collide_with_an_artifact_named_like_a_lock() {
        let directory = tempdir().unwrap();
        let root = directory.path().join("cache");
        let namespace = root.join("models-v04");
        let digest_dir = namespace.join(sha256_bytes(b"shared payload"));
        let source = directory.path().join("source.gguf");
        let bytes = b"shared payload";
        fs::create_dir_all(&digest_dir).unwrap();
        fs::create_dir_all(namespace.join(".locks")).unwrap();
        fs::write(&source, bytes).unwrap();
        let first = test_model_with_artifact("first", &source, "model.gguf", bytes);
        let second = test_model_with_artifact("second", &source, "model.gguf.lock", bytes);
        let first_destination = model_cache_path(&root, &first);
        let second_destination = model_cache_path(&root, &second);
        fs::write(&first_destination, bytes).unwrap();
        let lock_path = digest_lock_path(&namespace, &first);

        let reader = ensure_locked_model_at_lock(&first, &first_destination, &lock_path).unwrap();
        assert_eq!(
            lock_path,
            namespace
                .join(".locks")
                .join(format!("{}.lock", first.sha256))
        );
        assert!(ensure_locked_model_at_lock(&second, &second_destination, &lock_path).is_err());
        drop(reader);

        ensure_locked_model_at_lock(&second, &second_destination, &lock_path).unwrap();
        assert_eq!(fs::read(second_destination).unwrap(), bytes);
    }

    #[test]
    fn partial_identity_does_not_store_url_credentials() {
        let directory = tempdir().unwrap();
        let partial = directory.path().join("model.gguf.part");
        let userinfo = "user:credential";
        let host = "example.invalid";
        let query_value = "query-value";
        let model = ModelEntry {
            id: "test".to_owned(),
            aliases: Vec::new(),
            artifact: "model.gguf".to_owned(),
            url: format!("https://{userinfo}@{host}/model.gguf?access_key={query_value}"),
            sha256: sha256_bytes(b"model"),
        };

        prepare_partial(&model, &partial).unwrap();
        let metadata = fs::read_to_string(partial_identity_path(&partial)).unwrap();
        assert!(!metadata.contains(userinfo));
        assert!(!metadata.contains(query_value));
        assert_eq!(
            redacted_url(&model.url),
            "https://example.invalid/model.gguf"
        );
    }

    #[test]
    fn rejected_resume_retries_from_an_empty_partial() {
        let directory = tempdir().unwrap();
        let partial = directory.path().join("model.gguf.part");
        let payload = b"complete model bytes";
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let mut saw_resume = false;
            for _ in 0..8 {
                let Ok((mut stream, _)) = listener.accept() else {
                    break;
                };
                let mut request = [0_u8; 4096];
                let length = stream.read(&mut request).unwrap();
                let request = String::from_utf8_lossy(&request[..length]);
                if request.contains("Range: bytes=") {
                    saw_resume = true;
                    stream
                        .write_all(
                            b"HTTP/1.1 416 Range Not Satisfiable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                        )
                        .unwrap();
                } else {
                    let header = format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        payload.len()
                    );
                    stream.write_all(header.as_bytes()).unwrap();
                    stream.write_all(payload).unwrap();
                    break;
                }
            }
            saw_resume
        });
        let model = ModelEntry {
            id: "test".to_owned(),
            aliases: Vec::new(),
            artifact: "model.gguf".to_owned(),
            url: format!("http://{address}/model.gguf"),
            sha256: sha256_bytes(payload),
        };
        fs::write(&partial, b"partial").unwrap();

        download_model(&model, &partial).unwrap();
        verify_download(&model, &partial).unwrap();

        assert_eq!(fs::read(partial).unwrap(), payload);
        assert!(server.join().unwrap());
    }

    #[test]
    fn hash_mismatch_quarantines_partial() {
        let directory = tempdir().unwrap();
        let partial = directory.path().join("model.gguf.part");
        let model = ModelEntry {
            id: "test".to_owned(),
            aliases: Vec::new(),
            artifact: "model.gguf".to_owned(),
            url: "file:///source".to_owned(),
            sha256: sha256_bytes(b"expected"),
        };
        fs::write(&partial, b"wrong").unwrap();

        let error = verify_download(&model, &partial).unwrap_err().to_string();
        let actual = sha256_bytes(b"wrong");
        let quarantine = directory
            .path()
            .join(format!("model.gguf.part.invalid-{}", &actual[..12]));
        assert!(error.contains(&actual));
        assert!(quarantine.exists());
        assert!(!partial.exists());
    }

    #[test]
    fn invalid_destination_is_quarantined_before_install() {
        let directory = tempdir().unwrap();
        let source = directory.path().join("source.gguf");
        let destination = directory.path().join("model.gguf");
        let payload = b"verified source";
        fs::write(&source, payload).unwrap();
        fs::write(&destination, b"wrong destination").unwrap();
        let model = ModelEntry {
            id: "test".to_owned(),
            aliases: Vec::new(),
            artifact: "model.gguf".to_owned(),
            url: format!("file://{}", source.display()),
            sha256: sha256_bytes(payload),
        };

        ensure_locked_model(&model, &destination).unwrap();

        assert_eq!(fs::read(&destination).unwrap(), payload);
        let invalid = directory.path().join(format!(
            "model.gguf.invalid-{}",
            &sha256_bytes(b"wrong destination")[..12]
        ));
        assert_eq!(fs::read(invalid).unwrap(), b"wrong destination");
    }

    #[test]
    fn install_recovery_quarantines_an_invalid_existing_destination() {
        let directory = tempdir().unwrap();
        let destination = directory.path().join("model.gguf");
        let partial = destination.with_extension("gguf.part");
        let payload = b"verified partial";
        let invalid = b"invalid destination";
        let model = ModelEntry {
            id: "test".to_owned(),
            aliases: Vec::new(),
            artifact: "model.gguf".to_owned(),
            url: "file:///source".to_owned(),
            sha256: sha256_bytes(payload),
        };
        fs::write(&destination, invalid).unwrap();
        fs::write(&partial, payload).unwrap();
        fs::write(partial_identity_path(&partial), b"identity").unwrap();

        recover_install_error(
            &model,
            &partial,
            &destination,
            io::Error::new(io::ErrorKind::AlreadyExists, "destination exists"),
        )
        .unwrap();

        assert_eq!(fs::read(&destination).unwrap(), payload);
        let quarantine = directory.path().join(format!(
            "model.gguf.invalid-{}",
            &sha256_bytes(invalid)[..12]
        ));
        assert_eq!(fs::read(quarantine).unwrap(), invalid);
        assert!(!partial.exists());
        assert!(!partial_identity_path(&partial).exists());
    }

    #[test]
    fn stale_partial_identity_starts_a_new_download() {
        let directory = tempdir().unwrap();
        let source = directory.path().join("source.gguf");
        let destination = directory.path().join("model.gguf");
        let partial = destination.with_extension("gguf.part");
        let old = b"old model bytes";
        let current = b"current model bytes";
        fs::write(&source, current).unwrap();
        let old_model = test_model("test", &source, old);
        let current_model = test_model("test", &source, current);
        prepare_partial(&old_model, &partial).unwrap();
        fs::write(&partial, old).unwrap();

        ensure_locked_model(&current_model, &destination).unwrap();

        assert_eq!(fs::read(destination).unwrap(), current);
    }

    #[test]
    fn failed_source_does_not_publish_an_artifact() {
        let directory = tempdir().unwrap();
        let source = directory.path().join("source.gguf");
        let destination = directory.path().join("model.gguf");
        let payload = b"verified source";
        let model = ModelEntry {
            id: "test".to_owned(),
            aliases: Vec::new(),
            artifact: "model.gguf".to_owned(),
            url: format!("file://{}", source.display()),
            sha256: sha256_bytes(payload),
        };

        assert!(ensure_locked_model(&model, &destination).is_err());
        assert!(!destination.exists());
        fs::write(&source, payload).unwrap();
        ensure_locked_model(&model, &destination).unwrap();
        assert_eq!(fs::read(destination).unwrap(), payload);
    }

    fn test_model(id: &str, source: &Path, bytes: &[u8]) -> ModelEntry {
        test_model_with_artifact(id, source, "model.gguf", bytes)
    }

    fn test_model_with_artifact(
        id: &str,
        source: &Path,
        artifact: &str,
        bytes: &[u8],
    ) -> ModelEntry {
        ModelEntry {
            id: id.to_owned(),
            aliases: Vec::new(),
            artifact: artifact.to_owned(),
            url: format!("file://{}", source.display()),
            sha256: sha256_bytes(bytes),
        }
    }

    #[cfg(unix)]
    #[test]
    fn lock_symlink_is_rejected_without_touching_target() {
        let directory = tempdir().unwrap();
        let target = directory.path().join("outside");
        let link = directory.path().join("model.lock");
        fs::write(&target, b"outside").unwrap();
        symlink(&target, &link).unwrap();

        assert!(acquire_lock(&link, LockMode::Exclusive).is_err());
        assert_eq!(fs::read(target).unwrap(), b"outside");
    }

    #[cfg(unix)]
    #[test]
    fn destination_symlink_is_rejected_without_touching_target() {
        let directory = tempdir().unwrap();
        let target = directory.path().join("outside");
        let destination = directory.path().join("model.gguf");
        fs::write(&target, b"outside").unwrap();
        symlink(&target, &destination).unwrap();
        let model = test_model("test", &target, b"expected");

        assert!(ensure_locked_model(&model, &destination).is_err());
        assert_eq!(fs::read(target).unwrap(), b"outside");
    }

    #[cfg(unix)]
    #[test]
    fn partial_symlink_is_rejected_without_touching_target() {
        let directory = tempdir().unwrap();
        let target = directory.path().join("outside");
        let destination = directory.path().join("model.gguf");
        let partial = destination.with_extension("gguf.part");
        fs::write(&target, b"outside").unwrap();
        symlink(&target, &partial).unwrap();
        let model = test_model("test", &target, b"expected");

        assert!(ensure_locked_model(&model, &destination).is_err());
        assert_eq!(fs::read(target).unwrap(), b"outside");
    }

    #[cfg(unix)]
    #[test]
    fn partial_identity_symlink_is_rejected_without_touching_target() {
        let directory = tempdir().unwrap();
        let target = directory.path().join("outside");
        let destination = directory.path().join("model.gguf");
        let partial = destination.with_extension("gguf.part");
        let identity = partial_identity_path(&partial);
        fs::write(&target, b"outside").unwrap();
        fs::write(&partial, b"partial").unwrap();
        symlink(&target, &identity).unwrap();
        let model = test_model("test", &target, b"expected");

        assert!(ensure_locked_model(&model, &destination).is_err());
        assert_eq!(fs::read(target).unwrap(), b"outside");
    }
}
