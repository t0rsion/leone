use leone_receipt::sha256_file;
use serde::Deserialize;
use std::collections::BTreeSet;
use std::error::Error;
use std::ffi::OsStr;
use std::fs::{self, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

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

struct PullLock {
    path: PathBuf,
}

impl Drop for PullLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

pub fn pull(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let (name, registry_path, rest) = parse_target(arguments)?;
    if !rest.is_empty() {
        return Err(invalid_data(format!("pull argument is invalid: {}", rest[0])).into());
    }
    let registry = load_registry(registry_path.as_deref())?;
    let model = registry.resolve(&name)?;
    let path = ensure_model(model)?;
    println!("ready: {}", path.display());
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
    let registry = load_registry(registry_path.as_deref())?;
    let model = registry.resolve(&name)?;
    let path = ensure_model(model)?;
    let mut resolved = vec!["-m".to_owned(), path.to_string_lossy().into_owned()];
    resolved.extend(rest);
    if serve {
        crate::server::run(&resolved)
    } else {
        crate::chat::run(&resolved)
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
        if self.schema_version != 1 {
            return Err(invalid_data(format!(
                "model registry schema {} is unsupported",
                self.schema_version
            )));
        }
        if self.models.is_empty() {
            return Err(invalid_data(
                "model registry must contain at least one model",
            ));
        }
        let mut names = BTreeSet::new();
        for model in &self.models {
            if model.id.is_empty() || !names.insert(model.id.as_str()) {
                return Err(invalid_data(
                    "model registry identifiers must be nonempty and unique",
                ));
            }
            for alias in &model.aliases {
                if alias.is_empty() || !names.insert(alias.as_str()) {
                    return Err(invalid_data(
                        "model registry aliases must be nonempty and unique",
                    ));
                }
            }
            let artifact = Path::new(&model.artifact);
            if artifact.file_name() != Some(OsStr::new(&model.artifact)) {
                return Err(invalid_data("model artifact must be one file name"));
            }
            if !valid_sha256(&model.sha256) {
                return Err(invalid_data(
                    "model SHA-256 must be 64 lowercase hexadecimal digits",
                ));
            }
            if !model.url.starts_with("https://")
                && !model.url.starts_with("http://")
                && !model.url.starts_with("file://")
            {
                return Err(invalid_data("model URL must use https, http, or file"));
            }
        }
        Ok(())
    }

    fn resolve(&self, name: &str) -> Result<&ModelEntry, io::Error> {
        self.models
            .iter()
            .find(|model| model.id == name || model.aliases.iter().any(|alias| alias == name))
            .ok_or_else(|| invalid_data(format!("model {name:?} is not in the selected registry")))
    }
}

fn load_registry(path: Option<&Path>) -> Result<Registry, io::Error> {
    match path {
        Some(path) => Registry::parse(&fs::read_to_string(path)?),
        None => Registry::parse(BUILTIN_REGISTRY),
    }
}

fn ensure_model(model: &ModelEntry) -> Result<PathBuf, Box<dyn Error>> {
    let directory = data_home()?.join("models").join(storage_name(&model.id));
    fs::create_dir_all(&directory)?;
    let destination = directory.join(&model.artifact);
    if destination.exists() {
        let actual = sha256_file(&destination)?;
        if actual == model.sha256 {
            print_verified(model, &destination);
            return Ok(destination);
        }
        let quarantine = quarantine_path(&destination, &actual);
        fs::rename(&destination, &quarantine)?;
        eprintln!(
            "quarantined: {} (found sha256 {})",
            quarantine.display(),
            actual
        );
    }

    let lock_path = destination.with_extension(format!("{}.lock", extension(&destination)));
    let _lock = acquire_lock(&lock_path)?;
    if destination.exists() && sha256_file(&destination)? == model.sha256 {
        print_verified(model, &destination);
        return Ok(destination);
    }
    let partial = destination.with_extension(format!("{}.part", extension(&destination)));
    if let Some(source) = model.url.strip_prefix("file://") {
        if partial.exists() {
            fs::remove_file(&partial)?;
        }
        fs::hard_link(source, &partial).or_else(|_| fs::copy(source, &partial).map(|_| ()))?;
    } else {
        let status = Command::new("curl")
            .args([
                "--fail",
                "--location",
                "--retry",
                "3",
                "--continue-at",
                "-",
                "--output",
            ])
            .arg(&partial)
            .arg(&model.url)
            .status()
            .map_err(|error| invalid_data(format!("failed to start curl: {error}")))?;
        if !status.success() {
            return Err(invalid_data(format!("curl failed with status {status}")).into());
        }
    }
    let actual = sha256_file(&partial)?;
    if actual != model.sha256 {
        let quarantine = quarantine_path(&partial, &actual);
        fs::rename(&partial, &quarantine)?;
        return Err(invalid_data(format!(
            "downloaded model has sha256 {actual}, expected {}. The invalid file is {}",
            model.sha256,
            quarantine.display()
        ))
        .into());
    }
    fs::rename(&partial, &destination)?;
    print_verified(model, &destination);
    Ok(destination)
}

fn print_verified(model: &ModelEntry, path: &Path) {
    println!("model: {}", model.id);
    println!("artifact: {}", path.display());
    println!("sha256: {}", model.sha256);
    println!("source: {}", model.url);
}

fn acquire_lock(path: &Path) -> Result<PullLock, io::Error> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| {
            if error.kind() == io::ErrorKind::AlreadyExists {
                invalid_data(format!("another pull holds {}", path.display()))
            } else {
                error
            }
        })?;
    Ok(PullLock {
        path: path.to_owned(),
    })
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

fn storage_name(id: &str) -> String {
    id.chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '-' || character == '_' {
                character
            } else {
                '_'
            }
        })
        .collect()
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
    fn storage_names_do_not_create_subdirectories() {
        assert_eq!(storage_name("qwen3:8b/instruct"), "qwen3_8b_instruct");
    }
}
