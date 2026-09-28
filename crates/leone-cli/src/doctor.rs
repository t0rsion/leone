use crate::backend_choice::{default_backend, BackendChoice};
use leone::{CpuBackend, Runtime};
#[cfg(feature = "cuda")]
use leone_cuda::CudaBackend;
use leone_gguf::model::ModelConfig;
use leone_gguf::Gguf;
#[cfg(feature = "metal")]
use leone_metal::MetalBackend;
use std::error::Error;
use std::io;
use std::path::{Path, PathBuf};
#[cfg(feature = "cuda")]
use std::process::Command;

#[derive(Debug, PartialEq, Eq)]
struct DoctorArgs {
    model: Option<PathBuf>,
    probe: bool,
    backend: BackendChoice,
}

impl BackendChoice {
    const fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Metal => "metal",
        }
    }
}

pub fn run(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    run_with_backend_validator(arguments, validate_probe_backend)
}

fn run_with_backend_validator<F>(
    arguments: &[String],
    validate_backend: F,
) -> Result<(), Box<dyn Error>>
where
    F: FnOnce(BackendChoice) -> Result<(), Box<dyn Error>>,
{
    let arguments = parse(arguments)?;
    print_platform();
    print_backends();
    println!("selected backend: {}", arguments.backend.name());
    validate_backend(arguments.backend)?;
    if let Some(model) = arguments.model.as_deref() {
        check_model_metadata(model)?;
        if arguments.probe {
            probe_model(model, arguments.backend)?;
        } else {
            println!("model load: skipped (use --probe to load weights)");
        }
    } else if arguments.probe {
        return Err(invalid("doctor --probe requires -m <gguf>").into());
    }
    Ok(())
}

fn print_platform() {
    println!(
        "platform: {} {}",
        std::env::consts::OS,
        std::env::consts::ARCH
    );
}

fn print_backends() {
    println!("cpu backend: available");
    #[cfg(feature = "cuda")]
    match gpu_record() {
        Ok(fields) => {
            println!("cuda device: detected");
            println!("gpu: {}", fields.name);
            println!("compute capability: {}", fields.compute_capability);
            println!("memory MiB: {}", fields.memory_mib);
            println!("driver: {}", fields.driver);
        }
        Err(error) => println!("cuda device metadata: unavailable ({error})"),
    }
    #[cfg(not(feature = "cuda"))]
    println!("cuda backend: unavailable (not compiled)");
    #[cfg(feature = "metal")]
    match MetalBackend::new() {
        Ok(_) => println!("metal device: detected"),
        Err(error) => println!("metal backend: unavailable ({error})"),
    }
    #[cfg(not(feature = "metal"))]
    println!("metal backend: unavailable (not compiled)");
}

fn check_model_metadata(path: &Path) -> Result<(), Box<dyn Error>> {
    let gguf = Gguf::open(path).map_err(|error| {
        invalid(format!(
            "model metadata check failed for {}: {error}",
            path.display()
        ))
    })?;
    let architecture = gguf
        .metadata()
        .get("general.architecture")
        .and_then(|value| value.as_str())
        .ok_or_else(|| {
            invalid(format!(
                "model metadata check failed for {}: GGUF metadata key \"general.architecture\" is missing or not a string; use `leone gguf inspect {}` for container metadata",
                path.display(),
                path.display()
            ))
        })?;
    let support = leone_gguf::architecture_support(architecture);
    if support.metadata_probe && !support.text_runtime {
        return Err(invalid(format!(
            "model metadata check failed for {}: architecture {architecture:?} has no text runtime; use `leone gguf inspect {}` for metadata-only inspection",
            path.display(),
            path.display()
        ))
        .into());
    }
    let config = ModelConfig::from_metadata(gguf.metadata()).map_err(|error| {
        invalid(format!(
            "model metadata check failed for {}: {error}; use `leone gguf inspect {}` for container metadata",
            path.display(),
            path.display(),
        ))
    })?;
    println!("model: {}", path.display());
    println!("architecture: {}", config.architecture);
    println!("context limit: {}", config.context_length);
    println!("model metadata: ready");
    Ok(())
}

fn probe_model(path: &Path, backend: BackendChoice) -> Result<(), Box<dyn Error>> {
    match backend {
        BackendChoice::Cpu => probe_cpu_model(path),
        BackendChoice::Cuda => probe_cuda_model(path),
        BackendChoice::Metal => probe_metal_model(path),
    }
}

fn validate_probe_backend(backend: BackendChoice) -> Result<(), Box<dyn Error>> {
    match backend {
        BackendChoice::Cpu => Ok(()),
        BackendChoice::Cuda => {
            #[cfg(feature = "cuda")]
            return probe_cuda_device();
            #[cfg(not(feature = "cuda"))]
            return Err(invalid("the cuda backend is unavailable in this build").into());
        }
        BackendChoice::Metal => {
            #[cfg(feature = "metal")]
            return MetalBackend::new()
                .map(|_| ())
                .map_err(|error| invalid(format!("Metal probe failed: {error}")).into());
            #[cfg(not(feature = "metal"))]
            return Err(invalid("the metal backend is unavailable in this build").into());
        }
    }
}

#[cfg(feature = "cuda")]
fn probe_cuda_device() -> Result<(), Box<dyn Error>> {
    let backend =
        CudaBackend::new(0).map_err(|error| invalid(format!("CUDA probe failed: {error}")))?;
    let info = backend
        .device_info()
        .map_err(|error| invalid(format!("CUDA probe failed: {error}")))?;
    validate_cuda_compute_capability(info.compute_major, info.compute_minor)
        .map_err(|error| invalid(format!("CUDA probe failed: {error}")).into())
}

fn probe_cpu_model(path: &Path) -> Result<(), Box<dyn Error>> {
    Runtime::load(CpuBackend::new(), path).map_err(|error| {
        invalid(format!(
            "CPU model probe failed for {}: {error}",
            path.display()
        ))
    })?;
    println!("cpu model probe: ready");
    Ok(())
}

#[cfg(feature = "cuda")]
fn probe_cuda_model(path: &Path) -> Result<(), Box<dyn Error>> {
    let backend = CudaBackend::new(0).map_err(|error| {
        invalid(format!(
            "CUDA probe failed: {error}. Check the NVIDIA driver and CUDA 13 runtime libraries"
        ))
    })?;
    Runtime::load(backend, path).map_err(|error| {
        invalid(format!(
            "CUDA model probe failed for {}: {error}",
            path.display()
        ))
    })?;
    println!("cuda model probe: ready");
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn probe_cuda_model(_path: &Path) -> Result<(), Box<dyn Error>> {
    Err(invalid("the cuda backend is unavailable in this build").into())
}

#[cfg(feature = "metal")]
fn probe_metal_model(path: &Path) -> Result<(), Box<dyn Error>> {
    let backend =
        MetalBackend::new().map_err(|error| invalid(format!("Metal probe failed: {error}")))?;
    Runtime::load(backend, path).map_err(|error| {
        invalid(format!(
            "Metal model probe failed for {}: {error}",
            path.display()
        ))
    })?;
    println!("metal model probe: ready");
    Ok(())
}

#[cfg(not(feature = "metal"))]
fn probe_metal_model(_path: &Path) -> Result<(), Box<dyn Error>> {
    Err(invalid("the metal backend is unavailable in this build").into())
}

#[cfg(any(feature = "cuda", test))]
fn validate_cuda_compute_capability(major: i32, minor: i32) -> Result<(), io::Error> {
    if matches!((major, minor), (8, 9) | (12, 0)) {
        return Ok(());
    }
    Err(invalid(format!(
        "GPU compute capability {major}.{minor} is unsupported. Leone supports SM89 and checks SM120 for correctness"
    )))
}

#[cfg(feature = "cuda")]
#[derive(Debug, PartialEq, Eq)]
struct GpuRecord {
    name: String,
    compute_capability: String,
    memory_mib: String,
    driver: String,
}

#[cfg(feature = "cuda")]
fn gpu_record() -> Result<GpuRecord, io::Error> {
    let identity = nvidia_smi(&[
        "--query-gpu=name,compute_cap,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ])?;
    let line = identity
        .lines()
        .next()
        .ok_or_else(|| invalid("nvidia-smi did not report a GPU"))?;
    let fields = line.split(',').map(str::trim).collect::<Vec<_>>();
    if fields.len() != 4 {
        return Err(invalid("nvidia-smi returned an unexpected GPU record"));
    }
    let (compute_major, compute_minor) = parse_compute_capability(fields[1])?;
    validate_cuda_compute_capability(compute_major, compute_minor)?;
    Ok(GpuRecord {
        name: fields[0].to_owned(),
        compute_capability: format!("{compute_major}.{compute_minor}"),
        memory_mib: fields[2].to_owned(),
        driver: fields[3].to_owned(),
    })
}

#[cfg(feature = "cuda")]
fn parse_compute_capability(value: &str) -> Result<(i32, i32), io::Error> {
    let (major, minor) = value.split_once('.').ok_or_else(|| {
        invalid(format!(
            "nvidia-smi returned invalid compute capability {value}"
        ))
    })?;
    let major = major.parse().map_err(|_| {
        invalid(format!(
            "nvidia-smi returned invalid compute capability {value}"
        ))
    })?;
    let minor = minor.parse().map_err(|_| {
        invalid(format!(
            "nvidia-smi returned invalid compute capability {value}"
        ))
    })?;
    Ok((major, minor))
}

#[cfg(feature = "cuda")]
fn nvidia_smi(arguments: &[&str]) -> Result<String, io::Error> {
    let output = Command::new("nvidia-smi")
        .args(arguments)
        .output()
        .map_err(|error| invalid(format!("nvidia-smi is unavailable: {error}")))?;
    if !output.status.success() {
        return Err(invalid(format!(
            "nvidia-smi failed with status {}",
            output.status
        )));
    }
    String::from_utf8(output.stdout).map_err(|_| invalid("nvidia-smi output is not UTF-8"))
}

fn parse(arguments: &[String]) -> Result<DoctorArgs, io::Error> {
    let mut parsed = DoctorArgs {
        model: None,
        probe: false,
        backend: default_backend(),
    };
    let mut index = 0;
    while index < arguments.len() {
        if parse_argument(&mut parsed, arguments, &mut index)? {
            index += 1;
            continue;
        }
        return Err(invalid(format!(
            "usage: leone doctor [-m <gguf>] [--backend cpu|cuda|metal] [--probe], found {}",
            arguments[index]
        )));
    }
    Ok(parsed)
}

fn parse_argument(
    parsed: &mut DoctorArgs,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    match arguments[*index].as_str() {
        "-m" | "--model" => {
            *index += 1;
            let value = arguments
                .get(*index)
                .ok_or_else(|| invalid("doctor model flag is missing its value"))?;
            parsed.model = Some(PathBuf::from(value));
            Ok(true)
        }
        "--probe" => {
            parsed.probe = true;
            Ok(true)
        }
        "--backend" => {
            *index += 1;
            let value = arguments
                .get(*index)
                .ok_or_else(|| invalid("doctor backend flag is missing its value"))?;
            parsed.backend = parse_backend(value)?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

fn parse_backend(value: &str) -> Result<BackendChoice, io::Error> {
    match value {
        "cpu" => Ok(BackendChoice::Cpu),
        "cuda" => Ok(BackendChoice::Cuda),
        "metal" => Ok(BackendChoice::Metal),
        value => Err(invalid(format!("doctor backend is invalid: {value}"))),
    }
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

#[cfg(test)]
mod tests {
    use super::{
        default_backend, parse, parse_backend, run_with_backend_validator, BackendChoice,
        DoctorArgs,
    };
    use leone_gguf::ValueType;
    use std::io::Write;
    use std::path::PathBuf;
    use tempfile::NamedTempFile;

    #[test]
    fn doctor_defaults_to_metadata_only() {
        assert_eq!(
            parse(&[]).expect("default doctor arguments"),
            DoctorArgs {
                model: None,
                probe: false,
                backend: default_backend(),
            }
        );
    }

    #[test]
    fn doctor_accepts_explicit_probe() {
        assert_eq!(
            parse(&[
                "-m".to_owned(),
                "model.gguf".to_owned(),
                "--probe".to_owned()
            ])
            .expect("probe arguments"),
            DoctorArgs {
                model: Some(PathBuf::from("model.gguf")),
                probe: true,
                backend: default_backend(),
            }
        );
    }

    #[test]
    fn doctor_accepts_backend_selector() {
        assert_eq!(
            parse(&["--backend".to_owned(), "cpu".to_owned()]).expect("backend arguments"),
            DoctorArgs {
                model: None,
                probe: false,
                backend: BackendChoice::Cpu,
            }
        );
        assert_eq!(
            parse_backend("metal").expect("metal backend"),
            BackendChoice::Metal
        );
    }

    #[test]
    fn default_doctor_rejects_an_injected_unavailable_selected_backend() {
        let result = run_with_backend_validator(&[], |backend| {
            assert_eq!(backend, default_backend());
            Err(super::invalid("injected backend unavailable").into())
        });
        assert!(result.is_err());
    }

    #[test]
    fn cpu_metadata_check_runs_through_the_doctor_entrypoint() {
        let fixture = minimal_qwen3_gguf();
        let arguments = vec![
            "--backend".to_owned(),
            "cpu".to_owned(),
            "-m".to_owned(),
            fixture.path().display().to_string(),
        ];
        run_with_backend_validator(&arguments, |backend| {
            assert_eq!(backend, BackendChoice::Cpu);
            Ok(())
        })
        .expect("CPU metadata check");
    }

    #[test]
    fn accelerator_availability_is_injectable_without_gpu_hardware() {
        for (name, expected) in [
            ("cuda", BackendChoice::Cuda),
            ("metal", BackendChoice::Metal),
        ] {
            let arguments = vec!["--backend".to_owned(), name.to_owned()];
            let result = run_with_backend_validator(&arguments, |backend| {
                assert_eq!(backend, expected);
                Err(super::invalid("injected accelerator unavailable").into())
            });
            assert!(result.is_err());
        }
    }

    #[test]
    fn cuda_capability_policy_accepts_sm89_and_sm120_only() {
        assert!(super::validate_cuda_compute_capability(8, 9).is_ok());
        assert!(super::validate_cuda_compute_capability(12, 0).is_ok());
        for (major, minor) in [(8, 6), (9, 0)] {
            let error = super::validate_cuda_compute_capability(major, minor)
                .expect_err("unsupported CUDA capability");
            assert_eq!(
                error.to_string(),
                format!(
                    "GPU compute capability {major}.{minor} is unsupported. Leone supports SM89 and checks SM120 for correctness"
                )
            );
        }
    }

    fn minimal_qwen3_gguf() -> NamedTempFile {
        let mut file = NamedTempFile::new().expect("temporary GGUF");
        let metadata = [
            ("general.architecture", MetadataFixture::String("qwen3")),
            ("qwen3.block_count", MetadataFixture::Uint64(1)),
            ("qwen3.attention.head_count", MetadataFixture::Uint64(1)),
            ("qwen3.attention.head_count_kv", MetadataFixture::Uint64(1)),
            ("qwen3.embedding_length", MetadataFixture::Uint64(4)),
            ("qwen3.feed_forward_length", MetadataFixture::Uint64(8)),
            (
                "qwen3.rope.freq_base",
                MetadataFixture::Float64(1_000_000.0),
            ),
            (
                "qwen3.attention.layer_norm_rms_epsilon",
                MetadataFixture::Float64(0.000001),
            ),
            ("qwen3.context_length", MetadataFixture::Uint64(128)),
            ("tokenizer.ggml.model", MetadataFixture::String("gpt2")),
        ];
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        bytes.extend_from_slice(&11_u64.to_le_bytes());
        for (key, value) in metadata {
            append_string(&mut bytes, key);
            match value {
                MetadataFixture::String(value) => {
                    bytes.extend_from_slice(&(ValueType::String as u32).to_le_bytes());
                    append_string(&mut bytes, value);
                }
                MetadataFixture::Uint64(value) => {
                    bytes.extend_from_slice(&(ValueType::Uint64 as u32).to_le_bytes());
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                MetadataFixture::Float64(value) => {
                    bytes.extend_from_slice(&(ValueType::Float64 as u32).to_le_bytes());
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
            }
        }
        append_metadata_string_array(&mut bytes, "tokenizer.ggml.tokens", &["a"]);
        file.write_all(&bytes).expect("write GGUF");
        file.flush().expect("flush GGUF");
        file
    }

    enum MetadataFixture {
        String(&'static str),
        Uint64(u64),
        Float64(f64),
    }

    fn append_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }

    fn append_metadata_string_array(bytes: &mut Vec<u8>, key: &str, values: &[&str]) {
        append_string(bytes, key);
        bytes.extend_from_slice(&(ValueType::Array as u32).to_le_bytes());
        bytes.extend_from_slice(&(ValueType::String as u32).to_le_bytes());
        bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for value in values {
            append_string(bytes, value);
        }
    }
}
