use leone::{ModelArchitecture, Runtime};
use leone_cuda::CudaBackend;
use std::error::Error;
use std::io;
use std::path::PathBuf;
use std::process::Command;

pub fn run(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let model = parse(arguments)?;
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
        return Err(invalid("nvidia-smi returned an unexpected GPU record").into());
    }
    match fields[1] {
        "8.9" | "12.0" => {}
        capability => {
            return Err(invalid(format!(
                "GPU compute capability {capability} is unsupported. Leone supports SM89 and checks SM120 for correctness"
            ))
            .into());
        }
    }
    println!("gpu: {}", fields[0]);
    println!("compute capability: {}", fields[1]);
    println!("memory MiB: {}", fields[2]);
    println!("driver: {}", fields[3]);

    let backend = CudaBackend::new(0).map_err(|error| {
        invalid(format!(
            "CUDA initialization failed: {error}. Check the NVIDIA driver and CUDA 13 runtime libraries"
        ))
    })?;
    println!("cuda context: ready");
    if let Some(path) = model {
        let runtime = Runtime::load(backend, &path).map_err(|error| {
            invalid(format!(
                "model check failed for {}: {error}",
                path.display()
            ))
        })?;
        let config = runtime.model().config();
        println!("model: {}", path.display());
        println!("architecture: {}", config.architecture.name());
        println!(
            "decode execution: {}",
            if config.architecture == ModelArchitecture::Qwen3 {
                "CUDA graph or eager"
            } else {
                "eager"
            }
        );
        println!("context limit: {}", config.context_length);
        println!("model load: ready");
    }
    Ok(())
}

fn parse(arguments: &[String]) -> Result<Option<PathBuf>, io::Error> {
    match arguments {
        [] => Ok(None),
        [flag, path] if flag == "-m" || flag == "--model" => Ok(Some(PathBuf::from(path))),
        _ => Err(invalid("usage: leone doctor [-m <gguf>]")),
    }
}

fn nvidia_smi(arguments: &[&str]) -> Result<String, io::Error> {
    let output = Command::new("nvidia-smi")
        .args(arguments)
        .output()
        .map_err(|error| {
            invalid(format!(
                "nvidia-smi is unavailable: {error}. Install an NVIDIA driver before using CUDA"
            ))
        })?;
    if !output.status.success() {
        return Err(invalid(format!(
            "nvidia-smi failed with status {}. Check the NVIDIA driver",
            output.status
        )));
    }
    String::from_utf8(output.stdout).map_err(|_| invalid("nvidia-smi output is not UTF-8"))
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}
