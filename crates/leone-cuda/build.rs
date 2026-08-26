use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let lineinfo = env::var("LEONE_CUDA_LINEINFO").is_ok_and(|value| value == "1");
    let cuda_root = env::var_os("CUDA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            ["/opt/cuda", "/usr/local/cuda"]
                .into_iter()
                .map(PathBuf::from)
                .find(|root| root.join("bin/nvcc").is_file())
                .unwrap_or_else(|| PathBuf::from("/usr/local/cuda"))
        });
    let nvcc = cuda_root.join("bin/nvcc");
    let include = cuda_root.join("include");
    let lib = cuda_root.join("targets/x86_64-linux/lib");
    let output = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo sets OUT_DIR"));
    let archive = output.join("libie_cuda_kernels.a");

    println!("cargo:rerun-if-changed=cuda/ie_cuda.cu");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=LEONE_CUDA_LINEINFO");
    println!("cargo:rustc-link-search=native={}", output.display());
    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib=static=ie_cuda_kernels");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublasLt");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    let mut command = Command::new(nvcc);
    command
        .arg("-lib")
        .arg("-arch=sm_89")
        .arg("-O3")
        .arg("--std=c++17")
        .arg("-Xcompiler=-fPIC")
        .arg(format!("-I{}", include.display()))
        .arg("cuda/ie_cuda.cu")
        .arg("-o")
        .arg(archive);
    if lineinfo {
        command.arg("-lineinfo");
    }
    let status = command.status().expect("run nvcc");
    assert!(status.success(), "nvcc failed with {status}");
}
