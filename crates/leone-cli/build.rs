use std::env;
use std::path::Path;
use std::process::Command;

fn git(root: &Path, arguments: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .current_dir(root)
        .args(arguments)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn main() {
    let manifest = env::var("CARGO_MANIFEST_DIR").expect("Cargo sets the manifest directory");
    let root = Path::new(&manifest)
        .parent()
        .and_then(Path::parent)
        .expect("the CLI crate is under crates");
    watch_sources(root);
    let commit = git(root, &["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".to_owned());
    let status = git(
        root,
        &[
            "status",
            "--porcelain",
            "--untracked-files=normal",
            "--",
            "Cargo.toml",
            "Cargo.lock",
            "crates",
            "fixtures",
        ],
    );
    let dirty = status.as_ref().is_none_or(|status| !status.is_empty());
    println!("cargo:rustc-env=LEONE_SOURCE_COMMIT={commit}");
    println!("cargo:rustc-env=LEONE_SOURCE_DIRTY={dirty}");
    println!(
        "cargo:rustc-env=LEONE_BUILD_TARGET={}",
        env::var("TARGET").expect("Cargo sets TARGET")
    );
    println!(
        "cargo:rustc-env=LEONE_BUILD_PROFILE={}",
        env::var("PROFILE").expect("Cargo sets PROFILE")
    );
}

fn watch_sources(root: &Path) {
    for relative in ["Cargo.toml", "Cargo.lock", "crates", "fixtures"] {
        println!("cargo:rerun-if-changed={}", root.join(relative).display());
    }
    for name in ["HEAD", "index", "packed-refs"] {
        watch_git_path(root, name);
    }
    if let Some(reference) = git(root, &["symbolic-ref", "-q", "HEAD"]) {
        watch_git_path(root, &reference);
    }
}

fn watch_git_path(root: &Path, name: &str) {
    if let Some(path) = git(root, &["rev-parse", "--git-path", name]) {
        let path = root.join(path);
        if path.exists() {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}
