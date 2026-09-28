use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::path::Path;

#[path = "../../build-support/provenance.rs"]
mod provenance;

fn main() {
    let manifest = env::var("CARGO_MANIFEST_DIR").expect("Cargo sets CARGO_MANIFEST_DIR");
    let root = Path::new(&manifest)
        .parent()
        .and_then(Path::parent)
        .expect("the Metal crate is under crates");
    provenance::emit(root);
    println!("cargo:rerun-if-changed=metal/leone.metal");
    println!("cargo:rerun-if-changed=metal/metal_bridge.m");
    println!("cargo:rerun-if-changed=metal/metal_bridge_stub.c");
    let shader = fs::read("metal/leone.metal").expect("read Metal shader");
    println!(
        "cargo:rustc-env=LEONE_METAL_SHADER_SHA256={:x}",
        Sha256::digest(&shader)
    );

    if cfg!(target_os = "macos") {
        cc::Build::new()
            .file("metal/metal_bridge.m")
            .flag("-fobjc-arc")
            .flag("-fmodules")
            .flag("-Wno-deprecated-declarations")
            .compile("leone_metal_bridge");
        println!("cargo:rustc-link-framework=Metal");
        println!("cargo:rustc-link-framework=Foundation");
    } else {
        cc::Build::new()
            .file("metal/metal_bridge_stub.c")
            .compile("leone_metal_bridge");
    }
}
