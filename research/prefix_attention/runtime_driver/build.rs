use std::path::Path;

#[path = "../../../build-support/provenance.rs"]
mod provenance;

fn main() {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("Cargo sets the manifest directory");
    let root = Path::new(&manifest)
        .ancestors()
        .nth(3)
        .expect("the driver is under research/prefix_attention");
    provenance::emit(root);
}
