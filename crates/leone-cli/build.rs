use std::env;
use std::path::Path;

#[path = "../../build-support/provenance.rs"]
mod provenance;

fn main() {
    let manifest = env::var("CARGO_MANIFEST_DIR").expect("Cargo sets the manifest directory");
    let root = Path::new(&manifest)
        .parent()
        .and_then(Path::parent)
        .expect("the CLI crate is under crates");
    provenance::emit(root);
}
