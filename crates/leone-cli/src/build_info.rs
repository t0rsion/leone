pub fn print() {
    println!(
        "{}",
        serde_json::json!({
            "schema_version": "leone.build-info.v1",
            "version": env!("CARGO_PKG_VERSION"),
            "source_commit": env!("LEONE_SOURCE_COMMIT"),
            "source_tree_dirty": env!("LEONE_SOURCE_DIRTY") == "true",
            "source_paths": ["Cargo.toml", "Cargo.lock", "crates", "fixtures"],
            "target": env!("LEONE_BUILD_TARGET"),
            "profile": env!("LEONE_BUILD_PROFILE"),
        })
    );
}
