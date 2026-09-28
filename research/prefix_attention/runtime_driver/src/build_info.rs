pub fn value() -> serde_json::Value {
    serde_json::json!({
        "target": env!("LEONE_BUILD_TARGET"),
        "host": env!("LEONE_BUILD_HOST"),
        "profile": env!("LEONE_BUILD_PROFILE"),
        "features": env!("LEONE_BUILD_FEATURES"),
        "rustc": env!("LEONE_BUILD_RUSTC"),
        "rustc_sha256": env!("LEONE_BUILD_RUSTC_SHA256"),
        "toolchain_sha256": env!("LEONE_BUILD_TOOLCHAIN_SHA256"),
        "tool_versions": env!("LEONE_BUILD_TOOL_VERSIONS"),
        "build_flags": env!("LEONE_BUILD_FLAGS"),
        "build_flags_sha256": env!("LEONE_BUILD_FLAGS_SHA256"),
        "build_flags_raw_sha256": env!("LEONE_BUILD_FLAGS_RAW_SHA256"),
        "profile_inputs": env!("LEONE_BUILD_PROFILE_INPUTS"),
        "profile_inputs_sha256": env!("LEONE_BUILD_PROFILE_INPUTS_SHA256"),
        "linker_inputs": env!("LEONE_BUILD_LINKER_INPUTS"),
        "linker_inputs_sha256": env!("LEONE_BUILD_LINKER_INPUTS_SHA256"),
        "native_provenance": env!("LEONE_BUILD_NATIVE_PROVENANCE"),
        "native_tools_sha256": env!("LEONE_BUILD_NATIVE_TOOLS_SHA256"),
        "native_tool_versions": env!("LEONE_BUILD_NATIVE_TOOL_VERSIONS"),
        "build_config_sha256": env!("LEONE_BUILD_CONFIG_SHA256"),
        "source_tree_sha256": env!("LEONE_SOURCE_TREE_SHA256"),
        "source_tree_dirty": env!("LEONE_SOURCE_DIRTY") == "true",
        "provenance_unknown": env!("LEONE_BUILD_PROVENANCE_UNKNOWN") == "true",
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn features_name_the_compiled_backend() {
        let backend = if cfg!(feature = "cuda") {
            "cuda"
        } else {
            "metal"
        };
        assert_eq!(value()["features"], backend);
    }

    #[test]
    fn source_identity_fields_are_typed() {
        let info = value();
        assert!(info["source_tree_sha256"].is_string());
        assert!(info["source_tree_dirty"].is_boolean());
        assert!(info["provenance_unknown"].is_boolean());
    }
}
