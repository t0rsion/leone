use serde::Serialize;

#[derive(Debug, Serialize)]
struct BuildInfo {
    schema_version: &'static str,
    version: &'static str,
    source_commit: &'static str,
    source_tree_dirty: bool,
    source_paths: Vec<&'static str>,
    target: &'static str,
    profile: &'static str,
    source_tree_sha256: &'static str,
    host: &'static str,
    rustc: &'static str,
    rustc_sha256: &'static str,
    toolchain_sha256: &'static str,
    tool_versions: &'static str,
    features: &'static str,
    build_flags: &'static str,
    build_flags_sha256: &'static str,
    build_flags_raw_sha256: &'static str,
    build_flags_redacted: bool,
    build_flags_redaction_count: usize,
    profile_inputs: &'static str,
    profile_inputs_sha256: &'static str,
    linker_inputs: &'static str,
    linker_inputs_sha256: &'static str,
    native_provenance: &'static str,
    native_tools_sha256: &'static str,
    native_tool_versions: &'static str,
    build_config_sha256: &'static str,
    provenance_unknown: bool,
    #[cfg(feature = "metal")]
    metal_shader: serde_json::Value,
}

pub fn print() {
    println!(
        "{}",
        serde_json::to_string(&value()).expect("build info is serializable")
    );
}

pub(crate) fn value() -> serde_json::Value {
    let info = BuildInfo {
        schema_version: "leone.build-info.v1",
        version: env!("CARGO_PKG_VERSION"),
        source_commit: env!("LEONE_SOURCE_COMMIT"),
        source_tree_dirty: env!("LEONE_SOURCE_DIRTY") == "true",
        source_paths: env!("LEONE_SOURCE_PATHS").split(',').collect(),
        target: env!("LEONE_BUILD_TARGET"),
        profile: env!("LEONE_BUILD_PROFILE"),
        source_tree_sha256: env!("LEONE_SOURCE_TREE_SHA256"),
        host: env!("LEONE_BUILD_HOST"),
        rustc: env!("LEONE_BUILD_RUSTC"),
        rustc_sha256: env!("LEONE_BUILD_RUSTC_SHA256"),
        toolchain_sha256: env!("LEONE_BUILD_TOOLCHAIN_SHA256"),
        tool_versions: env!("LEONE_BUILD_TOOL_VERSIONS"),
        features: env!("LEONE_BUILD_FEATURES"),
        build_flags: env!("LEONE_BUILD_FLAGS"),
        build_flags_sha256: env!("LEONE_BUILD_FLAGS_SHA256"),
        build_flags_raw_sha256: env!("LEONE_BUILD_FLAGS_RAW_SHA256"),
        build_flags_redacted: env!("LEONE_BUILD_FLAGS_REDACTED") == "true",
        build_flags_redaction_count: env!("LEONE_BUILD_FLAGS_REDACTION_COUNT")
            .parse()
            .expect("build script emits a numeric redaction count"),
        profile_inputs: env!("LEONE_BUILD_PROFILE_INPUTS"),
        profile_inputs_sha256: env!("LEONE_BUILD_PROFILE_INPUTS_SHA256"),
        linker_inputs: env!("LEONE_BUILD_LINKER_INPUTS"),
        linker_inputs_sha256: env!("LEONE_BUILD_LINKER_INPUTS_SHA256"),
        native_provenance: env!("LEONE_BUILD_NATIVE_PROVENANCE"),
        native_tools_sha256: env!("LEONE_BUILD_NATIVE_TOOLS_SHA256"),
        native_tool_versions: env!("LEONE_BUILD_NATIVE_TOOL_VERSIONS"),
        build_config_sha256: env!("LEONE_BUILD_CONFIG_SHA256"),
        provenance_unknown: env!("LEONE_BUILD_PROVENANCE_UNKNOWN") == "true",
        #[cfg(feature = "metal")]
        metal_shader: {
            let shader = leone_metal::shader_identity();
            serde_json::json!({
                "name": shader.name,
                "sha256": shader.sha256,
                "bytes": shader.bytes,
            })
        },
    };
    serde_json::to_value(info).expect("build info is serializable")
}
