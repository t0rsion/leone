use sha2::{Digest, Sha256};
use std::env;
use std::ffi::{OsStr, OsString};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const UNKNOWN: &str = "unknown";
const SOURCE_PATHS: [&str; 8] = [
    "Cargo.toml",
    "Cargo.lock",
    "crates",
    "fixtures",
    ".cargo",
    "rust-toolchain",
    "rust-toolchain.toml",
    "build-support",
];
const CONFIG_INPUTS: [&str; 35] = [
    "RUSTC",
    "RUSTC_WRAPPER",
    "RUSTC_WORKSPACE_WRAPPER",
    "PATH",
    "RUSTFLAGS",
    "CARGO_ENCODED_RUSTFLAGS",
    "TARGET",
    "HOST",
    "PROFILE",
    "OPT_LEVEL",
    "DEBUG",
    "CARGO_CFG_TARGET_FEATURE",
    "RUSTC_LINKER",
    "LD",
    "LDFLAGS",
    "CPPFLAGS",
    "OBJCFLAGS",
    "ARFLAGS",
    "LIBRARY_PATH",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    "CUDA_HOME",
    "CUDA_PATH",
    "LEONE_CUDA_LINEINFO",
    "CC",
    "CFLAGS",
    "CXX",
    "CXXFLAGS",
    "AR",
    "RANLIB",
    "SDKROOT",
    "MACOSX_DEPLOYMENT_TARGET",
    "IPHONEOS_DEPLOYMENT_TARGET",
    "TVOS_DEPLOYMENT_TARGET",
    "WATCHOS_DEPLOYMENT_TARGET",
];
const LINKER_INPUTS: [&str; 16] = [
    "RUSTC_LINKER",
    "LD",
    "LDFLAGS",
    "CPPFLAGS",
    "OBJCFLAGS",
    "ARFLAGS",
    "LIBRARY_PATH",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    "CC",
    "CFLAGS",
    "CXX",
    "CXXFLAGS",
    "AR",
    "RANLIB",
    "SDKROOT",
];
const PROFILE_KINDS: [&str; 4] = ["DEV", "RELEASE", "TEST", "BENCH"];
const PROFILE_KEYS: [&str; 11] = [
    "CODEGEN_UNITS",
    "DEBUG",
    "DEBUG_ASSERTIONS",
    "INCREMENTAL",
    "LTO",
    "OPT_LEVEL",
    "OVERFLOW_CHECKS",
    "PANIC",
    "RPATH",
    "SPLIT_DEBUGINFO",
    "STRIP",
];
const TOOL_ENV_INPUTS: [&str; 9] = [
    "RUSTC",
    "RUSTC_WRAPPER",
    "RUSTC_WORKSPACE_WRAPPER",
    "RUSTC_LINKER",
    "LD",
    "CC",
    "CXX",
    "AR",
    "RANLIB",
];

#[derive(Debug)]
struct BuildProvenance {
    source_commit: String,
    source_tree_dirty: bool,
    source_tree_sha256: String,
    target: String,
    host: String,
    profile: String,
    rustc: String,
    rustc_sha256: String,
    toolchain_sha256: String,
    tool_versions: String,
    features: String,
    flags: String,
    flags_sha256: String,
    flags_raw_sha256: String,
    flags_redacted: bool,
    flags_redaction_count: usize,
    profile_inputs: String,
    profile_inputs_sha256: String,
    linker_inputs: String,
    linker_inputs_sha256: String,
    native_provenance: String,
    native_tools_sha256: String,
    native_tool_versions: String,
    config_sha256: String,
    unknown: bool,
}

#[derive(Debug)]
struct ToolIdentity {
    name: String,
    path: String,
    sha256: String,
    version: String,
    complete: bool,
}

#[derive(Debug)]
struct ToolSummary {
    sha256: String,
    versions: String,
    complete: bool,
}

#[derive(Debug)]
struct NativeProvenance {
    status: String,
    tools_sha256: String,
    tool_versions: String,
}

#[derive(Debug)]
struct NativeCompilerSelection {
    source: String,
    value: OsString,
    exact: bool,
}

#[derive(Debug)]
struct ScopedInputs {
    names: String,
    sha256: String,
}

struct ConfigInputs<'a> {
    features: &'a str,
    flag_input: &'a FlagInput,
    rustc: &'a str,
    flags_raw_sha256: &'a str,
    profile_inputs: &'a ScopedInputs,
    linker_inputs: &'a ScopedInputs,
    toolchain: &'a ToolSummary,
    native: &'a NativeProvenance,
}

pub fn emit(root: &Path) {
    watch_inputs(root);
    let provenance = collect(root);
    let source_paths = SOURCE_PATHS.join(",");
    emit_value("LEONE_SOURCE_COMMIT", &provenance.source_commit);
    emit_value(
        "LEONE_SOURCE_DIRTY",
        &provenance.source_tree_dirty.to_string(),
    );
    emit_value("LEONE_SOURCE_PATHS", &source_paths);
    emit_value("LEONE_SOURCE_TREE_SHA256", &provenance.source_tree_sha256);
    emit_value("LEONE_BUILD_TARGET", &provenance.target);
    emit_value("LEONE_BUILD_HOST", &provenance.host);
    emit_value("LEONE_BUILD_PROFILE", &provenance.profile);
    emit_value("LEONE_BUILD_RUSTC", &provenance.rustc);
    emit_value("LEONE_BUILD_RUSTC_SHA256", &provenance.rustc_sha256);
    emit_value("LEONE_BUILD_TOOLCHAIN_SHA256", &provenance.toolchain_sha256);
    emit_value("LEONE_BUILD_TOOL_VERSIONS", &provenance.tool_versions);
    emit_value("LEONE_BUILD_FEATURES", &provenance.features);
    emit_value("LEONE_BUILD_FLAGS", &provenance.flags);
    emit_value("LEONE_BUILD_FLAGS_SHA256", &provenance.flags_sha256);
    emit_value("LEONE_BUILD_FLAGS_RAW_SHA256", &provenance.flags_raw_sha256);
    emit_value(
        "LEONE_BUILD_FLAGS_REDACTED",
        &provenance.flags_redacted.to_string(),
    );
    emit_value(
        "LEONE_BUILD_FLAGS_REDACTION_COUNT",
        &provenance.flags_redaction_count.to_string(),
    );
    emit_value("LEONE_BUILD_PROFILE_INPUTS", &provenance.profile_inputs);
    emit_value(
        "LEONE_BUILD_PROFILE_INPUTS_SHA256",
        &provenance.profile_inputs_sha256,
    );
    emit_value("LEONE_BUILD_LINKER_INPUTS", &provenance.linker_inputs);
    emit_value(
        "LEONE_BUILD_LINKER_INPUTS_SHA256",
        &provenance.linker_inputs_sha256,
    );
    emit_value(
        "LEONE_BUILD_NATIVE_PROVENANCE",
        &provenance.native_provenance,
    );
    emit_value(
        "LEONE_BUILD_NATIVE_TOOLS_SHA256",
        &provenance.native_tools_sha256,
    );
    emit_value(
        "LEONE_BUILD_NATIVE_TOOL_VERSIONS",
        &provenance.native_tool_versions,
    );
    emit_value("LEONE_BUILD_CONFIG_SHA256", &provenance.config_sha256);
    emit_value(
        "LEONE_BUILD_PROVENANCE_UNKNOWN",
        &provenance.unknown.to_string(),
    );
}

fn collect(root: &Path) -> BuildProvenance {
    let source_commit = source_commit(root);
    let source_tree_sha256 = source_tree_hash(root).unwrap_or_else(|| UNKNOWN.to_owned());
    let source_tree_dirty = source_dirty(root);
    let target = environment_value("TARGET");
    let host = environment_value("HOST");
    let profile = environment_value("PROFILE");
    let features = feature_list();
    let profile_inputs = scoped_profile_inputs();
    let linker_inputs = scoped_linker_inputs();
    let rustc_identity = executable_identity("rustc", env::var_os("RUSTC"), &["-Vv"], true);
    let rustc = rustc_identity.version.clone();
    let rustc_sha256 = rustc_identity.sha256.clone();
    let wrapper_identity = executable_identity(
        "rustc-wrapper",
        env::var_os("RUSTC_WRAPPER"),
        &["--version"],
        false,
    );
    let workspace_wrapper_identity = executable_identity(
        "rustc-workspace-wrapper",
        env::var_os("RUSTC_WORKSPACE_WRAPPER"),
        &["--version"],
        false,
    );
    let linker_identity = selected_linker_identity(&target);
    let toolchain = summarize_tools(&[
        rustc_identity,
        wrapper_identity,
        workspace_wrapper_identity,
        linker_identity,
    ]);
    let native = native_provenance(&target, &features);
    let flag_input = flag_input();
    let normalized = normalized_flags(&flag_input.tokens);
    let flags_sha256 = sha256_hex(normalized.text.as_bytes());
    let flags_raw_sha256 = digest_fields(&[
        ("flag-source", flag_input.source),
        ("flag-input", &flag_input.raw),
    ]);
    let config_sha256 = build_config_sha256(&ConfigInputs {
        features: &features,
        flag_input: &flag_input,
        rustc: &rustc,
        flags_raw_sha256: &flags_raw_sha256,
        profile_inputs: &profile_inputs,
        linker_inputs: &linker_inputs,
        toolchain: &toolchain,
        native: &native,
    });
    let unknown = source_commit == UNKNOWN
        || source_tree_sha256 == UNKNOWN
        || target == UNKNOWN
        || host == UNKNOWN
        || profile == UNKNOWN
        || rustc == UNKNOWN
        || !toolchain.complete
        || native.status == "incomplete";
    BuildProvenance {
        source_commit,
        source_tree_dirty: source_tree_dirty || unknown,
        source_tree_sha256,
        target,
        host,
        profile,
        rustc: public_tool_version(&rustc),
        rustc_sha256,
        toolchain_sha256: toolchain.sha256,
        tool_versions: toolchain.versions,
        features,
        flags: normalized.text,
        flags_sha256,
        flags_raw_sha256,
        flags_redacted: normalized.redaction_count != 0,
        flags_redaction_count: normalized.redaction_count,
        profile_inputs: profile_inputs.names,
        profile_inputs_sha256: profile_inputs.sha256,
        linker_inputs: linker_inputs.names,
        linker_inputs_sha256: linker_inputs.sha256,
        native_provenance: native.status,
        native_tools_sha256: native.tools_sha256,
        native_tool_versions: native.tool_versions,
        config_sha256,
        unknown,
    }
}

fn source_commit(root: &Path) -> String {
    let Some(commit) = git_text(root, &["rev-parse", "HEAD"]) else {
        return UNKNOWN.to_owned();
    };
    if is_commit(&commit) && git_commit_exists(root, &commit) {
        commit
    } else {
        UNKNOWN.to_owned()
    }
}

fn source_dirty(root: &Path) -> bool {
    if ignored_source_files(root).unwrap_or(true) {
        return true;
    }
    let mut arguments = vec!["status", "--porcelain=v1", "--untracked-files=all", "--"];
    arguments.extend(SOURCE_PATHS);
    match git_text(root, &arguments) {
        Some(status) => !status.is_empty(),
        None => true,
    }
}

// The digest covers SOURCE_PATHS with sorted UTF-8 paths and file bytes with u64 lengths.
fn source_tree_hash(root: &Path) -> Option<String> {
    if ignored_source_files(root)? {
        return None;
    }
    source_tree_hash_files(root)
}

fn source_tree_hash_files(root: &Path) -> Option<String> {
    let mut arguments = vec![
        "ls-files",
        "-z",
        "--cached",
        "--others",
        "--exclude-standard",
        "--",
    ];
    arguments.extend(SOURCE_PATHS);
    let paths = git_bytes(root, &arguments)?;
    let mut files = paths
        .split(|byte| *byte == 0)
        .filter(|path| !path.is_empty())
        .map(|raw_path| {
            let relative = std::str::from_utf8(raw_path).ok()?;
            let bytes = fs::read(root.join(relative)).ok()?;
            Some((raw_path.to_vec(), bytes))
        })
        .collect::<Option<Vec<_>>>()?;
    files.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    let mut hasher = Sha256::new();
    hasher.update((files.len() as u64).to_le_bytes());
    for (path, bytes) in files {
        update_length_prefixed(&mut hasher, &path);
        update_length_prefixed(&mut hasher, &bytes);
    }
    Some(hex_digest(hasher.finalize()))
}

fn ignored_source_files(root: &Path) -> Option<bool> {
    let mut arguments = vec![
        "ls-files",
        "-z",
        "--others",
        "--ignored",
        "--exclude-standard",
        "--",
    ];
    arguments.extend(SOURCE_PATHS);
    let paths = git_bytes(root, &arguments)?;
    Some(paths.split(|byte| *byte == 0).any(|path| !path.is_empty()))
}

fn environment_value(name: &str) -> String {
    env::var(name)
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| UNKNOWN.to_owned())
}

fn scoped_profile_inputs() -> ScopedInputs {
    let names = env::vars()
        .filter_map(|(name, _)| name.starts_with("CARGO_PROFILE_").then_some(name))
        .collect();
    scoped_inputs(names)
}

fn scoped_linker_inputs() -> ScopedInputs {
    let mut names = LINKER_INPUTS
        .iter()
        .map(|name| (*name).to_owned())
        .collect::<Vec<_>>();
    names.extend(
        env::vars()
            .filter_map(|(name, _)| {
                (name.starts_with("CARGO_TARGET_") && name.ends_with("_LINKER")).then_some(name)
            })
            .collect::<Vec<_>>(),
    );
    scoped_inputs(names)
}

fn scoped_inputs(mut names: Vec<String>) -> ScopedInputs {
    names.sort_unstable();
    names.dedup();
    let fields = names
        .iter()
        .map(|name| (name.clone(), environment_input(name)))
        .collect::<Vec<_>>();
    ScopedInputs {
        names: if names.is_empty() {
            "none".to_owned()
        } else {
            names.join(",")
        },
        sha256: digest_owned_fields(&fields),
    }
}

fn executable_identity(
    name: &str,
    value: Option<OsString>,
    version_arguments: &[&str],
    required: bool,
) -> ToolIdentity {
    let Some(value) = value.filter(|value| !value.is_empty()) else {
        return missing_tool_identity(name, required);
    };
    let Some(path) = resolve_executable(&value) else {
        return unknown_tool_identity(name);
    };
    let path_text = path.to_string_lossy().into_owned();
    let sha256 = fs::read(&path)
        .ok()
        .map(sha256_hex)
        .unwrap_or_else(|| UNKNOWN.to_owned());
    let version = command_version(&path, version_arguments).unwrap_or_else(|| UNKNOWN.to_owned());
    ToolIdentity {
        name: name.to_owned(),
        path: path_text,
        complete: sha256 != UNKNOWN,
        sha256,
        version,
    }
}

fn missing_tool_identity(name: &str, required: bool) -> ToolIdentity {
    let value = if required { UNKNOWN } else { "not-selected" };
    ToolIdentity {
        name: name.to_owned(),
        path: value.to_owned(),
        sha256: value.to_owned(),
        version: value.to_owned(),
        complete: !required,
    }
}

fn unknown_tool_identity(name: &str) -> ToolIdentity {
    ToolIdentity {
        name: name.to_owned(),
        path: UNKNOWN.to_owned(),
        sha256: UNKNOWN.to_owned(),
        version: UNKNOWN.to_owned(),
        complete: false,
    }
}

fn resolve_executable(value: &OsStr) -> Option<PathBuf> {
    let direct = PathBuf::from(value);
    if direct.is_file() {
        return Some(direct);
    }
    let command = value
        .to_string_lossy()
        .split_whitespace()
        .next()?
        .to_owned();
    let command = Path::new(&command);
    if command.components().count() > 1 {
        return command.is_file().then(|| command.to_owned());
    }
    env::var_os("PATH")
        .into_iter()
        .flat_map(|path| env::split_paths(&path).collect::<Vec<_>>())
        .map(|directory| directory.join(command))
        .find(|candidate| candidate.is_file())
}

fn command_version(path: &Path, arguments: &[&str]) -> Option<String> {
    let output = Command::new(path).args(arguments).output().ok()?;
    if !output.status.success() {
        return None;
    }
    normalized_output(&output.stdout).or_else(|| normalized_output(&output.stderr))
}

fn normalized_output(bytes: &[u8]) -> Option<String> {
    let text = std::str::from_utf8(bytes)
        .ok()?
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    (!text.is_empty()).then_some(text)
}

fn summarize_tools(tools: &[ToolIdentity]) -> ToolSummary {
    let fields = tools
        .iter()
        .flat_map(|tool| {
            [
                (format!("{}.path", tool.name), tool.path.clone()),
                (format!("{}.sha256", tool.name), tool.sha256.clone()),
                (format!("{}.version", tool.name), tool.version.clone()),
            ]
        })
        .collect::<Vec<_>>();
    let versions = tools
        .iter()
        .map(|tool| format!("{}={}", tool.name, public_tool_version(&tool.version)))
        .collect::<Vec<_>>()
        .join(";");
    let complete = tools.iter().all(|tool| tool.complete);
    ToolSummary {
        sha256: if complete {
            digest_owned_fields(&fields)
        } else {
            UNKNOWN.to_owned()
        },
        versions,
        complete,
    }
}

fn public_tool_version(version: &str) -> String {
    version
        .split_whitespace()
        .map(|word| {
            if word.contains(['/', '\\', '@']) {
                "<redacted>"
            } else {
                word
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn selected_linker_identity(target: &str) -> ToolIdentity {
    let value = env::var_os("RUSTC_LINKER").filter(|value| !value.is_empty());
    let value = value.or_else(|| {
        let name = target_linker_name(target);
        env::var_os(name).filter(|value| !value.is_empty())
    });
    executable_identity("linker", value, &["--version"], false)
}

fn target_linker_name(target: &str) -> String {
    format!(
        "CARGO_TARGET_{}_LINKER",
        target.replace('-', "_").to_ascii_uppercase()
    )
}

fn native_provenance(target: &str, features: &str) -> NativeProvenance {
    let apple = target.contains("apple");
    let cuda = feature_enabled(features, "cuda");
    let metal = feature_enabled(features, "metal") || apple;
    if !apple && !cuda && !metal {
        return NativeProvenance {
            status: "unavailable".to_owned(),
            tools_sha256: "not-applicable".to_owned(),
            tool_versions: "none".to_owned(),
        };
    }
    let compiler = metal.then(|| native_compiler_selection(target));
    let tools = native_tool_identities(target, cuda, apple, compiler.as_ref());
    let summary = summarize_tools(&tools);
    let compiler_complete = compiler.as_ref().is_none_or(|selection| selection.exact);
    let complete = summary.complete && compiler_complete;
    NativeProvenance {
        status: if complete {
            "complete".to_owned()
        } else {
            "incomplete".to_owned()
        },
        tools_sha256: native_tools_sha256(&summary, compiler.as_ref(), complete),
        tool_versions: native_tool_versions(&summary, compiler.as_ref()),
    }
}

fn native_tool_identities(
    target: &str,
    cuda: bool,
    apple: bool,
    compiler: Option<&NativeCompilerSelection>,
) -> Vec<ToolIdentity> {
    let mut tools = Vec::new();
    if cuda {
        tools.push(executable_identity(
            "nvcc",
            Some(cuda_compiler_path()),
            &["--version"],
            true,
        ));
    }
    if let Some(compiler) = compiler {
        tools.push(executable_identity(
            "cc",
            Some(compiler.value.clone()),
            &["--version"],
            true,
        ));
    }
    if apple {
        tools.push(executable_identity(
            "xcrun",
            Some(OsString::from("xcrun")),
            &["--version"],
            true,
        ));
        tools.push(sdk_identity(target));
    }
    tools
}

fn feature_enabled(features: &str, feature: &str) -> bool {
    features.split(',').any(|value| value == feature)
}

fn native_compiler_selection(target: &str) -> NativeCompilerSelection {
    for source in native_compiler_env_names(target) {
        if let Some(value) = env::var_os(&source) {
            if value.is_empty() {
                return default_native_compiler_selection(target, format!("empty={source}"));
            }
            return NativeCompilerSelection {
                exact: exact_compiler_value(&value) && !cc_wrapper_fallback_selected(),
                source,
                value,
            };
        }
    }
    default_native_compiler_selection(
        target,
        format!("default={}", default_native_compiler(target)),
    )
}

fn default_native_compiler_selection(target: &str, source: String) -> NativeCompilerSelection {
    let value = OsString::from(default_native_compiler(target));
    NativeCompilerSelection {
        source,
        exact: (target.contains("apple") || env::var("HOST").ok().as_deref() == Some(target))
            && !cc_wrapper_fallback_selected(),
        value,
    }
}

fn native_compiler_env_names(target: &str) -> Vec<String> {
    let target_with_underscores = target.replace(['-', '.'], "_");
    let kind = if env::var("HOST").ok().as_deref() == Some(target) {
        "HOST"
    } else {
        "TARGET"
    };
    vec![
        format!("CC_{target}"),
        format!("CC_{target_with_underscores}"),
        format!("{kind}_CC"),
        "CC".to_owned(),
    ]
}

fn exact_compiler_value(value: &OsStr) -> bool {
    let path = PathBuf::from(value);
    path.is_file() || value.to_string_lossy().split_whitespace().count() == 1
}

fn cc_wrapper_fallback_selected() -> bool {
    let Some(wrapper) = env::var_os("RUSTC_WRAPPER") else {
        return false;
    };
    let Some(stem) = Path::new(&wrapper).file_stem().and_then(OsStr::to_str) else {
        return false;
    };
    matches!(stem, "sccache" | "cachepot" | "buildcache" | "kache")
}

fn default_native_compiler(target: &str) -> &'static str {
    let apple_platform = ["-ios", "-tvos", "-watchos", "-visionos"]
        .iter()
        .any(|os| target.contains(os));
    let linker_plugin_lto = env::var_os("CARGO_ENCODED_RUSTFLAGS")
        .is_some_and(|flags| flags.to_string_lossy().contains("linker-plugin-lto"));
    if apple_platform || linker_plugin_lto {
        "clang"
    } else {
        "cc"
    }
}

fn native_tools_sha256(
    summary: &ToolSummary,
    compiler: Option<&NativeCompilerSelection>,
    complete: bool,
) -> String {
    if !complete {
        return UNKNOWN.to_owned();
    }
    let mut fields = vec![("tool-summary".to_owned(), summary.sha256.clone())];
    if let Some(compiler) = compiler {
        fields.push(("compiler-source".to_owned(), compiler.source.clone()));
    }
    digest_owned_fields(&fields)
}

fn native_tool_versions(
    summary: &ToolSummary,
    compiler: Option<&NativeCompilerSelection>,
) -> String {
    compiler.map_or_else(
        || summary.versions.clone(),
        |compiler| format!("compiler-source={};{}", compiler.source, summary.versions),
    )
}

fn cuda_compiler_path() -> OsString {
    if let Some(root) = env::var_os("CUDA_HOME") {
        return PathBuf::from(root).join("bin/nvcc").into_os_string();
    }
    for root in ["/opt/cuda", "/usr/local/cuda"] {
        let path = Path::new(root).join("bin/nvcc");
        if path.is_file() {
            return path.into_os_string();
        }
    }
    PathBuf::from("/usr/local/cuda/bin/nvcc").into_os_string()
}

fn sdk_identity(target: &str) -> ToolIdentity {
    let sdk = sdk_name(target);
    let path = sdk_path(&sdk);
    let version = sdk_version(&path);
    let complete = path != UNKNOWN && version != UNKNOWN && Path::new(&path).is_dir();
    let metadata = format!("{path}\0{version}");
    ToolIdentity {
        name: "sdk".to_owned(),
        path,
        sha256: if complete {
            sha256_hex(metadata.as_bytes())
        } else {
            UNKNOWN.to_owned()
        },
        version,
        complete,
    }
}

fn sdk_path(sdk: &str) -> String {
    let sdk_root = env::var_os("SDKROOT")
        .map(PathBuf::from)
        .filter(|path| sdk_root_is_usable(sdk, path));
    sdk_root
        .map(|path| path.to_string_lossy().into_owned())
        .or_else(|| {
            resolve_executable(OsStr::new("xcrun")).and_then(|xcrun| {
                command_text(
                    &xcrun,
                    &[
                        "--show-sdk-path".to_owned(),
                        "--sdk".to_owned(),
                        sdk.to_owned(),
                    ],
                )
            })
        })
        .unwrap_or_else(|| UNKNOWN.to_owned())
}

fn sdk_version(sdk_path: &str) -> String {
    resolve_executable(OsStr::new("xcrun"))
        .and_then(|xcrun| {
            command_text(
                &xcrun,
                &[
                    "--show-sdk-version".to_owned(),
                    "--sdk".to_owned(),
                    sdk_path.to_owned(),
                ],
            )
        })
        .unwrap_or_else(|| UNKNOWN.to_owned())
}

fn sdk_name(target: &str) -> String {
    if target.contains("-ios-sim") {
        "iphonesimulator".to_owned()
    } else if target.contains("-ios-macabi") {
        "macosx".to_owned()
    } else if target.contains("-ios") {
        "iphoneos".to_owned()
    } else if target.contains("-tvos-sim") {
        "appletvsimulator".to_owned()
    } else if target.contains("-tvos") {
        "appletvos".to_owned()
    } else if target.contains("-watchos-sim") {
        "watchsimulator".to_owned()
    } else if target.contains("-watchos") {
        "watchos".to_owned()
    } else if target.contains("-visionos-sim") {
        "xrsimulator".to_owned()
    } else if target.contains("-visionos") {
        "xros".to_owned()
    } else {
        "macosx".to_owned()
    }
}

fn sdk_root_is_usable(sdk: &str, path: &Path) -> bool {
    if !path.is_absolute() || path == Path::new("/") || !path.is_dir() {
        return false;
    }
    let text = path.to_string_lossy();
    !sdk_root_has_wrong_platform(sdk, &text)
}

fn sdk_root_has_wrong_platform(sdk: &str, path: &str) -> bool {
    match sdk {
        "appletvos" => contains_any(path, &["TVSimulator.platform", "MacOSX.platform"]),
        "appletvsimulator" => contains_any(path, &["TVOS.platform", "MacOSX.platform"]),
        "iphoneos" => contains_any(path, &["iPhoneSimulator.platform", "MacOSX.platform"]),
        "iphonesimulator" => contains_any(path, &["iPhoneOS.platform", "MacOSX.platform"]),
        "watchos" => contains_any(path, &["WatchSimulator.platform", "MacOSX.platform"]),
        "watchsimulator" => contains_any(path, &["WatchOS.platform", "MacOSX.platform"]),
        "xros" => contains_any(path, &["XRSimulator.platform", "MacOSX.platform"]),
        "xrsimulator" => contains_any(path, &["XROS.platform", "MacOSX.platform"]),
        _ => false,
    }
}

fn contains_any(value: &str, candidates: &[&str]) -> bool {
    candidates.iter().any(|candidate| value.contains(candidate))
}

fn command_text(path: &Path, arguments: &[String]) -> Option<String> {
    let output = Command::new(path).args(arguments).output().ok()?;
    if !output.status.success() {
        return None;
    }
    normalized_output(&output.stdout).or_else(|| normalized_output(&output.stderr))
}

fn feature_list() -> String {
    let mut features = env::vars()
        .filter_map(|(name, value)| {
            (value == "1")
                .then(|| {
                    name.strip_prefix("CARGO_FEATURE_")
                        .map(str::to_ascii_lowercase)
                })
                .flatten()
        })
        .collect::<Vec<_>>();
    features.sort_unstable();
    if features.is_empty() {
        "none".to_owned()
    } else {
        features.join(",")
    }
}

struct FlagInput {
    source: &'static str,
    raw: String,
    tokens: Vec<String>,
}

struct NormalizedFlags {
    text: String,
    redaction_count: usize,
}

fn flag_input() -> FlagInput {
    if let Ok(value) = env::var("CARGO_ENCODED_RUSTFLAGS") {
        if !value.is_empty() {
            return FlagInput {
                source: "CARGO_ENCODED_RUSTFLAGS",
                tokens: value.split('\x1f').map(str::to_owned).collect(),
                raw: value,
            };
        }
    }
    let raw = env::var("RUSTFLAGS").unwrap_or_default();
    FlagInput {
        source: "RUSTFLAGS",
        tokens: raw.split_whitespace().map(str::to_owned).collect(),
        raw,
    }
}

fn normalized_flags(tokens: &[String]) -> NormalizedFlags {
    let mut normalized = Vec::new();
    let mut tokens = tokens.iter().peekable();
    let mut redaction_count = 0;
    while let Some(token) = tokens.next() {
        if takes_private_value(token) {
            normalized.push(token_option(token));
            if tokens.next().is_some() {
                normalized.push("<redacted>".to_owned());
                redaction_count += 1;
            }
            continue;
        }
        if private_flag(token) {
            normalized.push(inline_redaction(token));
            redaction_count += 1;
            continue;
        }
        if token == "-C" {
            normalized.push(token.clone());
            if let Some(value) = tokens.peek() {
                if private_flag(value) {
                    let _ = tokens.next();
                    normalized.push("<redacted>".to_owned());
                    redaction_count += 1;
                }
            }
            continue;
        }
        normalized.push(token.to_owned());
    }
    NormalizedFlags {
        text: normalized.join(" "),
        redaction_count,
    }
}

fn private_flag(flag: &str) -> bool {
    let lower = flag.to_ascii_lowercase();
    lower.starts_with("--remap-path-prefix")
        || lower.starts_with("--extern")
        || lower.starts_with("--out-dir")
        || lower.starts_with("--sysroot")
        || lower.starts_with("-l/")
        || lower.starts_with("-l\\")
        || flag.contains('/')
        || flag.contains('\\')
}

fn token_option(flag: &str) -> String {
    flag.split_once('=').map_or_else(
        || flag.to_owned(),
        |(option, _)| format!("{option}=<redacted>"),
    )
}

fn inline_redaction(flag: &str) -> String {
    flag.split_once('=').map_or_else(
        || "<redacted>".to_owned(),
        |(option, _)| format!("{option}=<redacted>"),
    )
}

fn build_config_sha256(inputs: &ConfigInputs<'_>) -> String {
    let mut fields = CONFIG_INPUTS
        .iter()
        .map(|name| (*name, environment_input(name)))
        .collect::<Vec<_>>();
    fields.push(("LEONE_BUILD_FEATURES", inputs.features.to_owned()));
    fields.push(("LEONE_BUILD_RUSTC_VERSION", inputs.rustc.to_owned()));
    fields.push((
        "LEONE_BUILD_FLAGS_SOURCE",
        inputs.flag_input.source.to_owned(),
    ));
    fields.push((
        "LEONE_BUILD_FLAGS_RAW_SHA256",
        inputs.flags_raw_sha256.to_owned(),
    ));
    fields.push((
        "LEONE_BUILD_PROFILE_INPUTS_SHA256",
        inputs.profile_inputs.sha256.clone(),
    ));
    fields.push((
        "LEONE_BUILD_LINKER_INPUTS_SHA256",
        inputs.linker_inputs.sha256.clone(),
    ));
    fields.push((
        "LEONE_BUILD_TOOLCHAIN_SHA256",
        inputs.toolchain.sha256.clone(),
    ));
    fields.push((
        "LEONE_BUILD_NATIVE_PROVENANCE",
        inputs.native.status.clone(),
    ));
    fields.push((
        "LEONE_BUILD_NATIVE_TOOLS_SHA256",
        inputs.native.tools_sha256.clone(),
    ));
    digest_mixed_fields(&fields)
}

fn environment_input(name: &str) -> String {
    env::var(name).unwrap_or_else(|_| "<unset>".to_owned())
}

fn takes_private_value(flag: &str) -> bool {
    matches!(
        flag,
        "-L" | "--library-path" | "--extern" | "--out-dir" | "--remap-path-prefix" | "--sysroot"
    )
}

fn is_commit(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn git_text(root: &Path, arguments: &[&str]) -> Option<String> {
    let output = git_output(root, arguments)?;
    Some(String::from_utf8(output).ok()?.trim().to_owned())
}

fn git_bytes(root: &Path, arguments: &[&str]) -> Option<Vec<u8>> {
    git_output(root, arguments)
}

fn git_commit_exists(root: &Path, commit: &str) -> bool {
    let reference = format!("{commit}^{{commit}}");
    git_output(root, &["cat-file", "-e", &reference]).is_some()
}

fn git_output(root: &Path, arguments: &[&str]) -> Option<Vec<u8>> {
    let output = Command::new("git")
        .current_dir(root)
        .args(arguments)
        .output()
        .ok()?;
    output.status.success().then_some(output.stdout)
}

fn digest_fields(fields: &[(&str, &str)]) -> String {
    let mut hasher = Sha256::new();
    hasher.update((fields.len() as u64).to_le_bytes());
    for (name, value) in fields {
        update_length_prefixed(&mut hasher, name.as_bytes());
        update_length_prefixed(&mut hasher, value.as_bytes());
    }
    hex_digest(hasher.finalize())
}

fn digest_mixed_fields(fields: &[(&str, String)]) -> String {
    let mut hasher = Sha256::new();
    hasher.update((fields.len() as u64).to_le_bytes());
    for (name, value) in fields {
        update_length_prefixed(&mut hasher, name.as_bytes());
        update_length_prefixed(&mut hasher, value.as_bytes());
    }
    hex_digest(hasher.finalize())
}

fn digest_owned_fields(fields: &[(String, String)]) -> String {
    let mut hasher = Sha256::new();
    hasher.update((fields.len() as u64).to_le_bytes());
    for (name, value) in fields {
        update_length_prefixed(&mut hasher, name.as_bytes());
        update_length_prefixed(&mut hasher, value.as_bytes());
    }
    hex_digest(hasher.finalize())
}

fn update_length_prefixed(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn sha256_hex(bytes: impl AsRef<[u8]>) -> String {
    hex_digest(Sha256::digest(bytes))
}

fn hex_digest(bytes: impl AsRef<[u8]>) -> String {
    bytes
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn emit_value(name: &str, value: &str) {
    println!("cargo:rustc-env={name}={value}");
}

fn known_profile_inputs() -> Vec<String> {
    PROFILE_KINDS
        .iter()
        .flat_map(|profile| {
            PROFILE_KEYS
                .iter()
                .map(move |key| format!("CARGO_PROFILE_{profile}_{key}"))
        })
        .collect()
}

fn native_requested(target: &str, features: &str) -> bool {
    target.contains("apple")
        || feature_enabled(features, "cuda")
        || feature_enabled(features, "metal")
}

fn watched_tool_paths(target: &str, features: &str) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    for name in TOOL_ENV_INPUTS {
        add_tool_path(&mut paths, env::var_os(name));
    }
    for name in scoped_linker_inputs()
        .names
        .split(',')
        .filter(|name| name.starts_with("CARGO_TARGET_") && name.ends_with("_LINKER"))
    {
        add_tool_path(&mut paths, env::var_os(name));
    }
    if native_requested(target, features) {
        if feature_enabled(features, "cuda") {
            add_tool_path(&mut paths, Some(cuda_compiler_path()));
        }
        if feature_enabled(features, "metal") || target.contains("apple") {
            let compiler = native_compiler_selection(target);
            add_tool_path(&mut paths, Some(compiler.value));
        }
        if target.contains("apple") {
            add_tool_path(&mut paths, Some(OsString::from("xcrun")));
        }
    }
    paths
}

fn add_tool_path(paths: &mut Vec<PathBuf>, value: Option<OsString>) {
    let Some(path) = value.and_then(|value| resolve_executable(&value)) else {
        return;
    };
    if !paths.iter().any(|other| other == &path) {
        paths.push(path);
    }
}

fn watch_inputs(root: &Path) {
    watch_source_inputs(root);
    watch_config_inputs();
    watch_feature_inputs();
    watch_git_inputs(root);
    watch_tool_inputs();
}

fn watch_source_inputs(root: &Path) {
    for relative in SOURCE_PATHS {
        println!("cargo:rerun-if-changed={}", root.join(relative).display());
    }
    println!("cargo:rerun-if-changed={}", root.join(".git").display());
}

fn watch_config_inputs() {
    for name in CONFIG_INPUTS {
        println!("cargo:rerun-if-env-changed={name}");
    }
    println!("cargo:rerun-if-env-changed=PATH");
    for name in known_profile_inputs() {
        println!("cargo:rerun-if-env-changed={name}");
    }
    for name in scoped_profile_inputs()
        .names
        .split(',')
        .filter(|name| !name.is_empty() && *name != "none")
    {
        println!("cargo:rerun-if-env-changed={name}");
    }
    for name in scoped_linker_inputs()
        .names
        .split(',')
        .filter(|name| !name.is_empty() && *name != "none")
    {
        println!("cargo:rerun-if-env-changed={name}");
    }
    let target = environment_value("TARGET");
    for name in native_compiler_env_names(&target) {
        println!("cargo:rerun-if-env-changed={name}");
    }
}

fn watch_feature_inputs() {
    for name in ["CARGO_FEATURE_CUDA", "CARGO_FEATURE_METAL"] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    for (name, _) in env::vars().filter(|(name, _)| name.starts_with("CARGO_FEATURE_")) {
        println!("cargo:rerun-if-env-changed={name}");
    }
}

fn watch_git_inputs(root: &Path) {
    for name in ["HEAD", "index", "packed-refs"] {
        watch_git_path(root, name);
    }
    if let Some(reference) = git_text(root, &["symbolic-ref", "-q", "HEAD"]) {
        watch_git_path(root, &reference);
    }
}

fn watch_tool_inputs() {
    let target = environment_value("TARGET");
    let features = feature_list();
    for path in watched_tool_paths(&target, &features) {
        println!("cargo:rerun-if-changed={}", path.display());
    }
}

fn watch_git_path(root: &Path, name: &str) {
    let Some(path) = git_text(root, &["rev-parse", "--git-path", name]) else {
        return;
    };
    let path = PathBuf::from(path);
    let path = if path.is_absolute() {
        path
    } else {
        root.join(path)
    };
    if path.exists() {
        println!("cargo:rerun-if-changed={}", path.display());
    }
}
