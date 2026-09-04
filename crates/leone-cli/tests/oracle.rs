use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("CLI crate is under the workspace root")
        .to_owned()
}

fn model() -> PathBuf {
    root().join("models/Qwen3-8B-Q4_K_M.gguf")
}

fn llama_model() -> PathBuf {
    root().join("models/Llama-3.2-1B-Instruct-Q4_K_M.gguf")
}

fn run_generation(backend: &str, token_path: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_leone"))
        .args([
            "generate",
            "-m",
            model().to_str().expect("model path is UTF-8"),
            "-p",
            "Hello",
            "-n",
            "8",
            "--backend",
            backend,
            "--debug-tokens",
            token_path.to_str().expect("token path is UTF-8"),
        ])
        .output()
        .expect("generation process starts")
}

fn run_cuda_decode(token_path: &Path, logit_path: &Path, eager: bool) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_leone"));
    command.args([
        "generate",
        "-m",
        model().to_str().expect("model path is UTF-8"),
        "-p",
        "The capital of France is",
        "-n",
        "32",
        "--debug-tokens",
        token_path.to_str().expect("token path is UTF-8"),
        "--debug-logits",
        logit_path.to_str().expect("logit path is UTF-8"),
    ]);
    if eager {
        command.arg("--eager-decode");
    }
    command.output().expect("generation process starts")
}

fn run_llama_cuda_decode(token_path: &Path, logit_path: &Path, eager: bool) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_leone"));
    command.args([
        "generate",
        "-m",
        llama_model().to_str().expect("model path is UTF-8"),
        "-p",
        "The capital of Norway is",
        "-n",
        "32",
        "--debug-tokens",
        token_path.to_str().expect("token path is UTF-8"),
        "--debug-logits",
        logit_path.to_str().expect("logit path is UTF-8"),
    ]);
    if eager {
        command.arg("--eager-decode");
    }
    command.output().expect("Llama generation process starts")
}

#[test]
#[ignore = "requires the Qwen3 model and an idle RTX 4090"]
fn fixed_prompts_match_first_token_and_report_later_divergence() {
    let output = Command::new(root().join("scripts/token-match.sh"))
        .env("TOKENS", "64")
        .output()
        .expect("token oracle starts");
    assert!(
        output.status.success(),
        "token oracle failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
#[ignore = "loads the full Qwen3 model into the CPU and CUDA backends"]
fn cpu_and_cuda_match_for_eight_tokens() {
    let temporary = std::env::temp_dir().join(format!("leone-oracle-{}", uuid::Uuid::new_v4()));
    fs::create_dir(&temporary).expect("temporary directory is created");
    let cuda_path = temporary.join("cuda.json");
    let cpu_path = temporary.join("cpu.json");
    let cuda = run_generation("cuda", &cuda_path);
    assert!(
        cuda.status.success(),
        "CUDA generation failed: {}",
        String::from_utf8_lossy(&cuda.stderr)
    );
    let cpu = run_generation("cpu", &cpu_path);
    assert!(
        cpu.status.success(),
        "CPU generation failed: {}",
        String::from_utf8_lossy(&cpu.stderr)
    );
    assert_eq!(
        fs::read(&cpu_path).expect("CPU tokens are readable"),
        fs::read(&cuda_path).expect("CUDA tokens are readable")
    );
    fs::remove_dir_all(&temporary).expect("temporary directory is removed");
}

#[test]
#[ignore = "requires the Qwen3 model and an idle RTX 4090"]
fn cuda_graph_and_eager_decode_match_for_32_tokens() {
    let temporary = std::env::temp_dir().join(format!("leone-graph-{}", uuid::Uuid::new_v4()));
    fs::create_dir(&temporary).expect("temporary directory is created");
    let graph_path = temporary.join("graph.json");
    let eager_path = temporary.join("eager.json");
    let graph_logits = temporary.join("graph.logits");
    let eager_logits = temporary.join("eager.logits");
    let graph = run_cuda_decode(&graph_path, &graph_logits, false);
    assert!(
        graph.status.success(),
        "CUDA graph generation failed: {}",
        String::from_utf8_lossy(&graph.stderr)
    );
    let eager = run_cuda_decode(&eager_path, &eager_logits, true);
    assert!(
        eager.status.success(),
        "eager CUDA generation failed: {}",
        String::from_utf8_lossy(&eager.stderr)
    );
    assert_eq!(
        fs::read(&eager_path).expect("eager tokens are readable"),
        fs::read(&graph_path).expect("graph tokens are readable")
    );
    assert_eq!(
        fs::read(&eager_logits).expect("eager logits are readable"),
        fs::read(&graph_logits).expect("graph logits are readable")
    );
    fs::remove_dir_all(&temporary).expect("temporary directory is removed");
}

#[test]
#[ignore = "requires the Llama 3.2 model and an idle RTX 4090"]
fn llama_cuda_graph_and_eager_decode_match_for_32_tokens() {
    let temporary =
        std::env::temp_dir().join(format!("leone-llama-graph-{}", uuid::Uuid::new_v4()));
    fs::create_dir(&temporary).expect("temporary directory is created");
    let graph_path = temporary.join("graph.json");
    let eager_path = temporary.join("eager.json");
    let graph_logits = temporary.join("graph.logits");
    let eager_logits = temporary.join("eager.logits");
    let graph = run_llama_cuda_decode(&graph_path, &graph_logits, false);
    assert!(
        graph.status.success(),
        "Llama CUDA graph generation failed: {}",
        String::from_utf8_lossy(&graph.stderr)
    );
    let eager = run_llama_cuda_decode(&eager_path, &eager_logits, true);
    assert!(
        eager.status.success(),
        "Llama eager CUDA generation failed: {}",
        String::from_utf8_lossy(&eager.stderr)
    );
    assert_eq!(
        fs::read(&eager_path).expect("eager tokens are readable"),
        fs::read(&graph_path).expect("graph tokens are readable")
    );
    assert_eq!(
        fs::read(&eager_logits).expect("eager logits are readable"),
        fs::read(&graph_logits).expect("graph logits are readable")
    );
    fs::remove_dir_all(&temporary).expect("temporary directory is removed");
}

#[test]
#[ignore = "requires the Qwen3 model and an idle RTX 4090"]
fn repeated_generation_reproduces_one_transcript() {
    let temporary =
        std::env::temp_dir().join(format!("leone-determinism-{}", uuid::Uuid::new_v4()));
    fs::create_dir(&temporary).expect("temporary directory is created");
    let first_path = temporary.join("first.json");
    let second_path = temporary.join("second.json");

    let first = run_generation("cuda", &first_path);
    assert!(
        first.status.success(),
        "first generation failed: {}",
        String::from_utf8_lossy(&first.stderr)
    );
    let second = run_generation("cuda", &second_path);
    assert!(
        second.status.success(),
        "second generation failed: {}",
        String::from_utf8_lossy(&second.stderr)
    );

    // From schema v6, a runtime receipt claims that one model artifact and one
    // prompt reproduce one token stream.
    assert_eq!(
        fs::read(&first_path).expect("first tokens are readable"),
        fs::read(&second_path).expect("second tokens are readable")
    );
    fs::remove_dir_all(&temporary).expect("temporary directory is removed");
}

/// Runs one generation and returns its token identifiers.
fn generation_tokens(extra: &[&str], path: &Path) -> Vec<u8> {
    let mut command = Command::new(root().join("target/release/leone"));
    command.args([
        "generate",
        "-m",
        root().join("models/Qwen3-8B-Q4_K_M.gguf").to_str().unwrap(),
        "-p",
        "The capital of France is Paris. The capital of Italy is Rome. \
         The capital of France is",
        "-n",
        "48",
        "--debug-tokens",
        path.to_str().unwrap(),
    ]);
    command.args(extra);
    let output = command.output().expect("generation process starts");
    assert!(
        output.status.success(),
        "generation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    fs::read(path).expect("tokens are readable")
}

#[test]
#[ignore = "requires the Qwen3 model and an idle RTX 4090"]
fn greedy_speculation_emits_the_same_tokens_as_plain_decoding() {
    // Under greedy sampling the target distribution is a point mass. A
    // proposal is accepted only when it equals the argmax, and rejection
    // replaces it with that argmax. The emitted stream must match plain
    // decode token for token.
    let temporary = std::env::temp_dir().join(format!("leone-spec-{}", uuid::Uuid::new_v4()));
    fs::create_dir(&temporary).expect("temporary directory is created");
    for draft in ["1", "4", "8"] {
        let plain = generation_tokens(&["--eager-decode"], &temporary.join("plain.json"));
        let speculated = generation_tokens(
            &["--eager-decode", "--draft", draft],
            &temporary.join("speculated.json"),
        );
        assert_eq!(
            plain, speculated,
            "speculation changed the greedy token stream at draft {draft}"
        );
    }
    let plain = generation_tokens(&["--eager-decode"], &temporary.join("plain.json"));
    let adaptive = generation_tokens(
        &["--eager-decode", "--adaptive-draft"],
        &temporary.join("adaptive.json"),
    );
    assert_eq!(
        plain, adaptive,
        "adaptive drafting changed the greedy stream"
    );
    fs::remove_dir_all(&temporary).expect("temporary directory is removed");
}

#[test]
#[ignore = "requires the Qwen3 model and an idle RTX 4090"]
fn stochastic_speculation_stays_reproducible() {
    // Speculative sampling reproduces the target distribution, not the token a
    // particular uniform draw would have selected without it. Accepting a
    // proposal couples the randomness differently from sampling directly.
    // Sampler unit tests check the distributional claim over 400,000 trials.
    // This test checks that a seeded speculative run is itself reproducible.
    let temporary = std::env::temp_dir().join(format!("leone-spec-{}", uuid::Uuid::new_v4()));
    fs::create_dir(&temporary).expect("temporary directory is created");
    for extra in [
        vec![
            "--temp", "0.8", "--top-p", "0.95", "--seed", "12345", "--draft", "4",
        ],
        vec![
            "--temp", "1.5", "--min-p", "0.05", "--seed", "777", "--draft", "4",
        ],
        vec![
            "--temp",
            "1.0",
            "--top-n-sigma",
            "1.0",
            "--seed",
            "99",
            "--draft",
            "6",
        ],
        vec![
            "--temp",
            "0.8",
            "--top-p",
            "0.95",
            "--seed",
            "12345",
            "--adaptive-draft",
        ],
    ] {
        let first = generation_tokens(&extra, &temporary.join("first.json"));
        let second = generation_tokens(&extra, &temporary.join("second.json"));
        assert_eq!(
            first, second,
            "speculative run was not reproducible for {extra:?}"
        );
    }
    fs::remove_dir_all(&temporary).expect("temporary directory is removed");
}

#[test]
#[ignore = "requires the Qwen3 model and an idle RTX 4090"]
fn stochastic_sampling_is_reproducible_and_seed_dependent() {
    let temporary = std::env::temp_dir().join(format!("leone-seed-{}", uuid::Uuid::new_v4()));
    fs::create_dir(&temporary).expect("temporary directory is created");
    let base = ["--temp", "1.0", "--top-p", "0.9", "--seed", "42"];
    let first = generation_tokens(&base, &temporary.join("first.json"));
    let second = generation_tokens(&base, &temporary.join("second.json"));
    assert_eq!(first, second, "the same seed produced a different stream");

    let other = ["--temp", "1.0", "--top-p", "0.9", "--seed", "43"];
    let third = generation_tokens(&other, &temporary.join("third.json"));
    assert_ne!(first, third, "a different seed produced the same stream");
    fs::remove_dir_all(&temporary).expect("temporary directory is removed");
}

#[test]
#[ignore = "requires the Qwen3 model and an idle RTX 4090"]
fn penalties_change_the_token_stream() {
    // An earlier version parsed and validated penalties but never applied them
    // on the host sampler. A penalty that reaches the logits must change the
    // emitted stream.
    let temporary = std::env::temp_dir().join(format!("leone-penalty-{}", uuid::Uuid::new_v4()));
    fs::create_dir(&temporary).expect("temporary directory is created");
    let plain = generation_tokens(&[], &temporary.join("plain.json"));
    for extra in [
        vec!["--repeat-penalty", "1.3"],
        vec!["--presence-penalty", "1.0"],
        vec!["--frequency-penalty", "1.0"],
        vec!["--dry-multiplier", "4.0", "--dry-base", "2.0"],
    ] {
        let penalized = generation_tokens(&extra, &temporary.join("penalized.json"));
        assert_ne!(
            plain, penalized,
            "{extra:?} did not change the token stream"
        );
    }
    fs::remove_dir_all(&temporary).expect("temporary directory is removed");
}
