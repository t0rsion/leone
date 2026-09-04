use chrono::Utc;
use ed25519_dalek::SigningKey;
use leone::runtime_service::{
    LeoneRuntimeDriver, RuntimeQuantumExecutor, ScheduledGenerationRequest,
};
use leone::scheduler::{AdmissionOutcome, RequestId, RequestSpec, RequestStatus, SchedulerPolicy};
use leone::service::{QuantumExecutor, QuantumOutput, ScheduledService};
use leone::{
    token_stream_sha256, AdaptiveDrafter, AdaptiveDrafterConfig, Backend, CpuBackend,
    DecodeExecution, GenerateOptions, GenerationSession, HibernatedSession, KvCacheDtype,
    MirostatConfig, OutputConstraint, Penalties, Runtime, RuntimeError, Sampler, SessionReuseClass,
    Speculation, SuffixDrafter, Temperature, Truncation,
};
use leone_cuda::CudaBackend;
use leone_receipt::{
    sha256_bytes, sha256_file, write_response_receipt, ResponseClaim, ResponseReceipt,
    SessionReplayRecord, RESPONSE_SCHEMA_VERSION,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::error::Error;
use std::fs::{self, OpenOptions};
use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use uuid::Uuid;

const DEFAULT_BIND: &str = "127.0.0.1:8080";
const DEFAULT_MAX_TOKENS: usize = 512;
const MAX_HEADER_BYTES: usize = 64 * 1024;
const MAX_BODY_BYTES: usize = 4 * 1024 * 1024;
const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>";
const LLAMA_BEGIN: &str = "<|begin_of_text|>";
const LLAMA_HEADER_START: &str = "<|start_header_id|>";
const LLAMA_HEADER_END: &str = "<|end_header_id|>";
const LLAMA_EOT: &str = "<|eot_id|>";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BackendChoice {
    Cuda,
    Cpu,
}

#[derive(Debug)]
struct ServeArgs {
    model: PathBuf,
    bind: SocketAddr,
    sessions: usize,
    hibernated_sessions: usize,
    backend: BackendChoice,
    kv_cache_dtype: KvCacheDtype,
    receipts: PathBuf,
    signing_key: PathBuf,
    session_store: Option<PathBuf>,
    allow_remote: bool,
    plan: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    max_completion_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    top_p: Option<f64>,
    #[serde(default)]
    min_p: Option<f64>,
    #[serde(default)]
    presence_penalty: Option<f64>,
    #[serde(default)]
    frequency_penalty: Option<f64>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    n: Option<usize>,
    #[serde(default)]
    stop: Option<Value>,
    #[serde(default)]
    tools: Option<Vec<ToolDefinition>>,
    #[serde(default)]
    tool_choice: Option<Value>,
    #[serde(default)]
    response_format: Option<Value>,
    #[serde(default)]
    logprobs: Option<bool>,
    #[serde(default)]
    top_logprobs: Option<usize>,
    #[serde(default)]
    user: Option<String>,
    #[serde(default)]
    leone_session: Option<String>,
    #[serde(default)]
    leone_fork_session: Option<String>,
    #[serde(default)]
    mirostat_tau: Option<f64>,
    #[serde(default)]
    mirostat_eta: Option<f64>,
    #[serde(default)]
    draft_tokens: Option<usize>,
    #[serde(default)]
    adaptive_speculation: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct Message {
    role: String,
    content: Value,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<RequestToolCall>>,
    #[serde(default)]
    tool_call_id: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ToolDefinition {
    #[serde(rename = "type")]
    kind: String,
    function: ToolFunction,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ToolFunction {
    name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    parameters: Option<Value>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RequestToolCall {
    #[serde(default)]
    id: Option<String>,
    #[serde(rename = "type")]
    kind: String,
    function: RequestToolCallFunction,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RequestToolCallFunction {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrlPart },
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ImageUrlPart {
    url: String,
    #[serde(default)]
    detail: Option<String>,
}

#[derive(Debug, Serialize)]
struct GeneratedToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    function: GeneratedToolCallFunction,
}

#[derive(Debug, Serialize)]
struct GeneratedToolCallFunction {
    name: String,
    arguments: String,
}

struct StoredSession<B: Backend> {
    generation: GenerationSession<B>,
    last_used: u64,
}

struct StoredHibernation {
    generation: HibernatedSession,
    last_used: u64,
}

#[derive(Debug, Clone)]
struct StoredArchive {
    archive: leone::SessionArchive,
    last_used: u64,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct SessionReference {
    schema_version: u32,
    session_id: String,
    blob_sha256: String,
    last_used: u64,
}

#[derive(Debug)]
struct SessionStore {
    root: PathBuf,
}

impl SessionStore {
    const REFERENCE_SCHEMA_VERSION: u32 = 1;

    fn open(
        root: PathBuf,
        model_sha256: &str,
    ) -> Result<(Self, HashMap<String, StoredArchive>), Box<dyn Error>> {
        fs::create_dir_all(root.join("blobs"))?;
        fs::create_dir_all(root.join("refs"))?;
        let store = Self { root };
        let archives = load_session_archives(&store, model_sha256)?;
        Ok((store, archives))
    }

    fn persist(
        &self,
        session_id: &str,
        model_sha256: &str,
        checkpoint: &leone::GenerationCheckpoint,
        last_used: u64,
    ) -> Result<StoredArchive, Box<dyn Error>> {
        let archive = leone::SessionArchive::new(model_sha256, checkpoint)?;
        let blob = archive.to_json()?;
        let blob_sha256 = sha256_bytes(&blob);
        let blob_path = self.root.join("blobs").join(format!("{blob_sha256}.json"));
        write_session_blob(&blob_path, &blob)?;
        write_session_reference(self, session_id, &blob_sha256, last_used)?;
        Ok(StoredArchive { archive, last_used })
    }

    fn remove(&self, session_id: &str) -> io::Result<()> {
        let key = sha256_bytes(session_id.as_bytes());
        let path = self.root.join("refs").join(format!("{key}.json"));
        match fs::remove_file(path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error),
        }
    }
}

fn load_session_archives(
    store: &SessionStore,
    model_sha256: &str,
) -> Result<HashMap<String, StoredArchive>, Box<dyn Error>> {
    let mut archives = HashMap::new();
    for entry in fs::read_dir(store.root.join("refs"))? {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        let (session_id, archive) = load_session_archive(store, &entry, model_sha256)?;
        archives.insert(session_id, archive);
    }
    Ok(archives)
}

fn load_session_archive(
    store: &SessionStore,
    entry: &std::fs::DirEntry,
    model_sha256: &str,
) -> Result<(String, StoredArchive), Box<dyn Error>> {
    let bytes = fs::read(entry.path())?;
    let reference: SessionReference = serde_json::from_slice(&bytes).map_err(invalid_json)?;
    validate_session_reference(entry, &reference)?;
    let blob = load_session_blob(store, &reference)?;
    let archive = leone::SessionArchive::from_json(&blob, model_sha256)?;
    Ok((
        reference.session_id.clone(),
        StoredArchive {
            archive,
            last_used: reference.last_used,
        },
    ))
}

fn validate_session_reference(
    entry: &std::fs::DirEntry,
    reference: &SessionReference,
) -> Result<(), Box<dyn Error>> {
    if reference.schema_version != SessionStore::REFERENCE_SCHEMA_VERSION {
        return Err(invalid_data(format!(
            "unsupported session reference schema {} in {}",
            reference.schema_version,
            entry.path().display()
        ))
        .into());
    }
    let key = sha256_bytes(reference.session_id.as_bytes());
    let expected_name = format!("{key}.json");
    if entry.file_name() != std::ffi::OsStr::new(&expected_name) {
        return Err(invalid_data(format!(
            "session reference name does not match its session ID: {}",
            entry.path().display()
        ))
        .into());
    }
    Ok(())
}

fn load_session_blob(
    store: &SessionStore,
    reference: &SessionReference,
) -> Result<Vec<u8>, Box<dyn Error>> {
    let blob_path = store
        .root
        .join("blobs")
        .join(format!("{}.json", reference.blob_sha256));
    let blob = fs::read(&blob_path)?;
    if sha256_bytes(&blob) != reference.blob_sha256 {
        return Err(invalid_data(format!(
            "session blob digest does not match its name: {}",
            blob_path.display()
        ))
        .into());
    }
    Ok(blob)
}

fn write_session_blob(path: &Path, blob: &[u8]) -> Result<(), Box<dyn Error>> {
    match OpenOptions::new().write(true).create_new(true).open(path) {
        Ok(mut file) => {
            file.write_all(blob)?;
            file.sync_all()?;
        }
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
            if fs::read(path)? != blob {
                return Err(
                    invalid_data(format!("session blob collision at {}", path.display())).into(),
                );
            }
        }
        Err(error) => return Err(error.into()),
    }
    Ok(())
}

fn write_session_reference(
    store: &SessionStore,
    session_id: &str,
    blob_sha256: &str,
    last_used: u64,
) -> Result<(), Box<dyn Error>> {
    let reference = SessionReference {
        schema_version: SessionStore::REFERENCE_SCHEMA_VERSION,
        session_id: session_id.to_owned(),
        blob_sha256: blob_sha256.to_owned(),
        last_used,
    };
    let key = sha256_bytes(session_id.as_bytes());
    let path = store.root.join("refs").join(format!("{key}.json"));
    let temporary = store
        .root
        .join("refs")
        .join(format!(".{key}.{}.tmp", Uuid::new_v4().simple()));
    fs::write(&temporary, serde_json::to_vec(&reference)?)?;
    fs::rename(&temporary, path)?;
    Ok(())
}

struct Server<B: Backend> {
    runtime: Runtime<B>,
    model_id: String,
    model_sha256: String,
    sessions: HashMap<String, StoredSession<B>>,
    hibernated: HashMap<String, StoredHibernation>,
    persisted: HashMap<String, StoredArchive>,
    session_store: Option<SessionStore>,
    max_sessions: usize,
    max_hibernated_sessions: usize,
    clock: u64,
    kv_cache_dtype: KvCacheDtype,
    execution_plan: Option<crate::execution_plan::PlanSelection>,
    receipts: PathBuf,
    signing_key: SigningKey,
}

struct HttpRequest {
    method: String,
    path: String,
    headers: BTreeMap<String, String>,
    body: Vec<u8>,
}

struct ChatPlan {
    request: ChatRequest,
    prompt_tokens: Vec<u32>,
    options: GenerateOptions,
    session_id: String,
    fork_parent: Option<String>,
    prefix_reused_tokens: usize,
    request_sha256: String,
    created: u64,
    completion_id: String,
}

struct PendingChat {
    plan: ChatPlan,
    stream: TcpStream,
}

struct ChatTask<B: Backend> {
    request: ChatRequest,
    prompt_tokens: Vec<u32>,
    transcript: Vec<u32>,
    options: GenerateOptions,
    remaining_tokens: usize,
    session_id: String,
    stored: StoredSession<B>,
    stream: TcpStream,
    peer: TcpStream,
    request_sha256: String,
    created: u64,
    completion_id: String,
    streamed: Utf8Stream,
    tokens: Vec<u32>,
    eos: bool,
    disconnected: bool,
    failure: Option<String>,
    headers_written: bool,
}

struct ChatAdmission(RefCell<Option<PendingChat>>);

struct ServerExecutor<B: Backend> {
    server: Server<B>,
    tasks: BTreeMap<RequestId, ChatTask<B>>,
}

type ChatService<B> = ScheduledService<ServerExecutor<B>>;

impl<B: Backend> QuantumExecutor for ServerExecutor<B> {
    type Request = ChatAdmission;
    type Error = io::Error;

    fn begin(&mut self, request_id: RequestId, request: &Self::Request) -> Result<(), Self::Error> {
        if self.tasks.contains_key(&request_id) {
            return Err(invalid_data(format!(
                "executor state already exists for request {}",
                request_id.0
            )));
        }
        let pending = request
            .0
            .borrow_mut()
            .take()
            .ok_or_else(|| invalid_data("chat admission was already consumed"))?;
        let peer = pending.stream.try_clone()?;
        peer.set_nonblocking(true)?;
        let stored = self
            .server
            .lease_session(&pending.plan)
            .map_err(|error| invalid_data(error.to_string()))?;
        let ChatPlan {
            request,
            prompt_tokens,
            options,
            session_id,
            fork_parent: _,
            prefix_reused_tokens: _,
            request_sha256,
            created,
            completion_id,
        } = pending.plan;
        let remaining_tokens = options.max_tokens;
        self.tasks.insert(
            request_id,
            ChatTask {
                request,
                transcript: prompt_tokens.clone(),
                prompt_tokens,
                options,
                remaining_tokens,
                session_id,
                stored,
                stream: pending.stream,
                peer,
                request_sha256,
                created,
                completion_id,
                streamed: Utf8Stream::default(),
                tokens: Vec::with_capacity(remaining_tokens),
                eos: false,
                disconnected: false,
                failure: None,
                headers_written: false,
            },
        );
        Ok(())
    }

    fn execute(
        &mut self,
        request_id: RequestId,
        token_budget: u32,
    ) -> Result<QuantumOutput, Self::Error> {
        let task = self
            .tasks
            .get_mut(&request_id)
            .ok_or_else(|| invalid_data(format!("unknown request {}", request_id.0)))?;
        let model_id = self.server.model_id.clone();
        write_executor_headers(task, &model_id)?;
        let budget = usize::try_from(token_budget)
            .expect("u32 fits usize")
            .min(task.remaining_tokens);
        let (tokens, cancelled) =
            match execute_task_quantum(&mut self.server.runtime, task, budget, &model_id) {
                Ok(result) => result,
                Err(error) => return Ok(quantum_error(task, error)),
            };
        if tokens.len() > budget {
            return Err(invalid_data(format!(
                "runtime emitted {} tokens for a {budget}-token quantum",
                tokens.len()
            )));
        }
        task.remaining_tokens -= tokens.len();
        task.transcript.extend_from_slice(&tokens);
        task.tokens.extend_from_slice(&tokens);
        task.eos = tokens.last().copied() == self.server.runtime.model().tokenizer().eos_token()
            || (!cancelled && tokens.len() < budget);
        Ok(QuantumOutput {
            tokens,
            eos: task.eos || task.remaining_tokens == 0,
            cancelled,
        })
    }

    fn finish(&mut self, request_id: RequestId, status: RequestStatus) -> Result<(), Self::Error> {
        let task = self
            .tasks
            .remove(&request_id)
            .ok_or_else(|| invalid_data(format!("unknown request {}", request_id.0)))?;
        match self.server.finish_scheduled_chat(task, status) {
            Ok(()) => Ok(()),
            Err(error) if is_disconnected(error.as_ref()) => Ok(()),
            Err(error) => Err(invalid_data(error.to_string())),
        }
    }
}

fn write_executor_headers<B: Backend>(
    task: &mut ChatTask<B>,
    model_id: &str,
) -> Result<(), io::Error> {
    if !task.request.stream || task.headers_written {
        return Ok(());
    }
    if let Err(error) = write_chat_headers_io(
        &mut task.stream,
        &task.session_id,
        &task.completion_id,
        task.created,
        model_id,
    ) {
        task.disconnected = true;
        return Err(error);
    }
    task.headers_written = true;
    Ok(())
}

fn execute_task_quantum<B: Backend>(
    runtime: &mut Runtime<B>,
    task: &mut ChatTask<B>,
    budget: usize,
    model_id: &str,
) -> Result<(Vec<u32>, bool), RuntimeError> {
    let mut options = task.options.clone();
    options.max_tokens = budget;
    let generation = &mut task.stored.generation;
    let transcript = &task.transcript;
    let mut stream_context = ChatStreamContext {
        stream_mode: task.request.stream,
        streamed: &mut task.streamed,
        stream: &mut task.stream,
        model_id,
        completion_id: &task.completion_id,
        created: task.created,
        disconnected: &mut task.disconnected,
    };
    let peer = &mut task.peer;
    let result = runtime.generate_session_tokens(
        generation,
        transcript,
        options,
        |token| stream_quantum_token(&mut stream_context, &token.bytes),
        || peer_closed(peer),
    )?;
    Ok((result.tokens, result.stats.cancelled))
}

fn stream_quantum_token(
    context: &mut ChatStreamContext<'_>,
    bytes: &[u8],
) -> Result<(), RuntimeError> {
    if !context.stream_mode {
        return Ok(());
    }
    for content in context.streamed.push(bytes) {
        if write_sse(
            context.stream,
            &json!({
                "id": context.completion_id,
                "object": "chat.completion.chunk",
                "created": context.created,
                "model": context.model_id,
                "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": null}]
            }),
        )
        .is_err()
        {
            *context.disconnected = true;
            return Err(RuntimeError::token_callback("client disconnected"));
        }
    }
    Ok(())
}

fn quantum_error<B: Backend>(task: &mut ChatTask<B>, error: RuntimeError) -> QuantumOutput {
    task.stored.generation.invalidate();
    if !task.disconnected {
        task.failure = Some(error.to_string());
    }
    QuantumOutput {
        tokens: Vec::new(),
        eos: false,
        cancelled: true,
    }
}

pub fn run(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let arguments = parse(arguments)?;
    validate_remote_access(&arguments)?;
    let interrupted = install_interrupt_handler()?;
    let (signing_key, model_sha256) = load_server_identity(&arguments)?;
    serve_backend(arguments, signing_key, model_sha256, interrupted)
}

fn validate_remote_access(arguments: &ServeArgs) -> Result<(), Box<dyn Error>> {
    if !arguments.allow_remote && !arguments.bind.ip().is_loopback() {
        return Err(
            invalid_data("serve refuses a non-loopback address without --allow-remote").into(),
        );
    }
    Ok(())
}

fn install_interrupt_handler() -> Result<Arc<AtomicBool>, Box<dyn Error>> {
    let interrupted = Arc::new(AtomicBool::new(false));
    let handler = Arc::clone(&interrupted);
    ctrlc::set_handler(move || handler.store(true, Ordering::Relaxed))?;
    Ok(interrupted)
}

fn load_server_identity(arguments: &ServeArgs) -> Result<(SigningKey, String), Box<dyn Error>> {
    let signing_key = load_or_create_key(&arguments.signing_key)?;
    let model_sha256 = sha256_file(&arguments.model)?;
    Ok((signing_key, model_sha256))
}

fn serve_backend(
    arguments: ServeArgs,
    signing_key: SigningKey,
    model_sha256: String,
    interrupted: Arc<AtomicBool>,
) -> Result<(), Box<dyn Error>> {
    match arguments.backend {
        BackendChoice::Cuda => serve(
            CudaBackend::new(0)?,
            arguments,
            signing_key,
            model_sha256,
            interrupted,
        ),
        BackendChoice::Cpu => serve(
            CpuBackend::new(),
            arguments,
            signing_key,
            model_sha256,
            interrupted,
        ),
    }
}

pub fn verify_response_receipt(path: &Path) -> Result<(), Box<dyn Error>> {
    let bytes = fs::read(path)?;
    let receipt = ResponseReceipt::from_json(&bytes)?;
    println!(
        "response receipt {} claim and signature agree",
        receipt.claim.receipt_id
    );
    println!("embedded signer: {}", receipt.public_key_ed25519);
    println!("model: {}", receipt.claim.model_sha256);
    println!("transcript: {}", receipt.claim.transcript_sha256);
    Ok(())
}

pub fn run_session_gate(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let arguments = parse_session_gate(arguments)?;
    let SessionGateSetup {
        mut runtime,
        prompt_tokens,
        policies,
        cuts,
        tokens,
        seed,
    } = prepare_session_gate(arguments)?;
    let (comparisons, mismatches) =
        run_session_policies(&mut runtime, &prompt_tokens, cuts, tokens, seed, policies)?;
    let cancellation_leaked = session_gate_cancellation(&mut runtime, &prompt_tokens, tokens)?;
    println!("comparisons={comparisons}");
    println!("mismatches={mismatches}");
    println!("cancellation_leaked={cancellation_leaked}");
    if mismatches != 0 || cancellation_leaked {
        return Err(invalid_data("session replay gate failed").into());
    }
    Ok(())
}

struct SessionPolicy {
    name: &'static str,
    mirostat: Option<MirostatConfig>,
}

struct SessionGateSetup {
    runtime: Runtime<CudaBackend>,
    prompt_tokens: Vec<u32>,
    policies: [SessionPolicy; 2],
    cuts: usize,
    tokens: usize,
    seed: u64,
}

fn prepare_session_gate(arguments: SessionGateArgs) -> Result<SessionGateSetup, Box<dyn Error>> {
    let model = arguments
        .model
        .ok_or_else(|| invalid_data("verify session requires -m <gguf>"))?;
    let runtime = Runtime::load(CudaBackend::new(0)?, model)?;
    let prompt =
        "Explain why exact replay matters for a local inference server. Use short sentences.";
    let prompt_tokens = runtime.model().tokenizer().encode(prompt)?;
    let policies = session_gate_policies()?;
    Ok(SessionGateSetup {
        runtime,
        prompt_tokens,
        policies,
        cuts: arguments.cuts,
        tokens: arguments.tokens,
        seed: arguments.seed,
    })
}

struct SessionGateArgs {
    model: Option<PathBuf>,
    cuts: usize,
    tokens: usize,
    seed: u64,
}

fn parse_session_gate(arguments: &[String]) -> Result<SessionGateArgs, Box<dyn Error>> {
    let mut parsed = SessionGateArgs {
        model: None,
        cuts: 32,
        tokens: 64,
        seed: 0x4c65_6f6e_652d_7633_u64,
    };
    let mut index = 0;
    while index < arguments.len() {
        if parse_session_model(&mut parsed, arguments, &mut index)?
            || parse_session_counts(&mut parsed, arguments, &mut index)?
            || parse_session_seed(&mut parsed, arguments, &mut index)?
        {
            index += 1;
            continue;
        }
        return Err(invalid_data(format!(
            "verify session argument is invalid: {}",
            arguments[index]
        ))
        .into());
    }
    if parsed.tokens < 2 {
        return Err(invalid_data("verify session needs at least two tokens").into());
    }
    Ok(parsed)
}

fn parse_session_model(
    parsed: &mut SessionGateArgs,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if !matches!(arguments[*index].as_str(), "-m" | "--model") {
        return Ok(false);
    }
    parsed.model = Some(PathBuf::from(flag_value(arguments, index)?));
    Ok(true)
}

fn parse_session_counts(
    parsed: &mut SessionGateArgs,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] == "--cuts" {
        parsed.cuts = nonzero_usize(flag_value(arguments, index)?, "cuts")?;
        return Ok(true);
    }
    if arguments[*index] == "--tokens" {
        parsed.tokens = nonzero_usize(flag_value(arguments, index)?, "tokens")?;
        return Ok(true);
    }
    Ok(false)
}

fn parse_session_seed(
    parsed: &mut SessionGateArgs,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] != "--seed" {
        return Ok(false);
    }
    let value = flag_value(arguments, index)?;
    parsed.seed = value
        .parse()
        .map_err(|_| invalid_data(format!("seed is invalid: {value}")))?;
    Ok(true)
}

fn session_gate_policies() -> Result<[SessionPolicy; 2], Box<dyn Error>> {
    Ok([
        SessionPolicy {
            name: "seeded",
            mirostat: None,
        },
        SessionPolicy {
            name: "mirostat-v2",
            mirostat: Some(MirostatConfig::new(5.0, 0.1).map_err(|error| {
                invalid_data(format!("Mirostat gate configuration failed: {error}"))
            })?),
        },
    ])
}

struct SessionReplaySummary {
    class: String,
    cached: usize,
    reused: usize,
    replayed: usize,
    computed: usize,
}

struct SessionCutEvidence {
    live_tokens: Vec<u32>,
    restored_tokens: Vec<u32>,
    live_replay: SessionReplaySummary,
    restored_replay: SessionReplaySummary,
}

fn run_session_policies(
    runtime: &mut Runtime<CudaBackend>,
    prompt_tokens: &[u32],
    cuts: usize,
    tokens: usize,
    seed: u64,
    policies: [SessionPolicy; 2],
) -> Result<(usize, usize), Box<dyn Error>> {
    let mut comparisons = 0;
    let mut mismatches = 0;
    for SessionPolicy { name, mirostat } in policies {
        let (policy_comparisons, policy_mismatches) =
            run_session_policy(runtime, prompt_tokens, cuts, tokens, seed, name, mirostat)?;
        comparisons += policy_comparisons;
        mismatches += policy_mismatches;
    }
    Ok((comparisons, mismatches))
}

fn run_session_policy(
    runtime: &mut Runtime<CudaBackend>,
    prompt_tokens: &[u32],
    cuts: usize,
    tokens: usize,
    seed: u64,
    policy: &str,
    mirostat: Option<MirostatConfig>,
) -> Result<(usize, usize), Box<dyn Error>> {
    let mut baseline_options = GenerateOptions::greedy(tokens);
    baseline_options.decode_execution = DecodeExecution::Eager;
    baseline_options.sampler = Sampler::temperature(1.0);
    baseline_options.seed = seed;
    baseline_options.mirostat = mirostat;
    let baseline =
        runtime.generate_tokens(prompt_tokens, baseline_options, |_| Ok(()), || false)?;
    if baseline.tokens.len() < 2 {
        return Err(invalid_data("session gate baseline stopped before two tokens").into());
    }
    let mut comparisons = 0;
    let mut live_mismatches = 0;
    let mut restored_mismatches = 0;
    let mut draw = seed;
    for _ in 0..cuts {
        draw = draw
            .wrapping_add(0x9e37_79b9_7f4a_7c15)
            .rotate_left(17)
            .wrapping_mul(0xbf58_476d_1ce4_e5b9);
        let cut = 1 + draw as usize % (baseline.tokens.len() - 1);
        let (live_mismatch, restored_mismatch) = session_gate_cut(
            runtime,
            prompt_tokens,
            &baseline.tokens,
            cut,
            seed,
            mirostat,
            policy,
        )?;
        comparisons += 2;
        live_mismatches += usize::from(live_mismatch);
        restored_mismatches += usize::from(restored_mismatch);
    }
    println!(
        "policy={policy} cuts={cuts} live_mismatches={} restored_mismatches={} baseline_tokens={} transcript={}",
        live_mismatches,
        restored_mismatches,
        baseline.tokens.len(),
        token_stream_sha256(&baseline.tokens)
    );
    Ok((comparisons, live_mismatches + restored_mismatches))
}

fn session_gate_cut(
    runtime: &mut Runtime<CudaBackend>,
    prompt_tokens: &[u32],
    baseline: &[u32],
    cut: usize,
    seed: u64,
    mirostat: Option<MirostatConfig>,
    policy: &str,
) -> Result<(bool, bool), Box<dyn Error>> {
    let evidence =
        session_gate_cut_evidence(runtime, prompt_tokens, baseline.len(), cut, seed, mirostat)?;
    let live_mismatch = evidence.live_tokens != baseline;
    let restored_mismatch = evidence.restored_tokens != baseline;
    print_session_mismatch(live_mismatch, "live", policy, cut, &evidence.live_replay);
    print_session_mismatch(
        restored_mismatch,
        "restored",
        policy,
        cut,
        &evidence.restored_replay,
    );
    Ok((live_mismatch, restored_mismatch))
}

fn session_gate_cut_evidence(
    runtime: &mut Runtime<CudaBackend>,
    prompt_tokens: &[u32],
    limit: usize,
    cut: usize,
    seed: u64,
    mirostat: Option<MirostatConfig>,
) -> Result<SessionCutEvidence, Box<dyn Error>> {
    let mut prefix_options = GenerateOptions::greedy(cut);
    prefix_options.decode_execution = DecodeExecution::Eager;
    prefix_options.sampler = Sampler::temperature(1.0);
    prefix_options.seed = seed;
    prefix_options.mirostat = mirostat;
    let mut live = GenerationSession::new();
    let prefix = runtime.generate_session_tokens(
        &mut live,
        prompt_tokens,
        prefix_options,
        |_| Ok(()),
        || false,
    )?;
    let checkpoint = live.checkpoint();
    let mut continuation_prompt = prompt_tokens.to_vec();
    continuation_prompt.extend_from_slice(&prefix.tokens);
    let mut tail_options = GenerateOptions::greedy(limit - cut);
    tail_options.decode_execution = DecodeExecution::Eager;
    tail_options.sampler = Sampler::temperature(1.0);
    tail_options.seed = seed;
    tail_options.mirostat = mirostat;
    let live_tail = runtime.generate_session_tokens(
        &mut live,
        &continuation_prompt,
        tail_options.clone(),
        |_| Ok(()),
        || false,
    )?;
    let mut restored = GenerationSession::new();
    restored.restore(checkpoint);
    let restored_tail = runtime.generate_session_tokens(
        &mut restored,
        &continuation_prompt,
        tail_options,
        |_| Ok(()),
        || false,
    )?;
    let mut live_tokens = prefix.tokens.clone();
    live_tokens.extend_from_slice(&live_tail.tokens);
    let mut restored_tokens = prefix.tokens;
    restored_tokens.extend_from_slice(&restored_tail.tokens);
    let live_replay = live.last_replay();
    let restored_replay = restored.last_replay();
    Ok(SessionCutEvidence {
        live_tokens,
        restored_tokens,
        live_replay: SessionReplaySummary {
            class: reuse_class_name(live_replay.reuse_class).to_owned(),
            cached: live_replay.cached_tokens,
            reused: live_replay.reused_tokens,
            replayed: live_replay.replayed_tokens,
            computed: live_replay.computed_tokens,
        },
        restored_replay: SessionReplaySummary {
            class: reuse_class_name(restored_replay.reuse_class).to_owned(),
            cached: restored_replay.cached_tokens,
            reused: restored_replay.reused_tokens,
            replayed: restored_replay.replayed_tokens,
            computed: restored_replay.computed_tokens,
        },
    })
}

fn print_session_mismatch(
    mismatch: bool,
    kind: &str,
    policy: &str,
    cut: usize,
    replay: &SessionReplaySummary,
) {
    if mismatch {
        println!(
            "mismatch={kind} policy={policy} cut={cut} class={} cached={} reused={} replayed={} computed={}",
            replay.class, replay.cached, replay.reused, replay.replayed, replay.computed,
        );
    }
}

fn session_gate_cancellation(
    runtime: &mut Runtime<CudaBackend>,
    prompt_tokens: &[u32],
    tokens: usize,
) -> Result<bool, Box<dyn Error>> {
    let mut cancelled_session = GenerationSession::new();
    let emitted = std::cell::Cell::new(0_usize);
    let mut cancellation_options = GenerateOptions::greedy(tokens);
    cancellation_options.decode_execution = DecodeExecution::Eager;
    let cancelled = runtime.generate_session_tokens(
        &mut cancelled_session,
        prompt_tokens,
        cancellation_options,
        |_| {
            emitted.set(emitted.get() + 1);
            Ok(())
        },
        || emitted.get() >= 3,
    )?;
    Ok(!cancelled.stats.cancelled || !cancelled_session.is_empty())
}

pub fn run_scheduler_gate(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let arguments = parse_scheduler_gate(arguments)?;
    let SchedulerGateSetup {
        runtime,
        prompt_tokens,
        options,
        isolated,
        policy,
        tokens,
    } = prepare_scheduler_gate(arguments)?;
    let executor = RuntimeQuantumExecutor::new(LeoneRuntimeDriver::new(runtime));
    let mut service = ScheduledService::new(policy, executor)?;
    admit_scheduler_requests(&mut service, &prompt_tokens, &options, tokens)?;
    finish_scheduler_requests(&mut service)?;
    let mismatches = compare_scheduler_requests(&service, &isolated)?;
    println!("comparisons={}", isolated.len());
    println!("mismatches={mismatches}");
    if mismatches != 0 {
        return Err(invalid_data("scheduler model gate failed").into());
    }
    Ok(())
}

struct SchedulerGateSetup {
    runtime: Runtime<CudaBackend>,
    prompt_tokens: Vec<Vec<u32>>,
    options: Vec<GenerateOptions>,
    isolated: Vec<Vec<u32>>,
    policy: SchedulerPolicy,
    tokens: usize,
}

fn prepare_scheduler_gate(
    arguments: SchedulerGateArgs,
) -> Result<SchedulerGateSetup, Box<dyn Error>> {
    let model = arguments
        .model
        .ok_or_else(|| invalid_data("verify scheduler requires -m <gguf>"))?;
    let mut runtime = Runtime::load(CudaBackend::new(0)?, model)?;
    let SchedulerGateWorkload {
        prompt_tokens,
        options,
    } = scheduler_gate_workload(&mut runtime, arguments.tokens, arguments.seed)?;
    let context = runtime.model().config().context_length;
    validate_scheduler_context(&prompt_tokens, arguments.tokens, context)?;
    let isolated = scheduler_isolated(&mut runtime, &prompt_tokens, &options)?;
    let policy = scheduler_gate_policy(
        &runtime,
        prompt_tokens.len(),
        arguments.tokens,
        context,
        arguments.quantum,
    )?;
    Ok(SchedulerGateSetup {
        runtime,
        prompt_tokens,
        options,
        isolated,
        policy,
        tokens: arguments.tokens,
    })
}

struct SchedulerGateArgs {
    model: Option<PathBuf>,
    tokens: usize,
    quantum: u32,
    seed: u64,
}

fn parse_scheduler_gate(arguments: &[String]) -> Result<SchedulerGateArgs, Box<dyn Error>> {
    let mut parsed = SchedulerGateArgs {
        model: None,
        tokens: 32,
        quantum: 4,
        seed: 0x4c65_6f6e_652d_7631_u64,
    };
    let mut index = 0;
    while index < arguments.len() {
        if parse_scheduler_model(&mut parsed, arguments, &mut index)?
            || parse_scheduler_counts(&mut parsed, arguments, &mut index)?
            || parse_scheduler_seed(&mut parsed, arguments, &mut index)?
        {
            index += 1;
            continue;
        }
        return Err(invalid_data(format!(
            "verify scheduler argument is invalid: {}",
            arguments[index]
        ))
        .into());
    }
    Ok(parsed)
}

fn parse_scheduler_model(
    parsed: &mut SchedulerGateArgs,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if !matches!(arguments[*index].as_str(), "-m" | "--model") {
        return Ok(false);
    }
    parsed.model = Some(PathBuf::from(flag_value(arguments, index)?));
    Ok(true)
}

fn parse_scheduler_counts(
    parsed: &mut SchedulerGateArgs,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] == "--tokens" {
        parsed.tokens = nonzero_usize(flag_value(arguments, index)?, "tokens")?;
        return Ok(true);
    }
    if arguments[*index] == "--quantum" {
        let value = nonzero_usize(flag_value(arguments, index)?, "quantum")?;
        parsed.quantum =
            u32::try_from(value).map_err(|_| invalid_data("quantum does not fit u32"))?;
        return Ok(true);
    }
    Ok(false)
}

fn parse_scheduler_seed(
    parsed: &mut SchedulerGateArgs,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] != "--seed" {
        return Ok(false);
    }
    let value = flag_value(arguments, index)?;
    parsed.seed = value
        .parse()
        .map_err(|_| invalid_data(format!("seed is invalid: {value}")))?;
    Ok(true)
}

struct SchedulerGateWorkload {
    prompt_tokens: Vec<Vec<u32>>,
    options: Vec<GenerateOptions>,
}

fn scheduler_gate_workload(
    runtime: &mut Runtime<CudaBackend>,
    tokens: usize,
    seed: u64,
) -> Result<SchedulerGateWorkload, Box<dyn Error>> {
    let prompts = [
        "Explain one invariant of exact local inference in two sentences.",
        "List three reasons to bound a server work quantum.",
        "Write a short definition of reproducible performance evidence.",
    ];
    let prompt_tokens = prompts
        .iter()
        .map(|prompt| runtime.model().tokenizer().encode(prompt))
        .collect::<Result<Vec<_>, _>>()?;
    let mut options = vec![GenerateOptions::greedy(tokens); prompts.len()];
    for option in &mut options {
        option.decode_execution = DecodeExecution::Eager;
    }
    options[1].sampler = Sampler::temperature(0.8);
    options[1].seed = seed;
    options[2].sampler = Sampler::temperature(1.0);
    options[2].seed = seed ^ 0x9e37_79b9_7f4a_7c15;
    options[2].mirostat = Some(
        MirostatConfig::new(5.0, 0.1)
            .map_err(|error| invalid_data(format!("Mirostat configuration failed: {error}")))?,
    );
    Ok(SchedulerGateWorkload {
        prompt_tokens,
        options,
    })
}

fn validate_scheduler_context(
    prompts: &[Vec<u32>],
    tokens: usize,
    context: usize,
) -> Result<(), Box<dyn Error>> {
    for prompt in prompts {
        if prompt.len().saturating_add(tokens) > context {
            return Err(invalid_data("scheduler gate workload exceeds model context").into());
        }
    }
    Ok(())
}

fn scheduler_isolated(
    runtime: &mut Runtime<CudaBackend>,
    prompts: &[Vec<u32>],
    options: &[GenerateOptions],
) -> Result<Vec<Vec<u32>>, Box<dyn Error>> {
    let mut isolated = Vec::with_capacity(prompts.len());
    for (prompt, options) in prompts.iter().zip(options) {
        let mut session = GenerationSession::new();
        isolated.push(
            runtime
                .generate_session_tokens(
                    &mut session,
                    prompt,
                    options.clone(),
                    |_| Ok(()),
                    || false,
                )?
                .tokens,
        );
    }
    Ok(isolated)
}

fn scheduler_gate_policy(
    runtime: &Runtime<CudaBackend>,
    request_count: usize,
    tokens: usize,
    context: usize,
    quantum: u32,
) -> Result<SchedulerPolicy, Box<dyn Error>> {
    let config = runtime.model().config();
    let (kv_bytes_per_token, request_count) = scheduler_gate_dimensions(config, request_count)?;
    let context_tokens = host_u64(context)?;
    let output_tokens = host_u64(tokens)?;
    let max_reserved_kv_bytes =
        scheduler_gate_capacity(kv_bytes_per_token, context_tokens, request_count)?;
    Ok(SchedulerPolicy {
        max_active_requests: request_count,
        max_queued_requests: request_count,
        max_reserved_kv_bytes,
        kv_bytes_per_token,
        max_prompt_tokens: context_tokens,
        max_output_tokens: output_tokens,
        service_quantum_tokens: quantum,
        urgent_window_ns: 0,
        max_prefix_credit_tokens: 0,
    })
}

fn scheduler_gate_dimensions(
    config: &leone::ModelConfig,
    request_count: usize,
) -> Result<(u64, u32), Box<dyn Error>> {
    let layers = host_u64(config.n_layer)?;
    let kv_heads = host_u64(config.n_head_kv)?;
    let head_dim = host_u64(config.head_dim)?;
    let request_count =
        u32::try_from(request_count).map_err(|_| invalid_data("request count does not fit u32"))?;
    let kv_bytes_per_token = layers
        .checked_mul(kv_heads)
        .and_then(|value| value.checked_mul(head_dim))
        .and_then(|value| value.checked_mul(8))
        .ok_or_else(|| invalid_data("scheduler gate KV bound overflowed"))?;
    Ok((kv_bytes_per_token, request_count))
}

fn scheduler_gate_capacity(
    kv_bytes_per_token: u64,
    context_tokens: u64,
    request_count: u32,
) -> Result<u64, Box<dyn Error>> {
    kv_bytes_per_token
        .checked_mul(context_tokens)
        .and_then(|value| value.checked_mul(u64::from(request_count)))
        .ok_or_else(|| invalid_data("scheduler gate KV capacity overflowed").into())
}

fn admit_scheduler_requests(
    service: &mut ScheduledService<RuntimeQuantumExecutor<LeoneRuntimeDriver<CudaBackend>>>,
    prompts: &[Vec<u32>],
    options: &[GenerateOptions],
    tokens: usize,
) -> Result<(), Box<dyn Error>> {
    for (offset, (prompt, options)) in prompts.iter().zip(options).enumerate() {
        let request_id = RequestId(u64::try_from(offset)? + 1);
        let request = ScheduledGenerationRequest::new(prompt.clone(), options.clone());
        let spec = RequestSpec {
            id: request_id,
            arrival_ns: 0,
            prompt_tokens: host_u64(prompt.len())?,
            prefix_reused_tokens: 0,
            max_output_tokens: host_u64(tokens)?,
            priority: 1,
            deadline_ns: None,
        };
        if !matches!(
            service.admit(spec, &request, 0)?,
            AdmissionOutcome::Admitted { .. }
        ) {
            return Err(invalid_data("scheduler gate admission failed").into());
        }
    }
    Ok(())
}

fn finish_scheduler_requests(
    service: &mut ScheduledService<RuntimeQuantumExecutor<LeoneRuntimeDriver<CudaBackend>>>,
) -> Result<(), Box<dyn Error>> {
    let mut now_ns = 1_u64;
    while service.has_runnable_requests() {
        service.tick(now_ns)?;
        now_ns = now_ns.saturating_add(1);
    }
    Ok(())
}

fn compare_scheduler_requests(
    service: &ScheduledService<RuntimeQuantumExecutor<LeoneRuntimeDriver<CudaBackend>>>,
    isolated: &[Vec<u32>],
) -> Result<usize, Box<dyn Error>> {
    let mut mismatches = 0;
    for (offset, expected) in isolated.iter().enumerate() {
        let request_id = RequestId(u64::try_from(offset)? + 1);
        let concurrent = &service
            .executor()
            .completed(request_id)
            .ok_or_else(|| invalid_data("scheduler gate lost a completed request"))?
            .tokens;
        let matches = concurrent == expected;
        mismatches += usize::from(!matches);
        println!(
            "request={} matches={} tokens={} transcript={}",
            request_id.0,
            matches,
            concurrent.len(),
            token_stream_sha256(concurrent)
        );
    }
    Ok(mismatches)
}

fn serve<B: Backend>(
    backend: B,
    arguments: ServeArgs,
    signing_key: SigningKey,
    model_sha256: String,
    interrupted: Arc<AtomicBool>,
) -> Result<(), Box<dyn Error>> {
    let setup = build_serve_setup(backend, arguments, signing_key, model_sha256)?;
    let policy = server_scheduler_policy(&setup.server, setup.max_sessions)?;
    let mut service = ScheduledService::new(
        policy,
        ServerExecutor {
            server: setup.server,
            tasks: BTreeMap::new(),
        },
    )?;
    run_serve_loop(&mut service, setup.bind, setup.isolated, interrupted)
}

struct ServeSetup<B: Backend> {
    server: Server<B>,
    bind: SocketAddr,
    isolated: bool,
    max_sessions: usize,
}

struct SessionPersistence {
    store: Option<SessionStore>,
    archives: HashMap<String, StoredArchive>,
}

fn build_serve_setup<B: Backend>(
    backend: B,
    arguments: ServeArgs,
    signing_key: SigningKey,
    model_sha256: String,
) -> Result<ServeSetup<B>, Box<dyn Error>> {
    let execution_plan = crate::execution_plan::load_optional(
        arguments.plan.as_deref(),
        &arguments.model,
        backend.name(),
    )?;
    let runtime = Runtime::load(backend, &arguments.model)?;
    let model_id = arguments
        .model
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("leone-model")
        .to_owned();
    let SessionPersistence {
        store: session_store,
        archives: persisted,
    } = load_session_persistence(&arguments, &model_sha256)?;
    let clock = persisted
        .values()
        .map(|archive| archive.last_used)
        .max()
        .unwrap_or(0);
    let max_sessions = arguments.sessions;
    let bind = arguments.bind;
    let isolated = std::env::var_os("LEONE_ISOLATED_SERVE").is_some();
    Ok(ServeSetup {
        server: Server {
            runtime,
            model_id,
            model_sha256,
            sessions: HashMap::new(),
            hibernated: HashMap::new(),
            persisted,
            session_store,
            max_sessions,
            max_hibernated_sessions: arguments.hibernated_sessions,
            clock,
            kv_cache_dtype: arguments.kv_cache_dtype,
            execution_plan,
            receipts: arguments.receipts,
            signing_key,
        },
        bind,
        isolated,
        max_sessions,
    })
}

fn load_session_persistence(
    arguments: &ServeArgs,
    model_sha256: &str,
) -> Result<SessionPersistence, Box<dyn Error>> {
    match arguments.session_store.clone() {
        Some(path) => {
            let (store, archives) = SessionStore::open(path, model_sha256)?;
            Ok(SessionPersistence {
                store: Some(store),
                archives,
            })
        }
        None => Ok(SessionPersistence {
            store: None,
            archives: HashMap::new(),
        }),
    }
}

fn run_serve_loop<B: Backend>(
    service: &mut ChatService<B>,
    bind: SocketAddr,
    isolated: bool,
    interrupted: Arc<AtomicBool>,
) -> Result<(), Box<dyn Error>> {
    let listener = TcpListener::bind(bind)?;
    let local_addr = listener.local_addr()?;
    listener.set_nonblocking(true)?;
    println!("Leone serves http://{local_addr}");
    println!("OpenAI base URL: http://{local_addr}/v1");
    println!(
        "execution: {}",
        if isolated { "isolated" } else { "scheduled" }
    );
    let mut next_request_id = 1_u64;
    while !interrupted.load(Ordering::Relaxed) {
        let accepted = accept_serve_connection(&listener, service, isolated, &mut next_request_id)?;
        let ticked = tick_serve_requests(service, isolated)?;
        let progressed = accepted || ticked;
        if !progressed {
            thread::sleep(Duration::from_millis(2));
        }
    }
    Ok(())
}

fn accept_serve_connection<B: Backend>(
    listener: &TcpListener,
    service: &mut ChatService<B>,
    isolated: bool,
    next_request_id: &mut u64,
) -> Result<bool, Box<dyn Error>> {
    let (mut stream, _) = match listener.accept() {
        Ok(connection) => connection,
        Err(error) if error.kind() == io::ErrorKind::WouldBlock => return Ok(false),
        Err(error) => return Err(error.into()),
    };
    let result = if isolated {
        handle_connection(&mut service.executor_mut().server, &mut stream)
    } else {
        let request_id = RequestId(*next_request_id);
        *next_request_id = next_request_id.wrapping_add(1).max(1);
        handle_scheduled_connection(service, stream, request_id)
    };
    if let Err(error) = result {
        eprintln!("request failed: {error}");
    }
    Ok(true)
}

fn tick_serve_requests<B: Backend>(
    service: &mut ChatService<B>,
    isolated: bool,
) -> Result<bool, Box<dyn Error>> {
    if !isolated && service.has_runnable_requests() {
        service.tick(scheduler_now_ns())?;
        return Ok(true);
    }
    Ok(false)
}

fn server_scheduler_policy<B: Backend>(
    server: &Server<B>,
    sessions: usize,
) -> Result<SchedulerPolicy, Box<dyn Error>> {
    let config = server.runtime.model().config();
    let kv_bytes_per_token = scheduler_kv_bytes(config)?;
    let max_active_requests = u32::try_from(sessions)
        .map_err(|_| invalid_data("session limit does not fit the scheduler"))?;
    let context_tokens = host_u64(config.context_length)?;
    let max_reserved_kv_bytes =
        scheduler_kv_capacity(kv_bytes_per_token, context_tokens, max_active_requests)?;
    Ok(SchedulerPolicy {
        max_active_requests,
        max_queued_requests: 64,
        max_reserved_kv_bytes,
        kv_bytes_per_token,
        max_prompt_tokens: context_tokens,
        max_output_tokens: context_tokens,
        service_quantum_tokens: 4,
        urgent_window_ns: 5_000_000,
        max_prefix_credit_tokens: context_tokens,
    }
    .validate()?)
}

fn scheduler_kv_bytes(config: &leone::ModelConfig) -> Result<u64, Box<dyn Error>> {
    let layers = host_u64(config.n_layer)?;
    let heads = host_u64(config.n_head_kv)?;
    let head_dim = host_u64(config.head_dim)?;
    layers
        .checked_mul(heads)
        .and_then(|value| value.checked_mul(head_dim))
        .and_then(|value| value.checked_mul(2))
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| invalid_data("scheduler KV byte bound overflowed").into())
}

fn scheduler_kv_capacity(
    kv_bytes_per_token: u64,
    context_tokens: u64,
    max_active_requests: u32,
) -> Result<u64, Box<dyn Error>> {
    kv_bytes_per_token
        .checked_mul(context_tokens)
        .and_then(|value| value.checked_mul(u64::from(max_active_requests)))
        .ok_or_else(|| invalid_data("scheduler KV capacity overflowed").into())
}

fn scheduler_now_ns() -> u64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    u64::try_from(nanos).unwrap_or(u64::MAX)
}

fn handle_scheduled_connection<B: Backend>(
    service: &mut ChatService<B>,
    mut stream: TcpStream,
    request_id: RequestId,
) -> Result<(), Box<dyn Error>> {
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    stream.set_write_timeout(Some(Duration::from_secs(30)))?;
    let Some(http) = read_request(&mut stream)? else {
        return Ok(());
    };
    if http.method == "OPTIONS" {
        return write_empty(&mut stream, 204, "No Content").map_err(Into::into);
    }
    handle_scheduled_route(service, stream, request_id, http)
}

fn handle_scheduled_route<B: Backend>(
    service: &mut ChatService<B>,
    mut stream: TcpStream,
    request_id: RequestId,
    http: HttpRequest,
) -> Result<(), Box<dyn Error>> {
    match (http.method.as_str(), http.path.as_str()) {
        ("GET", "/health") => write_json(&mut stream, 200, &json!({"status": "ok"}))?,
        ("GET", "/v1/models") => write_json(
            &mut stream,
            200,
            &json!({
                "object": "list",
                "data": [{
                    "id": service.executor().server.model_id,
                    "object": "model",
                    "owned_by": "local"
                }]
            }),
        )?,
        ("POST", "/v1/chat/completions") => {
            admit_scheduled_chat(service, stream, request_id, &http)?;
        }
        _ => write_error(&mut stream, 404, "route not found")?,
    }
    Ok(())
}

fn admit_scheduled_chat<B: Backend>(
    service: &mut ChatService<B>,
    mut stream: TcpStream,
    request_id: RequestId,
    http: &HttpRequest,
) -> Result<(), Box<dyn Error>> {
    let prepared = prepare_chat_plan(&service.executor().server, http);
    let Some(plan) = scheduled_request_or_error(&mut stream, prepared)? else {
        return Ok(());
    };
    if active_session_exists(service, &plan.session_id) {
        let mut stream = stream;
        write_error(
            &mut stream,
            409,
            "the requested Leone session is already running",
        )?;
        return Ok(());
    }
    let spec = scheduled_chat_spec(request_id, &plan)?;
    let admission = ChatAdmission(RefCell::new(Some(PendingChat { plan, stream })));
    admit_pending_chat(service, spec, &admission)
}

fn scheduled_request_or_error<T>(
    stream: &mut TcpStream,
    result: Result<T, Box<dyn Error>>,
) -> Result<Option<T>, Box<dyn Error>> {
    match result {
        Ok(value) => Ok(Some(value)),
        Err(error) => {
            write_error(stream, 400, &error.to_string())?;
            Ok(None)
        }
    }
}

fn active_session_exists<B: Backend>(service: &ChatService<B>, session_id: &str) -> bool {
    service
        .executor()
        .tasks
        .values()
        .any(|active| active.session_id == session_id)
}

fn scheduled_chat_spec(
    request_id: RequestId,
    plan: &ChatPlan,
) -> Result<RequestSpec, Box<dyn Error>> {
    Ok(RequestSpec {
        id: request_id,
        arrival_ns: scheduler_now_ns(),
        prompt_tokens: host_u64(plan.prompt_tokens.len())?,
        prefix_reused_tokens: host_u64(plan.prefix_reused_tokens)?,
        max_output_tokens: host_u64(plan.options.max_tokens)?,
        priority: 1,
        deadline_ns: None,
    })
}

fn admit_pending_chat<B: Backend>(
    service: &mut ChatService<B>,
    spec: RequestSpec,
    admission: &ChatAdmission,
) -> Result<(), Box<dyn Error>> {
    match service.admit(spec, admission, scheduler_now_ns())? {
        AdmissionOutcome::Admitted { .. } => {}
        AdmissionOutcome::Rejected { reason } => {
            let mut pending = admission
                .0
                .borrow_mut()
                .take()
                .expect("rejected admission does not create executor state");
            write_error(
                &mut pending.stream,
                429,
                &format!("scheduler rejected the request: {reason:?}"),
            )?;
        }
    }
    Ok(())
}

fn prepare_chat_plan<B: Backend>(
    server: &Server<B>,
    http: &HttpRequest,
) -> Result<ChatPlan, Box<dyn Error>> {
    let (request, prompt_tokens) = parse_chat_plan_input(server, http)?;
    let options = chat_plan_options(server, &request, &prompt_tokens)?;
    let requested_session = requested_chat_session(&request, http);
    let (session_id, fork_parent, prefix_session) = chat_plan_session(
        server,
        requested_session,
        request.leone_fork_session.as_deref(),
        &prompt_tokens,
    )?;
    let evaluated_tokens = chat_plan_evaluated_tokens(server, &prefix_session);
    let prefix_reused_tokens = common_prefix(&prompt_tokens, evaluated_tokens);
    let request_sha256 = sha256_bytes(&http.body);
    let created = unix_seconds()?;
    let completion_id = format!("chatcmpl-{}", Uuid::new_v4().simple());
    Ok(ChatPlan {
        request,
        prompt_tokens,
        options,
        session_id,
        fork_parent,
        prefix_reused_tokens,
        request_sha256,
        created,
        completion_id,
    })
}

fn parse_chat_plan_input<B: Backend>(
    server: &Server<B>,
    http: &HttpRequest,
) -> Result<(ChatRequest, Vec<u32>), Box<dyn Error>> {
    let request: ChatRequest = serde_json::from_slice(&http.body)?;
    validate_chat_request(&request, &server.model_id)?;
    let prompt_tokens = chat_tokens(
        server.runtime.model().tokenizer(),
        server.runtime.model().config().architecture,
        &request.messages,
        request.tools.as_deref(),
        request.tool_choice.as_ref(),
    )?;
    Ok((request, prompt_tokens))
}

fn chat_plan_options<B: Backend>(
    server: &Server<B>,
    request: &ChatRequest,
    prompt_tokens: &[u32],
) -> Result<GenerateOptions, Box<dyn Error>> {
    let mut options = chat_plan_base_options(server, request, prompt_tokens)?;
    configure_chat_plan_options(server, request, &mut options)?;
    Ok(options)
}

fn chat_plan_base_options<B: Backend>(
    server: &Server<B>,
    request: &ChatRequest,
    prompt_tokens: &[u32],
) -> Result<GenerateOptions, Box<dyn Error>> {
    let max_tokens = request
        .max_completion_tokens
        .or(request.max_tokens)
        .unwrap_or(DEFAULT_MAX_TOKENS);
    if max_tokens == 0 {
        return Err(invalid_data("max_tokens must be nonzero").into());
    }
    let capacity = server.runtime.model().config().context_length;
    let remaining = capacity
        .checked_sub(prompt_tokens.len())
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| invalid_data("the chat prompt exceeds model context"))?;
    let mut options = GenerateOptions::greedy(max_tokens.min(remaining));
    options.decode_execution = if server
        .runtime
        .model()
        .config()
        .architecture
        .decode_graph_supported()
    {
        DecodeExecution::Graph
    } else {
        DecodeExecution::Eager
    };
    Ok(options)
}

fn configure_chat_plan_options<B: Backend>(
    server: &Server<B>,
    request: &ChatRequest,
    options: &mut GenerateOptions,
) -> Result<(), Box<dyn Error>> {
    options.kv_cache_dtype = server.kv_cache_dtype;
    options.seed = request.seed.unwrap_or(0);
    options.sampler = sampler(request)?;
    options.penalties = penalties(request)?;
    options.mirostat = mirostat(request)?;
    options.output_constraint = response_constraint(request.response_format.as_ref())?;
    options.speculation = if options.output_constraint.is_some() {
        Speculation::Disabled
    } else {
        speculation(request)?
    };
    if let Some(plan) = server.execution_plan {
        plan.apply(options);
    }
    Ok(())
}

fn requested_chat_session<'a>(request: &'a ChatRequest, http: &'a HttpRequest) -> Option<&'a str> {
    request
        .leone_session
        .as_deref()
        .or_else(|| http.headers.get("x-leone-session").map(String::as_str))
        .or(request.user.as_deref())
}

fn chat_plan_session<B: Backend>(
    server: &Server<B>,
    requested_session: Option<&str>,
    fork_parent: Option<&str>,
    prompt_tokens: &[u32],
) -> Result<(String, Option<String>, String), Box<dyn Error>> {
    if let Some(parent_id) = fork_parent {
        return fork_chat_session(server, requested_session, parent_id);
    }
    let session_id = server.select_session(requested_session, prompt_tokens);
    Ok((session_id.clone(), None, session_id))
}

fn fork_chat_session<B: Backend>(
    server: &Server<B>,
    requested_session: Option<&str>,
    parent_id: &str,
) -> Result<(String, Option<String>, String), Box<dyn Error>> {
    if parent_id.is_empty() {
        return Err(invalid_data("leone_fork_session must not be empty").into());
    }
    let session_id = requested_session
        .map(str::to_owned)
        .unwrap_or_else(|| Uuid::new_v4().to_string());
    if session_id == parent_id {
        return Err(invalid_data("a fork must use a new session identifier").into());
    }
    if server.sessions.contains_key(&session_id)
        || server.hibernated.contains_key(&session_id)
        || server.persisted.contains_key(&session_id)
    {
        return Err(invalid_data("a fork must not replace an existing session").into());
    }
    if !server.sessions.contains_key(parent_id) && !server.persisted.contains_key(parent_id) {
        return Err(invalid_data("the fork parent session does not exist").into());
    }
    Ok((session_id, Some(parent_id.to_owned()), parent_id.to_owned()))
}

fn chat_plan_evaluated_tokens<'a, B: Backend>(
    server: &'a Server<B>,
    session_id: &str,
) -> &'a [u32] {
    server
        .sessions
        .get(session_id)
        .map(|stored| stored.generation.evaluated_tokens())
        .or_else(|| {
            server
                .hibernated
                .get(session_id)
                .map(|stored| stored.generation.evaluated_tokens())
        })
        .or_else(|| {
            server
                .persisted
                .get(session_id)
                .map(|stored| stored.archive.evaluated_tokens())
        })
        .unwrap_or_default()
}

fn handle_connection<B: Backend>(
    server: &mut Server<B>,
    stream: &mut TcpStream,
) -> Result<(), Box<dyn Error>> {
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    stream.set_write_timeout(Some(Duration::from_secs(30)))?;
    let Some(request) = read_request(stream)? else {
        return Ok(());
    };
    if request.method == "OPTIONS" {
        return write_empty(stream, 204, "No Content").map_err(Into::into);
    }
    handle_connection_route(server, stream, request)
}

fn handle_connection_route<B: Backend>(
    server: &mut Server<B>,
    stream: &mut TcpStream,
    request: HttpRequest,
) -> Result<(), Box<dyn Error>> {
    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/health") => write_json(stream, 200, &json!({"status": "ok"}))?,
        ("GET", "/v1/models") => write_json(
            stream,
            200,
            &json!({
                "object": "list",
                "data": [{"id": server.model_id, "object": "model", "owned_by": "local"}]
            }),
        )?,
        ("POST", "/v1/chat/completions") => handle_chat_request(server, stream, &request),
        _ => write_error(stream, 404, "route not found")?,
    }
    Ok(())
}

fn handle_chat_request<B: Backend>(
    server: &mut Server<B>,
    stream: &mut TcpStream,
    request: &HttpRequest,
) {
    if let Err(error) = handle_chat(server, stream, request) {
        if !is_disconnected(error.as_ref()) {
            let _ = write_error(stream, 400, &error.to_string());
        }
    }
}

fn handle_chat<B: Backend>(
    server: &mut Server<B>,
    stream: &mut TcpStream,
    http: &HttpRequest,
) -> Result<(), Box<dyn Error>> {
    let plan = prepare_chat_plan(server, http)?;
    let mut peer = stream.try_clone()?;
    peer.set_nonblocking(true)?;
    let mut stored = server.lease_session(&plan)?;
    let ChatPlan {
        request,
        prompt_tokens,
        options,
        session_id,
        fork_parent: _,
        prefix_reused_tokens: _,
        request_sha256,
        created,
        completion_id,
    } = plan;
    let mut streamed = Utf8Stream::default();
    let mut disconnected = false;
    let model_id = server.model_id.clone();

    if request.stream {
        if let Err(error) =
            write_chat_headers(stream, &session_id, &completion_id, created, &model_id)
        {
            stored.generation.invalidate();
            server.discard_session(&session_id)?;
            return Err(error);
        }
    }
    let mut generation_context = ChatGenerationContext {
        prompt_tokens: &prompt_tokens,
        options: options.clone(),
        stream: ChatStreamContext {
            stream_mode: request.stream,
            streamed: &mut streamed,
            stream,
            model_id: &model_id,
            completion_id: &completion_id,
            created,
            disconnected: &mut disconnected,
        },
        peer: &mut peer,
    };
    let Some(generated) =
        generate_chat_or_discard(server, &mut stored, &mut generation_context, &session_id)?
    else {
        return Ok(());
    };
    finish_chat_response(
        server,
        stream,
        ChatResponseContext {
            request: &request,
            session_id,
            completion_id,
            created,
            request_sha256,
            seed: options.seed,
            model_id,
        },
        &mut streamed,
        stored,
        generated,
    )
}

struct ChatGeneration {
    prompt_tokens: Vec<u32>,
    tokens: Vec<u32>,
    cancelled: bool,
    eos: bool,
}

struct ChatResponseDetails {
    decoded_content: Option<String>,
    generated_tool_calls: Option<Vec<GeneratedToolCall>>,
    finish_reason: String,
}

struct ChatStreamContext<'a> {
    stream_mode: bool,
    streamed: &'a mut Utf8Stream,
    stream: &'a mut TcpStream,
    model_id: &'a str,
    completion_id: &'a str,
    created: u64,
    disconnected: &'a mut bool,
}

struct ChatGenerationContext<'a> {
    prompt_tokens: &'a [u32],
    options: GenerateOptions,
    stream: ChatStreamContext<'a>,
    peer: &'a mut TcpStream,
}

struct ChatResponseContext<'a> {
    request: &'a ChatRequest,
    session_id: String,
    completion_id: String,
    created: u64,
    request_sha256: String,
    seed: u64,
    model_id: String,
}

fn write_chat_headers(
    stream: &mut TcpStream,
    session_id: &str,
    completion_id: &str,
    created: u64,
    model_id: &str,
) -> Result<(), Box<dyn Error>> {
    write_chat_headers_io(stream, session_id, completion_id, created, model_id)?;
    Ok(())
}

fn write_chat_headers_io(
    stream: &mut TcpStream,
    session_id: &str,
    completion_id: &str,
    created: u64,
    model_id: &str,
) -> io::Result<()> {
    write_stream_headers(stream, session_id)?;
    write_sse(
        stream,
        &json!({
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_id,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]
        }),
    )?;
    Ok(())
}

fn generate_chat_or_discard<B: Backend>(
    server: &mut Server<B>,
    stored: &mut StoredSession<B>,
    context: &mut ChatGenerationContext<'_>,
    session_id: &str,
) -> Result<Option<ChatGeneration>, Box<dyn Error>> {
    match generate_chat(&mut server.runtime, stored, &mut *context) {
        Ok(generated) => Ok(Some(generated)),
        Err(error) => {
            stored.generation.invalidate();
            server.discard_session(session_id)?;
            if *context.stream.disconnected {
                Ok(None)
            } else {
                Err(error.into())
            }
        }
    }
}

fn generate_chat<B: Backend>(
    runtime: &mut Runtime<B>,
    stored: &mut StoredSession<B>,
    context: &mut ChatGenerationContext<'_>,
) -> Result<ChatGeneration, RuntimeError> {
    let prompt_tokens = context.prompt_tokens;
    let options = context.options.clone();
    let stream = &mut context.stream;
    let peer = &mut *context.peer;
    let result = runtime.generate_session_tokens(
        &mut stored.generation,
        prompt_tokens,
        options,
        |token| stream_quantum_token(stream, &token.bytes),
        || peer_closed(peer),
    )?;
    let eos = result.tokens.last().copied() == runtime.model().tokenizer().eos_token();
    Ok(ChatGeneration {
        prompt_tokens: result.prompt_tokens,
        tokens: result.tokens,
        cancelled: result.stats.cancelled,
        eos,
    })
}

fn finish_chat_response<B: Backend>(
    server: &mut Server<B>,
    stream: &mut TcpStream,
    context: ChatResponseContext<'_>,
    streamed: &mut Utf8Stream,
    stored: StoredSession<B>,
    generated: ChatGeneration,
) -> Result<(), Box<dyn Error>> {
    let details = chat_response_details(server, context.request, &generated)?;
    let replay = stored.generation.last_replay();
    let session = response_session_record(
        &context.session_id,
        reuse_class_name(replay.reuse_class),
        replay.cached_tokens,
        replay.reused_tokens,
        replay.replayed_tokens,
        replay.computed_tokens,
    )?;
    let receipt = sign_chat_receipt(
        server,
        &context.request_sha256,
        context.seed,
        &generated,
        &details,
        session,
    )?;
    write_response_receipt(&server.receipts, &receipt)?;
    persist_chat_session(server, &context.session_id, stored, generated.cancelled)?;
    write_chat_response(stream, &context, streamed, &generated, &details, &receipt)
}

fn chat_response_details<B: Backend>(
    server: &Server<B>,
    request: &ChatRequest,
    generated: &ChatGeneration,
) -> Result<ChatResponseDetails, Box<dyn Error>> {
    let decoded_content = if request.stream {
        None
    } else {
        Some(
            server
                .runtime
                .model()
                .tokenizer()
                .decode(&generated.tokens)?,
        )
    };
    let generated_tool_calls = decoded_content
        .as_deref()
        .map(|content| parse_generated_tool_calls(content, request.tools.as_deref()))
        .transpose()?
        .flatten();
    let finish_reason = if generated.cancelled {
        "cancelled"
    } else if generated_tool_calls.is_some() {
        "tool_calls"
    } else if generated.eos {
        "stop"
    } else {
        "length"
    };
    Ok(ChatResponseDetails {
        decoded_content,
        generated_tool_calls,
        finish_reason: finish_reason.to_owned(),
    })
}

fn response_session_record(
    session_id: &str,
    reuse_class: &str,
    cached_tokens: usize,
    reused_tokens: usize,
    replayed_tokens: usize,
    computed_tokens: usize,
) -> Result<SessionReplayRecord, Box<dyn Error>> {
    Ok(SessionReplayRecord {
        session_id: session_id.to_owned(),
        reuse_class: reuse_class.to_owned(),
        cached_tokens: host_u64(cached_tokens)?,
        reused_tokens: host_u64(reused_tokens)?,
        replayed_tokens: host_u64(replayed_tokens)?,
        computed_tokens: host_u64(computed_tokens)?,
    })
}

fn sign_chat_receipt<B: Backend>(
    server: &Server<B>,
    request_sha256: &str,
    seed: u64,
    generated: &ChatGeneration,
    details: &ChatResponseDetails,
    session: SessionReplayRecord,
) -> Result<ResponseReceipt, Box<dyn Error>> {
    let response_tokens_sha256 = token_stream_sha256(&generated.tokens);
    let mut transcript = generated.prompt_tokens.clone();
    transcript.extend_from_slice(&generated.tokens);
    Ok(ResponseReceipt::sign(
        ResponseClaim {
            schema_version: RESPONSE_SCHEMA_VERSION,
            receipt_id: Uuid::new_v4(),
            created_utc: Utc::now(),
            engine_version: env!("CARGO_PKG_VERSION").to_owned(),
            model_sha256: server.model_sha256.clone(),
            request_sha256: request_sha256.to_owned(),
            prompt_tokens_sha256: token_stream_sha256(&generated.prompt_tokens),
            response_tokens_sha256,
            transcript_sha256: token_stream_sha256(&transcript),
            seed,
            prompt_tokens: host_u64(generated.prompt_tokens.len())?,
            generated_tokens: host_u64(generated.tokens.len())?,
            finish_reason: details.finish_reason.clone(),
            cancelled: generated.cancelled,
            session,
        },
        &server.signing_key,
    )?)
}

fn persist_chat_session<B: Backend>(
    server: &mut Server<B>,
    session_id: &str,
    mut stored: StoredSession<B>,
    cancelled: bool,
) -> Result<(), Box<dyn Error>> {
    if cancelled {
        stored.generation.invalidate();
        server.discard_session(session_id)?;
        return Ok(());
    }
    server.clock = server.clock.saturating_add(1);
    stored.last_used = server.clock;
    if let Some(store) = &server.session_store {
        let archive = store.persist(
            session_id,
            &server.model_sha256,
            &stored.generation.checkpoint(),
            stored.last_used,
        )?;
        server.persisted.insert(session_id.to_owned(), archive);
    }
    server
        .insert_session(session_id.to_owned(), stored)
        .map_err(|error| -> Box<dyn Error> { invalid_data(error.to_string()).into() })
}

fn write_chat_response(
    stream: &mut TcpStream,
    context: &ChatResponseContext<'_>,
    streamed: &mut Utf8Stream,
    generated: &ChatGeneration,
    details: &ChatResponseDetails,
    receipt: &ResponseReceipt,
) -> Result<(), Box<dyn Error>> {
    if context.request.stream {
        for content in streamed.finish() {
            write_sse(
                stream,
                &json!({
                    "id": context.completion_id,
                    "object": "chat.completion.chunk",
                    "created": context.created,
                    "model": context.model_id,
                    "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": null}]
                }),
            )?;
        }
        write_sse(
            stream,
            &json!({
                "id": context.completion_id,
                "object": "chat.completion.chunk",
                "created": context.created,
                "model": context.model_id,
                "choices": [{"index": 0, "delta": {}, "finish_reason": details.finish_reason}],
                "usage": {
                    "prompt_tokens": generated.prompt_tokens.len(),
                    "completion_tokens": generated.tokens.len(),
                    "total_tokens": generated.prompt_tokens.len() + generated.tokens.len()
                },
                "leone_receipt": receipt
            }),
        )?;
        write_chunk(stream, b"data: [DONE]\n\n")?;
        finish_chunks(stream)?;
        return Ok(());
    }
    let message = match details.generated_tool_calls.as_ref() {
        Some(tool_calls) => json!({
            "role": "assistant",
            "content": null,
            "tool_calls": tool_calls
        }),
        None => json!({
            "role": "assistant",
            "content": details
                .decoded_content
                .as_ref()
                .expect("non-streaming content is decoded")
        }),
    };
    write_json_with_session(
        stream,
        200,
        &context.session_id,
        &json!({
            "id": context.completion_id,
            "object": "chat.completion",
            "created": context.created,
            "model": context.model_id,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": details.finish_reason
            }],
            "usage": {
                "prompt_tokens": generated.prompt_tokens.len(),
                "completion_tokens": generated.tokens.len(),
                "total_tokens": generated.prompt_tokens.len() + generated.tokens.len()
            },
            "leone_receipt": receipt
        }),
    )?;
    Ok(())
}

impl<B: Backend> Server<B> {
    fn discard_session(&mut self, session_id: &str) -> Result<(), Box<dyn Error>> {
        self.sessions.remove(session_id);
        self.hibernated.remove(session_id);
        self.persisted.remove(session_id);
        if let Some(store) = &self.session_store {
            store.remove(session_id)?;
        }
        Ok(())
    }

    fn lease_session(&mut self, plan: &ChatPlan) -> Result<StoredSession<B>, Box<dyn Error>> {
        if let Some(parent_id) = plan.fork_parent.as_deref() {
            return self.lease_fork_session(parent_id);
        }
        if let Some(stored) = self.sessions.remove(&plan.session_id) {
            return Ok(stored);
        }
        if let Some(stored) = self.hibernated.remove(&plan.session_id) {
            return Ok(StoredSession {
                generation: self.runtime.wake_session(stored.generation)?,
                last_used: stored.last_used,
            });
        }
        if let Some(stored) = self.persisted.get(&plan.session_id) {
            return self.restore_persisted_session(stored);
        }
        Ok(StoredSession {
            generation: GenerationSession::new(),
            last_used: 0,
        })
    }

    fn lease_fork_session(&mut self, parent_id: &str) -> Result<StoredSession<B>, Box<dyn Error>> {
        let generation = if let Some(parent) = self.sessions.get(parent_id) {
            self.runtime.fork_session(&parent.generation)?
        } else if let Some(parent) = self.persisted.get(parent_id) {
            let mut generation = GenerationSession::new();
            generation.restore(parent.archive.checkpoint()?);
            generation
        } else {
            return Err(invalid_data("the fork parent session does not exist").into());
        };
        Ok(StoredSession {
            generation,
            last_used: 0,
        })
    }

    fn restore_persisted_session(
        &self,
        stored: &StoredArchive,
    ) -> Result<StoredSession<B>, Box<dyn Error>> {
        let mut generation = GenerationSession::new();
        generation.restore(stored.archive.checkpoint()?);
        Ok(StoredSession {
            generation,
            last_used: stored.last_used,
        })
    }

    fn finish_scheduled_chat(
        &mut self,
        mut task: ChatTask<B>,
        status: RequestStatus,
    ) -> Result<(), Box<dyn Error>> {
        if task.disconnected {
            task.stored.generation.invalidate();
            self.discard_session(&task.session_id)?;
            return Ok(());
        }
        if let Some(error) = task.failure.take() {
            task.stored.generation.invalidate();
            self.discard_session(&task.session_id)?;
            if task.request.stream {
                write_sse(&mut task.stream, &json!({"error": {"message": error}}))?;
                write_chunk(&mut task.stream, b"data: [DONE]\n\n")?;
                finish_chunks(&mut task.stream)?;
            } else {
                write_error(&mut task.stream, 400, &error)?;
            }
            return Ok(());
        }

        let cancelled = status != RequestStatus::Finished;
        let ChatTask {
            request,
            prompt_tokens,
            tokens,
            options,
            session_id,
            stored,
            mut stream,
            request_sha256,
            created,
            completion_id,
            mut streamed,
            eos,
            ..
        } = task;
        let generated = ChatGeneration {
            prompt_tokens,
            tokens,
            cancelled,
            eos,
        };
        let model_id = self.model_id.clone();
        finish_chat_response(
            self,
            &mut stream,
            ChatResponseContext {
                request: &request,
                session_id,
                completion_id,
                created,
                request_sha256,
                seed: options.seed,
                model_id,
            },
            &mut streamed,
            stored,
            generated,
        )
    }

    fn select_session(&self, requested: Option<&str>, prompt: &[u32]) -> String {
        if let Some(requested) = requested {
            return requested.to_owned();
        }
        self.sessions
            .iter()
            .map(|(id, session)| {
                (
                    common_prefix(prompt, session.generation.evaluated_tokens()),
                    session.last_used,
                    id,
                )
            })
            .chain(self.hibernated.iter().map(|(id, session)| {
                (
                    common_prefix(prompt, session.generation.evaluated_tokens()),
                    session.last_used,
                    id,
                )
            }))
            .chain(self.persisted.iter().map(|(id, session)| {
                (
                    common_prefix(prompt, session.archive.evaluated_tokens()),
                    session.last_used,
                    id,
                )
            }))
            .filter(|(prefix, _, _)| *prefix > 0)
            .max_by_key(|(prefix, used, _)| (*prefix, *used))
            .map(|(_, _, id)| id.clone())
            .unwrap_or_else(|| Uuid::new_v4().to_string())
    }

    fn insert_session(
        &mut self,
        id: String,
        session: StoredSession<B>,
    ) -> Result<(), RuntimeError> {
        self.sessions.insert(id, session);
        while self.sessions.len() > self.max_sessions {
            let Some(oldest) = self
                .sessions
                .iter()
                .min_by_key(|(_, session)| session.last_used)
                .map(|(id, _)| id.clone())
            else {
                break;
            };
            let Some(mut stored) = self.sessions.remove(&oldest) else {
                break;
            };
            let generation = self.runtime.hibernate_session(&mut stored.generation)?;
            self.hibernated.insert(
                oldest,
                StoredHibernation {
                    generation,
                    last_used: stored.last_used,
                },
            );
        }
        while self.hibernated.len() > self.max_hibernated_sessions {
            let Some(oldest) = self
                .hibernated
                .iter()
                .min_by_key(|(_, session)| session.last_used)
                .map(|(id, _)| id.clone())
            else {
                break;
            };
            self.hibernated.remove(&oldest);
        }
        Ok(())
    }
}

fn validate_chat_request(request: &ChatRequest, model_id: &str) -> Result<(), io::Error> {
    validate_chat_identity(request, model_id)?;
    validate_chat_tools(request)?;
    validate_chat_sampling(request)
}

fn validate_chat_identity(request: &ChatRequest, model_id: &str) -> Result<(), io::Error> {
    if request.model != model_id && request.model != "leone" {
        return Err(invalid_data(format!(
            "model {:?} is not served; use {model_id:?}",
            request.model
        )));
    }
    if request.messages.is_empty() {
        return Err(invalid_data("messages must not be empty"));
    }
    if request.n.unwrap_or(1) != 1 {
        return Err(invalid_data("the stable subset supports only n=1"));
    }
    if request.stop.is_some() {
        return Err(invalid_data("stop sequences are not in the stable subset"));
    }
    Ok(())
}

fn validate_chat_tools(request: &ChatRequest) -> Result<(), io::Error> {
    validate_tools(request)?;
    response_constraint(request.response_format.as_ref())?;
    if request.logprobs.unwrap_or(false) || request.top_logprobs.is_some() {
        return Err(invalid_data("logprobs are not in the stable subset"));
    }
    Ok(())
}

fn validate_chat_sampling(request: &ChatRequest) -> Result<(), io::Error> {
    if request.max_tokens.is_some() && request.max_completion_tokens.is_some() {
        return Err(invalid_data(
            "set max_tokens or max_completion_tokens, not both",
        ));
    }
    if request.mirostat_tau.is_some() != request.mirostat_eta.is_some() {
        return Err(invalid_data(
            "mirostat_tau and mirostat_eta must be set together",
        ));
    }
    if request.mirostat_tau.is_some() && request.draft_tokens.is_some() {
        return Err(invalid_data(
            "Mirostat cannot be combined with draft_tokens",
        ));
    }
    Ok(())
}

fn validate_tools(request: &ChatRequest) -> Result<(), io::Error> {
    let Some(tools) = request.tools.as_deref() else {
        if request.tool_choice.is_some() {
            return Err(invalid_data("tool_choice requires tools"));
        }
        return Ok(());
    };
    if request.stream {
        return Err(invalid_data(
            "streaming tool calls are not supported; set stream to false",
        ));
    }
    let names = validate_tool_definitions(tools)?;
    validate_tool_choice(request.tool_choice.as_ref(), &names)
}

fn validate_tool_definitions(
    tools: &[ToolDefinition],
) -> Result<std::collections::BTreeSet<&str>, io::Error> {
    let mut names = std::collections::BTreeSet::new();
    for tool in tools {
        if tool.kind != "function" {
            return Err(invalid_data("every tool type must be function"));
        }
        if tool.function.name.is_empty() || !names.insert(tool.function.name.as_str()) {
            return Err(invalid_data(
                "tool function names must be nonempty and unique",
            ));
        }
        if tool
            .function
            .parameters
            .as_ref()
            .is_some_and(|value| !value.is_object())
        {
            return Err(invalid_data(
                "tool function parameters must be one JSON object",
            ));
        }
    }
    Ok(names)
}

fn validate_tool_choice(
    choice: Option<&Value>,
    names: &std::collections::BTreeSet<&str>,
) -> Result<(), io::Error> {
    let Some(choice) = choice else {
        return Ok(());
    };
    match choice {
        Value::String(value) if matches!(value.as_str(), "auto" | "none" | "required") => {}
        Value::Object(object) => {
            let name = object
                .get("function")
                .and_then(Value::as_object)
                .and_then(|function| function.get("name"))
                .and_then(Value::as_str)
                .ok_or_else(|| invalid_data("named tool_choice requires function.name"))?;
            if object.get("type").and_then(Value::as_str) != Some("function")
                || !names.contains(name)
            {
                return Err(invalid_data(
                    "named tool_choice does not match a function tool",
                ));
            }
        }
        _ => return Err(invalid_data("tool_choice is invalid")),
    }
    Ok(())
}

fn response_constraint(value: Option<&Value>) -> Result<Option<OutputConstraint>, io::Error> {
    let Some(value) = value else {
        return Ok(None);
    };
    let object = value
        .as_object()
        .ok_or_else(|| invalid_data("response_format must be one object"))?;
    let kind = object
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid_data("response_format.type must be one string"))?;
    match kind {
        "text" => Ok(None),
        "json_object" => Ok(Some(OutputConstraint::JsonObject)),
        _ => Err(invalid_data(format!(
            "response_format.type {kind:?} is not supported; use text or json_object"
        ))),
    }
}

fn sampler(request: &ChatRequest) -> Result<Sampler, io::Error> {
    let temperature = request.temperature.unwrap_or(1.0);
    if !temperature.is_finite() || !(0.0..=2.0).contains(&temperature) {
        return Err(invalid_data("temperature must be finite and in [0, 2]"));
    }
    let mut sampler = Sampler {
        temperature: if temperature == 0.0 {
            Temperature::Greedy
        } else {
            Temperature::Scaled(temperature)
        },
        truncations: Vec::new(),
    };
    if let Some(top_p) = request.top_p {
        sampler.truncations.push(Truncation::TopP(top_p));
    }
    if let Some(min_p) = request.min_p {
        sampler.truncations.push(Truncation::MinP(min_p));
    }
    Ok(sampler)
}

fn penalties(request: &ChatRequest) -> Result<Penalties, io::Error> {
    let presence = request.presence_penalty.unwrap_or(0.0);
    let frequency = request.frequency_penalty.unwrap_or(0.0);
    if !presence.is_finite() || !frequency.is_finite() {
        return Err(invalid_data("penalties must be finite"));
    }
    let mut penalties = Penalties::none();
    penalties.presence = presence;
    penalties.frequency = frequency;
    penalties.window = usize::MAX;
    Ok(penalties)
}

fn mirostat(request: &ChatRequest) -> Result<Option<MirostatConfig>, io::Error> {
    request
        .mirostat_tau
        .zip(request.mirostat_eta)
        .map(|(target, learning)| {
            MirostatConfig::new(target, learning).map_err(|error| invalid_data(error.to_string()))
        })
        .transpose()
}

fn speculation(request: &ChatRequest) -> Result<Speculation, io::Error> {
    if request.mirostat_tau.is_some() {
        return Ok(Speculation::Disabled);
    }
    let Some(width) = request.draft_tokens else {
        if request.adaptive_speculation == Some(false) {
            return Ok(Speculation::Disabled);
        }
        return Ok(Speculation::Adaptive(AdaptiveDrafter::new(
            AdaptiveDrafterConfig::default(),
        )));
    };
    let proposal = NonZeroUsize::new(width)
        .filter(|value| value.get() <= 7)
        .ok_or_else(|| invalid_data("draft_tokens must be in [1, 7]"))?;
    Ok(Speculation::Suffix(
        SuffixDrafter::new(
            NonZeroUsize::new(16).expect("sixteen is nonzero"),
            NonZeroUsize::new(4).expect("four is nonzero"),
            proposal,
        )
        .map_err(|error| invalid_data(error.to_string()))?,
    ))
}

fn chat_tokens(
    tokenizer: &leone::Tokenizer,
    architecture: leone::ModelArchitecture,
    messages: &[Message],
    tools: Option<&[ToolDefinition]>,
    tool_choice: Option<&Value>,
) -> Result<Vec<u32>, RuntimeError> {
    let mut tokens = Vec::new();
    append_chat_prefix(tokenizer, architecture, &mut tokens)?;
    if let Some(prompt) = tool_system_prompt(tools, tool_choice)? {
        append_chat_message(tokenizer, architecture, &mut tokens, "system", &prompt)?;
    }
    append_chat_messages(tokenizer, architecture, messages, &mut tokens)?;
    append_chat_suffix(tokenizer, architecture, &mut tokens)?;
    Ok(tokens)
}

fn append_chat_prefix(
    tokenizer: &leone::Tokenizer,
    architecture: leone::ModelArchitecture,
    tokens: &mut Vec<u32>,
) -> Result<(), RuntimeError> {
    if architecture == leone::ModelArchitecture::Llama {
        tokens.push(tokenizer.token_id(LLAMA_BEGIN)?);
    }
    Ok(())
}

fn append_chat_messages(
    tokenizer: &leone::Tokenizer,
    architecture: leone::ModelArchitecture,
    messages: &[Message],
    tokens: &mut Vec<u32>,
) -> Result<(), RuntimeError> {
    for message in messages {
        let role = message_role(message)?;
        if message.name.is_some() && role != "tool" {
            return Err(RuntimeError::token_callback(
                "message names are supported only for tool results",
            ));
        }
        let rendered = render_message_content(message, role)?;
        let template_role = if role == "tool" { "user" } else { role };
        append_chat_message(tokenizer, architecture, tokens, template_role, &rendered)?;
    }
    Ok(())
}

fn message_role(message: &Message) -> Result<&'static str, RuntimeError> {
    match message.role.as_str() {
        "system" | "developer" => Ok("system"),
        "user" => Ok("user"),
        "assistant" => Ok("assistant"),
        "tool" => Ok("tool"),
        _ => Err(RuntimeError::token_callback(format!(
            "message role {:?} is not supported",
            message.role
        ))),
    }
}

fn render_message_content(message: &Message, role: &str) -> Result<String, RuntimeError> {
    let content = message_text(&message.content)?;
    if role == "tool" {
        if message.tool_call_id.as_deref().unwrap_or("").is_empty() {
            return Err(RuntimeError::token_callback(
                "a tool message requires tool_call_id",
            ));
        }
        return Ok(format!("<tool_response>\n{content}\n</tool_response>"));
    }
    if role == "assistant" && message.tool_calls.is_some() {
        return render_request_tool_calls(&content, message.tool_calls.as_deref().unwrap_or(&[]));
    }
    Ok(content)
}

fn append_chat_suffix(
    tokenizer: &leone::Tokenizer,
    architecture: leone::ModelArchitecture,
    tokens: &mut Vec<u32>,
) -> Result<(), RuntimeError> {
    match architecture {
        leone::ModelArchitecture::Qwen3 => {
            tokens.push(tokenizer.token_id(IM_START)?);
            tokens.extend(tokenizer.encode_piece("assistant")?);
            tokens.extend(tokenizer.encode_piece("\n")?);
        }
        leone::ModelArchitecture::Llama => {
            tokens.push(tokenizer.token_id(LLAMA_HEADER_START)?);
            tokens.extend(tokenizer.encode_piece("assistant")?);
            tokens.push(tokenizer.token_id(LLAMA_HEADER_END)?);
            tokens.extend(tokenizer.encode_piece("\n\n")?);
        }
    }
    Ok(())
}

fn message_text(content: &Value) -> Result<String, RuntimeError> {
    if let Some(text) = content.as_str() {
        return Ok(text.to_owned());
    }
    let parts: Vec<ContentPart> = serde_json::from_value(content.clone())
        .map_err(|_| RuntimeError::token_callback("message content parts are invalid"))?;
    let mut text = String::new();
    for part in parts {
        match part {
            ContentPart::Text { text: part } => text.push_str(&part),
            ContentPart::ImageUrl { image_url } => {
                let input = leone::ImageInput::new(image_url.url, image_url.detail.as_deref())
                    .map_err(|error| RuntimeError::token_callback(error.to_string()))?;
                return Err(RuntimeError::token_callback(format!(
                    "this model has no vision backend ({:?}, {:?})",
                    input.source(),
                    input.detail()
                )));
            }
        }
    }
    Ok(text)
}

fn append_chat_message(
    tokenizer: &leone::Tokenizer,
    architecture: leone::ModelArchitecture,
    tokens: &mut Vec<u32>,
    role: &str,
    content: &str,
) -> Result<(), RuntimeError> {
    match architecture {
        leone::ModelArchitecture::Qwen3 => append_qwen_message(tokenizer, tokens, role, content)?,
        leone::ModelArchitecture::Llama => append_llama_message(tokenizer, tokens, role, content)?,
    }
    Ok(())
}

fn append_qwen_message(
    tokenizer: &leone::Tokenizer,
    tokens: &mut Vec<u32>,
    role: &str,
    content: &str,
) -> Result<(), RuntimeError> {
    tokens.push(tokenizer.token_id(IM_START)?);
    tokens.extend(tokenizer.encode_piece(role)?);
    tokens.extend(tokenizer.encode_piece("\n")?);
    tokens.extend(tokenizer.encode_piece(content)?);
    tokens.push(tokenizer.token_id(IM_END)?);
    tokens.extend(tokenizer.encode_piece("\n")?);
    Ok(())
}

fn append_llama_message(
    tokenizer: &leone::Tokenizer,
    tokens: &mut Vec<u32>,
    role: &str,
    content: &str,
) -> Result<(), RuntimeError> {
    tokens.push(tokenizer.token_id(LLAMA_HEADER_START)?);
    tokens.extend(tokenizer.encode_piece(role)?);
    tokens.push(tokenizer.token_id(LLAMA_HEADER_END)?);
    tokens.extend(tokenizer.encode_piece("\n\n")?);
    tokens.extend(tokenizer.encode_piece(content)?);
    tokens.push(tokenizer.token_id(LLAMA_EOT)?);
    Ok(())
}

fn tool_system_prompt(
    tools: Option<&[ToolDefinition]>,
    tool_choice: Option<&Value>,
) -> Result<Option<String>, RuntimeError> {
    let Some(tools) = tools else {
        return Ok(None);
    };
    if tool_choice.and_then(Value::as_str) == Some("none") {
        return Ok(None);
    }
    let definitions = serde_json::to_string(tools)
        .map_err(|error| RuntimeError::token_callback(error.to_string()))?;
    let instruction = tool_instruction(tool_choice);
    let example = r#"<tool_call>{"name":"function_name","arguments":{}}</tool_call>"#;
    Ok(Some(format!(
        "You have access to the function tools below. {instruction}\n\
         Emit each call as {example}.\n<tools>\n{definitions}\n</tools>"
    )))
}

fn tool_instruction(tool_choice: Option<&Value>) -> String {
    match tool_choice {
        Some(Value::String(value)) if value == "required" => "You must call one tool.".to_owned(),
        Some(Value::Object(object)) => object
            .get("function")
            .and_then(Value::as_object)
            .and_then(|function| function.get("name"))
            .and_then(Value::as_str)
            .map(|name| format!("You must call the {name} tool."))
            .unwrap_or_else(|| "Use one named tool.".to_owned()),
        _ => "Call a tool only when it is needed.".to_owned(),
    }
}

fn render_request_tool_calls(
    content: &str,
    calls: &[RequestToolCall],
) -> Result<String, RuntimeError> {
    let mut rendered = content.to_owned();
    for call in calls {
        if call.kind != "function" || call.function.name.is_empty() {
            return Err(RuntimeError::token_callback(
                "assistant tool calls must name one function",
            ));
        }
        let arguments: Value = serde_json::from_str(&call.function.arguments)
            .map_err(|_| RuntimeError::token_callback("tool call arguments must be JSON"))?;
        let _ = &call.id;
        rendered.push_str("<tool_call>");
        rendered.push_str(
            &serde_json::to_string(&json!({
                "name": call.function.name,
                "arguments": arguments
            }))
            .map_err(|error| RuntimeError::token_callback(error.to_string()))?,
        );
        rendered.push_str("</tool_call>");
    }
    Ok(rendered)
}

fn parse_generated_tool_calls(
    content: &str,
    offered: Option<&[ToolDefinition]>,
) -> Result<Option<Vec<GeneratedToolCall>>, io::Error> {
    let mut remaining = content.trim();
    if !remaining.starts_with("<tool_call>") {
        return Ok(None);
    }
    let mut calls = Vec::new();
    while !remaining.is_empty() {
        let (call, rest) = parse_generated_tool_call(remaining, offered)?;
        calls.push(call);
        remaining = rest;
    }
    Ok(Some(calls))
}

fn parse_generated_tool_call<'a>(
    remaining: &'a str,
    offered: Option<&[ToolDefinition]>,
) -> Result<(GeneratedToolCall, &'a str), io::Error> {
    let body = remaining
        .strip_prefix("<tool_call>")
        .ok_or_else(|| invalid_data("tool call output contains text outside its tags"))?;
    let end = body
        .find("</tool_call>")
        .ok_or_else(|| invalid_data("tool call output is missing </tool_call>"))?;
    let (name, arguments) = parse_generated_tool_fields(body, end, offered)?;
    let rest = body[end + "</tool_call>".len()..].trim();
    let call = GeneratedToolCall {
        id: format!("call_{}", Uuid::new_v4().simple()),
        kind: "function",
        function: GeneratedToolCallFunction {
            name,
            arguments: serde_json::to_string(&arguments).map_err(invalid_json)?,
        },
    };
    Ok((call, rest))
}

fn parse_generated_tool_fields(
    body: &str,
    end: usize,
    offered: Option<&[ToolDefinition]>,
) -> Result<(String, Value), io::Error> {
    let value = parse_generated_tool_value(body, end)?;
    let name = generated_tool_name(&value, offered)?;
    let arguments = generated_tool_arguments(&value)?;
    Ok((name, arguments))
}

fn parse_generated_tool_value(body: &str, end: usize) -> Result<Value, io::Error> {
    serde_json::from_str(body[..end].trim()).map_err(invalid_json)
}

fn generated_tool_name(
    value: &Value,
    offered: Option<&[ToolDefinition]>,
) -> Result<String, io::Error> {
    let object = value
        .as_object()
        .ok_or_else(|| invalid_data("tool call output must be one JSON object"))?;
    let name = object
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .ok_or_else(|| invalid_data("tool call output requires one function name"))?;
    if !offered.is_some_and(|tools| tools.iter().any(|tool| tool.function.name == name)) {
        return Err(invalid_data(format!(
            "tool call output names an unoffered function: {name}"
        )));
    }
    Ok(name.to_owned())
}

fn generated_tool_arguments(value: &Value) -> Result<Value, io::Error> {
    let arguments = value
        .as_object()
        .and_then(|object| object.get("arguments"))
        .ok_or_else(|| invalid_data("tool call output requires arguments"))?;
    if !arguments.is_object() {
        return Err(invalid_data("tool call arguments must be one JSON object"));
    }
    Ok(arguments.clone())
}

fn read_request(stream: &mut TcpStream) -> Result<Option<HttpRequest>, io::Error> {
    let Some((mut bytes, header_end)) = read_request_headers(stream)? else {
        return Ok(None);
    };
    let (method, path, headers) = parse_request_head(&bytes[..header_end])?;
    let content_length = request_content_length(&headers)?;
    if content_length > MAX_BODY_BYTES {
        return Err(invalid_data("HTTP body exceeds 4 MiB"));
    }
    read_request_body(stream, &mut bytes, header_end, content_length)?;
    Ok(Some(HttpRequest {
        method,
        path,
        headers,
        body: bytes[header_end..header_end + content_length].to_vec(),
    }))
}

fn read_request_headers(stream: &mut TcpStream) -> Result<Option<(Vec<u8>, usize)>, io::Error> {
    let mut bytes = Vec::new();
    let mut buffer = [0_u8; 8192];
    let header_end = loop {
        let read = stream.read(&mut buffer)?;
        if read == 0 {
            return Ok(None);
        }
        bytes.extend_from_slice(&buffer[..read]);
        if bytes.len() > MAX_HEADER_BYTES {
            return Err(invalid_data("HTTP headers exceed 64 KiB"));
        }
        if let Some(end) = find_bytes(&bytes, b"\r\n\r\n") {
            break end + 4;
        }
    };
    Ok(Some((bytes, header_end)))
}

fn parse_request_head(
    bytes: &[u8],
) -> Result<(String, String, BTreeMap<String, String>), io::Error> {
    let head =
        std::str::from_utf8(bytes).map_err(|_| invalid_data("HTTP headers are not UTF-8"))?;
    let mut lines = head.split("\r\n");
    let (method, path) = parse_request_line(
        lines
            .next()
            .ok_or_else(|| invalid_data("HTTP request line is missing"))?,
    )?;
    let headers = parse_request_headers(lines)?;
    Ok((method, path, headers))
}

fn parse_request_line(line: &str) -> Result<(String, String), io::Error> {
    let mut parts = line.split_whitespace();
    let method = parts
        .next()
        .ok_or_else(|| invalid_data("HTTP method is missing"))?
        .to_owned();
    let path = parts
        .next()
        .ok_or_else(|| invalid_data("HTTP path is missing"))?
        .split('?')
        .next()
        .unwrap_or("/")
        .to_owned();
    Ok((method, path))
}

fn parse_request_headers<'a>(
    lines: impl Iterator<Item = &'a str>,
) -> Result<BTreeMap<String, String>, io::Error> {
    let mut headers = BTreeMap::new();
    for line in lines.filter(|line| !line.is_empty()) {
        let (name, value) = line
            .split_once(':')
            .ok_or_else(|| invalid_data("HTTP header is malformed"))?;
        headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_owned());
    }
    Ok(headers)
}

fn request_content_length(headers: &BTreeMap<String, String>) -> Result<usize, io::Error> {
    headers
        .get("content-length")
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|_| invalid_data("Content-Length is invalid"))
        })
        .transpose()
        .map(|length| length.unwrap_or(0))
}

fn read_request_body(
    stream: &mut TcpStream,
    bytes: &mut Vec<u8>,
    header_end: usize,
    content_length: usize,
) -> Result<(), io::Error> {
    let mut buffer = [0_u8; 8192];
    while bytes.len() - header_end < content_length {
        let read = stream.read(&mut buffer)?;
        if read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "HTTP body ended early",
            ));
        }
        bytes.extend_from_slice(&buffer[..read]);
    }
    Ok(())
}

fn write_json(stream: &mut TcpStream, status: u16, value: &Value) -> io::Result<()> {
    write_json_with_session(stream, status, "", value)
}

fn write_json_with_session(
    stream: &mut TcpStream,
    status: u16,
    session: &str,
    value: &Value,
) -> io::Result<()> {
    let body = serde_json::to_vec(value).map_err(invalid_json)?;
    let reason = if status == 200 { "OK" } else { "Error" };
    write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n",
        body.len()
    )?;
    if !session.is_empty() {
        write!(stream, "X-Leone-Session: {session}\r\n")?;
    }
    write!(stream, "\r\n")?;
    stream.write_all(&body)
}

fn write_error(stream: &mut TcpStream, status: u16, message: &str) -> io::Result<()> {
    write_json(
        stream,
        status,
        &json!({"error": {"message": message, "type": "invalid_request_error"}}),
    )
}

fn write_empty(stream: &mut TcpStream, status: u16, reason: &str) -> io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\nContent-Length: 0\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: authorization, content-type, x-leone-session\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nConnection: close\r\n\r\n"
    )
}

fn write_stream_headers(stream: &mut TcpStream, session: &str) -> io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\nX-Leone-Session: {session}\r\n\r\n"
    )
}

fn write_sse(stream: &mut TcpStream, value: &Value) -> io::Result<()> {
    let mut payload = b"data: ".to_vec();
    payload.extend(serde_json::to_vec(value).map_err(invalid_json)?);
    payload.extend_from_slice(b"\n\n");
    write_chunk(stream, &payload)
}

fn write_chunk(stream: &mut TcpStream, payload: &[u8]) -> io::Result<()> {
    write!(stream, "{:x}\r\n", payload.len())?;
    stream.write_all(payload)?;
    stream.write_all(b"\r\n")?;
    stream.flush()
}

fn finish_chunks(stream: &mut TcpStream) -> io::Result<()> {
    stream.write_all(b"0\r\n\r\n")?;
    stream.flush()
}

fn parse(arguments: &[String]) -> Result<ServeArgs, io::Error> {
    let mut parsed = ServeArgsBuilder::new()?;
    let mut index = 0;
    while index < arguments.len() {
        if parse_serve_argument(&mut parsed, arguments, &mut index)? {
            index += 1;
            continue;
        }
        return Err(invalid_data(format!(
            "serve argument is invalid: {}",
            arguments[index]
        )));
    }
    parsed.finish()
}

struct ServeArgsBuilder {
    model: Option<PathBuf>,
    bind: SocketAddr,
    sessions: usize,
    hibernated_sessions: usize,
    backend: BackendChoice,
    kv_cache_dtype: KvCacheDtype,
    receipts: PathBuf,
    signing_key: PathBuf,
    session_store: Option<PathBuf>,
    allow_remote: bool,
    plan: Option<PathBuf>,
}

impl ServeArgsBuilder {
    fn new() -> Result<Self, io::Error> {
        Ok(Self {
            model: None,
            bind: DEFAULT_BIND
                .parse::<SocketAddr>()
                .expect("default bind is valid"),
            sessions: 2,
            hibernated_sessions: 8,
            backend: BackendChoice::Cuda,
            kv_cache_dtype: KvCacheDtype::F16,
            receipts: PathBuf::from("receipts"),
            signing_key: default_key_path()?,
            session_store: None,
            allow_remote: false,
            plan: None,
        })
    }

    fn finish(self) -> Result<ServeArgs, io::Error> {
        Ok(ServeArgs {
            model: self
                .model
                .ok_or_else(|| invalid_data("serve requires -m <gguf>"))?,
            bind: self.bind,
            sessions: self.sessions,
            hibernated_sessions: self.hibernated_sessions,
            backend: self.backend,
            kv_cache_dtype: self.kv_cache_dtype,
            receipts: self.receipts,
            signing_key: self.signing_key,
            session_store: self.session_store,
            allow_remote: self.allow_remote,
            plan: self.plan,
        })
    }
}

type ServeArgumentParser =
    fn(&mut ServeArgsBuilder, &[String], &mut usize) -> Result<bool, io::Error>;

fn parse_serve_argument(
    parsed: &mut ServeArgsBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    let parsers: &[ServeArgumentParser] = &[
        parse_serve_model,
        parse_serve_bind,
        parse_serve_sessions,
        parse_serve_hibernated_sessions,
        parse_serve_backend,
        parse_serve_kv,
        parse_serve_receipts,
        parse_serve_signing_key,
        parse_serve_session_store,
        parse_serve_allow_remote,
        parse_serve_plan,
    ];
    for parser in parsers {
        if parser(parsed, arguments, index)? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn parse_serve_model(
    parsed: &mut ServeArgsBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if !matches!(arguments[*index].as_str(), "-m" | "--model") {
        return Ok(false);
    }
    parsed.model = Some(PathBuf::from(flag_value(arguments, index)?));
    Ok(true)
}

fn parse_serve_bind(
    parsed: &mut ServeArgsBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] != "--bind" {
        return Ok(false);
    }
    let value = flag_value(arguments, index)?;
    parsed.bind = value
        .parse()
        .map_err(|_| invalid_data(format!("bind address is invalid: {value}")))?;
    Ok(true)
}

fn parse_serve_sessions(
    parsed: &mut ServeArgsBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] != "--sessions" {
        return Ok(false);
    }
    parsed.sessions = parse_positive_serve_value(flag_value(arguments, index)?, "sessions")?;
    Ok(true)
}

fn parse_serve_hibernated_sessions(
    parsed: &mut ServeArgsBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] != "--hibernated-sessions" {
        return Ok(false);
    }
    parsed.hibernated_sessions =
        parse_positive_serve_value(flag_value(arguments, index)?, "hibernated sessions")?;
    Ok(true)
}

fn parse_positive_serve_value(value: &str, name: &str) -> Result<usize, io::Error> {
    value
        .parse()
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| invalid_data(format!("{name} must be nonzero")))
}

fn parse_serve_backend(
    parsed: &mut ServeArgsBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] != "--backend" {
        return Ok(false);
    }
    parsed.backend = match flag_value(arguments, index)? {
        "cuda" => BackendChoice::Cuda,
        "cpu" => BackendChoice::Cpu,
        value => return Err(invalid_data(format!("backend is invalid: {value}"))),
    };
    Ok(true)
}

fn parse_serve_kv(
    parsed: &mut ServeArgsBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] != "--kv" {
        return Ok(false);
    }
    parsed.kv_cache_dtype = match flag_value(arguments, index)? {
        "q8" => KvCacheDtype::Q8,
        "f16" => KvCacheDtype::F16,
        "f32" => KvCacheDtype::F32,
        value => return Err(invalid_data(format!("KV dtype is invalid: {value}"))),
    };
    Ok(true)
}

fn parse_serve_receipts(
    parsed: &mut ServeArgsBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] != "--receipt-dir" {
        return Ok(false);
    }
    parsed.receipts = PathBuf::from(flag_value(arguments, index)?);
    Ok(true)
}

fn parse_serve_signing_key(
    parsed: &mut ServeArgsBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] != "--signing-key" {
        return Ok(false);
    }
    parsed.signing_key = PathBuf::from(flag_value(arguments, index)?);
    Ok(true)
}

fn parse_serve_session_store(
    parsed: &mut ServeArgsBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] != "--session-store" {
        return Ok(false);
    }
    parsed.session_store = Some(PathBuf::from(flag_value(arguments, index)?));
    Ok(true)
}

fn parse_serve_allow_remote(
    parsed: &mut ServeArgsBuilder,
    _arguments: &[String],
    _index: &mut usize,
) -> Result<bool, io::Error> {
    if _arguments[*_index] != "--allow-remote" {
        return Ok(false);
    }
    parsed.allow_remote = true;
    Ok(true)
}

fn parse_serve_plan(
    parsed: &mut ServeArgsBuilder,
    arguments: &[String],
    index: &mut usize,
) -> Result<bool, io::Error> {
    if arguments[*index] != "--plan" {
        return Ok(false);
    }
    parsed.plan = Some(PathBuf::from(flag_value(arguments, index)?));
    Ok(true)
}

fn load_or_create_key(path: &Path) -> Result<SigningKey, io::Error> {
    if path.exists() {
        return read_signing_key(path);
    }
    create_signing_key(path)
}

fn read_signing_key(path: &Path) -> Result<SigningKey, io::Error> {
    let bytes = fs::read(path)?;
    let seed: [u8; 32] = bytes
        .try_into()
        .map_err(|_| invalid_data("Ed25519 key file must contain exactly 32 bytes"))?;
    Ok(SigningKey::from_bytes(&seed))
}

fn create_signing_key(path: &Path) -> Result<SigningKey, io::Error> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut seed = [0_u8; 32];
    fs::File::open("/dev/urandom")?.read_exact(&mut seed)?;
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options.open(path)?.write_all(&seed)?;
    Ok(SigningKey::from_bytes(&seed))
}

fn default_key_path() -> Result<PathBuf, io::Error> {
    let home = std::env::var_os("HOME")
        .ok_or_else(|| invalid_data("HOME is not set; pass --signing-key"))?;
    Ok(PathBuf::from(home).join(".config/leone/response.key"))
}

fn peer_closed(stream: &mut TcpStream) -> bool {
    let mut byte = [0_u8; 1];
    match stream.peek(&mut byte) {
        Ok(0) => true,
        Ok(_) => false,
        Err(error) if error.kind() == io::ErrorKind::WouldBlock => false,
        Err(_) => true,
    }
}

fn common_prefix(left: &[u32], right: &[u32]) -> usize {
    left.iter()
        .zip(right)
        .take_while(|(left, right)| left == right)
        .count()
}

fn reuse_class_name(class: SessionReuseClass) -> &'static str {
    match class {
        SessionReuseClass::Cold => "cold",
        SessionReuseClass::ExactRepeat => "exact-repeat",
        SessionReuseClass::AppendOnly => "append-only",
        SessionReuseClass::ArbitraryBranch => "arbitrary-branch",
        SessionReuseClass::RestoreReplay => "restore-replay",
        SessionReuseClass::DeviceFork => "device-fork",
        SessionReuseClass::HostWake => "host-wake",
    }
}

fn host_u64(value: usize) -> Result<u64, io::Error> {
    u64::try_from(value).map_err(|_| invalid_data("token count exceeds u64"))
}

fn unix_seconds() -> Result<u64, io::Error> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .map_err(|_| invalid_data("system clock is before 1970"))
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn flag_value<'a>(arguments: &'a [String], index: &mut usize) -> Result<&'a str, io::Error> {
    *index += 1;
    arguments
        .get(*index)
        .map(String::as_str)
        .ok_or_else(|| invalid_data("command flag is missing its value"))
}

fn nonzero_usize(value: &str, field: &str) -> Result<usize, io::Error> {
    value
        .parse()
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| invalid_data(format!("{field} must be nonzero: {value}")))
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn invalid_json(error: serde_json::Error) -> io::Error {
    invalid_data(error.to_string())
}

#[cfg(test)]
mod session_store_tests {
    use super::*;

    const MODEL: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    #[test]
    fn identical_sessions_share_one_immutable_blob() {
        let root = std::env::temp_dir().join(format!("leone-session-store-{}", Uuid::new_v4()));
        let (store, loaded) = SessionStore::open(root.clone(), MODEL).expect("new store");
        assert!(loaded.is_empty());
        let checkpoint = GenerationSession::<CpuBackend>::new().checkpoint();

        store
            .persist("left", MODEL, &checkpoint, 1)
            .expect("left session");
        store
            .persist("right", MODEL, &checkpoint, 2)
            .expect("right session");

        assert_eq!(fs::read_dir(root.join("blobs")).expect("blobs").count(), 1);
        let (_, reopened) = SessionStore::open(root.clone(), MODEL).expect("reopen store");
        assert_eq!(reopened.len(), 2);
        fs::remove_dir_all(root).expect("remove test store");
    }

    #[test]
    fn response_format_maps_json_object_to_the_runtime_constraint() {
        assert_eq!(
            response_constraint(Some(&json!({"type": "json_object"})))
                .expect("supported response format"),
            Some(OutputConstraint::JsonObject)
        );
        assert!(response_constraint(Some(&json!({"type": "json_schema"}))).is_err());
    }

    #[test]
    fn tool_prompt_and_response_use_typed_function_calls() {
        let tools = vec![ToolDefinition {
            kind: "function".to_owned(),
            function: ToolFunction {
                name: "weather".to_owned(),
                description: Some("Read the weather".to_owned()),
                parameters: Some(json!({
                    "type": "object",
                    "properties": {"city": {"type": "string"}}
                })),
            },
        }];
        let prompt = tool_system_prompt(Some(&tools), Some(&json!("required")))
            .expect("tool prompt")
            .expect("enabled tools");
        assert!(prompt.contains("You must call one tool."));
        assert!(prompt.contains("\"name\":\"weather\""));

        let calls = parse_generated_tool_calls(
            r#"<tool_call>{"name":"weather","arguments":{"city":"Oslo"}}</tool_call>"#,
            Some(&tools),
        )
        .expect("valid generated call")
        .expect("tool calls");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "weather");
        assert_eq!(calls[0].function.arguments, r#"{"city":"Oslo"}"#);
        assert!(parse_generated_tool_calls(
            r#"<tool_call>{"name":"unknown","arguments":{}}</tool_call>"#,
            Some(&tools),
        )
        .is_err());
    }

    #[test]
    fn content_parts_preserve_text_and_type_image_failures() {
        assert_eq!(
            message_text(&json!([
                {"type": "text", "text": "one"},
                {"type": "text", "text": " two"}
            ]))
            .expect("text parts"),
            "one two"
        );
        let error = message_text(&json!([{
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}
        }]))
        .expect_err("no vision backend");
        assert!(error.to_string().contains("no vision backend"));
    }

    #[test]
    fn scheduled_request_errors_use_the_openai_shape() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test listener");
        let mut client = TcpStream::connect(listener.local_addr().expect("listener address"))
            .expect("connect test client");
        let (mut server, _) = listener.accept().expect("accept test client");

        let result = scheduled_request_or_error::<()>(
            &mut server,
            Err(invalid_data("invalid request").into()),
        )
        .expect("write error response");
        assert!(result.is_none());
        drop(server);

        let mut response = String::new();
        client
            .read_to_string(&mut response)
            .expect("read error response");
        assert!(response.starts_with("HTTP/1.1 400 Error\r\n"));
        assert!(response.contains("\"type\":\"invalid_request_error\""));
        assert!(response.contains("\"message\":\"invalid request\""));
    }
}

fn is_disconnected(error: &(dyn Error + 'static)) -> bool {
    error
        .downcast_ref::<io::Error>()
        .map(|error| {
            matches!(
                error.kind(),
                io::ErrorKind::BrokenPipe
                    | io::ErrorKind::ConnectionReset
                    | io::ErrorKind::ConnectionAborted
            )
        })
        .unwrap_or(false)
}

#[derive(Default)]
struct Utf8Stream {
    pending: Vec<u8>,
}

impl Utf8Stream {
    fn push(&mut self, bytes: &[u8]) -> Vec<String> {
        self.pending.extend_from_slice(bytes);
        self.take(false)
    }

    fn finish(&mut self) -> Vec<String> {
        self.take(true)
    }

    fn take(&mut self, final_chunk: bool) -> Vec<String> {
        let mut output = Vec::new();
        while take_utf8_piece(&mut self.pending, final_chunk, &mut output) {}
        output
    }
}

fn take_utf8_piece(pending: &mut Vec<u8>, final_chunk: bool, output: &mut Vec<String>) -> bool {
    match std::str::from_utf8(pending) {
        Ok(text) => {
            if !text.is_empty() {
                output.push(text.to_owned());
                pending.clear();
            }
            false
        }
        Err(error) if error.valid_up_to() > 0 => {
            let count = error.valid_up_to();
            let text = String::from_utf8(pending.drain(..count).collect())
                .expect("validated UTF-8 prefix");
            output.push(text);
            true
        }
        Err(error) if error.error_len().is_some() => {
            let count = error.error_len().expect("checked above");
            pending.drain(..count);
            output.push("\u{fffd}".to_owned());
            true
        }
        Err(_) if final_chunk => {
            output.push(String::from_utf8_lossy(pending).into_owned());
            pending.clear();
            false
        }
        Err(_) => false,
    }
}
