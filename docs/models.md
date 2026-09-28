# Model workflows

Leone acquires registry models with `pull` and stores each verified artifact in
the Leone data directory. `run` combines acquisition with `chat` or `serve`.
`models` lists the entries in the selected registry.

## List the registry

The built-in registry contains the release model entries. List it with:

```sh
leone models
```

Each output row contains the model ID, artifact file name, and lowercase
SHA-256 digest, separated by tabs. A custom registry can replace the built-in
registry:

```sh
leone models --registry <registry.toml>
```

The registry uses schema version `1`:

```toml
schema_version = 1

[[models]]
id = "my-model"
aliases = ["my-alias"]
artifact = "my-model.gguf"
url = "https://example.invalid/my-model.gguf"
sha256 = "<64 lowercase hexadecimal digits>"
```

Each model ID and alias is nonempty and unique. `artifact` contains one file
name. `url` starts with `https://`, `http://`, or `file://`. The digest must
contain 64 lowercase hexadecimal digits.

## Pull a model

Pull a built-in entry by ID or alias:

```sh
leone pull qwen3:8b
leone pull qwen3-8b
```

Use a custom registry with `--registry <toml>`:

```sh
leone pull my-model --registry <registry.toml>
```

`pull` accepts one model name and an optional `--registry <toml>`. Other
arguments produce an error. The command prints the model ID, artifact path,
SHA-256 digest, source URL, and a final `ready` path after verification.

## Run a model

`run` pulls the selected entry, then starts interactive chat:

```sh
leone run qwen3:8b --backend cpu --tokens 64
```

The command accepts `--registry <toml>` and `--serve`. Without `--serve`, all
remaining options use the `chat` parser. With `--serve`, all remaining options
use the `serve` parser:

```sh
leone run qwen3:8b --serve --backend cuda --bind 127.0.0.1:8080
```

`--serve` has no value. `--registry <toml>` and `--serve` are removed before
the selected parser receives the remaining arguments. `run` supplies the
verified `-m` path. Passing `-m` or `--model` to `run` returns an error.

## Chat options

Use a local artifact directly with `chat`:

```sh
leone chat -m <model.gguf> [options]
```

The parser accepts these path and runtime flags:

| Flag | Value or default |
| --- | --- |
| `-m`, `--model` | GGUF path. Required. |
| `-n`, `--tokens` | Maximum response length. Default `512`. |
| `--backend` | `cuda`, `cpu`, or `metal`. Default follows the build. |
| `--eager-decode` | Disable decode graph replay. |
| `--kv` | `q8`, `f16`, or `f32`. Default `f16`. |
| `--plan` | JSON execution plan path. |

Truncation stages run in flag order. Temperature and seed are separate controls:

| Flag | Value |
| --- | --- |
| `--temp`, `--temperature` | Temperature. The default is greedy sampling. |
| `--seed` | Unsigned sampler seed. Default `0`. |
| `--top-k` | Nonzero token count. |
| `--top-p` | Probability mass. |
| `--min-p` | Minimum probability relative to the maximum. |
| `--top-a` | Probability threshold relative to the maximum squared. |
| `--tfs` | Tail-free sampling value. |
| `--typical` | Typical sampling mass. |
| `--epsilon` | Probability floor. |
| `--eta` | Entropy-relaxed probability floor. |
| `--min-k` | Raw-logit minimum-k value. |
| `--top-n-sigma` | Raw-logit deviation bound. |

Penalty flags rewrite logits before truncation:

| Flag | Value or default |
| --- | --- |
| `--repeat-penalty` | Repetition value. `1.0` is off. |
| `--presence-penalty` | Presence value. |
| `--frequency-penalty` | Frequency value. |
| `--penalty-window` | History length. Default `64`. |
| `--dry-multiplier` | DRY multiplier. `0.0` is off. |
| `--dry-base` | DRY growth per matching token. Default `1.75`. |
| `--dry-allowed-length` | Free matching length. Default `2`. |
| `--dry-window` | History length searched by DRY. Default `64`. |

Speculation flags are:

| Flag | Meaning |
| --- | --- |
| `--draft` | Use suffix drafting with a nonzero proposal length. |
| `--no-draft` | Disable the default adaptive drafter. |

## Serve options through `run`

When `run` receives `--serve`, the server parser accepts these flags:

| Flag | Value or default |
| --- | --- |
| `-m`, `--model` | GGUF path. `run` supplies the verified path. |
| `--bind` | Listen address. Default `127.0.0.1:8080`. |
| `--sessions` | Limit for each: admitted requests and resident idle sessions. Default `2`. |
| `--context-limit` | Per-session context bound. Default is the model context. |
| `--prefill-chunk` | Prompt tokens per scheduler chunk. Default is the plan or `4096`. |
| `--batch-size` | Maximum requests per decode pass. Default `min(8, backend maximum)`. |
| `--hibernated-sessions` | Maximum host sessions. Default `8`. |
| `--backend` | `cuda`, `cpu`, or `metal`. Default follows the build. |
| `--kv` | `q8`, `f16`, or `f32`. Default `f16`. |
| `--plan` | JSON execution plan path. |
| `--receipt-dir` | Signed response receipt directory. Default `receipts`. |
| `--signing-key` | 32-byte Ed25519 key file. Default `$HOME/.config/leone/response.key`. |
| `--session-store` | Model-bound session replay directory. |
| `--allow-remote` | Permit a non-loopback listen address. |
| `--max-connections` | Maximum socket workers. Default `64`. |
| `--max-connections-per-client` | Maximum workers for one client IP. Default `4`. |
| `--max-pending-requests` | Maximum queued engine requests. Default `64`. |
| `--max-output-bytes` | Maximum bytes in one response queue. Default `262144`. |
| `--request-timeout-ms` | Request and service deadline. Default `30000`. |
| `--cors-origin` | Explicit browser origin. Repeatable. |
| `--proxy-origin` | Browser origin allowed at a trusted proxy boundary. |
| `--trusted-proxy` | Proxy IP trusted for `X-Forwarded-For`. Repeatable. |

## Cache and recovery

The cache root is selected in this order:

1. `LEONE_HOME`, when set and nonempty.
2. `$XDG_DATA_HOME/leone`, when `XDG_DATA_HOME` is set and nonempty.
3. `$HOME/.local/share/leone`.

Leone stores an artifact at
`<cache-root>/models-v04/<sha256>/<artifact>`. The digest directory makes the
artifact path independent of model ID spelling and separates it from older
cache layouts.

Every pull and run hashes an existing artifact against the registry SHA-256.
The cache does not use modification time or file size as identity. A matching
artifact skips network access. A mismatch moves the artifact to
`<artifact>.invalid-<12-hex-digits>` before a new download starts.

Downloads use a same-directory partial file named `<artifact>.part`. HTTPS and
HTTP downloads resume with `curl --continue-at -` after an interrupted run. A
sidecar records hashes of the URL and expected SHA-256. Leone resumes a partial
only when that identity matches the selected registry entry. A partial with
another or invalid identity starts again. If the server rejects the resume,
Leone retries once from an empty partial. A local `file://` source copies from
the start unless the partial already has the expected digest. For an artifact
named `model` without an extension, the partial path is `model.model.part`; the
lock remains in the digest lock namespace.

The completed partial is hashed before installation. A mismatch moves it to
`<artifact>.part.invalid-<12-hex-digits>` and returns an error. Leone never
publishes an artifact before this check passes.

The `source` line removes URL userinfo, query parameters, and fragments. The
partial identity stores a URL digest, so credentials and signed query values do
not enter the cache metadata.

The lock file is `<cache-root>/models-v04/.locks/<sha256>.lock`. Every artifact
with the same digest uses that digest-level lock. It remains after a pull, but
the active lock belongs to the operating system. Valid pulls and runs use a
shared lock. Repairs and installs use an exclusive lock. If a process exits,
the operating system releases its lock, so a later pull can reuse the existing
lock file. A repair requested while a reader holds the lock returns an error.
Leone does not trust a PID in the file.

Installation uses an atomic same-directory rename. If another artifact already
exists when the rename fails, Leone hashes it. A matching artifact is retained.
An invalid artifact is quarantined, then the verified partial is installed. A
rename or quarantine error leaves the source file for a later recovery attempt.

The cache can contain quarantined files and partial files after failures. They
are retained so the digest and failed input remain available for diagnosis.
