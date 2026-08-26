# OpenAI API subset

Leone `v0.1.0` implements a stable subset of the OpenAI chat API. Unsupported
fields fail with a client error.

## Endpoints

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

The server answers `OPTIONS` for local browser clients.

## Chat request

Required fields:

- `model`
- `messages`

Supported controls:

- `max_tokens` or `max_completion_tokens`
- `seed`
- `temperature`
- `top_k`, `top_p`, `min_p`, `top_a`, `tfs_z`, and `typical_p`
- `presence_penalty`, `frequency_penalty`, and repetition controls
- `stream`
- `tools` and `tool_choice`
- JSON object response constraints
- Leone session and speculation extensions

Limits:

- `n` must equal `1`.
- `stop`, `logprobs`, and `top_logprobs` are not supported.
- Mirostat and draft speculation cannot be combined.
- Output constraints disable speculation.

## Sessions

Set a session with `leone_session`, the `x-leone-session` header, or `user`.
The response returns `x-leone-session`.

Set `leone_fork_session` to fork an existing live or persisted session. The new
session identifier must differ from the parent and must not already exist.

With `--session-store <path>`, Leone persists model-bound replay checkpoints.
A cancelled or failed generation invalidates the session and removes its
persisted reference.

## Streaming

Set `stream` to `true` for server-sent events. Leone emits a role chunk, content
chunks, a terminal chunk, and `[DONE]`. A disconnected client cancels at the
next runtime boundary.

## Tools

Leone accepts function tools. The generated assistant text must contain the
selected tool call in the documented JSON form. Leone parses that text into a
`tool_calls` response. It does not execute tools.

## Receipts

Each completed response includes `leone_receipt`. The same signed receipt is
written under the configured receipt directory.

The receipt contains its Ed25519 public key. Self-verification proves that its
claim and signature agree. To establish signer identity, compare that key with
a trusted key distributed separately.

## Security

The default bind address is `127.0.0.1:8080`. A non-loopback address requires
`--allow-remote`.

Leone does not implement authentication, authorization, quotas, or TLS. Before
exposing it to a network, put a trusted proxy in front of it.
