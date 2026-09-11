# OpenAI API subset

Leone implements a stable subset of the OpenAI chat API. Unknown top-level
request fields fail with a client error.

## Endpoints

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

The server answers `OPTIONS` for local browser clients.
The diagnostic `GET /debug/service-trace` endpoint reports bounded scheduler
events and [memory samples](memory-accounting.md).

## Chat request

Required fields:

- `model`
- `messages`

Supported controls:

- `max_tokens` or `max_completion_tokens`
- `seed`
- `temperature`
- `top_k`, `top_p`, `min_p`, `top_a`, `tfs_z`, and `typical_p`
- `presence_penalty`, `frequency_penalty`, `repetition_penalty`, and `repetition_window`
- `stream`
- `stream_options: {"include_usage": true}` with `stream: true`
- `tools` and `tool_choice`
- JSON object response constraints
- Leone session and speculation extensions

Limits:

- `n` must equal `1`.
- `stop`, `logprobs`, and `top_logprobs` are not supported.
- Mirostat and draft speculation cannot be combined.
- Output constraints disable speculation.
- `top_k` is positive. Probability truncations are greater than zero and at most one.
- `repetition_penalty` is positive. `repetition_window` bounds penalty history; zero disables it.
- Streaming function tools are not supported.

## Scheduling and backpressure

The server batches compatible active requests up to `--batch-size`. Attention,
sampling, and session state remain separate. Explicit speculation and output
constraints use the serial decode path.

`--prefill-chunk` sets the prompt positions evaluated before returning to the
service loop. It overrides the execution plan's prefill chunk size. The service
trace records prefill chunks and resident decode progress separately.

`--sessions` bounds resident KV state. KV reservations round up to fixed token
pages. If admission would exceed a configured bound, the server returns HTTP
`429` with this shape:

```json
{
  "error": {
    "message": "the server cannot admit this request",
    "type": "server_overloaded",
    "code": "kv-capacity"
  }
}
```

The `code` is one of `active-limit`, `queue-limit`, `kv-capacity`,
`prompt-limit`, `output-limit`, `invalid-prefix`, `invalid-priority`, or
`expired-deadline`.

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

Cancellation checks run between prefill chunks and decode quanta. Socket reads
and writes remain blocking. Responsiveness assumes clients send requests and
consume responses promptly.

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

## Context and terminal state

`--context-limit <n>` caps each session at a nonzero value within the model context.
A request whose prompt or output budget exceeds that limit receives HTTP 400.
The server does not shorten a requested output budget.

Prefix admission credit uses the runtime reuse check, including context capacity
and KV type. Host wake and archive replay receive no credit before leasing.
The server removes terminal scheduler records and output copies after each loop.
Cumulative counters and bounded debug traces remain available.

If no output limit is supplied, the same context check applies to the default
512-token budget. A failed host wake invalidates that in-memory session;
a configured session store can retain its replay archive.
