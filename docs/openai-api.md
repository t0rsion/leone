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
`GET /debug/service-metrics` reports cumulative request, prefix reuse, and memory
counters. These diagnostic endpoints are outside the OpenAI subset.

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
- `stream_options: {"include_usage": false}` with `stream: true`
- `stop` as one nonempty string or an array of at most four nonempty strings,
  with a total limit of 4096 UTF-8 bytes
- `tools` and `tool_choice`
- JSON object response constraints
- `leone_template`: `legacy` or `official`, default `legacy`
- `leone_session` and `leone_fork_session`
- `draft_tokens`
- `adaptive_speculation`, default `false`

Limits:

- `n` must equal `1`.
- The JSON body is at most 4 MiB, with at most 32,768 JSON nodes and 64 levels
  of nesting. Duplicate object fields fail before request parsing.
- `logprobs` and `top_logprobs` are not supported.
- `draft_tokens` is an integer in `[1, 7]`. It cannot be combined with
  `adaptive_speculation: true`.
- Output constraints, user stop sequences, and official Llama tool-header
  termination disable speculation.
- `top_k` is positive. Probability truncations are greater than zero and at most one.
- `repetition_penalty` is positive. An omitted `repetition_window` uses full
  history. Zero disables it. A positive value bounds penalty history.
- `presence_penalty` and `frequency_penalty` are finite values in `[-2, 2]`.
- `mirostat_tau` and `mirostat_eta` cannot be combined with draft or adaptive
  speculation.
- `tool_choice: "required"` needs at least one function tool. A named choice
  needs the named function. `tool_choice: "none"` disables tool calls.
- `response_format: {"type": "json_object"}` cannot be combined with function
  tools.
- Each assistant tool call needs one matching later tool result before the next
  message. Call IDs and result IDs must match and remain unique.
- Official Llama history supports one tool call per assistant message.
- `function.strict: true` is rejected because constrained tool decoding is not
  part of this subset. `strict: false` is accepted.
- Response-only message fields `refusal`, `annotations`, `audio`, and
  `function_call` may be null for OpenAI SDK round trips.
- `stop` cannot be combined with `tools`.

## Prompt rendering

The default `legacy` mode preserves the documented prompt format. Qwen tool calls
use the XML markers documented in the tool response format. The mode keeps
these bytes across sessions.

Set `leone_template` to `official` during migration. This mode renders the
model's pinned GGUF template. Qwen keeps XML tool calls. Llama adds its
knowledge and current date lines, puts tool definitions in the first user
message, and uses native `parameters` and `ipython` fields.

An assistant message that ends a request remains open for continuation in both
modes. A new assistant header is added only after a user or tool message.

Official mode matches the pinned template for the independent fixture cases.
Qwen required and named choices add Leone instructions. Llama named choices
filter the rendered definitions to the selected function. These choice
extensions are outside the byte-identical fixture cases.

The source tree's `fixtures/openai-chat-template-tokens.json` records the
official model output from pinned llama.cpp CPU `/apply-template` and
`/tokenize` calls. The fixture stores the GGUF template hash, the pinned
tokenizer source hash, a fixed date, and tool history for both model families.
The independent fixture catches changes in model templates.

The source tree records the default adapter's bytes separately in
`fixtures/openai-chat-template-leone.json`. Its Qwen JSON spacing is a
stable byte contract. Leone generated this fixture, so it pins the adapter's
tokens and does not compare them with the official template. Use `official`
when the model's exact native bytes are required.

## Scheduling and backpressure

The server batches compatible active requests up to `--batch-size`. The default
uses the backend limit, capped at eight decode rows. CUDA accepts eight rows.
The CPU and Metal backends accept one. An explicit larger request fails before
model load. Attention, sampling, and session state remain separate.
Explicit speculation and output constraints use the serial decode path.

`--prefill-chunk` sets the prompt positions evaluated before returning to the
service loop. It overrides the execution plan's prefill chunk size. The service
trace records prefill chunks and resident decode progress separately.

`--sessions` separately limits admitted requests and resident idle sessions.
Memory budgets cap their combined allocations. KV reservations round up to
fixed token pages. If admission would exceed a configured bound, the server
returns HTTP `429` with this shape:

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
`session-capacity`, `prompt-limit`, `output-limit`, `invalid-prefix`,
`invalid-priority`, or `expired-deadline`.

## Sessions

Set a session with `leone_session` or the `x-leone-session` header. Session
identifiers must be nonempty visible ASCII and at most 128 bytes. The `user`
field is request metadata and does not select a session.
The response returns `x-leone-session`.

Set `leone_fork_session` to fork an existing live or persisted session. The new
session identifier must differ from the parent and must not already exist.

With `--session-store <path>`, Leone persists model-bound replay checkpoints.
A cancelled or failed generation invalidates the session and removes its
persisted reference. A matched stop retains committed session state.

### Session store disk recovery

The session store does not reclaim unreferenced checkpoint blobs or
crash-left temporary files. Its disk use can grow and is not bounded by
`--sessions` or the memory budgets. Do not delete individual files while Leone
uses the store. An operating-system disk quota can limit the store's disk use.

If the store fills its filesystem or prevents startup, replace it as a unit:

1. Stop Leone.
2. Move the directory passed to `--session-store` to a backup. If the filesystem
   is full, move the backup to another filesystem.
3. Start Leone with the original path. Leone creates a new empty store.
4. Keep the backup unchanged until it is no longer needed for recovery or
   diagnosis.

The new store cannot continue or fork sessions from the backup. Prefix reuse
starts cold.

## Streaming

Set `stream` to `true` for server-sent events. Leone emits a role chunk, content
or tool-call chunks, a terminal chunk, and `[DONE]`. A TCP reset, failed
response write, or output queue failure cancels generation at the next runtime
boundary. Read EOF alone permits the client to finish sending its request and
keep reading the response.

The terminal chunk contains `finish_reason`. Completion chunks include
`usage: null`. With `include_usage: true`, a later chunk has an empty `choices`
array and usage counts. False, null, or an omitted option omits that usage
chunk. `stop` removes the matched sequence from content, including matches
across token and UTF-8 fragment boundaries. The generated token count includes
tokens evaluated through the matched sequence.

If validation fails after stream headers, the server sends one SSE error event
and `[DONE]`. It does not send a second HTTP response.

Cancellation checks run between prefill chunks and decode quanta. Socket reads
and writes run in bounded transport workers. A slow sender uses one connection
slot. A slow reader fills its response queue and cancels at the next boundary.
The default request deadline is 30 seconds from accept. Queue saturation returns
HTTP 429, and deadline expiry before a response starts returns HTTP 408. Set
`--request-timeout-ms` to change the read, service, and ordinary write
deadlines. A terminal HTTP or SSE frame has a fixed 250 millisecond grace period
after the request deadline. The transport writes a fallback deadline response
without waiting for executor cleanup. If stream headers were sent, the fallback
is an SSE error, `[DONE]`, and the final chunk. The writer queues complete HTTP
responses and complete chunked frames. A socket write can still end partway
through a frame when its absolute deadline expires. It drops frames queued after
the deadline and closes when the grace period ends.

Each response queue holds at most `--max-output-bytes` payload bytes and 256
complete frames. The application-owned transport memory is bounded by
`max_connections * (64 KiB + 8 MiB + max-output-bytes + 256 frame slots)` plus
request metadata. The 8 MiB term covers the transient input copy while a
4 MiB body is parsed. Kernel socket buffers, thread stacks, and the bounded
rejection socket queue are outside this bound.

## Tools

Leone accepts function tools. The generated assistant text must contain the
selected tool call in the documented JSON form. Legacy mode uses the XML
`<tool_call>` form. Official Llama mode also accepts its native JSON
`parameters` form. Leone parses the text into a `tool_calls` response.
Streaming requests send parsed calls in `delta.tool_calls`. Leone does not
execute tools. Assistant tool-call history must include matching tool results
before another message.

## Receipts

Each completed response includes `leone_receipt`. The same signed receipt is
written under the configured receipt directory.

The receipt records server-side inference completion. It does not prove client
delivery. If Leone cannot queue the final response, it invalidates the session
and keeps the immutable receipt. A later delivery failure leaves the completed
session and receipt available.

The receipt contains its Ed25519 public key. Self-verification proves that its
claim and signature agree. To establish signer identity, compare that key with
a trusted key distributed separately.

## Security

The default bind address is `127.0.0.1:8080`. A non-loopback address requires
`--allow-remote`.

Leone does not implement authentication, authorization, application quotas, or
TLS. Before exposing it to a network, put a trusted proxy in front of it.
Browser origins are denied by default. Set `--cors-origin <origin>` for each allowed origin.
Set `--proxy-origin <origin>` only when the proxy boundary controls that origin.
Allowed origins receive `Access-Control-Expose-Headers: X-Leone-Session`, so
browser code can read the session identifier returned by the server.
Connections are bounded per client IP. Set `--trusted-proxy <ip>` only for a
proxy address that appends `X-Forwarded-For`. Leone scans that header from
right to left, skips configured trusted proxy addresses, and uses the first
untrusted address. Missing, invalid, or all-trusted values use the proxy IP.

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

## Prefix reuse metrics

Inspect the prompt token counts with:

```sh
curl -sS http://127.0.0.1:8080/debug/service-metrics | jq '.realized_prompt_tokens'
```

The counters include requests that ended with a recorded `SessionReplay`,
including requests cancelled after prefill. In-flight requests and requests
without a replay record contribute nothing.

| Field | Count |
|---|---|
| `reused_tokens` | Prompt tokens served from retained KV state. |
| `replayed_tokens` | Archived prompt tokens recomputed during restore. |
| `computed_tokens` | Remaining prompt tokens computed by prefill. |
| `planned_credit_tokens` | Admission credit for the same recorded requests. |
| `credit_shortfall_tokens` | Sum of positive differences between planned credit and realized reuse. |

`prefix_reuse_sources.reused_tokens` records plan-time credit for leased
requests. Host wake can reuse more tokens than admission credited. These
counters do not measure hardware cache hits or prove client delivery.
If `counter_overflowed` is true, cumulative counts have saturated.
