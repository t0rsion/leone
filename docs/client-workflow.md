# Check the OpenAI Python client

The client check sends requests to a local Leone HTTP server. It covers two
concurrent streaming forks, response signatures, a matched stop sequence, an
official-template streamed function call, a truncated tool request, and
recovery after a client disconnect.

Run these commands from a source checkout. Use Python 3.11 or later. Create a
separate environment and install the pinned client:

```sh
python3 -m venv target/client-venv
target/client-venv/bin/python -m pip install -r scripts/client-requirements.txt
```

With the GPU idle, run:

```sh
taskset -c 16-31 scripts/check-openai-client.sh \
  target/release/leone \
  models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  receipts/openai-client-check.json
```

The wrapper starts the supplied binary on loopback. It removes temporary
sessions and response files when the check ends. It preserves the server log
beside the output as `OUTPUT.server.log`. Neither output path may already exist.

The generated record includes the binary and model hashes, build provenance,
client version, installed environment versions, and every signed response
receipt. Streaming requests opt into the canonical usage event. The tool
workflow checks streamed deltas, uses the returned SDK assistant message for
the follow-up, and verifies that a one-token tool failure leaves the server
healthy.

Signature verification checks agreement between the signed claim and its
signature. It does not establish model quality.

The disconnect probe verifies subsequent service availability. It does not
observe cancellation cleanup. The CUDA lifecycle tests check buffer release.
This check does not measure performance.

The check also grows a conversation beyond its initial context bucket.
It sends TCP resets while another stream runs and rejects a cancelled survivor.
Reset timing does not identify the exact scheduler phase.
