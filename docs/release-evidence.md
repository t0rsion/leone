# Batched-service evidence

The candidate batches compatible decode rows. Attention, sampling,
cancellation, and retained sessions stay separate.
The [study receipt](../receipts/batched-service-study.json) records the source and inputs.

## Tested workload

| Field | Value |
|---|---|
| Model SHA-256 | `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785` |
| Concurrent clients | 4 |
| Maximum tokens per request | 64 |
| Repetitions | 5 |
| Shared batch limit | 4 |
| Baseline batch limit | 1 |

## Results

| Ratio | Minimum | Median | Maximum | Required |
|---|---:|---:|---:|---:|
| Aggregate completion throughput | 1.9540554695042753 | 1.9584065206215597 | 1.967212880696005 | greater than 1 |
| P95 completion latency | 0.49876198614840317 | 0.5000915391804645 | 0.5019858981457418 | less than 1 |

All batched transcripts match the baseline. Every run accepts a request after
a forced client disconnect.

The linked quality record is `db8f6bb8-9181-4f9b-ac2c-6df82844e42b`. Its mean KLD is 0.03085331714411758
nats, and its top-1 agreement is 0.9162150896206753. The study verifies that the
quality record and served model have the same SHA-256 digest.

## Limits

- The result covers one Qwen3 8B model, one RTX 4090, four clients, and one
  deterministic prompt.
- The baseline accepts concurrent requests but executes at most one decode row
  per pass.
- Non-streaming time to first HTTP byte is not time to first generated token.
- KV pages are admission units. The runtime does not remap physical KV storage.
