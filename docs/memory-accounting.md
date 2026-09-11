# Memory accounting

Leone records physical backend allocations separately from scheduler KV
reservations. A reservation is an admission bound. It is not a device allocation.

Each allocation has one class, byte count, and lifetime. Cloning creates a new
allocation identity. Dropping its owner releases the live count. Snapshots include
live bytes, peak live bytes, allocation counts, and free counts.

| Class | Allocation |
|---|---|
| `model_weight` | Imported weights without backend repacking |
| `repacked_weight` | Backend-repacked weights |
| `activation` | Per-session decode buffers |
| `kv_cache` | Per-session KV buffers and device position |
| `backend_scratch` | Kernel scratch and retained batch or verifier buffers |
| `prefill_scratch` | Prefill activations and cuBLASLt workspace |
| `contract_buffer` | Other buffers allocated through the backend contract |

CUDA snapshots cover direct device allocations. They do not measure process RAM
or total GPU memory reported by the driver. CUDA graphs, streams, events, and
library handles have object counts. Their byte sizes remain unreported.

Counters follow ownership release. A destructor cannot confirm cleanup after
a driver failure. Reclassification moves live bytes and allocation counts;
earlier class peaks remain recorded.

`GET /debug/service-trace` returns prefill and resident-decode events with memory
samples. Samples include `logical_reserved_kv_bytes` separately. The response
reports dropped event and sample counts when its bounded history fills.

Sessions own contiguous KV buffers. Fork copies the allocated context, including
unused capacity. Hibernation copies session buffers to host RAM and releases
their device owners. Waking restores those buffers. Persistent checkpoints store
tokens for replay, not KV bytes.

The lifecycle tests check that session allocation classes rise during use and
return to their baseline after cancellation, drop, and hibernation. Backend
scratch can remain allocated for later requests.
