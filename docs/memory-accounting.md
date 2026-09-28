# Memory accounting

Leone records physical backend allocations separately from scheduler KV
reservations. A reservation is an admission bound. It is not a device allocation.

Each allocation has one class, byte count, and lifetime. A shared allocation is
charged once until its final owner releases it. `Backend::clone_buffer` creates
an independent allocation. Snapshots include live bytes, peak live bytes,
allocation counts, and free counts.

| Class | Allocation |
|---|---|
| `model_weight` | Imported weights without backend repacking |
| `repacked_weight` | Backend-repacked weights |
| `activation` | Per-session decode buffers |
| `kv_cache` | Shared and private KV segments, and session device positions |
| `backend_scratch` | Kernel scratch and retained batch or verifier buffers |
| `prefill_scratch` | Prefill activations and backend workspace |
| `graph_buffer` | Buffers retained for graph execution |
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

Fork shares committed KV segments and copies session activations. Each child
appends to private storage. A partially used shared segment keeps its physical
capacity while the child maps only its committed prefix.

Hibernation copies session buffers to host RAM and releases the session's device
owners. Other sessions can retain shared KV allocations. Waking restores
independent buffers. Persistent checkpoints store tokens for replay, not KV bytes.
On Apple silicon, host and device buffers use the same physical memory. A host
copy does not establish a physical memory saving.

The lifecycle tests check that session allocation classes rise during use and
return to their baseline after cancellation, drop, and hibernation. Backend
scratch can remain allocated for later requests.
