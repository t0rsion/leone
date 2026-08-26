# Receipt index

This directory contains evidence for the public release candidate only.

A runtime receipt records the candidate commit, model digest, hardware, clocks,
workload, and sample summary. A quality receipt records the corpus, BF16 oracle,
comparison definition, and sample count. A runtime claim is quality-verified
only when it references a matching quality receipt.

These receipts bind the v0.1.0 candidate and comparator to the same BF16 oracle.

| Date | Kind | Receipt ID | Result |
|---|---|---|---|
| 2026-08-26 | quality | `d24c7c28-e9ba-4c30-aabb-bf2858818826` | 0.036081 mean KLD nats |
| 2026-08-26 | quality | `4e188eb5-091f-434b-abb1-23a11ca4b198` | 0.035593 mean KLD nats |
| 2026-08-26 | runtime | `67f91e44-0604-49d1-aa49-84457e9d50e1` | 172.565 decode tok/s |
| 2026-08-26 | runtime | `2d188d61-58c2-4a80-bfad-74af08dac061` | 168.317 decode tok/s |
| 2026-08-26 | correctable | `b937751f-87dc-4078-8dad-44d45465148a` | release gate pass |
