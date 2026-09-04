# Receipt index

This directory contains indexed evidence for public v0.1.0 and v0.2.0 releases.

A runtime receipt records the candidate commit, model digest, hardware, clocks,
workload, and sample summary. A quality receipt records the corpus, BF16 oracle,
comparison definition, and sample count. A runtime claim is quality-verified
only when it references a matching quality receipt.

The v0.1.0 receipts bind Leone and its comparator to the same BF16 oracle.
The v0.2.0 receipts add independent Qwen BF16 and Llama F16 oracles.

| Date | Kind | Receipt ID | Result |
|---|---|---|---|
| 2026-08-26 | quality | `d24c7c28-e9ba-4c30-aabb-bf2858818826` | 0.036081 mean KLD nats |
| 2026-08-26 | quality | `4e188eb5-091f-434b-abb1-23a11ca4b198` | 0.035593 mean KLD nats |
| 2026-08-26 | runtime | `67f91e44-0604-49d1-aa49-84457e9d50e1` | 172.565 decode tok/s |
| 2026-08-26 | runtime | `2d188d61-58c2-4a80-bfad-74af08dac061` | 168.317 decode tok/s |
| 2026-08-26 | correctable | `b937751f-87dc-4078-8dad-44d45465148a` | release gate pass |
| 2026-08-27 | quality | `e5c41c26-b7da-4eb6-abdb-501316b0db48` | 0.033925 mean KLD nats |
| 2026-08-27 | runtime | `2adeadc8-775c-4e4c-93ce-6ada0b9d9bfa` | 172.507 decode tok/s |
| 2026-08-27 | runtime | `38be3405-0d21-460b-a513-5adc0474d335` | 172.608 decode tok/s |
| 2026-08-27 | runtime | `694172f6-414c-4dc7-96ab-771233d9fb8f` | 172.500 decode tok/s |
| 2026-08-27 | runtime | `44de764b-fed1-4eeb-ba9f-7cdd72a2b487` | 169.793 decode tok/s |
| 2026-08-27 | quality | `3e668ae3-c295-402e-95ed-7b55b4267ded` | 0.056617 mean KLD nats |
