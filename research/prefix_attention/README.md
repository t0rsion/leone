# Shared-prefix attention prototype

This directory contains a standalone CPU and GPU experiment. It does not
change the default Leone runtime path or backend contract.

The prototype compares four paths:

- `per_row` computes each query against its K/V rows in token order.
- `fixed_tile_per_row` uses absolute token tiles and one fixed reduction order per query.
- `shared_read_unconstrained` loads each tile once per query group. Its tile order follows group scheduling.
- `shared_read_fixed_reduction` loads each shared tile once. Each query keeps the same absolute tile order and reduction order.

Each path computes every query-dependent output. The shared paths reuse K/V tile reads only. They do not reuse attention scores, probabilities, or output rows.

The CUDA and Metal harnesses generate FP32 queries and host-rounded FP16 K/V
values from the manifest. Each receipt stores the measured FP32 output arrays,
the shared input digest, and the common scalar FP64 oracle digest. Calibration
cases and evaluation cases use different names, shapes, and seeds. Evaluation
requires criteria frozen from a valid calibration receipt.

Absolute token tiles cover the full token range. The final tile may be partial. A missing tile raises `MissingTileError`; paths do not truncate or pad it silently. The unrelated case gives every query separate K/V rows and checks that shared paths cannot claim reuse.

Run the protocol from the repository root:

```text
python3 research/prefix_attention/run_experiment.py \
  --phase calibration \
  --output research/prefix_attention/receipts/calibration.json
python3 research/prefix_attention/run_experiment.py \
  --phase evaluation \
  --calibration-receipt research/prefix_attention/receipts/calibration.json \
  --output research/prefix_attention/receipts/evaluation.json
python3 -m unittest discover -s research/prefix_attention -p 'test_*.py'
python3 research/prefix_attention/check_gpu_complexity.py
```

The receipts record generated timings, operator counts, estimated working
bytes, output hashes, oracle error, and schedule checks. CPU counts estimate
operator traffic. They are not DRAM counter measurements. The prototype makes
no speed or novelty claim. End-to-end model evidence remains open.

## CUDA and Metal harness

`gpu_manifest.json` freezes the CUDA and Metal case names, shapes, seeds, tile
rule, and path names. Calibration and evaluation names are disjoint. The GPU
oracle uses host-rounded FP16 K/V values and scalar FP64 attention. A fixed
path must keep its output bitwise equal when row groups change. The
unconstrained path records schedule sensitivity without treating it as an
error.

Build the CUDA harness on the SM89 host:

```text
research/prefix_attention/cuda_fixed_reduction/build_cuda.sh \
  /tmp/prefix_attention_cuda
```

Generate a calibration receipt when the GPU is idle:

```text
python3 research/prefix_attention/run_cuda.py \
  --phase calibration --build \
  --output research/prefix_attention/receipts/tmp/prefix_attention_cuda_calibration.json
python3 research/prefix_attention/freeze_gpu_criteria.py \
  research/prefix_attention/receipts/tmp/prefix_attention_cuda_calibration.json \
  --backend cuda \
  --output research/prefix_attention/receipts/tmp/prefix_attention_cuda_criteria.json
python3 research/prefix_attention/run_cuda.py \
  --phase evaluation --build \
  --criteria research/prefix_attention/receipts/tmp/prefix_attention_cuda_criteria.json \
  --output research/prefix_attention/receipts/tmp/prefix_attention_cuda_evaluation.json
python3 research/prefix_attention/validate_gpu_receipt.py \
  research/prefix_attention/receipts/tmp/prefix_attention_cuda_evaluation.json \
  research/prefix_attention/receipts/tmp/prefix_attention_cuda_criteria.json \
  --calibration research/prefix_attention/receipts/tmp/prefix_attention_cuda_calibration.json \
  --backend cuda
```

The Metal source uses the same descriptors and reduction contract. Run the
same calibration, freeze, evaluation, and validation sequence with
`run_metal.py` on a macOS host with Xcode Metal tools.

Both wrappers write generated records under `receipts/tmp`. They rebuild the
operator, hash the executable or Metal artifacts before execution, and record
source, manifest, compiler, device, and command provenance. The wrapper and
validator recompute quality from the stored output arrays and common oracle.
The criteria file derives quality limits from calibration output before
evaluation. It records paired baseline and candidate medians and derives a
reviewable effect threshold from the nearest-rank 95th percentile of paired
split-half variation.

The default validator accepts a negative timing result as a valid experiment.
Pass `--require-positive` when the caller gates a speed claim.

The CPU path remains exploratory and is not GPU evidence.

## Leone prefill differential

The opt-in CUDA driver in `production_driver/` calls the public
`leone_cuda::attention_prefill_f16` path with one contiguous causal KV cache.
It uses the manifest's context length, query chunk, GQA shape, and FP16 K/V
inputs. Its Rust FP64 oracle and the independent `production_oracle.py` output
check cover the same causal rows. The receipt keeps the driver's local oracle
digest and quality under `backend_*` fields. The driver does not enable a
shared-read kernel in the Leone runtime.

Build and run it when CUDA is available and the GPU is idle:

```text
python3 research/prefix_attention/run_production.py \
  --phase calibration \
  --output research/prefix_attention/receipts/tmp/production_calibration.json
python3 research/prefix_attention/freeze_production_criteria.py \
  research/prefix_attention/receipts/tmp/production_calibration.json \
  --output research/prefix_attention/receipts/tmp/production_criteria.json
python3 research/prefix_attention/run_production.py \
  --phase evaluation \
  --criteria research/prefix_attention/receipts/tmp/production_criteria.json \
  --output research/prefix_attention/receipts/tmp/production_evaluation.json
python3 research/prefix_attention/validate_production_receipt.py \
  research/prefix_attention/receipts/tmp/production_evaluation.json \
  research/prefix_attention/receipts/tmp/production_criteria.json \
  --calibration research/prefix_attention/receipts/tmp/production_calibration.json
```

This differential records the current causal prefill path. It does not measure
segmented span storage, graph recapture, a full model, or a llama.cpp control.

## Leone Runtime shared prefix driver

`runtime_driver/` runs the pinned cached workload through the Runtime batch
attention seam. `--backend cuda` selects the CUDA backend. `--backend metal`
selects the Metal backend. The four `--path` values select the per-row,
fixed-tile, shared-read, and fixed-reduction paths.

The driver requires the model digest and vocabulary from the source workload.
It records raw logits, predicted positions, token digests, memory snapshots,
batch counters, source hashes, and the selected backend. It reports
`quality: unverified` until an independent quality receipt is attached.

Build and run one calibration path with the pinned Qwen3 model:

```text
python3 research/prefix_attention/run_runtime.py \
  --backend cuda \
  --phase calibration \
  --path shared_read_fixed_reduction \
  --model models/Qwen3-8B-Q4_K_M.gguf \
  --output research/prefix_attention/receipts/tmp/runtime_cuda_calibration.json
```

The Metal command uses `--backend metal` on a macOS host. The driver remains
opt-in and does not change the default Runtime path.

The runner refuses an existing `--output` and a missing or non-regular
`--criteria` file before it builds. If publication fails after a run, it keeps
that run's raw receipt and sidecars and prints their directory. The build record
carries `source_tree_sha256`, `source_tree_dirty`, and `provenance_unknown`. The
runner requires both flags to be false. With `--phase evaluation`, the receipt
records the digest of `--criteria` and derives no verdict from it. Quality stays
`unverified` until an independent common oracle and a frozen validation are
attached.

## Runtime freeze and validation

`freeze_runtime_criteria.py` and `validate_runtime.py` turn four calibration
receipts into frozen criteria, then score four evaluation receipts against them.
Both tools run offline on CPU. They read the receipts, their sidecars, the
runtime manifest, the source workload, the token fixture, the BF16 oracle
bundles, and the optional GPU sample records. They never write to an input. The
freeze and the report refuse to replace an existing file.

Each phase needs one receipt for each of the four paths, with identical model,
device, build, repetition, and affinity records. `fixed_tile_per_row` is the
control. The tools score every other path against it on the same logit rows:

- Calibration cases come from `runtime_manifest.json`. The validator recomputes
  each input digest and each logit binding from the manifest and the token
  fixture, and requires the driver's values to match.
- Evaluation cases come from `llama_cached_workload.json`. The validator derives
  the expected logit rows from that workload alone and requires the driver's
  scored rows to match, in order, with no missing, repeated, or extra row.
- The receipts must run the source workload's subject model.
- The unrelated case must count no multi-row group on any path. The shared
  paths must count at least one on the shared cases.
- The frozen limit for each metric is the largest calibration value plus one
  representable FP64 step. The metrics are maximum absolute logit difference,
  mean KLD, and maximum KLD. Each non-control path is scored against the
  control. The control is scored against its own other repetitions, so a control
  that does not repeat exactly fails its own limit.
- The report gives each path its own numerical verdict and role. `per_row`
  differs from the control by design. `shared_read_unconstrained` leaves the
  reduction order free, so a numerical `fail` on it is an expected outcome and
  not a defect signal. `shared_read_fixed_reduction` is the candidate that the
  numerical gate targets.

The calibration and evaluation workloads use different cases. They share fixture
token positions, and the freeze records that count as
`fixture_positions_shared_by_phases`.

### Digest scope

For calibration receipts, the validator recomputes `logits_digest` and
`raw_logit_position_digest` from the sidecar of each repetition. Every step is
captured and no setup event exists, so the sidecar holds exactly the digested
rows. For evaluation receipts, it does not. Setup events write sidecar rows that
the digests exclude, and the receipts do not mark which rows they are. The
sidecar hashes and the row header identities are the only linkage there.
`token_digest` is not recomputed in either phase. The freeze and the report
record this scope as `digest_scope`.

### Build and source identity

The build record holds `source_tree_sha256` and a dirty flag. They cover only
`Cargo.toml`, `Cargo.lock`, `crates`, `fixtures`, `.cargo`, `rust-toolchain*`,
and `build-support`. Nothing under `research/` enters them. The driver crate,
the manifests, the fixture, and the wrapper scripts enter through
`source_sha256`, which the validator requires to match across all receipts.

The freeze pins the validator tools by content hash: the two new scripts,
`runner_support.py`, `run_llama_cached_comparator.py`, `common_oracle.py`,
`gpu_receipt.py`, and `generate_manifest.py`. These are every local module that
the validator imports. They are not driver inputs, so a tool edit voids the
criteria but not the calibration receipts. A driver or compiler source edit
requires a new calibration.

Every receipt comes from its own build. The report lists the executable hash of
each path and the run order that the receipts record. It states whether the four
paths ran from four executables. Builds are not reproducible, so equal hashes
are not required. Drift between builds, processes, and run order is not bounded.

### Timing rule

The timing rule is new for this study. Neither `runtime_manifest.json` nor
`gpu_manifest.json` declares a practical effect rule, and `gpu_receipt.py` needs
paired baseline, candidate, baseline samples that a runtime run lacks. The freeze
records that origin as `timing_rule_origin`. The timed value is `setup_ms` plus
`decode_ms` per repetition. The signed effect is `1 - candidate median / control
median` for each case. The review threshold is the nearest-rank 95th percentile
of the relative deviation from the median, pooled over every path, case, and
repetition.

A case is labelled `faster_than_control_beyond_review_threshold` when the effect
is at or above the threshold, `slower_than_control_beyond_review_threshold` when
it is at or below the negative threshold, and `within_review_threshold`
otherwise. The overall label is `mixed_beyond_review_threshold` when both
directions occur. A run without a bound GPU sample record appends
`_unmonitored` to every label. `--require-positive` exits 1 unless the overall
label is the faster label with no suffix.

The threshold measures repeat jitter inside one process. Each path ran in its
own process, so the threshold excludes drift between processes, builds, and run
order, and it does not bound the difference between two paths. An effect that
clears it is a review flag. It does not support a speed claim, and no timing
label does. The report records `speed_claim: none`.

### GPU monitoring

`--gpu-samples` names an externally sampled clock and power file. `--collection`
names the record that lists the receipt and driver hashes. The freeze records
both content hashes and the observed clock, power, and temperature ranges. The
validator takes the same pair for the evaluation phase through
`--evaluation-gpu-samples` and `--evaluation-collection`. The samples come from
outside the driver and are not attributed to a path, case, or repetition. The
receipts hold no DRAM counter and no layer, head, or context geometry, so the
report sets both measured and estimated traffic to null. Without a sample pair,
the criteria and the report record that monitoring is unbound.

The binding checks content hashes, not whether sample timestamps cover the run.
The parser accepts `nvidia-smi` columns only. Metal runs remain unmonitored, so
`--require-positive` exits 1 for Metal.

### Quality references

Quality has two separately named references.

The BF16 oracle is the pinned llama.cpp CPU run of `Qwen3-8B-BF16.gguf`. It is
the only reference the report calls BF16. The oracle is an independent
implementation run on the pinned BF16 file. No record shows that the BF16 and Q4
files share one original checkpoint, so every absolute distance to the oracle
includes Q4 quantization error and the difference between two implementations.
The report carries that statement as `limit`.

`run_llama_cached_comparator.py` enters oracle mode only when both
`--subject-model` and `--oracle-identity` are given. In that mode it requires
the subject and oracle file hashes to match one pair in
`common_oracle.LOGIT_ORACLE_PAIRS`, the GGUF metadata of both files to equal the
independent pin in `common_oracle.LOGIT_ORACLE_METADATA`, and the engine file
type to be BF16. It does not check the backend. The validator enforces the CPU
backend and rejects a receipt from another one. The default mode keeps its
receipt format and output files. New receipts record the changed collector hash.

Oracle mode reads GGUF metadata only. The reader rejects a repeated key, a
string that is not UTF-8, a bad bool byte, an array of arrays, and a length
beyond its bound. It never replaces a byte. The identity digest is the sha256 of
the sorted lines `key=sha256(value bytes)` over the architecture keys and over
`GGUF_TOKENIZER_KEYS`, so it fixes the key set. Its value hash omits the type
code of a top-level scalar, and the recorded identity records use this scheme, so
it stays fixed. The model file hash proves the exact bytes, types included.

The metadata pin adds one check to the file hash. A validator that never opens
the model checks the identity record against digests read by the pinned
llama.cpp `GGUFReader`, a second implementation. It requires an exact field set
and the exact `tokenizer_keys` list. BOS and padding ids are not compared. The
workloads pass fixed token ids and no BOS, so those ids change no logit.

The calibration cases are not in the source workload.
`run_llama_cached_comparator.py --write-calibration-workload` generates their
workload from `runtime_manifest.json`. Each branch is a prefill event with no
output, followed by one `decode_stepwise` event that forces the manifest's
teacher tokens. The model contract, fixture, and context come from the source
workload. The validator rebuilds every row identity and requires the oracle rows
to match the driver's rows in order.

The freeze needs both BF16 oracle bundles, calibration and evaluation. Each
bundle is the comparator receipt, its logit file, and its identity record. The
gate is control relative. For each non-control path, the difference between its
distance to the BF16 rows and the control's distance on the same rows is
frozen from calibration only. The limit is the largest calibration difference,
floored at zero, plus one representable FP64 step. A path that sits closer to
the oracle than the control never fails. Absolute distances are report only and
gate nothing. The freeze reads the evaluation bundle for its hash and rows and
never scores it. Calibration holds four short cases, so the limits describe that
sample. The `fixed_tile_per_row` control stays the numerical reference. The BF16
verdict is a second gate.

Both bundles must share one CPU origin: host system and machine, CPU model,
compiler, linked llama.cpp libraries, backend module, comparator binary content,
and the recorded collector sources. The binary name and the kernel release are
not compared. The origin check also omits thread count, CPU affinity, and
requested context. The report names the origin `same_host` or `common_bundle`.
The oracle is a CPU run, so its rows do not depend on the receipts' backend. One
bundle from the Linux host that holds the BF16 model serves both CUDA and Metal
receipts, and the Mac never loads the BF16 model. The freeze records the
collector source hashes that each bundle recorded. The current
`run_llama_cached_comparator.py` may differ from the collector that produced an
older bundle. The validator does not claim otherwise.

The `--oracle-receipt` option keeps the earlier comparison against the subject's
own Q4 weights on the source workload. The report names it
`q4_subject_cpu_comparison`, never BF16. It has no calibration rows, so it has no
frozen limit and never gates. Without BF16 bundles, the freeze needs
`--quality-unverified` and the report prints `quality: unverified`.

The report makes no speed, novelty, or end-to-end quality claim. A negative
timing result or a failed numerical or BF16 verdict stays in the report. Allocator
live bytes after setup measure shared storage. Row groups counted by the batch
path show shared reads.

### Commands

Run the whole sequence on the CUDA host with the pinned Qwen3 models:

```text
STUDY=research/prefix_attention/receipts/tmp/runtime_cuda
PA=research/prefix_attention
SUBJECT=models/Qwen3-8B-Q4_K_M.gguf
ORACLE=models/Qwen3-8B-BF16.gguf
for path in per_row fixed_tile_per_row shared_read_unconstrained shared_read_fixed_reduction; do
  python3 $PA/run_runtime.py --backend cuda --phase calibration --path "$path" \
    --model "$SUBJECT" --output "$STUDY/calibration/$path.json"
done
python3 $PA/run_llama_cached_comparator.py \
  --write-calibration-workload "$STUDY/oracle/calibration_workload.json"
bf16() {  # bf16 NAME WORKLOAD
  python3 $PA/run_llama_cached_comparator.py --llama-dir external/llama.cpp \
    --model "$ORACLE" --subject-model "$SUBJECT" --backend cpu --manifest "$2" \
    --output "$STUDY/oracle/$1.json" --logits "$STUDY/oracle/$1.logits.f32" \
    --oracle-identity "$STUDY/oracle/$1.identity.json"
}
bf16 calibration "$STUDY/oracle/calibration_workload.json"
bf16 evaluation "$PA/llama_cached_workload.json"
BF16="--bf16-calibration $STUDY/oracle/calibration.json $STUDY/oracle/calibration.logits.f32 $STUDY/oracle/calibration.identity.json \
      --bf16-evaluation $STUDY/oracle/evaluation.json $STUDY/oracle/evaluation.logits.f32 $STUDY/oracle/evaluation.identity.json"
python3 $PA/freeze_runtime_criteria.py --backend cuda --calibration "$STUDY"/calibration/*.json \
  $BF16 --gpu-samples "$STUDY/calibration-gpu-samples.csv" --collection "$STUDY/calibration-collection.json" \
  --output "$STUDY/criteria.json"
for path in per_row fixed_tile_per_row shared_read_unconstrained shared_read_fixed_reduction; do
  python3 $PA/run_runtime.py --backend cuda --phase evaluation --path "$path" \
    --criteria "$STUDY/criteria.json" --model "$SUBJECT" --output "$STUDY/evaluation/$path.json"
done
python3 $PA/validate_runtime.py --backend cuda --criteria "$STUDY/criteria.json" \
  --calibration "$STUDY"/calibration/*.json --evaluation "$STUDY"/evaluation/*.json \
  $BF16 --gpu-samples "$STUDY/calibration-gpu-samples.csv" --collection "$STUDY/calibration-collection.json" \
  --evaluation-gpu-samples "$STUDY/evaluation-gpu-samples.csv" \
  --evaluation-collection "$STUDY/evaluation-collection.json" \
  --report "$STUDY/report.json"
```

The two GPU sample pairs come from an external sampler and a collection record
that names the receipts of that phase. Drop the four `--*gpu-samples` and
`--*collection` options to run with monitoring unbound. Drop the `--bf16-*`
options and add `--quality-unverified` to the freeze to run with
`quality: unverified`.

On the Metal host, set `--backend metal` and `STUDY=.../runtime_metal`. Copy the
BF16 bundle from the host that holds the model, and pass it in `$BF16`. The
report labels its origin `common_bundle`. Run the calibration and evaluation
receipts on the Mac, or copy them to any host with this checkout and run only the
freeze and validation. Freeze first, and run the evaluation from the same source
tree as the calibration. The validator rejects a receipt with another
`source_sha256` or `source_tree_sha256`.
