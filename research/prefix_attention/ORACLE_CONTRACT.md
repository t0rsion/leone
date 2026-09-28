# Prefix attention oracle contract

The GPU harnesses use `scalar-fp64-attention-v2` as the quality oracle. The
oracle reads the generated FP32 query values and host-rounded FP16 K/V values.
For each query head, it computes every score in FP64, finds the maximum score,
computes the FP64 exponential weights, and divides the FP64 weighted value sum
by the FP64 normalizer. It uses the GQA map from the case descriptor.

The host implementation in `common_oracle.py` independently regenerates each
manifest case and checks the input and oracle digests. FP64 sums use explicit
left-to-right accumulation. Exponential values use fixed binary64 range
reduction and 40 Taylor terms for softmax deltas in `[-16, 0]`. Square-root
values accept integer head dimensions in `[1, 64]`; known dimensions use fixed
binary64 constants, and other dimensions use fixed Newton iterations. The
exponential and square-root routines do not call platform libm. Version 1
receipts remain historical. New receipts must declare version 2. Each backend
receipt includes the FP32 output arrays for the measured path and its schedule
copy.
The wrappers retain the backend's local FP64 digest and quality values under
`backend_oracle_digest` and `backend_quality_*`, then compute the canonical
quality values from those arrays and the common host oracle. The canonical
values are recorded under `oracle_digest` and `quality_*`. The input digest is
`fnv1a64-le-q-f32-k-f16-v-f16-v1`. It feeds the little
endian bytes of every query FP32 bit pattern, then every key FP16 bit pattern,
then every value FP16 bit pattern into FNV-1a 64. The digest identifies the
rounded input consumed by a backend. It does not identify an output or a
timing sample.

CUDA and Metal record the oracle ID, input digest algorithm, input digest, and
maximum absolute and relative error. Backend output rounding may differ. The
oracle is independent of the online FP32 softmax recurrence used by the four
GPU paths. The validator recomputes every recorded error from the output
arrays. The opt-in production driver uses the same generator with a causal
single-cache layout. Its current cuBLASLt path rounds Q to FP16 before the
matmul, so `production_oracle.py` applies that rounding and validates its
digests and output values.

Provenance uses `wrapper-observed-structural-v1`. The wrapper hashes source
files around the build and run, then hashes generated artifacts after the
build. The validator checks those records and does not provide a signed
attestation or defend against a malicious binary.
