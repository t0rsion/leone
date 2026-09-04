#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 BASE_COMMIT CANDIDATE_COMMIT BASE_F16_LOG BASE_Q8_LOG CANDIDATE_F16_RECEIPT CANDIDATE_Q8_RECEIPT OUTPUT" >&2
  exit 2
fi

base_commit=$1
candidate_commit=$2
base_f16_log=$3
base_q8_log=$4
candidate_f16_receipt=$5
candidate_q8_receipt=$6
output=$7

if [[ -e $output ]]; then
  echo "output already exists: $output" >&2
  exit 2
fi

for input in "$base_f16_log" "$base_q8_log" "$candidate_f16_receipt" "$candidate_q8_receipt"; do
  if [[ ! -f $input ]]; then
    echo "input does not exist: $input" >&2
    exit 2
  fi
done

for commit in "$base_commit" "$candidate_commit"; do
  resolved=$(git rev-parse --verify "$commit^{commit}" 2>/dev/null) || {
    echo "revision is not a commit: $commit" >&2
    exit 2
  }
  if [[ $resolved != "$commit" ]]; then
    echo "revision must be a full commit ID: $commit" >&2
    exit 2
  fi
done

for receipt in "$candidate_f16_receipt" "$candidate_q8_receipt"; do
  measured_commit=$(jq -er '.workload.engine.git_commit' "$receipt")
  if [[ $measured_commit != "$candidate_commit" ]]; then
    echo "candidate revision does not match $receipt" >&2
    exit 2
  fi
done

log_metric() {
  local log=$1
  local pattern=$2
  local field=$3
  awk -v pattern="$pattern" -v field="$field" '$0 ~ pattern { print $field; found=1; exit } END { if (!found) exit 1 }' "$log"
}

base_f16_ttft=$(log_metric "$base_f16_log" '^TTFT at 4096 tokens:' 6)
base_f16_decode=$(log_metric "$base_f16_log" '^decode median' 3)
base_f16_workspace=$(log_metric "$base_f16_log" '^prefill workspace:' 3)
base_q8_ttft=$(log_metric "$base_q8_log" '^TTFT at 4096 tokens:' 6)
base_q8_decode=$(log_metric "$base_q8_log" '^decode median' 3)
base_q8_workspace=$(log_metric "$base_q8_log" '^prefill workspace:' 3)

candidate_f16_ttft=$(jq -er '.results.ttft_ms.median' "$candidate_f16_receipt")
candidate_f16_decode=$(jq -er '.results.decode_tok_s.median' "$candidate_f16_receipt")
candidate_f16_workspace=$(jq -er '.notes[] | select(startswith("Prefill workspace:")) | capture("Prefill workspace: (?<bytes>[0-9]+) bytes").bytes | tonumber' "$candidate_f16_receipt")
candidate_q8_ttft=$(jq -er '.results.ttft_ms.median' "$candidate_q8_receipt")
candidate_q8_decode=$(jq -er '.results.decode_tok_s.median' "$candidate_q8_receipt")
candidate_q8_workspace=$(jq -er '.notes[] | select(startswith("Prefill workspace:")) | capture("Prefill workspace: (?<bytes>[0-9]+) bytes").bytes | tonumber' "$candidate_q8_receipt")

mkdir -p "$(dirname "$output")"
jq -n \
  --arg created_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg base_commit "$base_commit" \
  --arg candidate_commit "$candidate_commit" \
  --arg base_f16_log "$base_f16_log" \
  --arg base_q8_log "$base_q8_log" \
  --arg candidate_f16_receipt "$candidate_f16_receipt" \
  --arg candidate_q8_receipt "$candidate_q8_receipt" \
  --arg base_f16_sha "$(sha256sum "$base_f16_log" | cut -d' ' -f1)" \
  --arg base_q8_sha "$(sha256sum "$base_q8_log" | cut -d' ' -f1)" \
  --arg candidate_f16_sha "$(sha256sum "$candidate_f16_receipt" | cut -d' ' -f1)" \
  --arg candidate_q8_sha "$(sha256sum "$candidate_q8_receipt" | cut -d' ' -f1)" \
  --argjson base_f16_ttft "$base_f16_ttft" \
  --argjson base_f16_decode "$base_f16_decode" \
  --argjson base_f16_workspace "$base_f16_workspace" \
  --argjson base_q8_ttft "$base_q8_ttft" \
  --argjson base_q8_decode "$base_q8_decode" \
  --argjson base_q8_workspace "$base_q8_workspace" \
  --argjson candidate_f16_ttft "$candidate_f16_ttft" \
  --argjson candidate_f16_decode "$candidate_f16_decode" \
  --argjson candidate_f16_workspace "$candidate_f16_workspace" \
  --argjson candidate_q8_ttft "$candidate_q8_ttft" \
  --argjson candidate_q8_decode "$candidate_q8_decode" \
  --argjson candidate_q8_workspace "$candidate_q8_workspace" \
  '{
    schema_version: "leone.prefill-study.v1",
    created_utc: $created_utc,
    workload: {
      model: "Qwen3-8B-Q4_K_M.gguf",
      context_tokens: 4096,
      repetitions: 5,
      hardware: "NVIDIA GeForce RTX 4090 (SM89)"
    },
    revisions: {
      baseline: $base_commit,
      candidate: $candidate_commit
    },
    inputs: [
      {role: "baseline-f16", path: $base_f16_log, sha256: $base_f16_sha},
      {role: "baseline-q8", path: $base_q8_log, sha256: $base_q8_sha},
      {role: "candidate-f16", path: $candidate_f16_receipt, sha256: $candidate_f16_sha},
      {role: "candidate-q8", path: $candidate_q8_receipt, sha256: $candidate_q8_sha}
    ],
    results: {
      f16: {
        baseline: {ttft_ms: $base_f16_ttft, decode_tok_s: $base_f16_decode, workspace_bytes: $base_f16_workspace},
        candidate: {ttft_ms: $candidate_f16_ttft, decode_tok_s: $candidate_f16_decode, workspace_bytes: $candidate_f16_workspace},
        ttft_reduction_percent: ((1 - $candidate_f16_ttft / $base_f16_ttft) * 100),
        decode_change_percent: (($candidate_f16_decode / $base_f16_decode - 1) * 100)
      },
      q8: {
        baseline: {ttft_ms: $base_q8_ttft, decode_tok_s: $base_q8_decode, workspace_bytes: $base_q8_workspace},
        candidate: {ttft_ms: $candidate_q8_ttft, decode_tok_s: $candidate_q8_decode, workspace_bytes: $candidate_q8_workspace},
        ttft_speedup: ($base_q8_ttft / $candidate_q8_ttft),
        ttft_reduction_percent: ((1 - $candidate_q8_ttft / $base_q8_ttft) * 100),
        decode_change_percent: (($candidate_q8_decode / $base_q8_decode - 1) * 100)
      }
    },
    limits: [
      "This study measures one model, one GPU, and one 4096-token prompt.",
      "The baseline logs predate runtime receipt emission and are retained by digest.",
      "Quality evidence is linked by the candidate F16 runtime receipt. The Q8 runtime receipt reports quality as unverified."
    ]
  }' >"$output"

echo "$output"
