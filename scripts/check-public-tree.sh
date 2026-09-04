#!/usr/bin/env bash
set -euo pipefail

root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"

tracked=$(mktemp)
trap 'rm -f "$tracked"' EXIT HUP INT TERM
while IFS= read -r path; do
  [[ -e $path ]] && printf '%s\n' "$path"
done < <(git ls-files) >"$tracked"

forbidden_paths='(^|/)(AGENTS|CLAUDE|PLAN)\.md$|(^|/)\.claude/|^grok_writeup\.md$|^research/v[0-9]+\.[0-9]+-(intervention|prior-art)\.md$|\.(gguf|f16|f32|pem|key)$'
if grep -En "$forbidden_paths" "$tracked"; then
  echo "public tree contains a private-work, model, logit, or key file" >&2
  exit 1
fi

check_content() {
  local description=$1
  local pattern=$2
  local matches
  # The rule source contains the literals that it rejects.
  matches=$(git grep -nI -E "$pattern" -- . \
    ':(exclude)scripts/check-public-tree.sh' || true)
  if [[ -n $matches ]]; then
    printf '%s\n' "$matches" >&2
    echo "public tree contains $description" >&2
    exit 1
  fi
}

check_content "an absolute personal path" "(/home/[^/[:space:]\"]+|/Users/[^/[:space:]\"]+|[A-Za-z]:\\\\Users\\\\[^\\\\[:space:]\"]+)"
check_content "an email address" '[[:alnum:]._%+-]+@[[:alnum:].-]+\.[A-Za-z]{2,}'
check_content "a private key or credential token" '(BEGIN [A-Z ]*PRIVATE KEY|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{30,}|gh[pousr]_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,}|xox[baprs]-[A-Za-z0-9-]{10,})'
check_content "an internal agent record" '(Subagent id|Grok Build TUI|Produced by.*Grok)'

echo "public tree hygiene gate passed"
