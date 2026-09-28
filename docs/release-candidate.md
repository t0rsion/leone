# Check the release candidate

For runtime packages, follow the focused [release checks](release.md).
The optional complete-evidence workflow runs from a clean source tree:

```sh
scripts/check-release.sh
```

The command runs source hygiene, dependency, static, CPU, CUDA, documentation,
and archive checks. It extracts the Linux runtime and evidence archives and
checks their manifests. It checks the binary's build identity, linkage, and
execution. A second build must produce the same archive checksums.

The gate validates both quality comparisons, the frozen streaming study, and
the recorded Python client check. It repeats the client check against the
extracted binary. Install the [client environment](client-workflow.md) first.

The extracted binary must return typed HTTP errors for malformed JSON, unknown
models, unknown routes, and rejected admission. It must remain available after
each request. The installed binary must load its packaged plan from an unrelated
directory.

The gate audits source and extracted archives for private-work files, credentials,
personal paths, and unnecessary release references. It rejects models, full
logits, runtime search paths, and unresolved libraries.

The command needs an idle NVIDIA GPU and both gated quantized models under
`models/`. It does not tag, push, install outside a temporary directory, or
publish anything.
