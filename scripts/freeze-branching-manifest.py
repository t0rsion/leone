#!/usr/bin/env python3
"""Write a frozen branching manifest from a service calibration receipt and a quality record.

Every service threshold and tokenizer pin derives from the calibration receipt
by a fixed rule. The script types no bound. The quality record is bound by path
and SHA-256 under the policy the template declares. The script derives no
quality tolerance and does not recompute KLD, top-1 agreement, or the needle
argmax: `study-branching-service.sh check-quality` does. The output path must
not exist.
"""

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path

RULES = {
    "ttft_p95_ms": ("upper_10_percent", "<="),
    "inter_token_latency_p95_ms": ("upper_10_percent", "<="),
    "fork_latency_p95_ms": ("upper_10_percent", "<="),
    "cancel_latency_p95_ms": ("upper_10_percent", "<="),
    "physical_memory_peak_bytes": ("upper_10_percent", "<="),
    "history_reuse_min_tokens": ("lower_10_percent", ">="),
    "backpressure_observed": ("boolean_true", "=="),
    "schedule_overlap_observed": ("boolean_true", "=="),
}
FACTORS = {"identity": 1.0, "upper_10_percent": 1.1, "lower_10_percent": 0.9}
TOKENIZER_FIELDS = ("tokenizer_metadata_sha256", "special_tokens_policy_sha256", "vocab_size")
TOKENIZER_IDENTITY_FIELDS = {
    "producer_source_commit": "source_commit",
    "producer_executable_sha256": "executable_sha256",
    "producer_model_sha256": "gguf_sha256",
    "producer_loaded_library_sha256": "loaded_library_sha256",
    "template_config_sha256": "template_config_sha256",
    "template_bytes_sha256": "template_bytes_sha256",
}


class FreezeError(Exception):
    """Raised when a receipt cannot support a frozen manifest."""


def load_harness():
    path = Path(__file__).resolve().parent / "study-branching-service.py"
    spec = importlib.util.spec_from_file_location("study_branching_service", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise FreezeError(f"cannot read {path.name}: {error}") from error


def threshold_item(harness, engine, role, scenario, metric):
    rule, operator = RULES[metric]
    return {
        "id": f"{engine['id']}-{role}-{metric}", "engine": engine["id"], "role": role,
        "scenario": scenario, "metric": metric, "model": engine["model"],
        "backend": engine["backend"], "device": engine["device"],
        "unit": harness.THRESHOLD_METRIC_UNITS[metric], "operator": operator, "rule": rule,
    }


def decide(harness, item, observed, receipt_sha256):
    """Fix one bound from its observation by the declared rule."""

    rule = item.pop("rule")
    derived = True if rule == "boolean_true" else observed * FACTORS[rule]
    item["value"] = derived
    decision = {"receipt_sha256": receipt_sha256, "observation": observed, "rule": rule, "derived_value": derived}
    decision["decision_sha256"] = harness._threshold_decision_digest(item, decision)
    item["calibration"] = decision
    return item


def derive_thresholds(harness, template, receipt, receipt_sha256):
    grouped = harness._records_by_engine_role(receipt)
    minimum = receipt["budgets"]["minimum_samples_for_quantiles"]
    thresholds, missing = [], []
    for engine in template["engines"]:
        unsupported = harness._unsupported_metrics(harness.engine_kind(engine))
        for role, scenario in sorted(harness.REQUIRED_THRESHOLD_SCOPES):
            for metric in sorted(harness.REQUIRED_THRESHOLD_METRICS[(role, scenario)] - unsupported):
                item = threshold_item(harness, engine, role, scenario, metric)
                observed = harness._threshold_observation(item, grouped.get((engine["id"], role), []), receipt["runs"], minimum)
                if observed is None or (RULES[metric][0] == "boolean_true" and observed is not True):
                    missing.append(f"{engine['id']}/{role}/{metric}")
                else:
                    thresholds.append(decide(harness, item, observed, receipt_sha256))
    if missing:
        raise FreezeError("the calibration receipt does not support: " + ", ".join(missing))
    return thresholds


def load_quality_record(harness, root, template, arguments, model_sha256):
    """Load one comparison record and check its structure. Return it with its path and SHA-256."""

    reference = {"path": arguments.quality_record, "sha256": harness.sha256_file(root / arguments.quality_record)}
    record, directory, problems = harness._load_quality_record(reference, root, "evaluation")
    if record:
        problems += harness._quality_record_errors(record, directory, harness._quality_backend(template), model_sha256, "evaluation")
    if problems:
        raise FreezeError("; ".join(problems))
    return reference


def oracle_pins(receipt, engine_id):
    """Yield the tokenizer oracle record of every branch of one engine."""

    for run in receipt["runs"]:
        for branch in run.get("branches", []) if run["engine"] == engine_id else []:
            yield ((branch.get("history_reuse") or {}).get("tokenization") or {}).get("oracle") or {}


def derive_tokenizer(receipt, engine_id):
    """Read the tokenizer pins from one engine's calibration runs. Every run must agree."""

    pins = list(oracle_pins(receipt, engine_id))
    values = {field: {json.dumps(pin.get(field)) for pin in pins} for field in TOKENIZER_FIELDS}
    if not pins or any(len(items) != 1 or "null" in items for items in values.values()):
        raise FreezeError("the calibration runs do not agree on one tokenizer identity")
    return {field: json.loads(next(iter(items))) for field, items in values.items()}


def derive_tokenizer_identity(receipt, engine_id):
    """Read independent producer identity pins from every branch of one engine."""

    pins = list(oracle_pins(receipt, engine_id))
    values = {
        declaration_field: {json.dumps(pin.get(oracle_field)) for pin in pins}
        for declaration_field, oracle_field in TOKENIZER_IDENTITY_FIELDS.items()
    }
    if not pins or any(len(items) != 1 or "null" in items for items in values.values()):
        raise FreezeError("the calibration runs do not agree on one tokenizer producer identity")
    return {field: json.loads(next(iter(items))) for field, items in values.items()}


def build_manifest(harness, root, template, arguments):
    receipt_path = root / arguments.calibration_receipt
    errors = harness.validate_receipt(receipt_path, root, offline=arguments.offline)
    if errors:
        raise FreezeError("calibration receipt is invalid: " + "; ".join(errors[:5]))
    receipt, receipt_sha256 = read_json(receipt_path), harness.sha256_file(receipt_path)
    model_sha256 = receipt["artifacts"]["model"]["sha256"]
    quality_record = load_quality_record(harness, root, template, arguments, model_sha256)
    manifest = copy.deepcopy(template)
    manifest.update({
        "phase": "frozen", "freeze_status": "frozen",
        "name": template["name"].removesuffix("-template"),
        "workload_id": template["workload_id"].replace("-pending", ""),
        "purpose": "Frozen workload. Thresholds derive from the calibration receipt by fixed rules. Quality is the declared policy applied to one comparison record.",
    })
    manifest["calibration_receipt"] = {"path": arguments.calibration_receipt, "sha256": receipt_sha256}
    manifest["evaluation"] = {
        "thresholds": derive_thresholds(harness, template, receipt, receipt_sha256), "quality": "observed",
        "quality_record": quality_record, "quality_policy": copy.deepcopy(template["evaluation"].get("quality_policy")),
        "calibration_receipt_sha256": receipt_sha256,
    }
    for engine in manifest["engines"]:
        declaration = engine.get("history_tokenization")
        if not isinstance(declaration, dict):
            raise FreezeError(f"engine {engine.get('id')} has no history tokenizer declaration")
        declaration.update(derive_tokenizer(receipt, engine["id"]))
        declaration.update(derive_tokenizer_identity(receipt, engine["id"]))
    return manifest, model_sha256


def freeze(arguments):
    root = arguments.root.resolve()
    harness = load_harness()
    manifest, model_sha256 = build_manifest(harness, root, read_json(root / arguments.template), arguments)
    errors = harness.validate_manifest(manifest)
    errors += harness._frozen_manifest_reference(manifest, root, model_sha256)[0]
    if errors:
        raise FreezeError("the derived manifest does not validate: " + "; ".join(errors[:5]))
    with arguments.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps(manifest, indent=2) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--template", required=True, help="pending frozen manifest, relative to --root")
    parser.add_argument("--calibration-receipt", required=True, help="service study receipt of the calibration run, relative to --root")
    parser.add_argument("--quality-record", required=True, help="canonical v2 comparison record, relative to --root")
    parser.add_argument("--offline", action="store_true", help="check the calibration receipt by recorded digests")
    parser.add_argument("--output", type=Path, required=True)
    try:
        freeze(parser.parse_args(argv))
    except (FreezeError, FileExistsError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
