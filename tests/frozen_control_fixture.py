import copy
import json
import pathlib
import sys
import tempfile

sys.path.insert(0, "tests")
import test_study_branching_service as T
H = T.HARNESS


def token_sha256(values):
    return H.sha256_bytes(b"".join(value.to_bytes(4, "little") for value in values))


def receipt(claim):
    return {"claim": claim, "public_key_ed25519": "0" * 64, "signature_ed25519": "0" * 128}


def put(root, path, data):
    p = root / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return {"path": path, "sha256": H.sha256_file(p)}


def record(request_id, service_id, role, start, status="completed", parent_content=None):
    end = start + 100
    r = {
        "role": role, "request_id": request_id, "service_request_id": service_id,
        "status": status, "history_complete": True,
        "request_start_ns": start, "first_content_ns": start + 10, "request_end_ns": end,
        "usage": {"prompt_tokens": 512, "completion_tokens": 2},
        "metrics": {
            "request_start_ns": start, "content_receive_ns": [start + 10, start + 20],
            "token_boundary_receive_ns": [start + 10, start + 20],
            "token_boundary_indexes": [0, 1], "token_boundaries_verified": True,
        },
    }
    if parent_content is not None:
        r["_content_text"] = parent_content
        r["content_sha256"] = H.sha256_bytes(parent_content.encode())
    if status == "completed":
        r["token_boundary_complete"] = True
        r["finish_reason"] = "length"
    if role == "parent":
        unmeasured = {"status": "unavailable", "reason": "nonstream_parent_unmeasured"}
        r.update(metrics=unmeasured, first_content_ns=None, token_boundary_complete=False)
    return r


def add_history(parent, branch, branch_prompt, manifest, engine, workload_epoch):
    parent_messages = H._prompt_messages(manifest["prompts"]["parent"])
    branch_entry = next(item for item in manifest["prompts"]["branches"] if item["id"] == branch_prompt)
    prefix = H._history_messages(parent_messages, parent)
    request = prefix + H._prompt_messages(branch_entry)
    body = H.request_body(manifest, request, branch["request_id"], "leone_session" if engine == "leone" else None)
    if engine == "leone":
        body.update({"leone_fork_session": parent["request_id"], "leone_session": branch["request_id"]})
    else:
        body.update({"llama_fork_parent": parent["request_id"], "llama_fork_session": branch["request_id"]})
    prompt_ids = list(range(512))
    generated_ids = [512, 513, 514]
    evaluated_ids = [*prompt_ids, *generated_ids][:-1]
    request_ids = [*evaluated_ids, 600, 601]
    oracle_engine = "llama.cpp"
    source = LEONE_COMMIT if engine == "leone" else "d" * 40
    oracle = {
        "engine": oracle_engine, "source_commit": source,
        "executable_sha256": "e" * 64, "loaded_library_sha256": "f" * 64,
        "gguf_sha256": "1" * 64, "template_config_sha256": "2" * 64,
        "template_bytes_sha256": "3" * 64, "special_tokens_policy_sha256": "3" * 64,
        "vocab_size": 1000, "apply_template_request_sha256": "5" * 64,
        "apply_template_response_sha256": "6" * 64, "tokenize_request_sha256": "7" * 64,
        "tokenize_response_sha256": "8" * 64, "tokenizer_metadata_sha256": "1" * 64,
    }
    parent_receipt = receipt({
        "request_sha256": parent["request_body_sha256"],
        "prompt_tokens_sha256": token_sha256(prompt_ids),
        "response_tokens_sha256": token_sha256(generated_ids),
        "transcript_sha256": token_sha256([*prompt_ids, *generated_ids]),
        "prompt_tokens": len(prompt_ids), "generated_tokens": len(generated_ids),
        "finish_reason": "length", "cancelled": False,
        "session": {"session_id": parent["service_request_id"]},
    })
    parent["response_receipt"] = parent_receipt
    parent["_response_receipt_sha256"] = H.sha256_bytes(H.canonical_json(parent_receipt))
    branch_receipt = receipt({
        "request_sha256": H.sha256_bytes(H.canonical_json(body)),
        "prompt_tokens_sha256": token_sha256(request_ids),
        "prompt_tokens": len(request_ids), "generated_tokens": 1,
        "finish_reason": "length", "cancelled": False,
        "session": {"session_id": branch["service_request_id"], "cached_tokens": len(evaluated_ids),
                    "reused_tokens": len(evaluated_ids)},
    })
    token = {
        "schema_version": H.HISTORY_TOKEN_SCHEMA, "status": "observed", "scope": H.HISTORY_TOKEN_SCOPE,
        "parent_service_request_id": parent["service_request_id"],
        "service_request_id": branch["service_request_id"],
        "process_instance_id": "process-1000", "workload_epoch": workload_epoch,
        "parent_receipt_sha256": parent["_response_receipt_sha256"],
        "branch_receipt_sha256": H.sha256_bytes(H.canonical_json(branch_receipt)),
        "parent_request_sha256": parent["request_body_sha256"],
        "request_sha256": H.sha256_bytes(H.canonical_json(body)),
        "canonical_prefix_sha256": H.sha256_bytes(H.canonical_json(prefix)),
        "canonical_request_sha256": H.sha256_bytes(H.canonical_json(request)),
        "parent_prompt_token_ids": prompt_ids,
        "parent_generated_token_ids": generated_ids,
        "parent_evaluated_token_ids": evaluated_ids,
        "request_token_ids": request_ids,
        "parent_evaluated_token_count": len(evaluated_ids),
        "expected_reused_token_count": len(evaluated_ids),
        "observed_reused_token_count": len(evaluated_ids),
        "oracle": oracle,
    }
    parent["usage"] = {"prompt_tokens": len(prompt_ids), "completion_tokens": len(generated_ids)}
    branch["response_receipt"] = branch_receipt
    branch["_response_receipt_sha256"] = H.sha256_bytes(H.canonical_json(branch_receipt))
    branch["_request_bytes_hex"] = H.canonical_json(body).hex()
    material = {"branch_prompt_id": branch_prompt, "prefix_messages": prefix,
                "request_messages": request, "request_body": body}
    history = {
        "status": "observed", "prefix_verified": True, "reuse_verified": True,
        "prefix_message_count": 2, "request_message_count": 3,
        "prefix_sha256": H.sha256_bytes(H.canonical_json(prefix)),
        "request_sha256": H.sha256_bytes(H.canonical_json(request)),
        "parent_content_sha256": parent["content_sha256"],
        "request_body_sha256": H.sha256_bytes(H.canonical_json(body)),
        "reuse_count": {"status": "observed", "values": [len(evaluated_ids)]},
        "required_reuse_tokens": len(evaluated_ids), "request_material": material, "tokenization": token,
    }
    branch["branch_id"] = branch["request_id"]
    branch["branch_mode"] = "fork"
    branch["request_body_sha256"] = history["request_body_sha256"]
    branch["history_reuse"] = history


GEN_SUFFIX = "<|im_start|>assistant\n"
LLAMA_THINKING = {"mode": "explicit_legacy_chatml", "reasoning_format": "none", "chat_template_kwargs": {},
                  "generation_prompt_suffix": GEN_SUFFIX}
LLAMA_POLICY = {"prompt": {"add_special": False, "parse_special": True},
                **H._thinking_policy_fields(LLAMA_THINKING)}
LLAMA_POLICY_SHA = H.sha256_bytes(H.canonical_json(LLAMA_POLICY))
CLOCK_DOMAIN = "caller_monotonic_ns"
LEONE_COMMIT = "c" * 40
LLAMA_MODEL = "model.gguf"
LLAMA_TEMPLATE_SHA = "3" * 64
LLAMA_PREFIX = ""
PARENT_RENDERED = "parent-rendered" + GEN_SUFFIX
BRANCH_RENDERED = "branch-rendered" + GEN_SUFFIX
PARENT_PROMPT_IDS = list(range(512))
PARENT_GENERATED_IDS = [512, 513, 514]
EVALUATED_IDS = [*PARENT_PROMPT_IDS, *PARENT_GENERATED_IDS][:-1]
T0 = 1000
LLAMA_SOURCE = "llama.cpp:git:" + "d" * 40


def engine_source(engine_id):
    """The build source id each engine reports, as the harness derives it."""

    return LLAMA_SOURCE if engine_id == "llama_cpp" else engine_id + ":source"


def llama_process_identity(identity, declared):
    """Extend a base identity with the bindings a local llama.cpp process records."""

    return {**identity, "workload_epoch": identity["process_instance_id"], "argv_sha256": "4" * 64,
            "argv_roles_sha256": "5" * 64, "flags": {"slot_save_path": True, "parallel": declared["slot_count"], "alias": LLAMA_MODEL},
            "template_sha256": LLAMA_TEMPLATE_SHA}


def hex_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode().hex()


def slot_exchange(action, slot, filename, tokens, size, start):
    fields = ("n_saved", "n_written") if action == "save" else ("n_restored", "n_read")
    reply = {"id_slot": slot, "filename": filename, fields[0]: tokens, fields[1]: size}
    return {"action": action, "slot": slot, "http_status": 200, "request_hex": hex_json({"filename": filename}),
            "response_hex": hex_json(reply), "request_start_ns": start, "request_end_ns": start + 10}


def llama_slot_copy(parent_id, branch_slots):
    """Retain the sequential save and restores between the parent and the timed barrier."""

    filename = parent_id + ".slot"
    plan = {"parent_slot": 0, "filename": filename, "cleanup": {"status": "not_owned"},
            "save": slot_exchange("save", 0, filename, len(EVALUATED_IDS), 4096, 200), "restore": {}}
    for index, slot in enumerate(branch_slots):
        plan["restore"][str(slot)] = slot_exchange(
            "restore", slot, filename, len(EVALUATED_IDS), 4096, 220 + 20 * index
        )
    return plan


def llama_raw_events(response, start=0):
    return [{"event": event, "received_ns": start + 10 * (index + 1)} for index, event in enumerate(response)]


def finalize_llama_stream(record, events, data):
    """Derive a streamed llama.cpp record from its retained wire bytes, as the harness does."""

    record["_raw_events"] = llama_raw_events(events, record["request_start_ns"])
    pairs = [(row["event"], row["received_ns"]) for row in record["_raw_events"]]
    content, times, usage, _ = H._content_and_timestamps(pairs)
    record.update(
        _request_bytes_hex=data.hex(), request_body_sha256=H.sha256_bytes(data), usage=usage,
        finish_reason=H._finish_reason([event for event, _ in pairs]),
        content_sha256=H.sha256_bytes(content.encode()), first_content_ns=times[0],
        metrics=H.latency_metrics(record["request_start_ns"], times, False), token_boundary_complete=False,
    )


def llama_stream_events(tokens, final_reason="length"):
    """One content chunk per token, a finish chunk, and a usage chunk."""

    events = [{"model": LLAMA_MODEL, "choices": [{"index": 0, "delta": {"content": "x"}, "finish_reason": None}]}
              for _ in range(tokens)]
    events.append({"model": LLAMA_MODEL, "choices": [{"index": 0, "delta": {}, "finish_reason": final_reason}]})
    events.append({"model": LLAMA_MODEL, "choices": [],
                   "usage": {"prompt_tokens": 40, "completion_tokens": tokens,
                             "prompt_tokens_details": {"cached_tokens": 0}}})
    return events


def add_llama_history(parent, branch, branch_prompt, manifest, engine, slot):
    """Retain llama.cpp raw requests and responses for one cache-history branch."""

    parent_messages = H._prompt_messages(manifest["prompts"]["parent"])
    branch_entry = next(item for item in manifest["prompts"]["branches"] if item["id"] == branch_prompt)
    prefix = H._history_messages(parent_messages, parent)
    request = prefix + H._prompt_messages(branch_entry)
    fields = {"cache_prompt": True}
    body = H._engine_request_body(
        engine, H.request_body(manifest, request, branch["request_id"], None), slot, proof=True
    )
    data = H.canonical_json({**body, **fields})
    verbose_prompt = BRANCH_RENDERED
    events = [
        {"model": LLAMA_MODEL, "choices": [{"index": 0, "delta": {"content": "x"}, "finish_reason": None}]},
        {"model": LLAMA_MODEL, "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
         "__verbose": {"prompt": verbose_prompt, "id_slot": slot, "stop_type": "limit", "truncated": False}},
        {"model": LLAMA_MODEL, "choices": [],
         "usage": {"prompt_tokens": len(EVALUATED_IDS) + 2, "completion_tokens": 1,
                   "prompt_tokens_details": {"cached_tokens": len(EVALUATED_IDS)}}},
    ]
    finalize_llama_stream(branch, events, data)
    branch["branch_id"] = branch["request_id"]
    branch["branch_mode"] = "cached_history"
    parent["usage"] = {"prompt_tokens": len(PARENT_PROMPT_IDS), "completion_tokens": len(PARENT_GENERATED_IDS)}
    material = {"branch_prompt_id": branch_prompt, "prefix_messages": prefix, "request_messages": request,
                "request_body": {**body, **fields}}
    branch["history_reuse"] = {
        "status": "observed", "prefix_verified": True, "reuse_verified": True,
        "prefix_message_count": 2, "request_message_count": 3,
        "prefix_sha256": H.sha256_bytes(H.canonical_json(prefix)),
        "request_sha256": H.sha256_bytes(H.canonical_json(request)),
        "parent_content_sha256": parent["content_sha256"], "request_body_sha256": branch["request_body_sha256"],
        "reuse_count": {"status": "observed", "values": [len(EVALUATED_IDS)]},
        "required_reuse_tokens": len(EVALUATED_IDS), "request_material": material,
    }


def llama_parent_events(parent_slot):
    return llama_raw_events([{
        "model": LLAMA_MODEL,
        "choices": [{"index": 0, "finish_reason": "length", "message": {"role": "assistant", "content": "answer"}}],
        "usage": {"prompt_tokens": len(PARENT_PROMPT_IDS), "completion_tokens": len(PARENT_GENERATED_IDS)},
        "__verbose": {
            "tokens": PARENT_GENERATED_IDS, "prompt": PARENT_RENDERED, "id_slot": parent_slot,
            "tokens_evaluated": len(PARENT_PROMPT_IDS), "tokens_predicted": len(PARENT_GENERATED_IDS),
            "tokens_cached": len(EVALUATED_IDS), "stop_type": "limit", "truncated": False,
            "generation_settings": {"speculative.types": "none"},
        },
    }])


def tokenizer_exchanges(prepared, parent_request, branch_request):
    """Retain the apply-template and tokenize exchanges the producer would make."""

    module = H._load_history_producer_module()

    def exchange(request, response):
        raw_request = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode()
        raw_response = json.dumps(response, ensure_ascii=False, separators=(",", ":")).encode()
        return {"request_sha256": H.sha256_bytes(raw_request), "response_sha256": H.sha256_bytes(raw_response),
                "request_hex": raw_request.hex(), "response_hex": raw_response.hex()}

    def prompt(request, rendered, ids):
        return {
            "apply_template": exchange(module._apply_body(prepared, request), {"prompt": rendered}),
            "tokenize": exchange(module._tokenize_body(prepared, rendered, "prompt"), {"tokens": ids}),
        }

    return {
        "parent": prompt(parent_request, PARENT_RENDERED, PARENT_PROMPT_IDS),
        "branch": prompt(branch_request, BRANCH_RENDERED, [*EVALUATED_IDS, 600, 601]),
        "generated": {"tokenize": exchange(
            module._tokenize_body(prepared, "answer", "generated"), {"tokens": PARENT_GENERATED_IDS}
        )},
    }


def attach_llama_tokenization(run, manifest, identity):
    """Build each branch's evidence with the producer's own deterministic path."""

    module = H._load_history_producer_module()
    engine = next(item for item in manifest["engines"] if item["id"] == "llama_cpp")
    context = H._history_run_engine(engine, run)
    parent = run["parent"]
    parent_id = parent["service_request_id"]
    for branch in run["branches"]:
        oracle = {
            "engine": "llama.cpp", "source_commit": "d" * 40,
            "executable_sha256": identity["executable_sha256"], "loaded_library_sha256": "f" * 64,
            "gguf_sha256": identity["model_sha256"], "tokenizer_metadata_sha256": "1" * 64,
            "tokenizer_metadata_hash_scheme": "tokenizer_config_json_bytes",
            "template_config_sha256": "2" * 64, "template_config_hash_scheme": "tokenizer_config_json_bytes",
            "template_bytes_sha256": LLAMA_TEMPLATE_SHA, "special_tokens_policy": LLAMA_POLICY,
            "special_tokens_policy_sha256": LLAMA_POLICY_SHA, "vocab_size": 1000,
        }
        prepared = module._prepared_from_oracle(context, oracle)
        oracle["apply_template_tokenize"] = tokenizer_exchanges(
            prepared, json.loads(bytes.fromhex(parent["_request_bytes_hex"])),
            json.loads(bytes.fromhex(branch["_request_bytes_hex"])),
        )
        evidence = module.recompute_llama_history(
            context, oracle,
            H._record_request_bytes(parent), H._history_raw_events(parent, context),
            H._record_request_bytes(branch), H._history_raw_events(branch, context, parent_id),
        )
        assert evidence["status"] == "observed", evidence
        branch["history_reuse"]["tokenization"] = {field: evidence[field] for field in H.HISTORY_TOKEN_FIELDS}


def snapshots(run, expected, epoch, process, source_id):
    rows = [{"request_id": item["service_request_id"], "numeric_request_id": i + 1,
             "outcome": H.SERVICE_OUTCOMES[item["status"]], "reclaimed": True, "terminal_at_ns": 10 * (i + 1),
             "delivery_status": "queued"} for i, item in enumerate(expected)]
    outcomes = {}
    for row in rows:
        outcomes[row["outcome"]] = outcomes.get(row["outcome"], 0) + 1
    loss = {field: 0 for field in H.MEMORY_COLLECTION_LOSS_FIELDS - {"overflowed"}}
    loss.update({"terminal_outputs": 0, "overflowed": False})
    history = {"values": rows, "capacity": 16, "sample_count": len(rows), "dropped_count": 0}
    empty = {"values": [], "capacity": 16, "sample_count": 0, "dropped_count": 0}
    common = {
        "workload_epoch": epoch, "process_instance_id": f"process-{process}", "source_id": source_id,
        "request_count": len(rows),
        "outcomes": outcomes, "requests": history,
        "memory_topology": "cpu_parent", "physical_tracker_ledger": "parent_memory_tracker_root",
        "physical_peak_definition": H.PHYSICAL_PEAK_DEFINITION,
        "physical_tracker_peak_bytes": {"value": 100, "status": "observed", "reason": None},
        "collection_errors": 0, "collection_losses": loss, "counter_overflowed": False,
        "degraded": False, "memory_topology_conflict": False,
        "physical_bytes": {"capacity": 16, "sample_count": 1, "dropped_count": 0,
                            "values": [{"at_ns": 1, "bytes_by_class": {"weights": 100}}]},
        "clock_domain": CLOCK_DOMAIN,
        "slow_client_intervals": {"values": [{
            "numeric_request_id": 6, "request_id": run["probes"][2]["service_request_id"],
            "reason": "write_blocked", "start_ns": 40, "end_ns": 80,
        }], "capacity": 16, "sample_count": 1, "dropped_count": 0},
        "cancellation": {"values": [{
            "request_id": run["probes"][1]["service_request_id"], "reason": "cancelled",
            "latency_ns": {"value": 1_000_000, "status": "observed", "reason": None},
        }], "capacity": 16, "sample_count": 1, "dropped_count": 0},
    }
    start = copy.deepcopy(common)
    start.update({"request_count": 0, "outcomes": {}, "requests": empty})
    start["slow_client_intervals"] = {"values": [], "capacity": 16, "sample_count": 0, "dropped_count": 0}
    start["cancellation"] = {"values": [], "capacity": 16, "sample_count": 0, "dropped_count": 0}
    return {"start": {"status": "observed", "body": start}, "end": {"status": "observed", "body": common}}


def parent_request_body(manifest, declared, parent_id):
    """Build the parent request with the controls its engine declares."""

    messages = H._prompt_messages(manifest["prompts"]["parent"])
    if declared["id"] == "llama_cpp":
        body = H.request_body(manifest, messages, parent_id, None, stream=False)
        return H._engine_request_body(declared, body, 0, proof=True)
    return H.request_body(manifest, messages, parent_id, "leone_session")


def make_branch(parent, parent_id, manifest, declared, epoch, index, branch_prompt):
    branch_id = parent_id + "-" + branch_prompt
    branch = record(branch_id, branch_id + "-service", "branch", T0 + 10 + index * 2)
    if declared["id"] == "llama_cpp":
        add_llama_history(parent, branch, branch_prompt, manifest, declared, 1 + index)
    else:
        add_history(parent, branch, branch_prompt, manifest, declared["id"], epoch)
    if declared["id"] != "llama_cpp":
        branch["service_metrics"] = {"fork_latency_ns": {"status": "observed", "value": 1_000_000}}
    return branch


def leone_probes(parent_id, epoch):
    """Leone probes carry the service telemetry Leone declares."""

    probes = [
        record(parent_id + "-new-prompt", parent_id + "-new-prompt-service", "new_prompt", T0 + 20),
        record(parent_id + "-cancel-probe", parent_id + "-cancel-probe-service", "cancel", T0 + 30, "cancelled"),
        record(parent_id + "-slow-reader-probe", parent_id + "-slow-reader-probe-service", "slow_reader", T0 + 40),
    ]
    probes[1]["cancel_ack"] = {"status": "observed", "request_id": probes[1]["service_request_id"]}
    probes[1]["service_metrics"] = {"cancel_latency_ns": {"status": "observed", "value": 1_000_000}}
    probes[2]["backpressure"] = {
        "status": "observed", "reason": "write_blocked", "request_id": probes[2]["service_request_id"],
        "interval_start_ns": 40, "interval_end_ns": 80,
        "clock_domain": CLOCK_DOMAIN, "workload_epoch": epoch,
    }
    return probes


def llama_probes(manifest, declared, parent_id):
    """llama.cpp probes retain client wire bytes and no Leone telemetry."""

    specs = (("new_prompt", "-new-prompt", T0 + 20, 3), ("cancel", "-cancel-probe", T0 + 30, 4),
             ("slow_reader", "-slow-reader-probe", T0 + 40, 5))
    probes = []
    for role, suffix, start, slot in specs:
        cancelled = role == "cancel"
        probe = record(parent_id + suffix, parent_id + suffix + "-service", role, start,
                       "cancel_acknowledgement_unavailable" if cancelled else "completed")
        body = H._engine_request_body(
            declared, H.request_body(manifest, H._prompt_messages(manifest["schedule"][role]), probe["request_id"], None), slot
        )
        finalize_llama_stream(probe, llama_stream_events(2), H.canonical_json(body))
        if cancelled:
            probe["cancel_ack"] = {"status": "unavailable", "reason": "cancel_acknowledgement_endpoint_missing"}
        probes.append(probe)
    return probes


def leone_telemetry(run, all_records, probes, epoch, source_id):
    """Attach the Leone service snapshots and sibling trace."""

    run["service_metrics"] = snapshots(run, all_records, epoch, 1000, source_id)
    probes[2]["backpressure"] = H._snapshot_backpressure(
        run["service_metrics"]["end"], probes[2]["service_request_id"]
    )
    trace = {
        "schema_version": "leone.service-trace.v1", "clock_domain": CLOCK_DOMAIN,
        "source_id": source_id, "workload_epoch": epoch, "process_instance_id": "process-1000",
        "dropped_events": 0, "dropped_memory_samples": 0,
        "events": [
            {"kind": "prefill_chunk", "request_id": 6, "at_ns": 50, "token_budget": 1, "processed_tokens": 1,
             "ready": False, "resident_request_ids": [4]},
            {"kind": "resident_decode_progress", "request_id": 4, "at_ns": 60, "token_budget": 1,
             "emitted_tokens": 1, "during_prefill_request_id": 6, "during_prefill_request_ids": [6],
             "resident_request_ids": [4]},
        ], "memory_samples": [], "logical_reserved_kv_bytes": 0,
    }
    run["service_trace"] = {"status": "observed", "body": trace}
    run["sibling_progress"] = H._sibling_progress(
        all_records, probes[2]["backpressure"], run["service_trace"], run["service_metrics"]
    )


def llama_telemetry(run, all_records, probes):
    """llama.cpp exposes none of the Leone snapshots. The run records the harness's own typed absence."""

    missing = {"status": "unavailable", "reason": "metrics_endpoint_missing"}
    run["service_metrics"] = {"start": dict(missing), "end": dict(missing)}
    run["service_trace"] = None
    probes[2]["backpressure"] = H._backpressure_evidence(probes[2])
    run["sibling_progress"] = H._sibling_progress(all_records, probes[2]["backpressure"], None, run["service_metrics"]["end"])


def make_parent(manifest, declared, parent_id):
    parent = record(parent_id, parent_id + "-service", "parent", 0, parent_content="answer")
    parent["content_sha256"] = H.sha256_bytes(b"answer")
    parent_bytes = H.canonical_json(parent_request_body(manifest, declared, parent_id))
    parent["request_body_sha256"] = H.sha256_bytes(parent_bytes)
    parent["_request_bytes_hex"] = parent_bytes.hex()
    return parent


def run_identity(declared, identity):
    """The start and end identity a run records. llama.cpp adds its process bindings."""

    identity = identity or {"source_id": engine_source(declared["id"]), "executable_sha256": "0" * 64,
                            "model_sha256": "1" * 64, "process_start_ns": 1000,
                            "process_instance_id": "process-1000"}
    if declared["id"] == "llama_cpp":
        return llama_process_identity(identity, declared)
    return {"process_start_clock": "unix_epoch_ns", **identity}


def make_run(manifest, engine, rep, identity=None):
    parent_id = f"branch-parent-{engine}-{rep}"
    epoch = f"epoch-{engine}-{rep}"
    declared = next(item for item in manifest["engines"] if item["id"] == engine)
    is_llama = engine == "llama_cpp"
    parent = make_parent(manifest, declared, parent_id)
    branches = [make_branch(parent, parent_id, manifest, declared, epoch, index, prompt)
                for index, prompt in enumerate(("memory", "scheduling"))]
    probes = llama_probes(manifest, declared, parent_id) if is_llama else leone_probes(parent_id, epoch)
    all_records = [parent, *branches, *probes]
    kind = H.engine_kind(declared)
    run = {"engine": engine, "engine_kind": kind, "comparison_support": H.capability_record(kind),
           "repetition": rep, "parent": parent, "branches": branches, "probes": probes,
           "slot_copy": llama_slot_copy(parent_id, [1, 2, 4]) if is_llama else None,
           "overlap": H._schedule_overlap(all_records)}
    if is_llama:
        parent["_raw_events"] = llama_parent_events(0)
        llama_telemetry(run, all_records, probes)
    else:
        leone_telemetry(run, all_records, probes, epoch, (identity or {}).get("source_id") or engine_source(engine))
    identity = run_identity(declared, identity)
    run["running_identity"] = {"start": {"status": "observed", "identity": identity},
                                "end": {"status": "observed", "identity": identity}}
    if is_llama:
        attach_llama_tokenization(run, manifest, identity)
    return run


def _build_calibration(root):
    """Build a valid calibration manifest and receipt for the control fixture."""

    cal = T.calibration_manifest()
    for engine in cal["engines"]:
        engine["executable_path"] = f"bin/{engine['id']}"
        engine["model_artifact"] = "models/model.gguf"
        declaration = engine["history_tokenization"]
        declaration["producer_model_artifact"] = "models/model.gguf"
        declaration["producer_executable_path"] = "bin/llama-server"
        if "spawn" in engine:
            argv = engine["spawn"]["argv"]
            argv[argv.index("--alias") + 1] = LLAMA_MODEL
    model = put(root, "models/model.gguf", b"subject")
    put(root, "external/PINNED", (T.ROOT / "external/PINNED").read_bytes())
    artifacts = {
        "model": model,
        "plan": put(root, "plans/plan.json", b"plan"),
    }
    for engine in cal["engines"]:
        put(root, engine["executable_path"], engine["id"].encode())
    put(root, "bin/llama-server", b"independent tokenizer")
    cal["artifacts"] = artifacts
    cal_path = root / "calibration.json"
    cal_path.write_text(json.dumps(cal, sort_keys=True))
    cal_ref = {"path": "calibration.json", "canonical_sha256": H.sha256_bytes(H.canonical_json(cal))}
    runs = [
        make_run(cal, engine["id"], rep)
        for rep in range(cal["budgets"]["repetitions"])
        for engine in cal["engines"]
    ]
    receipt = {
        "schema_version": H.SCHEMA_VERSION, "phase": "calibration", "freeze_status": "calibration",
        "workload_id": cal["workload_id"], "manifest": cal_ref, "prompts": H.prompt_provenance(cal),
        "schedule": cal["schedule"], "budgets": cal["budgets"],
        "artifacts": {name: {**item, "status": "observed"} for name, item in artifacts.items()},
        "engines": [], "runs": runs,
    }
    for engine in cal["engines"]:
        digest = H.sha256_file(root / engine["executable_path"])
        row = H._engine_declaration_row(engine)
        row["provenance"] = {
            "status": "observed", "executable_path": engine["executable_path"],
            "executable_sha256": digest,
            "build_info": {"status": "observed", "source_id_status": "observed", "source_id": engine_source(engine["id"])},
        }
        if engine["quality_producer"] == "peer_adapter":
            row["provenance"]["linked_libraries"] = served_libraries()
        receipt["engines"].append(row)
    minimum = cal["budgets"]["minimum_samples_for_quantiles"]
    receipt["summary"] = H.outcome_summary(H._all_records(receipt), minimum, cal["budgets"]["max_history"])
    receipt["summary_by_engine_role"] = H.summary_by_engine_role(receipt, minimum, cal["budgets"]["max_history"])
    receipt["end_to_end"] = H.receipt_end_to_end(receipt)
    receipt["evaluation"] = {"calibration_evidence": {
        "status": "observed", "workload_id": cal["workload_id"],
        "prompt_hashes": [item["sha256"] for item in receipt["prompts"]],
        "summary_sha256": H.sha256_bytes(H.canonical_json(receipt["summary"])),
    }}
    receipt_path = root / "calibration-receipt.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True))
    return cal, model, receipt, receipt_path


def _producer_identity(engine_id):
    """Return the producer pins for one fixture engine."""

    if engine_id == "leone":
        return {
            "producer_source_commit": "c" * 40,
            "producer_executable_sha256": "e" * 64,
            "producer_model_sha256": "1" * 64,
            "producer_loaded_library_sha256": "f" * 64,
        }
    return {
        "producer_source_commit": "d" * 40,
        "producer_executable_sha256": H.sha256_bytes(b"llama_cpp"),
        "producer_model_sha256": H.sha256_bytes(b"subject"),
        "producer_loaded_library_sha256": "f" * 64,
    }


def _build_frozen_manifest(cal, receipt_path):
    """Build the frozen workload declaration used by the accepted control."""

    frozen = copy.deepcopy(cal)
    frozen.update({"phase": "frozen", "freeze_status": "frozen", "workload_id": "fixture-frozen-v1"})
    frozen["budgets"].update({"repetitions": 12, "minimum_samples_for_quantiles": 12})
    prompts = [frozen["prompts"]["parent"], *frozen["prompts"]["branches"], *frozen["schedule"].values()]
    for prompt in prompts:
        if isinstance(prompt, dict) and isinstance(prompt.get("messages"), list):
            prompt["messages"] = [
                {"role": message["role"], "content": "Frozen fixture " + prompt["id"] + " " + "z" * 1200}
                for message in prompt["messages"]
            ]
    for engine in frozen["engines"]:
        declaration = engine["history_tokenization"]
        declaration.update({
            "tokenizer_sha256": "1" * 64, "chat_template_sha256": "2" * 64,
            "special_tokens_policy_sha256": "3" * 64, "vocab_size": 1000,
            "tokenizer_metadata_sha256": "1" * 64,
            "template_config_sha256": "2" * 64,
            "template_bytes_sha256": LLAMA_TEMPLATE_SHA if engine["id"] == "llama_cpp" else "3" * 64,
            **_producer_identity(engine["id"]),
        })
        if engine["id"] == "llama_cpp":
            declaration["special_tokens_policy_sha256"] = LLAMA_POLICY_SHA
    frozen["calibration_receipt"] = {"path": "calibration-receipt.json", "sha256": H.sha256_file(receipt_path)}
    names = (
        "disjoint_workload", "required_thresholds", "quality_receipt_required",
        "calibration_receipt_validated", "thresholds_evaluated", "quality_artifact_binding",
        "running_identity_binding", "required_outcomes", "token_history_complete",
        "schedule_overlap", "backpressure_evidence", "sibling_progress",
    )
    frozen["freeze_criteria"] = {name: True for name in names}
    return frozen


LIBRARY_FAMILIES = ("libllama", "libggml", "libggml-base", "libggml-cpu", "libggml-cuda")


def libraries(suffix, extra=()):
    """Library records the way ldd resolves them: one hash per family, plus server-only extras."""

    return [{"name": f"{name}.so.{suffix}", "sha256": H.sha256_bytes(name.encode())} for name in (*LIBRARY_FAMILIES, *extra)]


def served_libraries():
    """The linkage the harness records for the llama.cpp server."""

    return {"status": "observed", "loaded_library_status": H.LOADED_LIBRARY_STATUS,
            "libraries": libraries("0", ("libllama-server-impl",))}


def _publish_quality_record(root, leone_sha256):
    """Publish a canonical CUDA quality record, built by the quality-stage builder, under root.

    Only the fields the study joins to its serving rows are rebound: the native
    Leone executable hash and the adapter linkage. The statistics executable
    keeps the builder hash, so it differs from the served native executable.
    """

    case = T.QC.CudaQualityTests("test_cuda_comparison_accepts_samples_only")
    case.setUp()
    record = json.loads(case.manifest.read_text())
    record["producers"]["leone"]["body"]["executable"]["sha256"] = leone_sha256
    record["producers"]["llama_cpp"]["body"]["executable"]["linked_libraries"] = libraries("0.21.0")
    directory = root / "quality"
    directory.mkdir(parents=True)
    for entry in record["quality"].values():
        sidecar = directory / entry["path"]
        sidecar.write_text(json.dumps(entry["receipt"]))
        entry["sha256"] = H.sha256_file(sidecar)
    path = directory / "record.json"
    path.write_text(json.dumps(record))
    case.doCleanups()
    return {"path": "quality/record.json", "sha256": H.sha256_file(path)}


def _attach_quality_evaluation(frozen, receipt_path, evaluation_ref):
    """Attach the one evaluation record under the declared policy."""

    frozen["evaluation"] = {
        "quality": "observed", "quality_record": evaluation_ref, "quality_policy": copy.deepcopy(H.QUALITY_POLICY),
        "calibration_receipt_sha256": H.sha256_file(receipt_path),
    }


def _build_threshold_item(engine, role, scenario, metric, grouped, receipt, receipt_path):
    """Build one frozen threshold with its calibration decision."""

    item = {"id": f"{engine['id']}-{role}-{metric}", "engine": engine["id"], "role": role,
            "scenario": scenario, "metric": metric, "model": engine["model"],
            "backend": engine["backend"], "device": engine["device"], "unit": H.THRESHOLD_METRIC_UNITS[metric]}
    observed = H._threshold_observation(item, grouped.get((engine["id"], role), []), receipt["runs"], 2)
    rule = "boolean_true" if isinstance(observed, bool) else "identity"
    operator = "==" if isinstance(observed, bool) else ">=" if metric == "history_reuse_min_tokens" else "<="
    item.update({"operator": operator, "value": observed})
    decision = {"receipt_sha256": H.sha256_file(receipt_path), "observation": observed,
                "rule": rule, "derived_value": observed}
    decision["decision_sha256"] = H._threshold_decision_digest(item, decision)
    item["calibration"] = decision
    return item


def _build_thresholds(frozen, receipt, receipt_path):
    """Build every required frozen threshold from calibration observations."""

    grouped = H._records_by_engine_role(receipt)
    return [
        _build_threshold_item(engine, role, scenario, metric, grouped, receipt, receipt_path)
        for engine in frozen["engines"]
        for role, scenario in sorted(H.REQUIRED_THRESHOLD_SCOPES)
        for metric in sorted(
            H.REQUIRED_THRESHOLD_METRICS[(role, scenario)] - H._unsupported_metrics(H.engine_kind(engine))
        )
    ]


def _build_frozen_receipt(root, frozen, thresholds, model):
    """Build and write the accepted frozen receipt."""

    fruns = [
        make_run(frozen, engine["id"], rep, {
            "source_id": "leone:" + LEONE_COMMIT if engine["id"] == "leone" else LLAMA_SOURCE,
            "executable_sha256": H.sha256_file(root / engine["executable_path"]),
            "model_sha256": model["sha256"], "process_start_ns": 1000,
            "process_instance_id": "process-1000",
        })
        for rep in range(12)
        for engine in frozen["engines"]
    ]
    receipt = {
        "schema_version": H.SCHEMA_VERSION, "phase": "frozen", "freeze_status": "frozen",
        "workload_id": frozen["workload_id"],
        "manifest": {"path": "frozen.json", "canonical_sha256": H.sha256_bytes(H.canonical_json(frozen))},
        "prompts": H.prompt_provenance(frozen), "budgets": frozen["budgets"],
        "artifacts": {name: {**item, "status": "observed"} for name, item in frozen["artifacts"].items()},
        "engines": [], "runs": fruns,
    }
    for engine in frozen["engines"]:
        digest = H.sha256_file(root / engine["executable_path"])
        source = "leone:" + LEONE_COMMIT if engine["id"] == "leone" else LLAMA_SOURCE
        row = H._engine_declaration_row(engine)
        row["provenance"] = {"status": "observed", "executable_path": engine["executable_path"],
                              "executable_sha256": digest,
                              "build_info": {"status": "observed", "source_id_status": "observed", "source_id": source}}
        if engine["quality_producer"] == "peer_adapter":
            row["provenance"]["linked_libraries"] = served_libraries()
        row["running_identities"] = [run["running_identity"] for run in fruns if run["engine"] == engine["id"]]
        row["running_identity"] = row["running_identities"][0]
        receipt["engines"].append(row)
    receipt["summary"] = H.outcome_summary(H._all_records(receipt), 12, frozen["budgets"]["max_history"])
    receipt["summary_by_engine_role"] = H.summary_by_engine_role(receipt, 12, frozen["budgets"]["max_history"])
    receipt["end_to_end"] = H.receipt_end_to_end(receipt)
    receipt["evaluation"] = {"thresholds": thresholds, "threshold_results": H.evaluate_thresholds(receipt, frozen),
                             "quality_binding": H._quality_binding_declaration(frozen)}
    path = root / "frozen-receipt.json"
    path.write_text(json.dumps(receipt, sort_keys=True))
    return path, receipt


def build_control(root):
    """Build the accepted control in `root` and return its manifests, receipts, and paths."""

    cal, model, calibration_receipt, calibration_path = _build_calibration(root)
    frozen = _build_frozen_manifest(cal, calibration_path)
    leone_sha256 = H.sha256_file(root / "bin/leone")
    _attach_quality_evaluation(frozen, calibration_path, _publish_quality_record(root, leone_sha256))
    thresholds = _build_thresholds(frozen, calibration_receipt, calibration_path)
    frozen["evaluation"]["thresholds"] = thresholds
    frozen_path = root / "frozen.json"
    frozen_path.write_text(json.dumps(frozen, sort_keys=True))
    frozen_receipt_path, frozen_receipt = _build_frozen_receipt(root, frozen, thresholds, model)
    return {
        "root": root, "calibration": cal, "calibration_receipt_path": calibration_path, "frozen": frozen,
        "frozen_path": frozen_path, "receipt_path": frozen_receipt_path,
        "receipt": frozen_receipt, "calibration_receipt": calibration_receipt,
    }


def rewrite_receipt(control, receipt):
    """Write a mutated frozen receipt over the accepted one and return the validator errors."""

    control["receipt_path"].write_text(json.dumps(receipt, sort_keys=True))
    return H.validate_receipt(control["receipt_path"], control["root"])


def main():
    with tempfile.TemporaryDirectory(dir="/dev/shm") as td:
        root = pathlib.Path(td)
        control = build_control(root)
        print("cal manifest", H.validate_manifest(control["calibration"]))
        print("cal receipt", H.validate_receipt(control["calibration_receipt_path"], root))
        print("frozen manifest", H.validate_manifest(control["frozen"]))
        print("quality", H._quality_reference_errors(control["frozen"], root, H.sha256_file(root / "models/model.gguf")))
        print("frozen receipt", H.validate_receipt(control["receipt_path"], root))


if __name__ == "__main__":
    main()
