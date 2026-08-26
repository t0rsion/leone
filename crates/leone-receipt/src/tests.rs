use super::*;
use chrono::{TimeZone, Utc};
use std::fs;
use uuid::Uuid;

fn runtime_receipt() -> RuntimeReceipt {
    let model_bytes_total = 5_000_000_000;
    let bytes_per_token_total = 4_750_000_000;
    let bandwidth_gbs_assumed = 1008.0;
    let ceiling_tok_s = bandwidth_gbs_assumed * 1e9 / bytes_per_token_total as f64;
    let decode_tok_s = summarize_samples(&[90.0, 95.0, 100.0, 105.0, 110.0]).unwrap();
    let decode_median = decode_tok_s.median;
    let mut bytes_per_token_by_class = TensorClass::zero_map();
    bytes_per_token_by_class.insert(TensorClass::Other, bytes_per_token_total);
    let mut weights_resident_bytes_by_class = TensorClass::zero_map();
    weights_resident_bytes_by_class.insert(TensorClass::Other, model_bytes_total);

    RuntimeReceipt {
        schema_version: RUNTIME_SCHEMA_VERSION,
        receipt_id: Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").unwrap(),
        created_utc: Utc.with_ymd_and_hms(2026, 8, 21, 12, 34, 56).unwrap(),
        machine: Machine {
            hostname: "receipt-lab".to_owned(),
            gpu_name: "NVIDIA GeForce RTX 4090".to_owned(),
            gpu_vram_mib: 24_564,
            compute_cap: "8.9".to_owned(),
            driver: "580.82.07".to_owned(),
            cuda: "13.3".to_owned(),
            cpu_model: "Example CPU".to_owned(),
            ram_gib: 188,
            gpu_clocks_mhz: GpuClocksMhz {
                graphics: 2_520,
                memory: 10_501,
            },
            gpu_power_limit_w: 450.0,
        },
        workload: Workload {
            engine: Engine {
                name: "llama.cpp".to_owned(),
                git_commit: "0123456789abcdef".to_owned(),
                build_flags: vec!["GGML_CUDA=ON".to_owned()],
            },
            model_artifact: ModelArtifact {
                path: "models/Qwen3-8B-Q4_K_M.gguf".to_owned(),
                sha256: "a".repeat(64),
                file_bytes: model_bytes_total,
                format: "GGUF Q4_K_M".to_owned(),
            },
            context_tokens: 512,
            generated_tokens: 128,
            batch: 1,
            reps: 5,
        },
        results: RuntimeResults {
            roofline: Roofline {
                bandwidth_gbs_assumed,
                ceiling_tok_s,
                eta: decode_tok_s.median / ceiling_tok_s,
                denominator_definition: Some(ROOFLINE_DENOMINATOR_DEFINITION.to_owned()),
            },
            decode_tok_s,
            prefill_tok_s: Some(
                summarize_samples(&[8_000.0, 8_100.0, 8_200.0, 8_300.0, 8_400.0]).unwrap(),
            ),
            ttft_ms: Some(
                summarize_duration_samples_ms(&[250.0, 245.0, 240.0, 235.0, 230.0]).unwrap(),
            ),
            ttft_context_tokens: Some(2_048),
            prefill_method: Some(PrefillMethod::ChunkedCublasLtFp16),
            usable_bar: Some(UsableBar {
                ttft_2k_under_1s: true,
                pp512_ratio_vs_comparator: 0.64,
                bar_met: true,
            }),
            tokens_emitted: 640,
            bytes_per_token_by_class,
            bytes_per_token_total: Some(bytes_per_token_total),
            weights_resident_bytes_by_class: Some(weights_resident_bytes_by_class),
            model_bytes_total,
            roofline_measured_achievable: Some(Roofline {
                bandwidth_gbs_assumed: 930.0,
                ceiling_tok_s: 930.0 * 1e9 / bytes_per_token_total as f64,
                eta: decode_median / (930.0 * 1e9 / bytes_per_token_total as f64),
                denominator_definition: Some(ROOFLINE_DENOMINATOR_DEFINITION.to_owned()),
            }),
        },
        quality_ref: None,
        quality_summary: None,
        speculation: None,
        determinism: Some(DeterminismClaim::Reproduced {
            order: ReductionOrder::FixedOrder,
            sampler: SamplerRecord::Greedy,
            prompt_sha256: "c".repeat(64),
            transcript_sha256: "d".repeat(64),
            identical_reps: 5,
        }),
        notes: vec!["Roofline bandwidth is assumed.".to_owned()],
    }
}

fn clear_runtime_v5_fields(receipt: &mut RuntimeReceipt) {
    clear_runtime_v6_fields(receipt);
    receipt.results.ttft_ms = None;
    receipt.results.ttft_context_tokens = None;
    receipt.results.prefill_method = None;
    receipt.results.usable_bar = None;
}

fn clear_runtime_v6_fields(receipt: &mut RuntimeReceipt) {
    receipt.determinism = None;
    receipt.speculation = None;
}

fn quality_receipt() -> QualityReceipt {
    QualityReceipt {
        schema_version: QUALITY_SCHEMA_VERSION,
        receipt_id: Uuid::parse_str("6ba7b810-9dad-41d1-80b4-00c04fd430c8").unwrap(),
        created_utc: Utc.with_ymd_and_hms(2026, 8, 21, 13, 0, 0).unwrap(),
        corpus: Corpus {
            name: "fixed-corpus-v1".to_owned(),
            sha256: "b".repeat(64),
            n_prompts: 8,
            n_tokens_scored: 4_096,
        },
        oracle: Oracle {
            description: "llama.cpp F16 execution".to_owned(),
            artifact_sha256: "c".repeat(64),
            engine: EngineRef {
                name: "llama.cpp".to_owned(),
                git_commit: "0123456789abcdef".to_owned(),
            },
            dtype: "F16".to_owned(),
        },
        subject: Subject {
            model_artifact: ArtifactRef {
                sha256: "d".repeat(64),
                path: "models/Qwen3-8B-Q4_K_M.gguf".to_owned(),
            },
            logits_artifact: Some(ArtifactRef {
                sha256: "e".repeat(64),
                path: "receipts/raw/subject.f32".to_owned(),
            }),
            engine: EngineRef {
                name: "llama.cpp".to_owned(),
                git_commit: "0123456789abcdef".to_owned(),
            },
        },
        metrics: Some(Metrics {
            kld: KldMetric {
                mean: 0.01,
                p50: Some(0.005),
                p99: 0.05,
                max: Some(0.1),
                definition: KLD_DEFINITION.to_owned(),
            },
            top1_agreement: 0.98,
        }),
        batch_invariance: None,
        sample_count: 4_096,
    }
}

#[test]
fn runtime_round_trip_is_byte_identical() {
    let first = runtime_receipt().to_json().unwrap();
    let parsed = RuntimeReceipt::from_json(&first).unwrap();
    let second = parsed.to_json().unwrap();
    assert_eq!(first, second);
}

#[test]
fn quality_round_trip_is_byte_identical() {
    let first = quality_receipt().to_json().unwrap();
    let parsed = QualityReceipt::from_json(&first).unwrap();
    let second = parsed.to_json().unwrap();
    assert_eq!(first, second);
}

#[test]
fn load_rejects_unknown_schema_versions() {
    let mut runtime = runtime_receipt();
    runtime.schema_version = RUNTIME_SCHEMA_VERSION + 1;
    let bytes = serde_json::to_vec(&runtime).unwrap();
    assert!(RuntimeReceipt::from_json(&bytes).is_err());

    let mut quality = quality_receipt();
    quality.schema_version = QUALITY_SCHEMA_VERSION + 1;
    let bytes = serde_json::to_vec(&quality).unwrap();
    assert!(QualityReceipt::from_json(&bytes).is_err());
}

#[test]
fn quality_schema_v1_remains_readable() {
    let mut receipt = quality_receipt();
    receipt.schema_version = 1;
    receipt.subject.logits_artifact = None;
    receipt.metrics.as_mut().unwrap().kld.p50 = None;
    receipt.metrics.as_mut().unwrap().kld.max = None;
    let encoded = receipt.to_json().unwrap();
    assert_eq!(QualityReceipt::from_json(&encoded).unwrap(), receipt);
}

#[test]
fn runtime_schema_v1_remains_readable() {
    let mut receipt = runtime_receipt();
    receipt.schema_version = 1;
    clear_runtime_v5_fields(&mut receipt);
    receipt.results.bytes_per_token_total = None;
    receipt.results.weights_resident_bytes_by_class = None;
    receipt.results.roofline_measured_achievable = None;
    receipt.results.roofline.denominator_definition = None;
    receipt.results.roofline.ceiling_tok_s = receipt.results.roofline.bandwidth_gbs_assumed * 1e9
        / receipt.results.model_bytes_total as f64;
    receipt.results.roofline.eta =
        receipt.results.decode_tok_s.median / receipt.results.roofline.ceiling_tok_s;
    let encoded = receipt.to_json().unwrap();
    assert_eq!(RuntimeReceipt::from_json(&encoded).unwrap(), receipt);
}

#[test]
fn runtime_schema_v2_remains_readable() {
    let mut receipt = runtime_receipt();
    receipt.schema_version = 2;
    clear_runtime_v5_fields(&mut receipt);
    receipt.results.roofline.denominator_definition = None;
    receipt
        .results
        .roofline_measured_achievable
        .as_mut()
        .unwrap()
        .denominator_definition = None;
    let encoded = receipt.to_json().unwrap();
    assert_eq!(RuntimeReceipt::from_json(&encoded).unwrap(), receipt);
}

#[test]
fn runtime_schema_v3_remains_readable() {
    let mut receipt = runtime_receipt();
    receipt.schema_version = 3;
    clear_runtime_v5_fields(&mut receipt);
    let encoded = receipt.to_json().unwrap();
    assert_eq!(RuntimeReceipt::from_json(&encoded).unwrap(), receipt);
}

#[test]
fn runtime_schema_v4_remains_readable() {
    let mut receipt = runtime_receipt();
    receipt.schema_version = 4;
    clear_runtime_v5_fields(&mut receipt);
    let encoded = receipt.to_json().unwrap();
    assert_eq!(RuntimeReceipt::from_json(&encoded).unwrap(), receipt);
}

#[test]
fn runtime_schema_v4_rejects_v5_prefill_fields() {
    let mut receipt = runtime_receipt();
    receipt.schema_version = 4;
    assert!(receipt.validate().is_err());
}

#[test]
fn runtime_schema_v5_recomputes_usable_prefill_bar() {
    let mut receipt = runtime_receipt();
    receipt.results.usable_bar.as_mut().unwrap().bar_met = false;
    assert!(receipt.validate().is_err());

    let mut receipt = runtime_receipt();
    receipt.results.ttft_ms.as_mut().unwrap().reps.fill(1_001.0);
    receipt.results.ttft_ms =
        Some(summarize_duration_samples_ms(&receipt.results.ttft_ms.unwrap().reps).unwrap());
    assert!(receipt.validate().is_err());
}

#[test]
fn runtime_schema_v2_requires_honest_byte_fields() {
    let mut receipt = runtime_receipt();
    receipt.results.bytes_per_token_total = None;
    assert!(receipt.validate().is_err());

    let mut receipt = runtime_receipt();
    receipt
        .results
        .bytes_per_token_by_class
        .insert(TensorClass::Kv, 1);
    assert!(receipt.validate().is_err());

    let mut receipt = runtime_receipt();
    receipt.results.roofline_measured_achievable = None;
    assert!(receipt.validate().is_err());
}

#[test]
fn runtime_schema_v3_requires_the_fixed_denominator_definition() {
    let mut receipt = runtime_receipt();
    receipt.results.roofline.denominator_definition = None;
    assert!(receipt.validate().is_err());

    let mut receipt = runtime_receipt();
    receipt.results.roofline.denominator_definition = Some("different bytes".to_owned());
    assert!(receipt.validate().is_err());

    let mut receipt = runtime_receipt();
    receipt
        .results
        .roofline_measured_achievable
        .as_mut()
        .unwrap()
        .denominator_definition = None;
    assert!(receipt.validate().is_err());
}

#[test]
fn load_rejects_unknown_fields_at_each_level() {
    let mut value = serde_json::to_value(runtime_receipt()).unwrap();
    value["machine"]["gpu_serial"] = serde_json::json!("not-recorded");
    let error = RuntimeReceipt::from_json(&serde_json::to_vec(&value).unwrap()).unwrap_err();
    assert!(error.to_string().contains("unknown field `gpu_serial`"));
}

#[test]
fn map_keys_serialize_in_tensor_class_order() {
    let json = String::from_utf8(runtime_receipt().to_json().unwrap()).unwrap();
    let attn = json.find("\"attn\"").unwrap();
    let ffn = json.find("\"ffn\"").unwrap();
    let embed = json.find("\"embed\"").unwrap();
    let head = json.find("\"head\"").unwrap();
    let kv = json.find("\"kv\"").unwrap();
    let other = json.find("\"other\"").unwrap();
    assert!(attn < ffn && ffn < embed && embed < head && head < kv && kv < other);
}

#[test]
fn classifies_canonical_gguf_tensor_names() {
    let cases = [
        ("token_embd.weight", TensorClass::Embed),
        ("position_embd.weight", TensorClass::Embed),
        ("output.weight", TensorClass::Head),
        ("blk.0.attn_q.weight", TensorClass::Attn),
        ("blk.0.attn_kv_b.weight", TensorClass::Attn),
        ("blk.0.ffn_down.weight", TensorClass::Ffn),
        ("kv.layer.0", TensorClass::Kv),
        ("cache_k.layer.0", TensorClass::Kv),
        ("blk.0.ssm_a", TensorClass::Other),
    ];
    for (name, expected) in cases {
        assert_eq!(TensorClass::from_gguf_name(name), expected, "{name}");
    }
}

#[test]
fn runtime_without_quality_reference_prints_unverified() {
    let rendered = runtime_receipt().to_string();
    assert!(rendered.contains("quality: unverified"));
}

#[test]
fn linked_runtime_prints_its_quality_summary() {
    let mut receipt = runtime_receipt();
    receipt.quality_ref = Some(Uuid::parse_str("6ba7b810-9dad-41d1-80b4-00c04fd430c8").unwrap());
    receipt.quality_summary = Some(QualitySummary {
        kld_mean: 0.01,
        kld_p99: 0.05,
        top1_agreement: 0.98,
    });
    let rendered = receipt.to_string();
    assert!(rendered.contains("quality: KLD mean 0.010000, p99 0.050000, top1 0.980000"));
}

#[test]
fn validation_recomputes_ceiling_and_eta() {
    let mut receipt = runtime_receipt();
    receipt.results.roofline.ceiling_tok_s *= 1.000_000_002;
    assert!(receipt.validate().is_err());

    let mut receipt = runtime_receipt();
    receipt.results.roofline.eta *= 1.000_000_002;
    assert!(receipt.validate().is_err());
}

#[test]
fn validation_accepts_derived_values_within_tolerance() {
    let mut receipt = runtime_receipt();
    receipt.results.roofline.ceiling_tok_s *= 1.000_000_000_5;
    receipt.results.roofline.eta *= 1.000_000_000_5;
    receipt.validate().unwrap();
}

#[test]
fn sha256_helpers_match_known_digest() {
    let expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    assert_eq!(sha256_bytes(b"abc"), expected);

    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("data");
    fs::write(&path, b"abc").unwrap();
    assert_eq!(sha256_file(path).unwrap(), expected);
}

#[test]
fn storage_uses_versioned_names_and_appends_index_lines() {
    let directory = tempfile::tempdir().unwrap();
    let runtime_path = write_runtime_receipt(directory.path(), &runtime_receipt()).unwrap();
    let quality_path = write_quality_receipt(directory.path(), &quality_receipt()).unwrap();

    assert_eq!(
        runtime_path.file_name().unwrap(),
        "2026-08-21T12:34:56Z-runtime-550e8400.json"
    );
    assert_eq!(
        quality_path.file_name().unwrap(),
        "2026-08-21T13:00:00Z-quality-6ba7b810.json"
    );
    let index = fs::read_to_string(directory.path().join("INDEX.md")).unwrap();
    assert!(index.contains("| 2026-08-21 | runtime |"));
    assert!(index.contains("100.000 decode tok/s"));
    assert!(index.contains("0.010000 mean KLD nats"));
}

#[test]
fn quality_validation_enforces_kld_definition() {
    let mut receipt = quality_receipt();
    receipt.metrics.as_mut().unwrap().kld.definition = "a different definition".to_owned();
    assert!(receipt.validate().is_err());
}

#[test]
fn runtime_schema_v5_remains_readable() {
    let mut receipt = runtime_receipt();
    receipt.schema_version = 5;
    clear_runtime_v6_fields(&mut receipt);
    let encoded = receipt.to_json().unwrap();
    assert_eq!(RuntimeReceipt::from_json(&encoded).unwrap(), receipt);
}

#[test]
fn runtime_schema_v6_requires_a_determinism_record() {
    let mut receipt = runtime_receipt();
    receipt.determinism = None;
    assert!(receipt.validate().is_err());
}

#[test]
fn runtime_schema_v5_rejects_a_determinism_record() {
    let mut receipt = runtime_receipt();
    receipt.schema_version = 5;
    assert!(receipt.validate().is_err());
}

#[test]
fn determinism_rejects_a_bad_digest_or_zero_reps() {
    let mut receipt = runtime_receipt();
    receipt.determinism = Some(DeterminismClaim::Reproduced {
        order: ReductionOrder::FixedOrder,
        sampler: SamplerRecord::Greedy,
        prompt_sha256: "c".repeat(64),
        transcript_sha256: "not-a-digest".to_owned(),
        identical_reps: 5,
    });
    assert!(receipt.validate().is_err());

    let mut receipt = runtime_receipt();
    receipt.determinism = Some(DeterminismClaim::Reproduced {
        order: ReductionOrder::FixedOrder,
        sampler: SamplerRecord::Greedy,
        prompt_sha256: "c".repeat(64),
        transcript_sha256: "d".repeat(64),
        identical_reps: 0,
    });
    assert!(receipt.validate().is_err());
}

#[test]
fn determinism_not_measured_needs_a_reason() {
    let mut receipt = runtime_receipt();
    receipt.determinism = Some(DeterminismClaim::NotMeasured {
        reason: String::new(),
    });
    assert!(receipt.validate().is_err());

    let mut receipt = runtime_receipt();
    receipt.determinism = Some(DeterminismClaim::NotMeasured {
        reason: "llama-bench does not report a token stream".to_owned(),
    });
    assert!(receipt.validate().is_ok());
}

#[test]
fn speculation_counts_must_be_consistent() {
    let mut receipt = runtime_receipt();
    receipt.speculation = Some(SpeculationRecord {
        drafter: "suffix".to_owned(),
        rounds: 4,
        proposed: 12,
        accepted: 13,
        evaluations: 20,
        draft_width: 4,
        verify_passes: 0,
        verified_positions: 0,
        verify_duration_ms: 0.0,
        adaptive_plain_rounds: 0,
        adaptive_speculative_rounds: 0,
        adaptive_below_gate_rounds: 0,
        adaptive_regret_limited_rounds: 0,
        adaptive_suffix_proposals: 0,
        adaptive_recycling_proposals: 0,
        adaptive_controller_duration_ms: 0.0,
        adaptive_width_rounds: [0; 9],
        correctable_plain_rounds: 0,
        correctable_speculative_rounds: 0,
        correctable_below_gate_rounds: 0,
        correctable_regret_limited_rounds: 0,
        correctable_plan_rounds: [0; 4],
        correctable_width_rounds: [0; 9],
        correctable_controller_duration_ms: 0.0,
        correctable_overlap_sum: 0.0,
        correctable_overlap_proposals: 0,
    });
    assert!(receipt.validate().is_err());

    let mut receipt = runtime_receipt();
    receipt.speculation = Some(SpeculationRecord {
        drafter: "suffix".to_owned(),
        rounds: 4,
        proposed: 12,
        accepted: 9,
        evaluations: 20,
        draft_width: 4,
        verify_passes: 2,
        verified_positions: 10,
        verify_duration_ms: 12.5,
        adaptive_plain_rounds: 0,
        adaptive_speculative_rounds: 0,
        adaptive_below_gate_rounds: 0,
        adaptive_regret_limited_rounds: 0,
        adaptive_suffix_proposals: 0,
        adaptive_recycling_proposals: 0,
        adaptive_controller_duration_ms: 0.0,
        adaptive_width_rounds: [0; 9],
        correctable_plain_rounds: 0,
        correctable_speculative_rounds: 0,
        correctable_below_gate_rounds: 0,
        correctable_regret_limited_rounds: 0,
        correctable_plan_rounds: [0; 4],
        correctable_width_rounds: [0; 9],
        correctable_controller_duration_ms: 0.0,
        correctable_overlap_sum: 0.0,
        correctable_overlap_proposals: 0,
    });
    assert!(receipt.validate().is_ok());
}
