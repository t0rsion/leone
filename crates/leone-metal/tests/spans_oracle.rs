mod oracle_support;

use half::f16;
use leone::{
    AttentionShape, Backend, BufferLayout, CpuBackend, KvReadSpan, KvReadView, KvWriteSpan,
    MemoryBudget, MemoryClass, RopePairing, RopeShape, VectorShape,
};

use oracle_support::{assert_close, metal_backend, read_f16, read_f32, seeded_values, upload_f32};

#[test]
fn segmented_kv_append_and_attention_match_cpu_oracle() {
    let Some(mut metal) = metal_backend() else {
        return;
    };
    let mut cpu = CpuBackend::new();
    let shape = AttentionShape::new(4, 2, 2, 8).expect("attention shape");
    let projected = shape
        .projected_kv_elements()
        .expect("projected KV elements");
    let keys = seeded_values(5 * projected, 0x5350_414e);
    let values = seeded_values(5 * projected, 0x5641_4c55);

    let (cpu_key_first, cpu_value_first) = append_segment(
        &mut cpu,
        shape,
        &keys[..2 * projected],
        &values[..2 * projected],
        0,
        2,
        3,
    );
    let (cpu_key_second, cpu_value_second) = append_segment(
        &mut cpu,
        shape,
        &keys[2 * projected..],
        &values[2 * projected..],
        2,
        3,
        4,
    );
    let (metal_key_first, metal_value_first) = append_segment(
        &mut metal,
        shape,
        &keys[..2 * projected],
        &values[..2 * projected],
        0,
        2,
        3,
    );
    let (metal_key_second, metal_value_second) = append_segment(
        &mut metal,
        shape,
        &keys[2 * projected..],
        &values[2 * projected..],
        2,
        3,
        4,
    );

    let cpu_spans = [
        KvReadSpan::new(&cpu_key_first, &cpu_value_first, 0, 2, 3).expect("CPU first span"),
        KvReadSpan::new(&cpu_key_second, &cpu_value_second, 2, 3, 4).expect("CPU second span"),
    ];
    let metal_spans = [
        KvReadSpan::new(&metal_key_first, &metal_value_first, 0, 2, 3).expect("Metal first span"),
        KvReadSpan::new(&metal_key_second, &metal_value_second, 2, 3, 4)
            .expect("Metal second span"),
    ];
    let cpu_cache = KvReadView::new(&cpu_spans).expect("CPU KV view");
    let metal_cache = KvReadView::new(&metal_spans).expect("Metal KV view");

    let prefill_tokens = 3;
    let prefill_query_values = seeded_values(
        prefill_tokens * shape.query_elements().unwrap(),
        0x5155_4552,
    );
    let cpu_prefill_query = upload_f32(&mut cpu, &prefill_query_values);
    let metal_prefill_query = upload_f32(&mut metal, &prefill_query_values);
    let output_layout = BufferLayout::f32(prefill_query_values.len()).expect("output layout");
    let mut cpu_prefill = cpu.allocate(output_layout).expect("CPU prefill output");
    let mut metal_prefill = metal.allocate(output_layout).expect("Metal prefill output");
    cpu.attention_prefill_spans(
        &cpu_prefill_query,
        cpu_cache,
        &mut cpu_prefill,
        shape,
        2,
        prefill_tokens,
    )
    .expect("CPU segmented prefill attention");
    metal
        .attention_prefill_spans(
            &metal_prefill_query,
            metal_cache,
            &mut metal_prefill,
            shape,
            2,
            prefill_tokens,
        )
        .expect("Metal segmented prefill attention");
    let cpu_prefill_values = read_f32(&mut cpu, &cpu_prefill, prefill_query_values.len());
    let metal_prefill_values = read_f32(&mut metal, &metal_prefill, prefill_query_values.len());
    assert_close(
        "segmented prefill attention",
        &metal_prefill_values,
        &cpu_prefill_values,
        5e-3,
        5e-3,
    );

    let decode_query_values = &prefill_query_values[2 * shape.query_elements().unwrap()..];
    let cpu_decode_query = upload_f32(&mut cpu, decode_query_values);
    let metal_decode_query = upload_f32(&mut metal, decode_query_values);
    let decode_layout = BufferLayout::f32(shape.query_elements().unwrap()).expect("decode layout");
    let mut cpu_decode = cpu.allocate(decode_layout).expect("CPU decode output");
    let mut metal_decode = metal.allocate(decode_layout).expect("Metal decode output");
    cpu.attention_decode_spans(
        &cpu_decode_query,
        cpu_cache,
        &mut cpu_decode,
        shape,
        leone::Position::Host(4),
    )
    .expect("CPU segmented decode attention");
    metal
        .attention_decode_spans(
            &metal_decode_query,
            metal_cache,
            &mut metal_decode,
            shape,
            leone::Position::Host(4),
        )
        .expect("Metal segmented decode attention");
    let cpu_decode_values = read_f32(&mut cpu, &cpu_decode, shape.query_elements().unwrap());
    let metal_decode_values = read_f32(&mut metal, &metal_decode, shape.query_elements().unwrap());
    assert_close(
        "segmented decode attention",
        &metal_decode_values,
        &cpu_decode_values,
        5e-3,
        5e-3,
    );
}

#[test]
fn four_direct_spans_and_five_span_gather_match_cpu() {
    let Some(mut metal) = metal_backend() else {
        return;
    };
    let mut cpu = CpuBackend::new();
    let shape = AttentionShape::new(1, 1, 2, 5).expect("attention shape");
    let metal_caches = (0..4)
        .map(|index| {
            let values = [index as f32 + 0.25, index as f32 + 0.5];
            append_segment(&mut metal, shape, &values, &values, index, 1, 1)
        })
        .collect::<Vec<_>>();
    let cpu_caches = (0..4)
        .map(|index| {
            let values = [index as f32 + 0.25, index as f32 + 0.5];
            append_segment(&mut cpu, shape, &values, &values, index, 1, 1)
        })
        .collect::<Vec<_>>();
    let metal_spans = [
        KvReadSpan::new(&metal_caches[0].0, &metal_caches[0].1, 0, 1, 1).unwrap(),
        KvReadSpan::new(&metal_caches[1].0, &metal_caches[1].1, 1, 1, 1).unwrap(),
        KvReadSpan::new(&metal_caches[2].0, &metal_caches[2].1, 2, 1, 1).unwrap(),
        KvReadSpan::new(&metal_caches[3].0, &metal_caches[3].1, 3, 1, 1).unwrap(),
    ];
    let cpu_spans = [
        KvReadSpan::new(&cpu_caches[0].0, &cpu_caches[0].1, 0, 1, 1).unwrap(),
        KvReadSpan::new(&cpu_caches[1].0, &cpu_caches[1].1, 1, 1, 1).unwrap(),
        KvReadSpan::new(&cpu_caches[2].0, &cpu_caches[2].1, 2, 1, 1).unwrap(),
        KvReadSpan::new(&cpu_caches[3].0, &cpu_caches[3].1, 3, 1, 1).unwrap(),
    ];
    let query_values = [0.75_f32, -0.25];
    let metal_query = upload_f32(&mut metal, &query_values);
    let cpu_query = upload_f32(&mut cpu, &query_values);
    let mut metal_output = metal.allocate(BufferLayout::f32(2).unwrap()).unwrap();
    let mut cpu_output = cpu.allocate(BufferLayout::f32(2).unwrap()).unwrap();
    metal
        .attention_decode_spans(
            &metal_query,
            KvReadView::new(&metal_spans).unwrap(),
            &mut metal_output,
            shape,
            leone::Position::Host(3),
        )
        .unwrap();
    cpu.attention_decode_spans(
        &cpu_query,
        KvReadView::new(&cpu_spans).unwrap(),
        &mut cpu_output,
        shape,
        leone::Position::Host(3),
    )
    .unwrap();
    assert_close(
        "four-span direct decode",
        &read_f32(&mut metal, &metal_output, 2),
        &read_f32(&mut cpu, &cpu_output, 2),
        5e-3,
        5e-3,
    );

    let over_limit_shape = AttentionShape::new(1, 1, 2, 6).unwrap();
    let over_limit_caches =
        five_gather_segments(&mut metal, over_limit_shape, 1, 0x4b45_5931, 0x5641_4c31);
    let cpu_over_limit_caches =
        five_gather_segments(&mut cpu, over_limit_shape, 1, 0x4b45_5931, 0x5641_4c31);
    let over_limit_spans = five_span_view(&over_limit_caches, 1);
    let cpu_over_limit_spans = five_span_view(&cpu_over_limit_caches, 1);
    let mut over_limit_output = metal
        .upload(BufferLayout::f32(2).unwrap(), &f32_bytes(&[42.0, 43.0]))
        .unwrap();
    let mut cpu_over_limit_output = cpu
        .upload(BufferLayout::f32(2).unwrap(), &f32_bytes(&[42.0, 43.0]))
        .unwrap();
    metal
        .attention_decode_spans(
            &metal_query,
            KvReadView::new(&over_limit_spans).unwrap(),
            &mut over_limit_output,
            over_limit_shape,
            leone::Position::Host(4),
        )
        .expect("five-span GPU gather decode");
    cpu.attention_decode_spans(
        &cpu_query,
        KvReadView::new(&cpu_over_limit_spans).unwrap(),
        &mut cpu_over_limit_output,
        over_limit_shape,
        leone::Position::Host(4),
    )
    .expect("five-span CPU gather decode");
    assert_close(
        "five-span gathered decode",
        &read_f32(&mut metal, &over_limit_output, 2),
        &read_f32(&mut cpu, &cpu_over_limit_output, 2),
        5e-3,
        5e-3,
    );

    let prefill_query_values = [0.4_f32, -0.2, 0.9, 0.1];
    let metal_prefill_query = upload_f32(&mut metal, &prefill_query_values);
    let cpu_prefill_query = upload_f32(&mut cpu, &prefill_query_values);
    let prefill_layout = BufferLayout::f32(prefill_query_values.len()).unwrap();
    let mut metal_prefill = metal.allocate(prefill_layout).unwrap();
    let mut cpu_prefill = cpu.allocate(prefill_layout).unwrap();
    metal
        .attention_prefill_spans(
            &metal_prefill_query,
            KvReadView::new(&over_limit_spans).unwrap(),
            &mut metal_prefill,
            over_limit_shape,
            3,
            2,
        )
        .expect("five-span GPU gather prefill");
    cpu.attention_prefill_spans(
        &cpu_prefill_query,
        KvReadView::new(&cpu_over_limit_spans).unwrap(),
        &mut cpu_prefill,
        over_limit_shape,
        3,
        2,
    )
    .expect("five-span CPU gather prefill");
    assert_close(
        "five-span gathered prefill",
        &read_f32(&mut metal, &metal_prefill, prefill_query_values.len()),
        &read_f32(&mut cpu, &cpu_prefill, prefill_query_values.len()),
        5e-3,
        5e-3,
    );

    let small_shape = AttentionShape::new(1, 1, 1, 5).unwrap();
    let metal_small_caches =
        five_gather_segments(&mut metal, small_shape, 2, 0x4b45_5932, 0x5641_4c32);
    let cpu_small_caches = five_gather_segments(&mut cpu, small_shape, 2, 0x4b45_5932, 0x5641_4c32);
    let metal_small_spans = five_span_view(&metal_small_caches, 2);
    let cpu_small_spans = five_span_view(&cpu_small_caches, 2);
    let small_query_values = [0.75_f32];
    let metal_small_query = upload_f32(&mut metal, &small_query_values);
    let cpu_small_query = upload_f32(&mut cpu, &small_query_values);
    let mut metal_small_output = metal.allocate(BufferLayout::f32(1).unwrap()).unwrap();
    let mut cpu_small_output = cpu.allocate(BufferLayout::f32(1).unwrap()).unwrap();
    metal
        .attention_decode_spans(
            &metal_small_query,
            KvReadView::new(&metal_small_spans).unwrap(),
            &mut metal_small_output,
            small_shape,
            leone::Position::Host(4),
        )
        .expect("smaller five-span GPU gather after larger gather");
    cpu.attention_decode_spans(
        &cpu_small_query,
        KvReadView::new(&cpu_small_spans).unwrap(),
        &mut cpu_small_output,
        small_shape,
        leone::Position::Host(4),
    )
    .expect("smaller five-span CPU gather after larger gather");
    assert_close(
        "smaller five-span gathered decode",
        &read_f32(&mut metal, &metal_small_output, 1),
        &read_f32(&mut cpu, &cpu_small_output, 1),
        5e-3,
        5e-3,
    );
}

#[test]
fn gathered_scratch_stops_at_causal_end_for_mapped_future_tail() {
    let Some(mut metal) = metal_backend() else {
        return;
    };
    let mut cpu = CpuBackend::new();
    let shape = AttentionShape::new(1, 1, 2, 12).expect("attention shape");
    let metal_caches = five_full_gather_segments(&mut metal, shape, 2, 0x4b45_5933, 0x5641_4c33);
    let cpu_caches = five_full_gather_segments(&mut cpu, shape, 2, 0x4b45_5933, 0x5641_4c33);
    let metal_spans = five_full_span_view(&metal_caches, 2);
    let cpu_spans = five_full_span_view(&cpu_caches, 2);
    let metal_cache = KvReadView::new(&metal_spans).expect("Metal mapped cache");
    let cpu_cache = KvReadView::new(&cpu_spans).expect("CPU mapped cache");
    assert_eq!(metal_cache.mapped_tokens(), 10);

    let query_values = [0.75_f32, -0.25];
    let metal_query = upload_f32(&mut metal, &query_values);
    let cpu_query = upload_f32(&mut cpu, &query_values);
    let mut metal_output = metal.allocate(BufferLayout::f32(2).unwrap()).unwrap();
    let mut cpu_output = cpu.allocate(BufferLayout::f32(2).unwrap()).unwrap();
    let before_decode = metal.memory_accounting();
    metal
        .attention_decode_spans(
            &metal_query,
            metal_cache,
            &mut metal_output,
            shape,
            leone::Position::Host(4),
        )
        .expect("Metal gathered decode");
    cpu.attention_decode_spans(
        &cpu_query,
        cpu_cache,
        &mut cpu_output,
        shape,
        leone::Position::Host(4),
    )
    .expect("CPU gathered decode");
    assert_close(
        "causal gathered decode",
        &read_f32(&mut metal, &metal_output, 2),
        &read_f32(&mut cpu, &cpu_output, 2),
        5e-3,
        5e-3,
    );
    let after_decode = metal.memory_accounting();
    let expected_scratch = u64::try_from(
        BufferLayout::f16(shape.n_head_kv() * shape.head_dim() * 5)
            .unwrap()
            .bytes(),
    )
    .expect("causal scratch size fits accounting")
    .checked_mul(2)
    .expect("key and value scratch size");
    let scratch_before = before_decode.class(MemoryClass::BackendScratch).live_bytes;
    assert_eq!(
        after_decode
            .class(MemoryClass::BackendScratch)
            .live_bytes
            .checked_sub(scratch_before)
            .expect("gather scratch grows"),
        expected_scratch,
        "gather scratch includes only positions through decode position",
    );

    metal.synchronize().expect("release decode scratch");
    let prefill_query_values = [0.4_f32, -0.2, 0.9, 0.1];
    let metal_prefill_query = upload_f32(&mut metal, &prefill_query_values);
    let cpu_prefill_query = upload_f32(&mut cpu, &prefill_query_values);
    let mut metal_prefill = metal.allocate(BufferLayout::f32(4).unwrap()).unwrap();
    let mut cpu_prefill = cpu.allocate(BufferLayout::f32(4).unwrap()).unwrap();
    metal
        .attention_prefill_spans(
            &metal_prefill_query,
            KvReadView::new(&metal_spans).expect("Metal mapped cache"),
            &mut metal_prefill,
            shape,
            3,
            2,
        )
        .expect("Metal gathered prefill");
    cpu.attention_prefill_spans(
        &cpu_prefill_query,
        KvReadView::new(&cpu_spans).expect("CPU mapped cache"),
        &mut cpu_prefill,
        shape,
        3,
        2,
    )
    .expect("CPU gathered prefill");
    assert_close(
        "causal gathered prefill",
        &read_f32(&mut metal, &metal_prefill, 4),
        &read_f32(&mut cpu, &cpu_prefill, 4),
        5e-3,
        5e-3,
    );
    let after_prefill = metal.memory_accounting();
    assert_eq!(
        after_prefill.class(MemoryClass::BackendScratch).live_bytes,
        expected_scratch,
        "prefill gather scratch includes only its causal end",
    );
}

#[test]
fn tiled_span_attention_boundaries_match_f64_oracle() {
    let Some(mut metal) = metal_backend() else {
        return;
    };
    let mut cpu = CpuBackend::new();
    let shape = AttentionShape::new(4, 2, 256, 8).expect("tiled span shape");
    let projected = shape
        .projected_kv_elements()
        .expect("tiled projected KV elements");
    let keys = adversarial_span_values(6, projected, true);
    let values = adversarial_span_values(6, projected, false);
    let decode_query = adversarial_span_query(shape.query_elements().unwrap(), 0x4445_434f);
    let prefill_query = adversarial_span_query(3 * shape.query_elements().unwrap(), 0x5052_4546);
    let direct_specs = direct_span_specs(&keys, &values, projected);
    let gathered_specs = gathered_span_specs(&keys, &values, projected);

    let direct_cpu = run_span_case(
        &mut cpu,
        shape,
        &direct_specs,
        &decode_query,
        &prefill_query,
    );
    let direct_metal = run_span_case(
        &mut metal,
        shape,
        &direct_specs,
        &decode_query,
        &prefill_query,
    );
    let expected_direct_decode = span_attention_oracle(&decode_query, &keys, &values, shape, 5, 1);
    let expected_direct_prefill =
        span_attention_oracle(&prefill_query, &keys, &values, shape, 2, 3);
    assert_span_results(
        "direct four-span",
        &direct_cpu,
        &direct_metal,
        &expected_direct_decode,
        &expected_direct_prefill,
    );

    let gathered_cpu = run_span_case(
        &mut cpu,
        shape,
        &gathered_specs,
        &decode_query,
        &prefill_query,
    );
    let gathered_metal = run_span_case(
        &mut metal,
        shape,
        &gathered_specs,
        &decode_query,
        &prefill_query,
    );
    let expected_gathered_decode =
        span_attention_oracle(&decode_query, &keys, &values, shape, 5, 1);
    let expected_gathered_prefill =
        span_attention_oracle(&prefill_query, &keys, &values, shape, 2, 3);
    assert_span_results(
        "gathered five-span",
        &gathered_cpu,
        &gathered_metal,
        &expected_gathered_decode,
        &expected_gathered_prefill,
    );
}

#[test]
fn finite_budget_allows_large_to_small_span_gather_replacement() {
    let Some(mut metal) = metal_backend() else {
        return;
    };
    let mut cpu = CpuBackend::new();
    let large_shape = AttentionShape::new(1, 1, 2, 6).unwrap();
    let small_shape = AttentionShape::new(1, 1, 1, 5).unwrap();
    let metal_large_caches =
        five_gather_segments(&mut metal, large_shape, 1, 0x4c41_5247, 0x4c56_414c);
    let metal_small_caches =
        five_gather_segments(&mut metal, small_shape, 2, 0x534d_414c, 0x5356_414c);
    let cpu_small_caches = five_gather_segments(&mut cpu, small_shape, 2, 0x534d_414c, 0x5356_414c);
    let metal_large_spans = five_span_view(&metal_large_caches, 1);
    let metal_small_spans = five_span_view(&metal_small_caches, 2);
    let cpu_small_spans = five_span_view(&cpu_small_caches, 2);
    let metal_large_query = upload_f32(&mut metal, &[0.25, -0.5]);
    let metal_small_query = upload_f32(&mut metal, &[0.75]);
    let cpu_small_query = upload_f32(&mut cpu, &[0.75]);
    let mut metal_large_output = metal.allocate(BufferLayout::f32(2).unwrap()).unwrap();
    let mut metal_small_output = metal.allocate(BufferLayout::f32(1).unwrap()).unwrap();
    let mut cpu_small_output = cpu.allocate(BufferLayout::f32(1).unwrap()).unwrap();

    metal
        .attention_decode_spans(
            &metal_large_query,
            KvReadView::new(&metal_large_spans).unwrap(),
            &mut metal_large_output,
            large_shape,
            leone::Position::Host(4),
        )
        .unwrap();
    let before_replacement = metal.memory_accounting();
    metal
        .set_memory_budget(MemoryBudget::limited(before_replacement.live_bytes).unwrap())
        .unwrap();
    metal
        .attention_decode_spans(
            &metal_small_query,
            KvReadView::new(&metal_small_spans).unwrap(),
            &mut metal_small_output,
            small_shape,
            leone::Position::Host(4),
        )
        .expect("finite budget permits a smaller replacement");
    cpu.attention_decode_spans(
        &cpu_small_query,
        KvReadView::new(&cpu_small_spans).unwrap(),
        &mut cpu_small_output,
        small_shape,
        leone::Position::Host(4),
    )
    .unwrap();
    assert_close(
        "finite-budget smaller gathered decode",
        &read_f32(&mut metal, &metal_small_output, 1),
        &read_f32(&mut cpu, &cpu_small_output, 1),
        5e-3,
        5e-3,
    );
    let after_replacement = metal.memory_accounting();
    assert_eq!(after_replacement.reserved_bytes, 0);
    assert!(
        after_replacement
            .class(MemoryClass::BackendScratch)
            .live_bytes
            < before_replacement
                .class(MemoryClass::BackendScratch)
                .live_bytes
    );
}

#[test]
fn failed_span_gather_replacement_restores_ledger_baseline() {
    let Some(mut metal) = metal_backend() else {
        return;
    };
    let large_shape = AttentionShape::new(1, 1, 2, 6).unwrap();
    let small_shape = AttentionShape::new(1, 1, 1, 5).unwrap();
    let metal_large_caches =
        five_gather_segments(&mut metal, large_shape, 1, 0x4c41_5247, 0x4c56_414c);
    let metal_small_caches =
        five_gather_segments(&mut metal, small_shape, 2, 0x534d_414c, 0x5356_414c);
    let metal_large_spans = five_span_view(&metal_large_caches, 1);
    let metal_small_spans = five_span_view(&metal_small_caches, 2);
    let metal_large_query = upload_f32(&mut metal, &[0.25, -0.5]);
    let metal_small_query = upload_f32(&mut metal, &[0.75]);
    let mut metal_large_output = metal.allocate(BufferLayout::f32(2).unwrap()).unwrap();
    let mut metal_small_output = metal.allocate(BufferLayout::f32(1).unwrap()).unwrap();
    let baseline = metal.memory_accounting();
    metal
        .attention_decode_spans(
            &metal_small_query,
            KvReadView::new(&metal_small_spans).unwrap(),
            &mut metal_small_output,
            small_shape,
            leone::Position::Host(4),
        )
        .unwrap();
    let with_small_scratch = metal.memory_accounting();
    let mapped_tokens = KvReadView::new(&metal_large_spans)
        .expect("large spans are gap free")
        .mapped_tokens();
    let large_scratch_bytes = u64::try_from(
        BufferLayout::f16(large_shape.n_head_kv() * large_shape.head_dim() * mapped_tokens)
            .unwrap()
            .bytes(),
    )
    .expect("scratch size fits tracker accounting")
    .checked_mul(2)
    .expect("key and value scratch size");
    let failed_replacement_budget = baseline
        .live_bytes
        .checked_add(large_scratch_bytes)
        .and_then(|bytes| bytes.checked_sub(1))
        .expect("fixture budget");
    metal
        .set_memory_budget(MemoryBudget::limited(failed_replacement_budget).unwrap())
        .unwrap();
    let error = metal
        .attention_decode_spans(
            &metal_large_query,
            KvReadView::new(&metal_large_spans).unwrap(),
            &mut metal_large_output,
            large_shape,
            leone::Position::Host(4),
        )
        .expect_err("larger replacement must exceed the finite budget");
    assert!(matches!(
        error,
        leone::BackendError::Memory(leone::MemoryError::BudgetExceeded { .. })
    ));
    let after_failure = metal.memory_accounting();
    assert_eq!(after_failure.live_bytes, baseline.live_bytes);
    assert_eq!(after_failure.reserved_bytes, baseline.reserved_bytes);
    assert_eq!(
        after_failure.class(MemoryClass::BackendScratch).live_bytes,
        baseline.class(MemoryClass::BackendScratch).live_bytes
    );
    metal
        .attention_decode_spans(
            &metal_small_query,
            KvReadView::new(&metal_small_spans).unwrap(),
            &mut metal_small_output,
            small_shape,
            leone::Position::Host(4),
        )
        .expect("small replacement remains available after failure");
    let after_retry = metal.memory_accounting();
    assert_eq!(
        after_retry.class(MemoryClass::BackendScratch).live_bytes,
        with_small_scratch
            .class(MemoryClass::BackendScratch)
            .live_bytes
    );
    assert_eq!(after_retry.reserved_bytes, 0);
}

#[test]
fn odd_head_dim_spans_match_cpu_when_f16_copy_is_two_bytes() {
    let Some(mut metal) = metal_backend() else {
        return;
    };
    let mut cpu = CpuBackend::new();
    let shape = AttentionShape::new(1, 1, 1, 2).expect("attention shape");
    let key_values = [0.25_f32, -0.5];
    let value_values = [1.5_f32, -2.0];
    let (cpu_key_first, cpu_value_first) = append_segment(
        &mut cpu,
        shape,
        &key_values[..1],
        &value_values[..1],
        0,
        1,
        1,
    );
    let (cpu_key_second, cpu_value_second) = append_segment(
        &mut cpu,
        shape,
        &key_values[1..],
        &value_values[1..],
        1,
        1,
        1,
    );
    let (metal_key_first, metal_value_first) = append_segment(
        &mut metal,
        shape,
        &key_values[..1],
        &value_values[..1],
        0,
        1,
        1,
    );
    let (metal_key_second, metal_value_second) = append_segment(
        &mut metal,
        shape,
        &key_values[1..],
        &value_values[1..],
        1,
        1,
        1,
    );
    let cpu_spans = [
        KvReadSpan::new(&cpu_key_first, &cpu_value_first, 0, 1, 1).unwrap(),
        KvReadSpan::new(&cpu_key_second, &cpu_value_second, 1, 1, 1).unwrap(),
    ];
    let metal_spans = [
        KvReadSpan::new(&metal_key_first, &metal_value_first, 0, 1, 1).unwrap(),
        KvReadSpan::new(&metal_key_second, &metal_value_second, 1, 1, 1).unwrap(),
    ];
    let cpu_cache = KvReadView::new(&cpu_spans).unwrap();
    let metal_cache = KvReadView::new(&metal_spans).unwrap();
    let cpu_query = upload_f32(&mut cpu, &[0.5]);
    let metal_query = upload_f32(&mut metal, &[0.5]);
    let mut cpu_output = cpu.allocate(BufferLayout::f32(1).unwrap()).unwrap();
    let mut metal_output = metal.allocate(BufferLayout::f32(1).unwrap()).unwrap();
    cpu.attention_decode_spans(
        &cpu_query,
        cpu_cache,
        &mut cpu_output,
        shape,
        leone::Position::Host(1),
    )
    .unwrap();
    metal
        .attention_decode_spans(
            &metal_query,
            metal_cache,
            &mut metal_output,
            shape,
            leone::Position::Host(1),
        )
        .unwrap();
    let expected = read_f32(&mut cpu, &cpu_output, 1);
    let actual = read_f32(&mut metal, &metal_output, 1);
    assert_close(
        "odd head dimension segmented decode",
        &actual,
        &expected,
        5e-3,
        5e-3,
    );
}

#[test]
fn mapped_future_tail_does_not_enter_causal_attention() {
    let Some(mut metal) = metal_backend() else {
        return;
    };
    let mut cpu = CpuBackend::new();
    let shape = AttentionShape::new(1, 1, 1, 4).expect("attention shape");
    let key_values = [0.1_f32, 0.2, 100.0, 100.0];
    let value_values = [1.0_f32, 2.0, 300.0, 400.0];
    let key_bytes = f16_bytes(&key_values);
    let value_bytes = f16_bytes(&value_values);
    let cpu_key = cpu
        .upload(BufferLayout::f16(4).unwrap(), &key_bytes)
        .unwrap();
    let cpu_value = cpu
        .upload(BufferLayout::f16(4).unwrap(), &value_bytes)
        .unwrap();
    let metal_key = metal
        .upload(BufferLayout::f16(4).unwrap(), &key_bytes)
        .unwrap();
    let metal_value = metal
        .upload(BufferLayout::f16(4).unwrap(), &value_bytes)
        .unwrap();
    let cpu_spans = [KvReadSpan::new(&cpu_key, &cpu_value, 0, 4, 4).unwrap()];
    let metal_spans = [KvReadSpan::new(&metal_key, &metal_value, 0, 4, 4).unwrap()];
    let cpu_cache = KvReadView::new(&cpu_spans).unwrap();
    let metal_cache = KvReadView::new(&metal_spans).unwrap();
    let cpu_query = upload_f32(&mut cpu, &[1.0]);
    let metal_query = upload_f32(&mut metal, &[1.0]);
    let mut cpu_output = cpu.allocate(BufferLayout::f32(1).unwrap()).unwrap();
    let mut metal_output = metal.allocate(BufferLayout::f32(1).unwrap()).unwrap();
    cpu.attention_decode_spans(
        &cpu_query,
        cpu_cache,
        &mut cpu_output,
        shape,
        leone::Position::Host(1),
    )
    .unwrap();
    metal
        .attention_decode_spans(
            &metal_query,
            metal_cache,
            &mut metal_output,
            shape,
            leone::Position::Host(1),
        )
        .unwrap();
    let expected = read_f32(&mut cpu, &cpu_output, 1);
    let actual = read_f32(&mut metal, &metal_output, 1);
    assert_close("causal mapped tail", &actual, &expected, 5e-3, 5e-3);
}

#[test]
fn verifier_span_matches_cpu_sequence_and_rejects_bad_cache_before_writes() {
    let Some(mut metal) = metal_backend() else {
        return;
    };
    let shape = AttentionShape::new(1, 1, 2, 4).expect("attention shape");
    let vector_shape = VectorShape::new(1, 2).expect("vector shape");
    let batched_shape = VectorShape::new(2, 2).expect("batched vector shape");
    let rope_shape = RopeShape::new(2, 1, 2).expect("rope shape");
    let query_values = upload_f32(&mut metal, &[1.0, 2.0, 3.0, 4.0]);
    let key_values = upload_f32(&mut metal, &[0.5, 1.5, 2.5, 3.5]);
    let query_weight = upload_f32(&mut metal, &[1.0, 1.0]);
    let key_weight = upload_f32(&mut metal, &[1.0, 1.0]);
    let value = upload_f32(&mut metal, &[0.25, 0.75, 1.25, 1.75]);

    let mut bad_query_output = upload_f32(&mut metal, &[9.0; 4]);
    let mut bad_key_output = upload_f32(&mut metal, &[8.0; 4]);
    let mut bad_key_cache = metal
        .upload(BufferLayout::f16(1).unwrap(), &f16_bytes(&[6.0]))
        .unwrap();
    let mut bad_value_cache = metal
        .upload(BufferLayout::f16(8).unwrap(), &f16_bytes(&[7.0; 8]))
        .unwrap();
    let bad_target = KvWriteSpan::new(&mut bad_key_cache, &mut bad_value_cache, 0, 4).unwrap();
    let error = metal
        .verify_qk_norm_rope_kv_append_span(
            &query_values,
            &query_weight,
            &mut bad_query_output,
            vector_shape,
            &key_values,
            &key_weight,
            &mut bad_key_output,
            vector_shape,
            &value,
            bad_target,
            shape,
            0,
            2,
            1e-5,
            10_000.0,
        )
        .expect_err("undersized KV cache must fail before verifier writes");
    assert!(matches!(error, leone::BackendError::SizeMismatch { .. }));
    assert_eq!(read_f32(&mut metal, &bad_query_output, 4), [9.0; 4]);
    assert_eq!(read_f32(&mut metal, &bad_key_output, 4), [8.0; 4]);
    assert_eq!(read_f16(&mut metal, &bad_key_cache, 1), [6.0]);
    assert_eq!(read_f16(&mut metal, &bad_value_cache, 8), [7.0; 8]);

    let mut cpu = CpuBackend::new();
    cpu.configure_rope(2, 10_000.0, None, RopePairing::HalfSplit)
        .unwrap();
    metal
        .configure_rope(2, 10_000.0, None, RopePairing::HalfSplit)
        .unwrap();
    let cpu_query = upload_f32(&mut cpu, &[1.0, 2.0, 3.0, 4.0]);
    let cpu_key = upload_f32(&mut cpu, &[0.5, 1.5, 2.5, 3.5]);
    let cpu_query_weight = upload_f32(&mut cpu, &[1.0, 1.0]);
    let cpu_key_weight = upload_f32(&mut cpu, &[1.0, 1.0]);
    let cpu_value = upload_f32(&mut cpu, &[0.25, 0.75, 1.25, 1.75]);
    let mut cpu_query_output = cpu.allocate(BufferLayout::f32(4).unwrap()).unwrap();
    let mut cpu_key_output = cpu.allocate(BufferLayout::f32(4).unwrap()).unwrap();
    cpu.rms_norm(
        &cpu_query,
        &cpu_query_weight,
        &mut cpu_query_output,
        batched_shape,
        1e-5,
    )
    .unwrap();
    cpu.rms_norm(
        &cpu_key,
        &cpu_key_weight,
        &mut cpu_key_output,
        batched_shape,
        1e-5,
    )
    .unwrap();
    cpu.rope(&mut cpu_query_output, 0, rope_shape, 10_000.0)
        .unwrap();
    cpu.rope(&mut cpu_key_output, 0, rope_shape, 10_000.0)
        .unwrap();
    let mut cpu_key_cache = cpu.allocate(BufferLayout::f16(8).unwrap()).unwrap();
    let mut cpu_value_cache = cpu.allocate(BufferLayout::f16(8).unwrap()).unwrap();
    cpu.kv_append_chunk(
        &cpu_key_output,
        &cpu_value,
        &mut cpu_key_cache,
        &mut cpu_value_cache,
        shape,
        0,
        2,
    )
    .unwrap();

    let mut metal_query_output = metal.allocate(BufferLayout::f32(4).unwrap()).unwrap();
    let mut metal_key_output = metal.allocate(BufferLayout::f32(4).unwrap()).unwrap();
    let mut metal_key_cache = metal.allocate(BufferLayout::f16(8).unwrap()).unwrap();
    let mut metal_value_cache = metal.allocate(BufferLayout::f16(8).unwrap()).unwrap();
    let target = KvWriteSpan::new(&mut metal_key_cache, &mut metal_value_cache, 0, 4).unwrap();
    metal
        .verify_qk_norm_rope_kv_append_span(
            &query_values,
            &query_weight,
            &mut metal_query_output,
            vector_shape,
            &key_values,
            &key_weight,
            &mut metal_key_output,
            vector_shape,
            &value,
            target,
            shape,
            0,
            2,
            1e-5,
            10_000.0,
        )
        .unwrap();
    let expected_query = read_f32(&mut cpu, &cpu_query_output, 4);
    let expected_key = read_f32(&mut cpu, &cpu_key_output, 4);
    let actual_query = read_f32(&mut metal, &metal_query_output, 4);
    let actual_key = read_f32(&mut metal, &metal_key_output, 4);
    assert_close("verifier query", &actual_query, &expected_query, 5e-3, 5e-3);
    assert_close("verifier key", &actual_key, &expected_key, 5e-3, 5e-3);
    let expected_cache = read_f16(&mut cpu, &cpu_key_cache, 8);
    let actual_cache = read_f16(&mut metal, &metal_key_cache, 8);
    assert_close(
        "verifier key cache",
        &actual_cache,
        &expected_cache,
        5e-3,
        5e-3,
    );
    let expected_value_cache = read_f16(&mut cpu, &cpu_value_cache, 8);
    let actual_value_cache = read_f16(&mut metal, &metal_value_cache, 8);
    assert_close(
        "verifier value cache",
        &actual_value_cache,
        &expected_value_cache,
        5e-3,
        5e-3,
    );
}

#[test]
fn swapped_kv_span_write_invalidates_cached_attention() {
    let Some(mut metal) = metal_backend() else {
        return;
    };
    let mut cpu = CpuBackend::new();
    let shape = AttentionShape::new(1, 1, 1, 4).expect("attention shape");
    let initial_keys = upload_f32(&mut metal, &[0.1, 0.2]);
    let initial_values = upload_f32(&mut metal, &[1.0, 2.0]);
    let cpu_initial_keys = upload_f32(&mut cpu, &[0.1, 0.2]);
    let cpu_initial_values = upload_f32(&mut cpu, &[1.0, 2.0]);
    let cache_layout = BufferLayout::f16(shape.cache_elements().unwrap()).unwrap();
    let mut metal_key = metal.allocate(cache_layout).unwrap();
    let mut metal_value = metal.allocate(cache_layout).unwrap();
    let mut cpu_key = cpu.allocate(cache_layout).unwrap();
    let mut cpu_value = cpu.allocate(cache_layout).unwrap();
    metal
        .kv_append_chunk_span(
            &initial_keys,
            &initial_values,
            KvWriteSpan::new(&mut metal_key, &mut metal_value, 0, 4).unwrap(),
            shape,
            0,
            2,
        )
        .unwrap();
    cpu.kv_append_chunk_span(
        &cpu_initial_keys,
        &cpu_initial_values,
        KvWriteSpan::new(&mut cpu_key, &mut cpu_value, 0, 4).unwrap(),
        shape,
        0,
        2,
    )
    .unwrap();
    let initial_metal_spans = [KvReadSpan::new(&metal_key, &metal_value, 0, 2, 4).unwrap()];
    let initial_cpu_spans = [KvReadSpan::new(&cpu_key, &cpu_value, 0, 2, 4).unwrap()];
    let query = upload_f32(&mut metal, &[0.5]);
    let cpu_query = upload_f32(&mut cpu, &[0.5]);
    let mut output = metal.allocate(BufferLayout::f32(1).unwrap()).unwrap();
    let mut cpu_output = cpu.allocate(BufferLayout::f32(1).unwrap()).unwrap();
    metal
        .attention_decode_spans(
            &query,
            KvReadView::new(&initial_metal_spans).unwrap(),
            &mut output,
            shape,
            leone::Position::Host(1),
        )
        .unwrap();
    cpu.attention_decode_spans(
        &cpu_query,
        KvReadView::new(&initial_cpu_spans).unwrap(),
        &mut cpu_output,
        shape,
        leone::Position::Host(1),
    )
    .unwrap();

    // Keep the view signature fixed so the second read tests cache invalidation.
    let swapped_keys = upload_f32(&mut metal, &[9.0]);
    let swapped_values = upload_f32(&mut metal, &[7.0]);
    let cpu_swapped_keys = upload_f32(&mut cpu, &[9.0]);
    let cpu_swapped_values = upload_f32(&mut cpu, &[7.0]);
    metal
        .kv_append_chunk_span(
            &swapped_keys,
            &swapped_values,
            KvWriteSpan::new(&mut metal_value, &mut metal_key, 0, 4).unwrap(),
            shape,
            1,
            1,
        )
        .unwrap();
    cpu.kv_append_chunk_span(
        &cpu_swapped_keys,
        &cpu_swapped_values,
        KvWriteSpan::new(&mut cpu_value, &mut cpu_key, 0, 4).unwrap(),
        shape,
        1,
        1,
    )
    .unwrap();
    let metal_spans = [KvReadSpan::new(&metal_key, &metal_value, 0, 2, 4).unwrap()];
    let cpu_spans = [KvReadSpan::new(&cpu_key, &cpu_value, 0, 2, 4).unwrap()];
    metal
        .attention_decode_spans(
            &query,
            KvReadView::new(&metal_spans).unwrap(),
            &mut output,
            shape,
            leone::Position::Host(1),
        )
        .unwrap();
    cpu.attention_decode_spans(
        &cpu_query,
        KvReadView::new(&cpu_spans).unwrap(),
        &mut cpu_output,
        shape,
        leone::Position::Host(1),
    )
    .unwrap();
    assert_close(
        "swapped KV span write invalidation",
        &read_f32(&mut metal, &output, 1),
        &read_f32(&mut cpu, &cpu_output, 1),
        5e-3,
        5e-3,
    );
}

#[test]
fn foreign_metal_buffers_are_rejected_before_span_cache_lookup() {
    let Some(mut first) = metal_backend() else {
        return;
    };
    let Some(mut second) = metal_backend() else {
        return;
    };
    let shape = AttentionShape::new(1, 1, 1, 2).unwrap();
    let key = second
        .upload(BufferLayout::f16(2).unwrap(), &f16_bytes(&[0.1, 0.2]))
        .unwrap();
    let value = second
        .upload(BufferLayout::f16(2).unwrap(), &f16_bytes(&[1.0, 2.0]))
        .unwrap();
    let query = upload_f32(&mut second, &[0.5]);
    let mut output = first.allocate(BufferLayout::f32(1).unwrap()).unwrap();
    let spans = [KvReadSpan::new(&key, &value, 0, 2, 2).unwrap()];
    let error = first
        .attention_decode_spans(
            &query,
            KvReadView::new(&spans).unwrap(),
            &mut output,
            shape,
            leone::Position::Host(1),
        )
        .expect_err("foreign Metal buffers must fail before dispatch");
    assert!(matches!(error, leone::BackendError::Operation { .. }));
}

fn five_gather_segments<B: Backend>(
    backend: &mut B,
    shape: AttentionShape,
    capacity: usize,
    key_seed: u32,
    value_seed: u32,
) -> Vec<(B::Buffer, B::Buffer)> {
    let projected = shape
        .projected_kv_elements()
        .expect("projected KV elements");
    (0..5)
        .map(|index| {
            let keys = seeded_values(projected, key_seed.wrapping_add(index as u32));
            let values = seeded_values(projected, value_seed.wrapping_add(index as u32));
            append_segment(backend, shape, &keys, &values, index, 1, capacity)
        })
        .collect()
}

fn five_full_gather_segments<B: Backend>(
    backend: &mut B,
    shape: AttentionShape,
    capacity: usize,
    key_seed: u32,
    value_seed: u32,
) -> Vec<(B::Buffer, B::Buffer)> {
    let projected = shape
        .projected_kv_elements()
        .expect("projected KV elements");
    (0..5)
        .map(|index| {
            let keys = seeded_values(capacity * projected, key_seed.wrapping_add(index as u32));
            let values = seeded_values(capacity * projected, value_seed.wrapping_add(index as u32));
            append_segment(
                backend,
                shape,
                &keys,
                &values,
                index * capacity,
                capacity,
                capacity,
            )
        })
        .collect()
}

fn five_span_view<'a, T>(caches: &'a [(T, T)], capacity: usize) -> [KvReadSpan<'a, T>; 5] {
    std::array::from_fn(|index| {
        KvReadSpan::new(&caches[index].0, &caches[index].1, index, 1, capacity)
            .expect("five span descriptor")
    })
}

fn five_full_span_view<'a, T>(caches: &'a [(T, T)], capacity: usize) -> [KvReadSpan<'a, T>; 5] {
    std::array::from_fn(|index| {
        KvReadSpan::new(
            &caches[index].0,
            &caches[index].1,
            index * capacity,
            capacity,
            capacity,
        )
        .expect("full span descriptor")
    })
}

fn append_segment<B: Backend>(
    backend: &mut B,
    shape: AttentionShape,
    keys: &[f32],
    values: &[f32],
    logical_start: usize,
    tokens: usize,
    capacity: usize,
) -> (B::Buffer, B::Buffer) {
    let key = upload_f32(backend, keys);
    let value = upload_f32(backend, values);
    let physical_shape = AttentionShape::new(
        shape.n_head(),
        shape.n_head_kv(),
        shape.head_dim(),
        capacity,
    )
    .expect("physical shape");
    let cache_layout =
        BufferLayout::f16(physical_shape.cache_elements().unwrap()).expect("cache layout");
    let mut key_cache = backend.allocate(cache_layout).expect("key cache");
    let mut value_cache = backend.allocate(cache_layout).expect("value cache");
    let target = KvWriteSpan::new(&mut key_cache, &mut value_cache, logical_start, capacity)
        .expect("write span");
    backend
        .kv_append_chunk_span(&key, &value, target, shape, logical_start, tokens)
        .expect("KV span append");
    (key_cache, value_cache)
}

struct SpanSpec<'a> {
    keys: &'a [f32],
    values: &'a [f32],
    logical_start: usize,
    tokens: usize,
    capacity: usize,
}

struct SpanCaseResult {
    decode: Vec<f32>,
    prefill: Vec<f32>,
}

fn direct_span_specs<'a>(
    keys: &'a [f32],
    values: &'a [f32],
    projected: usize,
) -> [SpanSpec<'a>; 4] {
    [
        SpanSpec {
            keys: &keys[..2 * projected],
            values: &values[..2 * projected],
            logical_start: 0,
            tokens: 2,
            capacity: 3,
        },
        SpanSpec {
            keys: &keys[2 * projected..3 * projected],
            values: &values[2 * projected..3 * projected],
            logical_start: 2,
            tokens: 1,
            capacity: 2,
        },
        SpanSpec {
            keys: &keys[3 * projected..5 * projected],
            values: &values[3 * projected..5 * projected],
            logical_start: 3,
            tokens: 2,
            capacity: 3,
        },
        SpanSpec {
            keys: &keys[5 * projected..6 * projected],
            values: &values[5 * projected..6 * projected],
            logical_start: 5,
            tokens: 1,
            capacity: 2,
        },
    ]
}

fn gathered_span_specs<'a>(
    keys: &'a [f32],
    values: &'a [f32],
    projected: usize,
) -> [SpanSpec<'a>; 5] {
    [
        SpanSpec {
            keys: &keys[..projected],
            values: &values[..projected],
            logical_start: 0,
            tokens: 1,
            capacity: 2,
        },
        SpanSpec {
            keys: &keys[projected..2 * projected],
            values: &values[projected..2 * projected],
            logical_start: 1,
            tokens: 1,
            capacity: 2,
        },
        SpanSpec {
            keys: &keys[2 * projected..3 * projected],
            values: &values[2 * projected..3 * projected],
            logical_start: 2,
            tokens: 1,
            capacity: 2,
        },
        SpanSpec {
            keys: &keys[3 * projected..4 * projected],
            values: &values[3 * projected..4 * projected],
            logical_start: 3,
            tokens: 1,
            capacity: 2,
        },
        SpanSpec {
            keys: &keys[4 * projected..6 * projected],
            values: &values[4 * projected..6 * projected],
            logical_start: 4,
            tokens: 2,
            capacity: 3,
        },
    ]
}

fn run_span_case<B: Backend>(
    backend: &mut B,
    shape: AttentionShape,
    specs: &[SpanSpec<'_>],
    decode_query: &[f32],
    prefill_query: &[f32],
) -> SpanCaseResult {
    let caches = specs
        .iter()
        .map(|spec| {
            append_segment(
                backend,
                shape,
                spec.keys,
                spec.values,
                spec.logical_start,
                spec.tokens,
                spec.capacity,
            )
        })
        .collect::<Vec<_>>();
    let spans = specs
        .iter()
        .zip(&caches)
        .map(|(spec, (key, value))| {
            KvReadSpan::new(key, value, spec.logical_start, spec.tokens, spec.capacity)
                .expect("read span")
        })
        .collect::<Vec<_>>();
    let prefill_input = upload_f32(backend, prefill_query);
    let mut prefill = backend
        .allocate(BufferLayout::f32(prefill_query.len()).expect("span prefill layout"))
        .expect("span prefill output");
    backend
        .attention_prefill_spans(
            &prefill_input,
            KvReadView::new(&spans).expect("prefill span view"),
            &mut prefill,
            shape,
            2,
            3,
        )
        .expect("span prefill");
    let decode_input = upload_f32(backend, decode_query);
    let mut decode = backend
        .allocate(BufferLayout::f32(decode_query.len()).expect("span decode layout"))
        .expect("span decode output");
    backend
        .attention_decode_spans(
            &decode_input,
            KvReadView::new(&spans).expect("decode span view"),
            &mut decode,
            shape,
            leone::Position::Host(5),
        )
        .expect("span decode");
    SpanCaseResult {
        decode: read_f32(backend, &decode, decode_query.len()),
        prefill: read_f32(backend, &prefill, prefill_query.len()),
    }
}

fn assert_span_results(
    name: &str,
    cpu: &SpanCaseResult,
    metal: &SpanCaseResult,
    expected_decode: &[f32],
    expected_prefill: &[f32],
) {
    assert_close(
        &format!("{name} CPU decode oracle"),
        &cpu.decode,
        expected_decode,
        3e-3,
        3e-3,
    );
    assert_close(
        &format!("{name} Metal decode oracle"),
        &metal.decode,
        expected_decode,
        5e-3,
        5e-3,
    );
    assert_close(
        &format!("{name} CPU prefill oracle"),
        &cpu.prefill,
        expected_prefill,
        3e-3,
        3e-3,
    );
    assert_close(
        &format!("{name} Metal prefill oracle"),
        &metal.prefill,
        expected_prefill,
        5e-3,
        5e-3,
    );
}

fn adversarial_span_values(tokens: usize, projected: usize, key: bool) -> Vec<f32> {
    (0..tokens * projected)
        .map(|index| {
            let dimension = index % 256;
            let token = index / projected;
            let kv_head = (index % projected) / 256;
            let sign = if dimension.is_multiple_of(2) {
                1.0
            } else {
                -1.0
            };
            let drift = (token * 3 + kv_head) as f32 * 0.03125;
            if key {
                sign * (512.0 + (dimension % 7) as f32 * 0.25 + drift)
            } else {
                sign * (0.5 + (token + kv_head) as f32 * 0.25)
            }
        })
        .collect()
}

fn adversarial_span_query(elements: usize, seed: u32) -> Vec<f32> {
    seeded_values(elements, seed)
        .into_iter()
        .map(|value| 0.001 + value.abs() * 0.00001)
        .collect()
}

fn span_attention_oracle(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    shape: AttentionShape,
    start_position: usize,
    tokens: usize,
) -> Vec<f32> {
    let group = shape.n_head() / shape.n_head_kv();
    let projected = shape.projected_kv_elements().expect("oracle projected KV");
    let mut output = Vec::with_capacity(tokens * shape.query_elements().expect("oracle query"));
    for token in 0..tokens {
        let context = start_position + token + 1;
        for head in 0..shape.n_head() {
            let kv_head = head / group;
            let query_start =
                token * shape.query_elements().expect("oracle query") + head * shape.head_dim();
            let scores = (0..context)
                .map(|position| span_score(query, keys, shape, query_start, kv_head, position))
                .collect::<Vec<_>>();
            let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights = scores
                .iter()
                .map(|score| (*score - maximum).exp())
                .collect::<Vec<_>>();
            let denominator = weights.iter().sum::<f64>();
            for column in 0..shape.head_dim() {
                let value = (0..context)
                    .map(|position| {
                        let index = position * projected + kv_head * shape.head_dim() + column;
                        weights[position] * f64::from(f16::from_f32(values[index]).to_f32())
                    })
                    .sum::<f64>()
                    / denominator;
                output.push(value as f32);
            }
        }
    }
    output
}

fn span_score(
    query: &[f32],
    keys: &[f32],
    shape: AttentionShape,
    query_start: usize,
    kv_head: usize,
    position: usize,
) -> f64 {
    let projected = shape.projected_kv_elements().expect("oracle projected KV");
    let key_start = position * projected + kv_head * shape.head_dim();
    let dot = (0..shape.head_dim())
        .map(|column| {
            let key = f16::from_f32(keys[key_start + column]).to_f32();
            f64::from(query[query_start + column]) * f64::from(key)
        })
        .sum::<f64>();
    dot / (shape.head_dim() as f64).sqrt()
}

fn f16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| f16::from_f32(*value).to_bits().to_le_bytes())
        .collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_bits().to_le_bytes())
        .collect()
}
