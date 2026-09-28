use super::*;

fn routing_runtime() -> Runtime<crate::CpuBackend> {
    let mut runtime = super::tests::cpu_toy_runtime();
    runtime.backend.enable_prefill_routing_stub();
    assert!(runtime.backend.decode_equivalent_prefill_supported());
    assert_eq!(runtime.backend.prefill_method(), PrefillMethod::ChunkedGpu);
    runtime
}

fn prepare_generation(
    dtype: KvCacheDtype,
    prefill_tokens: usize,
    context_tokens: usize,
    reused_tokens: usize,
) -> PreparedPrefill<crate::CpuBackend> {
    let mut runtime = routing_runtime();
    let mut options = GenerateOptions::greedy(1);
    options.prefill_chunk_tokens = 2;
    options.kv_cache_dtype = dtype;
    runtime
        .prepare_generation_prefill(
            prefill_tokens,
            context_tokens,
            &options,
            reused_tokens,
            false,
        )
        .expect("routing stub prepares prefill")
}

fn prepared_numerics(prepared: &PreparedPrefill<crate::CpuBackend>) -> Option<PrefillNumerics> {
    match prepared {
        PreparedPrefill::Chunked { plan, .. } => Some(plan.numerics()),
        PreparedPrefill::Reused | PreparedPrefill::Sequential => None,
    }
}

#[test]
fn generation_prefill_selects_by_capability_and_kv_dtype() {
    assert_eq!(
        prepared_numerics(&prepare_generation(KvCacheDtype::F16, 4, 4, 0)),
        Some(PrefillNumerics::BackendPreferred)
    );
    assert_eq!(
        prepared_numerics(&prepare_generation(KvCacheDtype::F32, 4, 4, 0)),
        Some(PrefillNumerics::BackendPreferred)
    );
    assert!(matches!(
        prepare_generation(KvCacheDtype::Q8, 4, 4, 0),
        PreparedPrefill::Sequential
    ));

    assert_eq!(
        prepared_numerics(&prepare_generation(KvCacheDtype::F16, 2, 4, 2)),
        Some(PrefillNumerics::DecodeEquivalent)
    );
    assert!(matches!(
        prepare_generation(KvCacheDtype::F32, 2, 4, 2),
        PreparedPrefill::Sequential
    ));
    assert!(matches!(
        prepare_generation(KvCacheDtype::Q8, 2, 4, 2),
        PreparedPrefill::Sequential
    ));

    let reused = prepare_generation(KvCacheDtype::F16, 0, 4, 4);
    assert!(matches!(reused, PreparedPrefill::Reused));
}
