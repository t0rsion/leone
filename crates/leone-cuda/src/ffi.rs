use std::ffi::{c_char, c_int, c_uint, c_void};

// SAFETY: Each declaration matches the `extern "C"` definition in
// `cuda/ie_cuda.cu`. The `cuda` module checks pointer validity and sizes.
unsafe extern "C" {
    pub(crate) fn ie_cuda_error_string(code: c_int) -> *const c_char;
    pub(crate) fn ie_cuda_set_device(device: c_int) -> c_int;
    pub(crate) fn ie_cuda_initialize() -> c_int;
    pub(crate) fn ie_attention_split_count(
        f16_cache: c_int,
        n_head: usize,
        max_context: usize,
        split_count: *mut usize,
    ) -> c_int;
    pub(crate) fn ie_cuda_mem_get_info(free_bytes: *mut usize, total_bytes: *mut usize) -> c_int;
    pub(crate) fn ie_cuda_malloc(pointer: *mut *mut c_void, bytes: usize) -> c_int;
    pub(crate) fn ie_cuda_free(pointer: *mut c_void) -> c_int;
    pub(crate) fn ie_cuda_copy_h2d(
        destination: *mut c_void,
        source: *const c_void,
        bytes: usize,
    ) -> c_int;
    pub(crate) fn ie_cuda_copy_h2d_async(
        destination: *mut c_void,
        source: *const c_void,
        bytes: usize,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_cuda_copy_d2h(
        destination: *mut c_void,
        source: *const c_void,
        bytes: usize,
    ) -> c_int;
    pub(crate) fn ie_cuda_copy_d2h_async(
        destination: *mut c_void,
        source: *const c_void,
        bytes: usize,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_cuda_copy_d2d_async(
        destination: *mut c_void,
        source: *const c_void,
        bytes: usize,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_cuda_stream_create(stream: *mut *mut c_void) -> c_int;
    pub(crate) fn ie_cuda_stream_destroy(stream: *mut c_void) -> c_int;
    pub(crate) fn ie_cuda_stream_synchronize(stream: *mut c_void) -> c_int;
    pub(crate) fn ie_cuda_graph_capture_begin(stream: *mut c_void) -> c_int;
    pub(crate) fn ie_cuda_graph_capture_end(stream: *mut c_void, graph: *mut *mut c_void) -> c_int;
    pub(crate) fn ie_cuda_graph_launch(graph: *mut c_void, stream: *mut c_void) -> c_int;
    pub(crate) fn ie_cuda_graph_destroy(graph: *mut c_void) -> c_int;
    pub(crate) fn ie_cuda_event_create(event: *mut *mut c_void) -> c_int;
    pub(crate) fn ie_cuda_event_destroy(event: *mut c_void) -> c_int;
    pub(crate) fn ie_cuda_event_record(event: *mut c_void, stream: *mut c_void) -> c_int;
    pub(crate) fn ie_cuda_event_synchronize(event: *mut c_void) -> c_int;
    pub(crate) fn ie_cuda_event_elapsed_ms(
        milliseconds: *mut f32,
        start: *mut c_void,
        end: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_cublaslt_error_string(code: c_int) -> *const c_char;
    pub(crate) fn ie_cublaslt_create(handle: *mut *mut c_void) -> c_int;
    pub(crate) fn ie_cublaslt_destroy(handle: *mut c_void) -> c_int;

    pub(crate) fn ie_launch_dequant_k_f16(
        weights: *const u8,
        output: *mut u16,
        rows: usize,
        columns: usize,
        q4: c_int,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_cublaslt_prefill_gemm(
        handle: *mut c_void,
        weights: *const u8,
        input: *const f32,
        output: *mut f32,
        dequantized_weights: *mut u16,
        converted_input: *mut u16,
        rows: usize,
        columns: usize,
        tokens: usize,
        q4: c_int,
        workspace: *mut u8,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_cublaslt_attention_prefill_f16(
        handle: *mut c_void,
        query: *const f32,
        key_cache: *const u16,
        value_cache: *const u16,
        output: *mut f32,
        converted_query: *mut u16,
        scores: *mut f32,
        probabilities: *mut u16,
        head_output: *mut f32,
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        max_context: usize,
        start_position: usize,
        tokens: usize,
        workspace: *mut u8,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_cublaslt_attention_prefill_f32(
        handle: *mut c_void,
        query: *const f32,
        key_cache: *const f32,
        value_cache: *const f32,
        output: *mut f32,
        converted_query: *mut u16,
        scores: *mut f32,
        probabilities: *mut u16,
        head_output: *mut f32,
        converted_kv: *mut u16,
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        max_context: usize,
        start_position: usize,
        tokens: usize,
        workspace: *mut u8,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_quantize_q8_1(
        input: *const f32,
        output: *mut u8,
        quantized_sums: *mut u32,
        elements: usize,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_q4_k_gemv(
        weights: *const u8,
        input: *const u8,
        quantized_sums: *const u32,
        output: *mut f32,
        rows: usize,
        columns: usize,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_q4_k_gemv_residual(
        weights: *const u8,
        input: *const u8,
        residual: *const f32,
        output: *mut f32,
        rows: usize,
        columns: usize,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_q4_k_gemv_multi(
        weights: *const u8,
        input: *const u8,
        residual: *const f32,
        output: *mut f32,
        rows: usize,
        columns: usize,
        positions: usize,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_q4_k_gemv_pair(
        first_weights: *const u8,
        second_weights: *const u8,
        input: *const u8,
        quantized_sums: *const u32,
        first_output: *mut f32,
        second_output: *mut f32,
        first_rows: usize,
        second_rows: usize,
        columns: usize,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_q4_k_gemv_swiglu(
        gate_weights: *const u8,
        up_weights: *const u8,
        input: *const u8,
        gate_output: *mut f32,
        up_output: *mut f32,
        output: *mut f32,
        quantized_output: *mut u8,
        quantized_sums: *mut u32,
        epilogue_ready: *mut u32,
        rows: usize,
        columns: usize,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_q4_k_gemv_probe(
        weights: *const u8,
        input: *const u8,
        output: *mut f32,
        rows: usize,
        columns: usize,
        weight_set: usize,
        geometry: c_int,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_q4_k_gemv_ring_probe(
        weights: *const u8,
        input: *const u8,
        output: *mut f32,
        rows: usize,
        columns: usize,
        weight_sets: usize,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_q4_k_apron_pair_probe(
        first_weights: *const u8,
        second_weights: *const u8,
        third_weights: *const u8,
        input: *const u8,
        output: *mut f32,
        rows: usize,
        columns: usize,
        weight_set: usize,
        apron_bytes: usize,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_q6_k_gemv(
        weights: *const u8,
        input: *const u8,
        output: *mut f32,
        rows: usize,
        columns: usize,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_q6_k_gemv_residual(
        weights: *const u8,
        input: *const u8,
        residual: *const f32,
        output: *mut f32,
        rows: usize,
        columns: usize,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_q6_k_gemv_multi(
        weights: *const u8,
        input: *const u8,
        residual: *const f32,
        output: *mut f32,
        rows: usize,
        columns: usize,
        positions: usize,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_quant_gemv_group_multi(
        first_weights: *const u8,
        second_weights: *const u8,
        third_weights: *const u8,
        input: *const u8,
        first_output: *mut f32,
        second_output: *mut f32,
        third_output: *mut f32,
        first_rows: usize,
        second_rows: usize,
        third_rows: usize,
        columns: usize,
        first_q4: c_int,
        second_q4: c_int,
        third_q4: c_int,
        positions: usize,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_qkv_gemv(
        query_weights: *const u8,
        key_weights: *const u8,
        value_weights: *const u8,
        input: *const f32,
        quantized_input: *const u8,
        quantized_sums: *const u32,
        query: *mut f32,
        key: *mut f32,
        value: *mut f32,
        query_rows: usize,
        key_rows: usize,
        value_rows: usize,
        columns: usize,
        query_q4: c_int,
        key_q4: c_int,
        value_q4: c_int,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_rms_norm(
        input: *const f32,
        weight: *const f32,
        output: *mut f32,
        rows: usize,
        columns: usize,
        epsilon: f32,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_rms_norm_q8_parallel(
        input: *const f32,
        weight: *const f32,
        output: *mut f32,
        quantized_output: *mut u8,
        quantized_sums: *mut u32,
        rows: usize,
        columns: usize,
        epsilon: f32,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_rms_norm_residual(
        left: *const f32,
        right: *const f32,
        weight: *const f32,
        output: *mut f32,
        rows: usize,
        columns: usize,
        epsilon: f32,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_rms_norm_residual_store(
        left: *const f32,
        right: *const f32,
        weight: *const f32,
        residual: *mut f32,
        output: *mut f32,
        rows: usize,
        columns: usize,
        epsilon: f32,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_rms_norm_rope(
        input: *const f32,
        weight: *const f32,
        output: *mut f32,
        rows: usize,
        columns: usize,
        position: usize,
        epsilon: f32,
        theta: f32,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_rms_norm_rope_device_position(
        input: *const f32,
        weight: *const f32,
        output: *mut f32,
        rows: usize,
        columns: usize,
        position: *const c_uint,
        epsilon: f32,
        theta: f32,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_qk_norm_rope(
        query: *const f32,
        query_weight: *const f32,
        query_output: *mut f32,
        query_rows: usize,
        key: *const f32,
        key_weight: *const f32,
        key_output: *mut f32,
        key_rows: usize,
        columns: usize,
        rope_table: *const f32,
        epsilon: f32,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_qk_norm_rope_kv_append(
        query: *const f32,
        query_weight: *const f32,
        query_output: *mut f32,
        query_rows: usize,
        key: *const f32,
        key_weight: *const f32,
        key_output: *mut f32,
        key_rows: usize,
        value: *const f32,
        key_cache: *mut f32,
        value_cache: *mut f32,
        columns: usize,
        max_context: usize,
        position: usize,
        rope_table: *const f32,
        epsilon: f32,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_qk_norm_rope_kv_append_f16(
        query: *const f32,
        query_weight: *const f32,
        query_output: *mut f32,
        query_rows: usize,
        key: *const f32,
        key_weight: *const f32,
        key_output: *mut f32,
        key_rows: usize,
        value: *const f32,
        key_cache: *mut u16,
        value_cache: *mut u16,
        columns: usize,
        max_context: usize,
        position: usize,
        rope_table: *const f32,
        epsilon: f32,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_qk_norm_rope_kv_append_device_position(
        query: *const f32,
        query_weight: *const f32,
        query_output: *mut f32,
        query_rows: usize,
        key: *const f32,
        key_weight: *const f32,
        key_output: *mut f32,
        key_rows: usize,
        value: *const f32,
        key_cache: *mut f32,
        value_cache: *mut f32,
        columns: usize,
        max_context: usize,
        position: *const c_uint,
        rope_table: *const f32,
        epsilon: f32,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_qk_norm_rope_kv_append_f16_device_position(
        query: *const f32,
        query_weight: *const f32,
        query_output: *mut f32,
        query_rows: usize,
        key: *const f32,
        key_weight: *const f32,
        key_output: *mut f32,
        key_rows: usize,
        value: *const f32,
        key_cache: *mut u16,
        value_cache: *mut u16,
        columns: usize,
        max_context: usize,
        position: *const c_uint,
        rope_table: *const f32,
        epsilon: f32,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_verify_qk_norm_rope_kv_append(
        query: *const f32,
        query_weight: *const f32,
        query_output: *mut f32,
        query_rows: usize,
        key: *const f32,
        key_weight: *const f32,
        key_output: *mut f32,
        key_rows: usize,
        value: *const f32,
        key_cache: *mut f32,
        value_cache: *mut f32,
        columns: usize,
        max_context: usize,
        start_position: usize,
        positions: usize,
        inverse_frequencies: *const f64,
        rope_table: *mut f32,
        epsilon: f32,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_verify_qk_norm_rope_kv_append_f16(
        query: *const f32,
        query_weight: *const f32,
        query_output: *mut f32,
        query_rows: usize,
        key: *const f32,
        key_weight: *const f32,
        key_output: *mut f32,
        key_rows: usize,
        value: *const f32,
        key_cache: *mut u16,
        value_cache: *mut u16,
        columns: usize,
        max_context: usize,
        start_position: usize,
        positions: usize,
        inverse_frequencies: *const f64,
        rope_table: *mut f32,
        epsilon: f32,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_prepare_rope_table(
        inverse_frequencies: *const f64,
        table: *mut f32,
        pairs: usize,
        position: usize,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_prepare_rope_table_device_position(
        inverse_frequencies: *const f64,
        table: *mut f32,
        pairs: usize,
        position: *const c_uint,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_rope_neox(
        values: *mut f32,
        positions: *const c_uint,
        tokens: usize,
        heads: usize,
        head_dim: usize,
        theta: f32,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_rope_neox_at(
        values: *mut f32,
        position: usize,
        tokens: usize,
        heads: usize,
        head_dim: usize,
        theta: f32,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_rope_at_frequencies(
        values: *mut f32,
        position: usize,
        tokens: usize,
        heads: usize,
        head_dim: usize,
        inverse_frequencies: *const f64,
        adjacent_pairs: bool,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_swiglu(
        gate: *const f32,
        up: *const f32,
        output: *mut f32,
        elements: usize,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_swiglu_q8(
        gate: *const f32,
        up: *const f32,
        output: *mut f32,
        quantized_output: *mut u8,
        quantized_sums: *mut u32,
        elements: usize,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_residual_add(
        left: *const f32,
        right: *const f32,
        output: *mut f32,
        elements: usize,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_write_u32(
        output: *mut c_uint,
        value: c_uint,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_increment_u32(output: *mut c_uint, stream: *mut c_void) -> c_int;
    pub(crate) fn ie_launch_kv_append(
        key: *const f32,
        value: *const f32,
        key_cache: *mut f32,
        value_cache: *mut f32,
        n_head_kv: usize,
        head_dim: usize,
        max_context: usize,
        position: usize,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_kv_append_device_position(
        key: *const f32,
        value: *const f32,
        key_cache: *mut f32,
        value_cache: *mut f32,
        n_head_kv: usize,
        head_dim: usize,
        max_context: usize,
        position: *const c_uint,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_kv_append_f16(
        key: *const f32,
        value: *const f32,
        key_cache: *mut u16,
        value_cache: *mut u16,
        n_head_kv: usize,
        head_dim: usize,
        max_context: usize,
        position: usize,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_kv_append_f16_device_position(
        key: *const f32,
        value: *const f32,
        key_cache: *mut u16,
        value_cache: *mut u16,
        n_head_kv: usize,
        head_dim: usize,
        max_context: usize,
        position: *const c_uint,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_kv_append_q8(
        key: *const f32,
        value: *const f32,
        key_cache: *mut u8,
        value_cache: *mut u8,
        n_head_kv: usize,
        head_dim: usize,
        max_context: usize,
        position: usize,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_kv_append_q8_device_position(
        key: *const f32,
        value: *const f32,
        key_cache: *mut u8,
        value_cache: *mut u8,
        n_head_kv: usize,
        head_dim: usize,
        max_context: usize,
        position: *const c_uint,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_kv_append_chunk(
        key: *const f32,
        value: *const f32,
        key_cache: *mut f32,
        value_cache: *mut f32,
        n_head_kv: usize,
        head_dim: usize,
        max_context: usize,
        start_position: usize,
        tokens: usize,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_kv_append_chunk_f16(
        key: *const f32,
        value: *const f32,
        key_cache: *mut u16,
        value_cache: *mut u16,
        n_head_kv: usize,
        head_dim: usize,
        max_context: usize,
        start_position: usize,
        tokens: usize,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_embedding_q4_k(
        table: *const u8,
        output: *mut f32,
        rows: usize,
        columns: usize,
        row: usize,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_embedding_q6_k(
        table: *const u8,
        output: *mut f32,
        rows: usize,
        columns: usize,
        row: usize,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_embedding_q4_k_device_row(
        table: *const u8,
        row: *const c_uint,
        output: *mut f32,
        rows: usize,
        columns: usize,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_embedding_q6_k_device_row(
        table: *const u8,
        row: *const c_uint,
        output: *mut f32,
        rows: usize,
        columns: usize,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_embedding_q4_k_batch(
        table: *const u8,
        rows: *const c_uint,
        output: *mut f32,
        table_rows: usize,
        columns: usize,
        tokens: usize,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_embedding_q6_k_batch(
        table: *const u8,
        rows: *const c_uint,
        output: *mut f32,
        table_rows: usize,
        columns: usize,
        tokens: usize,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_copy_f32_row(
        input: *const f32,
        output: *mut f32,
        row: usize,
        columns: usize,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_attention_decode(
        query: *const f32,
        key_cache: *const f32,
        value_cache: *const f32,
        output: *mut f32,
        partial_max: *mut f32,
        partial_sum: *mut f32,
        partial_output: *mut f32,
        quantized_output: *mut u8,
        quantized_sums: *mut u32,
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        max_context: usize,
        context_length: usize,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_attention_decode_device_position(
        query: *const f32,
        key_cache: *const f32,
        value_cache: *const f32,
        output: *mut f32,
        partial_max: *mut f32,
        partial_sum: *mut f32,
        partial_output: *mut f32,
        quantized_output: *mut u8,
        quantized_sums: *mut u32,
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        max_context: usize,
        position: *const c_uint,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_attention_decode_f16(
        query: *const f32,
        key_cache: *const u16,
        value_cache: *const u16,
        output: *mut f32,
        partial_max: *mut f32,
        partial_sum: *mut f32,
        partial_output: *mut f32,
        quantized_output: *mut u8,
        quantized_sums: *mut u32,
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        max_context: usize,
        context_length: usize,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_attention_decode_f16_device_position(
        query: *const f32,
        key_cache: *const u16,
        value_cache: *const u16,
        output: *mut f32,
        partial_max: *mut f32,
        partial_sum: *mut f32,
        partial_output: *mut f32,
        quantized_output: *mut u8,
        quantized_sums: *mut u32,
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        max_context: usize,
        position: *const c_uint,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_attention_decode_q8(
        query: *const f32,
        key_cache: *const u8,
        value_cache: *const u8,
        output: *mut f32,
        partial_max: *mut f32,
        partial_sum: *mut f32,
        partial_output: *mut f32,
        quantized_output: *mut u8,
        quantized_sums: *mut u32,
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        max_context: usize,
        context_length: usize,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_attention_decode_q8_device_position(
        query: *const f32,
        key_cache: *const u8,
        value_cache: *const u8,
        output: *mut f32,
        partial_max: *mut f32,
        partial_sum: *mut f32,
        partial_output: *mut f32,
        quantized_output: *mut u8,
        quantized_sums: *mut u32,
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        max_context: usize,
        position: *const c_uint,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_verify_attention(
        query: *const f32,
        key_cache: *const f32,
        value_cache: *const f32,
        output: *mut f32,
        partial_max: *mut f32,
        partial_sum: *mut f32,
        partial_output: *mut f32,
        quantized_output: *mut u8,
        quantized_sums: *mut u32,
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        max_context: usize,
        start_position: usize,
        positions: usize,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_verify_attention_f16(
        query: *const f32,
        key_cache: *const u16,
        value_cache: *const u16,
        output: *mut f32,
        partial_max: *mut f32,
        partial_sum: *mut f32,
        partial_output: *mut f32,
        quantized_output: *mut u8,
        quantized_sums: *mut u32,
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        max_context: usize,
        start_position: usize,
        positions: usize,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn ie_launch_argmax(
        input: *const f32,
        elements: usize,
        output: *mut c_uint,
        partial_values: *mut f32,
        partial_indices: *mut c_uint,
        stream: *mut c_void,
    ) -> c_int;
}
