use half::f16;
use leone::{
    AttentionShape, Backend, BufferLayout, CpuBackend, MemoryBudget, MemoryClass, Position,
    QuantFormat, QuantMatrix, RopePairing, RopeShape, VectorShape,
};
use leone_gguf::ref_dequant;
use leone_metal::MetalBackend;

#[test]
fn shader_context_loads_on_supported_host() {
    match MetalBackend::new() {
        Ok(backend) => {
            let metadata = backend.device_metadata().expect("Metal device metadata");
            assert!(!metadata.device_name.is_empty());
            assert!(!metadata.architecture_name.is_empty());
            assert!(!metadata.os_version.is_empty());
            assert!(metadata.os_version.contains("Build"));
            assert_eq!(metadata.shader_source_hash.len(), 64);
            assert!(metadata
                .shader_source_hash
                .chars()
                .all(|character| character.is_ascii_hexdigit()));
            assert!(!metadata.fast_math_enabled);
        }
        Err(error) => {
            #[cfg(not(target_os = "macos"))]
            assert!(error.to_string().contains("requires macOS"));
            #[cfg(target_os = "macos")]
            panic!("Metal initialization failed on macOS: {error}");
        }
    }
}

#[test]
fn allocation_budget_rejects_before_native_buffer_creation() {
    let Ok(mut backend) = MetalBackend::new() else {
        #[cfg(target_os = "macos")]
        panic!("Metal initialization failed on macOS");
        #[cfg(not(target_os = "macos"))]
        return;
    };
    backend
        .set_memory_budget(MemoryBudget::limited(4).unwrap())
        .unwrap();
    let error = backend.allocate(BufferLayout::f32(2).unwrap()).unwrap_err();
    assert!(matches!(
        error,
        leone::BackendError::Memory(leone::MemoryError::BudgetExceeded { .. })
    ));
    assert_eq!(backend.memory_accounting().live_bytes, 0);

    backend
        .set_memory_budget(MemoryBudget::limited(16).unwrap())
        .unwrap();
    let buffer = backend.allocate(BufferLayout::f32(2).unwrap()).unwrap();
    assert_eq!(backend.memory_accounting().live_bytes, 8);
    drop(buffer);
    assert_eq!(backend.memory_accounting().live_bytes, 0);
}

#[test]
fn one_element_f16_clone_handles_two_byte_copy() {
    let Ok(mut backend) = MetalBackend::new() else {
        #[cfg(target_os = "macos")]
        panic!("Metal initialization failed on macOS");
        #[cfg(not(target_os = "macos"))]
        return;
    };
    let source = backend
        .upload(
            BufferLayout::f16(1).unwrap(),
            &f16::from_f32(-0.75).to_bits().to_le_bytes(),
        )
        .unwrap();
    let clone = backend
        .clone_buffer(&source)
        .expect("two-byte F16 copy must use the supported fallback");
    let mut actual = [0_u16];
    backend.read_f16(&clone, &mut actual).unwrap();
    assert_eq!(f16::from_bits(actual[0]), f16::from_f32(-0.75));
}

#[test]
fn typed_reads_preserve_little_endian_scalar_bits() {
    let Ok(mut backend) = MetalBackend::new() else {
        #[cfg(target_os = "macos")]
        panic!("Metal initialization failed on macOS");
        #[cfg(not(target_os = "macos"))]
        return;
    };
    let u32_bits = [0x0000_0001_u32, 0x7fff_fffe_u32];
    let u32_buffer = backend
        .upload(
            BufferLayout::u32(u32_bits.len()).unwrap(),
            &u32_bits
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let mut actual_u32 = [0_u32; 2];
    backend.read_u32(&u32_buffer, &mut actual_u32).unwrap();
    assert_eq!(actual_u32, u32_bits);

    let f16_bits = [0x0001_u16, 0x7e01_u16];
    let f16_buffer = backend
        .upload(
            BufferLayout::f16(f16_bits.len()).unwrap(),
            &f16_bits
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let mut actual_f16 = [0_u16; 2];
    backend.read_f16(&f16_buffer, &mut actual_f16).unwrap();
    assert_eq!(actual_f16, f16_bits);

    let f32_bits = [0x8000_0000_u32, 0x7fc0_1234_u32];
    let f32_buffer = backend
        .upload(
            BufferLayout::f32(f32_bits.len()).unwrap(),
            &f32_bits
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let mut actual_f32 = [0.0_f32; 2];
    backend.read_f32(&f32_buffer, &mut actual_f32).unwrap();
    assert_eq!(
        actual_f32.map(f32::to_bits),
        f32_bits,
        "f32 reads preserve NaN and signed-zero bit patterns"
    );
}

#[test]
fn typed_read_rejects_short_destination_without_writing_it() {
    let Ok(mut backend) = MetalBackend::new() else {
        #[cfg(target_os = "macos")]
        panic!("Metal initialization failed on macOS");
        #[cfg(not(target_os = "macos"))]
        return;
    };
    let source = backend
        .upload(BufferLayout::f32(2).unwrap(), &f32_bytes(&[1.0, 2.0]))
        .unwrap();
    let sentinel = f32::from_bits(0x7fc0_1234);
    let mut destination = [sentinel];
    let error = backend.read_f32(&source, &mut destination).unwrap_err();
    assert!(matches!(error, leone::BackendError::SizeMismatch { .. }));
    assert_eq!(destination[0].to_bits(), sentinel.to_bits());
}

#[test]
fn q4_gemv_dispatches_on_supported_host() {
    let Ok(mut backend) = MetalBackend::new() else {
        #[cfg(target_os = "macos")]
        panic!("Metal initialization failed on macOS");
        #[cfg(not(target_os = "macos"))]
        return;
    };
    let shape = QuantMatrix::new(1, 256, QuantFormat::Q4K).unwrap();
    let mut weights = vec![0_u8; shape.layout().unwrap().bytes()];
    weights[0] = 0x00;
    weights[1] = 0x3c;
    weights[4..12].fill(1);
    weights[16..].fill(0x22);
    let weights = backend.upload(shape.layout().unwrap(), &weights).unwrap();
    let input = backend
        .upload(
            BufferLayout::f32(256).unwrap(),
            &vec![1.0_f32; 256]
                .into_iter()
                .flat_map(|value| value.to_bits().to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let mut output = backend.allocate(BufferLayout::f32(1).unwrap()).unwrap();
    backend.gemv(&weights, &input, &mut output, shape).unwrap();
    let mut value = [0.0_f32];
    backend.read_f32(&output, &mut value).unwrap();
    assert!((value[0] - 256.0).abs() < 0.01, "got {}", value[0]);
}

#[test]
fn q6_gemv_and_dense_primitives_match_oracle() {
    let Ok(mut backend) = MetalBackend::new() else {
        #[cfg(target_os = "macos")]
        panic!("Metal initialization failed on macOS");
        #[cfg(not(target_os = "macos"))]
        return;
    };
    let shape = QuantMatrix::new(1, 256, QuantFormat::Q6K).unwrap();
    let mut weights = vec![0_u8; shape.layout().unwrap().bytes()];
    weights[208] = 0x00;
    weights[209] = 0x3c;
    weights[192..208].fill(1);
    weights[0..128].fill(0x11);
    weights[128..192].fill(0xaa);
    let weights = backend.upload(shape.layout().unwrap(), &weights).unwrap();
    let input_bytes = f32_bytes(&vec![1.0; 256]);
    let input = backend
        .upload(BufferLayout::f32(256).unwrap(), &input_bytes)
        .unwrap();
    let mut output = backend.allocate(BufferLayout::f32(1).unwrap()).unwrap();
    backend.gemv(&weights, &input, &mut output, shape).unwrap();
    let mut value = [0.0];
    backend.read_f32(&output, &mut value).unwrap();
    assert!((value[0] - 256.0).abs() < 0.05, "got {}", value[0]);

    let residual = backend
        .upload(BufferLayout::f32(1).unwrap(), &f32_bytes(&[2.5]))
        .unwrap();
    let mut residual_output = backend.allocate(BufferLayout::f32(1).unwrap()).unwrap();
    backend
        .gemv_residual(&weights, &input, &residual, &mut residual_output, shape)
        .unwrap();
    backend.read_f32(&residual_output, &mut value).unwrap();
    assert!((value[0] - 258.5).abs() < 0.05);

    let mut query = backend.allocate(BufferLayout::f32(1).unwrap()).unwrap();
    let mut key = backend.allocate(BufferLayout::f32(1).unwrap()).unwrap();
    let mut value_output = backend.allocate(BufferLayout::f32(1).unwrap()).unwrap();
    backend
        .qkv_gemv(
            &weights,
            shape,
            &weights,
            shape,
            &weights,
            shape,
            &input,
            &mut query,
            &mut key,
            &mut value_output,
        )
        .unwrap();
    let mut qkv = [0.0];
    backend.read_f32(&query, &mut qkv).unwrap();
    assert!((qkv[0] - 256.0).abs() < 0.05);
    backend.read_f32(&key, &mut qkv).unwrap();
    assert!((qkv[0] - 256.0).abs() < 0.05);
    backend.read_f32(&value_output, &mut qkv).unwrap();
    assert!((qkv[0] - 256.0).abs() < 0.05);

    let vector = backend
        .upload(
            BufferLayout::f32(4).unwrap(),
            &f32_bytes(&[1.0, 2.0, 3.0, 4.0]),
        )
        .unwrap();
    let weight = backend
        .upload(BufferLayout::f32(4).unwrap(), &f32_bytes(&[1.0; 4]))
        .unwrap();
    let mut normalized = backend.allocate(BufferLayout::f32(4).unwrap()).unwrap();
    backend
        .rms_norm(
            &vector,
            &weight,
            &mut normalized,
            VectorShape::new(1, 4).unwrap(),
            1e-5,
        )
        .unwrap();
    let mut values = [0.0; 4];
    backend.read_f32(&normalized, &mut values).unwrap();
    let norm = (7.5_f32 + 1e-5).sqrt();
    for (value, expected) in values.iter().zip([1.0, 2.0, 3.0, 4.0]) {
        assert!((*value - expected / norm).abs() < 1e-3);
    }

    let factors = [1.25, 0.75];
    backend
        .configure_rope(4, 10_000.0, Some(&factors), RopePairing::HalfSplit)
        .unwrap();
    let mut fused = backend.allocate(BufferLayout::f32(4).unwrap()).unwrap();
    backend
        .rms_norm_rope(
            &vector,
            &weight,
            &mut fused,
            VectorShape::new(1, 4).unwrap(),
            Position::Host(3),
            1e-5,
            10_000.0,
        )
        .unwrap();
    let mut fused_values = [0.0; 4];
    backend.read_f32(&fused, &mut fused_values).unwrap();
    let mut cpu = CpuBackend::new();
    cpu.configure_rope(4, 10_000.0, Some(&factors), RopePairing::HalfSplit)
        .unwrap();
    let cpu_vector = cpu
        .upload(
            BufferLayout::f32(4).unwrap(),
            &f32_bytes(&[1.0, 2.0, 3.0, 4.0]),
        )
        .unwrap();
    let cpu_weight = cpu
        .upload(BufferLayout::f32(4).unwrap(), &f32_bytes(&[1.0; 4]))
        .unwrap();
    let mut cpu_fused = cpu.allocate(BufferLayout::f32(4).unwrap()).unwrap();
    cpu.rms_norm_rope(
        &cpu_vector,
        &cpu_weight,
        &mut cpu_fused,
        VectorShape::new(1, 4).unwrap(),
        Position::Host(3),
        1e-5,
        10_000.0,
    )
    .unwrap();
    let mut expected_fused = [0.0; 4];
    cpu.read_f32(&cpu_fused, &mut expected_fused).unwrap();
    for (actual, expected) in fused_values.iter().zip(expected_fused) {
        assert!((actual - expected).abs() < 2e-4, "{actual} vs {expected}");
    }

    let up = backend
        .upload(BufferLayout::f32(4).unwrap(), &f32_bytes(&[2.0; 4]))
        .unwrap();
    let mut swiglu = backend.allocate(BufferLayout::f32(4).unwrap()).unwrap();
    backend.swiglu(&vector, &up, &mut swiglu).unwrap();
    backend.residual_add(&vector, &up, &mut normalized).unwrap();
    backend.read_f32(&normalized, &mut values).unwrap();
    assert_eq!(values, [3.0, 4.0, 5.0, 6.0]);

    let right = backend
        .upload(
            BufferLayout::f32(4).unwrap(),
            &f32_bytes(&[4.0, 3.0, 2.0, 1.0]),
        )
        .unwrap();
    let mut residual_norm = backend.allocate(BufferLayout::f32(4).unwrap()).unwrap();
    backend
        .rms_norm_residual(
            &vector,
            &right,
            &weight,
            &mut residual_norm,
            VectorShape::new(1, 4).unwrap(),
            1e-5,
        )
        .unwrap();
    backend.read_f32(&residual_norm, &mut values).unwrap();
    let residual_norm_value = (25.0_f32 + 1e-5).sqrt();
    for (value, expected) in values.iter().zip([5.0, 5.0, 5.0, 5.0]) {
        assert!(
            (*value - expected / residual_norm_value).abs() < 1e-3,
            "{value} vs {}",
            expected / residual_norm_value
        );
    }
    let mut stored_residual = backend.allocate(BufferLayout::f32(4).unwrap()).unwrap();
    backend
        .rms_norm_residual_store(
            &vector,
            &right,
            &weight,
            &mut stored_residual,
            &mut residual_norm,
            VectorShape::new(1, 4).unwrap(),
            1e-5,
        )
        .unwrap();
    backend.read_f32(&stored_residual, &mut values).unwrap();
    assert_eq!(values, [5.0, 5.0, 5.0, 5.0]);

    let snapshot = backend.download_buffer(&vector).unwrap();
    let clone = backend.clone_buffer(&vector).unwrap();
    let restored = backend
        .restore_buffer_classified(&snapshot, MemoryClass::Activation)
        .unwrap();
    let mut clone_values = [0.0; 4];
    backend.read_f32(&clone, &mut clone_values).unwrap();
    assert_eq!(clone_values, [1.0, 2.0, 3.0, 4.0]);
    backend.read_f32(&restored, &mut clone_values).unwrap();
    assert_eq!(clone_values, [1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn random_quantized_gemv_matches_scalar_decoder() {
    let Ok(mut backend) = MetalBackend::new() else {
        #[cfg(target_os = "macos")]
        panic!("Metal initialization failed on macOS");
        #[cfg(not(target_os = "macos"))]
        return;
    };
    let input_values = (0..256)
        .map(|index| (index as f32 * 0.03125).sin())
        .collect::<Vec<_>>();
    let input = backend
        .upload(
            BufferLayout::f32(input_values.len()).unwrap(),
            &f32_bytes(&input_values),
        )
        .unwrap();
    for format in [QuantFormat::Q4K, QuantFormat::Q6K] {
        let shape = QuantMatrix::new(2, 256, format).unwrap();
        let mut bytes = vec![0_u8; shape.layout().unwrap().bytes()];
        let mut state = 0x8d12_73a1_u32;
        for byte in &mut bytes {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *byte = (state >> 24) as u8;
        }
        let weights = backend.upload(shape.layout().unwrap(), &bytes).unwrap();
        let mut output = backend.allocate(BufferLayout::f32(2).unwrap()).unwrap();
        backend.gemv(&weights, &input, &mut output, shape).unwrap();
        let mut actual = [0.0_f32; 2];
        backend.read_f32(&output, &mut actual).unwrap();
        for (row, actual) in actual.iter().copied().enumerate() {
            let start = row * shape.layout().unwrap().bytes() / 2;
            let end = start + shape.layout().unwrap().bytes() / 2;
            let decoded = match format {
                QuantFormat::Q4K => ref_dequant::q4_k::dequant_row(&bytes[start..end], 256),
                QuantFormat::Q6K => ref_dequant::q6_k::dequant_row(&bytes[start..end], 256),
            }
            .unwrap();
            let expected = decoded
                .iter()
                .zip(&input_values)
                .map(|(weight, input)| weight * input)
                .sum::<f32>();
            assert!(
                (actual - expected).abs() < expected.abs() * 1e-5 + 0.1,
                "{format:?} row {row}: {actual} vs {expected}"
            );
        }
    }
}

#[test]
fn chunked_gemm_matches_scalar_decoder_for_multiple_rows_and_tokens() {
    let Ok(mut backend) = MetalBackend::new() else {
        #[cfg(target_os = "macos")]
        panic!("Metal initialization failed on macOS");
        #[cfg(not(target_os = "macos"))]
        return;
    };
    let shape = QuantMatrix::new(3, 256, QuantFormat::Q6K).unwrap();
    let mut bytes = vec![0_u8; shape.layout().unwrap().bytes()];
    let mut state = 0x2718_2818_u32;
    for byte in &mut bytes {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *byte = (state >> 24) as u8;
    }
    let weights = backend.upload(shape.layout().unwrap(), &bytes).unwrap();
    let tokens = 4;
    let input_values = (0..tokens * 256)
        .map(|index| (index as f32 * 0.017).cos())
        .collect::<Vec<_>>();
    let input = backend
        .upload(
            BufferLayout::f32(input_values.len()).unwrap(),
            &f32_bytes(&input_values),
        )
        .unwrap();
    let mut output = backend
        .allocate(BufferLayout::f32(tokens * shape.rows()).unwrap())
        .unwrap();
    backend
        .prefill_gemm(&weights, &input, &mut output, shape, tokens)
        .unwrap();
    let mut actual = vec![0.0; tokens * shape.rows()];
    backend.read_f32(&output, &mut actual).unwrap();
    let row_bytes = shape.layout().unwrap().bytes() / shape.rows();
    for token in 0..tokens {
        for row in 0..shape.rows() {
            let start = row * row_bytes;
            let decoded =
                ref_dequant::q6_k::dequant_row(&bytes[start..start + row_bytes], 256).unwrap();
            let expected = decoded
                .iter()
                .zip(&input_values[token * 256..(token + 1) * 256])
                .map(|(weight, input)| weight * input)
                .sum::<f32>();
            let index = token * shape.rows() + row;
            assert!(
                (actual[index] - expected).abs() < expected.abs() * 1e-5 + 0.1,
                "token {token} row {row}: {} vs {expected}",
                actual[index]
            );
        }
    }
}

#[test]
fn chunked_kv_attention_and_rows_match_oracle() {
    let Ok(mut backend) = MetalBackend::new() else {
        #[cfg(target_os = "macos")]
        panic!("Metal initialization failed on macOS");
        #[cfg(not(target_os = "macos"))]
        return;
    };
    let shape = AttentionShape::new(2, 1, 2, 4).unwrap();
    let key = backend
        .upload(
            BufferLayout::f32(4).unwrap(),
            &f32_bytes(&[1.0, 0.0, 0.0, 1.0]),
        )
        .unwrap();
    let value = backend
        .upload(
            BufferLayout::f32(4).unwrap(),
            &f32_bytes(&[2.0, 3.0, 4.0, 5.0]),
        )
        .unwrap();
    let mut key_cache = backend.allocate(BufferLayout::f16(8).unwrap()).unwrap();
    let mut value_cache = backend.allocate(BufferLayout::f16(8).unwrap()).unwrap();
    backend
        .kv_append_chunk(&key, &value, &mut key_cache, &mut value_cache, shape, 0, 2)
        .unwrap();
    let query = backend
        .upload(
            BufferLayout::f32(8).unwrap(),
            &f32_bytes(&[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]),
        )
        .unwrap();
    let mut output = backend.allocate(BufferLayout::f32(8).unwrap()).unwrap();
    backend
        .attention_prefill(&query, &key_cache, &value_cache, &mut output, shape, 0, 2)
        .unwrap();
    let mut attention_values = [0.0; 8];
    backend.read_f32(&output, &mut attention_values).unwrap();
    assert!((attention_values[0] - 2.0).abs() < 1e-3);
    assert!((attention_values[1] - 3.0).abs() < 1e-3);
    assert!(attention_values[4] > 2.0 && attention_values[4] < 4.0);
    assert!(attention_values[5] > 3.0 && attention_values[5] < 5.0);

    let rows = backend
        .upload(
            BufferLayout::f32(8).unwrap(),
            &f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        )
        .unwrap();
    let mut row = backend.allocate(BufferLayout::f32(4).unwrap()).unwrap();
    backend.copy_f32_row(&rows, 1, 4, &mut row).unwrap();
    let mut row_values = [0.0; 4];
    backend.read_f32(&row, &mut row_values).unwrap();
    assert_eq!(row_values, [5.0, 6.0, 7.0, 8.0]);
    let source = backend
        .upload(
            BufferLayout::f32(4).unwrap(),
            &f32_bytes(&[9.0, 10.0, 11.0, 12.0]),
        )
        .unwrap();
    let mut row_output = backend.allocate(BufferLayout::f32(8).unwrap()).unwrap();
    backend
        .write_f32_row(&source, &mut row_output, 0, 4)
        .unwrap();
    backend
        .read_f32(&row_output, &mut attention_values)
        .unwrap();
    assert_eq!(attention_values[..4], [9.0, 10.0, 11.0, 12.0]);
}

#[test]
fn multi_token_rope_matches_cpu_oracle_at_nonzero_positions() {
    let Ok(mut backend) = MetalBackend::new() else {
        #[cfg(target_os = "macos")]
        panic!("Metal initialization failed on macOS");
        #[cfg(not(target_os = "macos"))]
        return;
    };
    let shape = RopeShape::new(2, 2, 4).unwrap();
    let source_values = (0..shape.elements().unwrap())
        .map(|index| (index as f32 * 0.19).sin())
        .collect::<Vec<_>>();
    for pairing in [RopePairing::HalfSplit, RopePairing::Adjacent] {
        let factors = [1.25, 0.75];
        backend
            .configure_rope(4, 10_000.0, Some(&factors), pairing)
            .unwrap();
        let mut values = backend
            .upload(
                BufferLayout::f32(source_values.len()).unwrap(),
                &f32_bytes(&source_values),
            )
            .unwrap();
        backend.rope(&mut values, 7, shape, 10_000.0).unwrap();
        let mut actual = vec![0.0; source_values.len()];
        backend.read_f32(&values, &mut actual).unwrap();

        let mut cpu = CpuBackend::new();
        cpu.configure_rope(4, 10_000.0, Some(&factors), pairing)
            .unwrap();
        let mut cpu_values = cpu
            .upload(
                BufferLayout::f32(source_values.len()).unwrap(),
                &f32_bytes(&source_values),
            )
            .unwrap();
        cpu.rope(&mut cpu_values, 7, shape, 10_000.0).unwrap();
        let mut expected = vec![0.0; source_values.len()];
        cpu.read_f32(&cpu_values, &mut expected).unwrap();
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 2e-4,
                "{pairing:?}: {actual} vs {expected}"
            );
        }
    }

    let mut mismatched = backend.allocate(BufferLayout::f32(8).unwrap()).unwrap();
    let error = backend
        .rope(
            &mut mismatched,
            0,
            RopeShape::new(1, 1, 8).unwrap(),
            10_000.0,
        )
        .unwrap_err();
    assert!(matches!(error, leone::BackendError::SizeMismatch { .. }));

    let odd_input = backend
        .upload(BufferLayout::f32(3).unwrap(), &f32_bytes(&[1.0, 2.0, 3.0]))
        .unwrap();
    let odd_weight = backend
        .upload(BufferLayout::f32(3).unwrap(), &f32_bytes(&[1.0; 3]))
        .unwrap();
    let mut odd_output = backend.allocate(BufferLayout::f32(3).unwrap()).unwrap();
    let error = backend
        .rms_norm_rope(
            &odd_input,
            &odd_weight,
            &mut odd_output,
            VectorShape::new(1, 3).unwrap(),
            Position::Host(0),
            1e-5,
            10_000.0,
        )
        .unwrap_err();
    assert!(matches!(error, leone::BackendError::NotDivisible { .. }));

    let Ok(mut unconfigured) = MetalBackend::new() else {
        #[cfg(target_os = "macos")]
        panic!("Metal initialization failed on macOS");
        #[cfg(not(target_os = "macos"))]
        return;
    };
    let mut unconfigured_values = unconfigured
        .upload(BufferLayout::f32(4).unwrap(), &f32_bytes(&[1.0; 4]))
        .unwrap();
    let error = unconfigured
        .rope(
            &mut unconfigured_values,
            0,
            RopeShape::new(1, 1, 4).unwrap(),
            10_000.0,
        )
        .unwrap_err();
    assert!(matches!(
        error,
        leone::BackendError::SizeMismatch {
            name: "RoPE inverse frequencies",
            ..
        }
    ));
}

#[test]
fn f16_gqa_attention_matches_f64_oracle_at_boundary_positions() {
    let Ok(mut backend) = MetalBackend::new() else {
        #[cfg(target_os = "macos")]
        panic!("Metal initialization failed on macOS");
        #[cfg(not(target_os = "macos"))]
        return;
    };
    let shape = AttentionShape::new(4, 2, 2, 6).unwrap();
    let key_values = (0..5 * shape.projected_kv_elements().unwrap())
        .map(|index| (index as f32 * 0.11 - 0.8).sin())
        .collect::<Vec<_>>();
    let value_values = (0..5 * shape.projected_kv_elements().unwrap())
        .map(|index| (index as f32 * 0.07 + 0.3).cos())
        .collect::<Vec<_>>();
    let key_prefix = backend
        .upload(
            BufferLayout::f32(2 * shape.projected_kv_elements().unwrap()).unwrap(),
            &f32_bytes(&key_values[..2 * shape.projected_kv_elements().unwrap()]),
        )
        .unwrap();
    let value_prefix = backend
        .upload(
            BufferLayout::f32(2 * shape.projected_kv_elements().unwrap()).unwrap(),
            &f32_bytes(&value_values[..2 * shape.projected_kv_elements().unwrap()]),
        )
        .unwrap();
    let key_tail = backend
        .upload(
            BufferLayout::f32(3 * shape.projected_kv_elements().unwrap()).unwrap(),
            &f32_bytes(&key_values[2 * shape.projected_kv_elements().unwrap()..]),
        )
        .unwrap();
    let value_tail = backend
        .upload(
            BufferLayout::f32(3 * shape.projected_kv_elements().unwrap()).unwrap(),
            &f32_bytes(&value_values[2 * shape.projected_kv_elements().unwrap()..]),
        )
        .unwrap();
    let mut key_cache = backend
        .allocate(BufferLayout::f16(shape.cache_elements().unwrap()).unwrap())
        .unwrap();
    let mut value_cache = backend
        .allocate(BufferLayout::f16(shape.cache_elements().unwrap()).unwrap())
        .unwrap();
    backend
        .kv_append_chunk(
            &key_prefix,
            &value_prefix,
            &mut key_cache,
            &mut value_cache,
            shape,
            0,
            2,
        )
        .unwrap();
    backend
        .kv_append_chunk(
            &key_tail,
            &value_tail,
            &mut key_cache,
            &mut value_cache,
            shape,
            2,
            3,
        )
        .unwrap();

    let query_values = (0..3 * shape.query_elements().unwrap())
        .map(|index| (index as f32 * 0.13 - 0.5).cos())
        .collect::<Vec<_>>();
    let query = backend
        .upload(
            BufferLayout::f32(query_values.len()).unwrap(),
            &f32_bytes(&query_values),
        )
        .unwrap();
    let mut output = backend
        .allocate(BufferLayout::f32(query_values.len()).unwrap())
        .unwrap();
    backend
        .attention_prefill(&query, &key_cache, &value_cache, &mut output, shape, 2, 3)
        .unwrap();
    let mut actual = vec![0.0; query_values.len()];
    backend.read_f32(&output, &mut actual).unwrap();
    let expected = attention_oracle(&query_values, &key_values, &value_values, shape, 2, 3);
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() < 2e-3, "{actual} vs {expected}");
    }
    let mut repeat = backend
        .allocate(BufferLayout::f32(query_values.len()).unwrap())
        .unwrap();
    backend
        .attention_prefill(&query, &key_cache, &value_cache, &mut repeat, shape, 2, 3)
        .unwrap();
    let mut repeat_values = vec![0.0; query_values.len()];
    backend.read_f32(&repeat, &mut repeat_values).unwrap();
    assert_eq!(actual, repeat_values);
}

fn attention_oracle(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    shape: AttentionShape,
    start_position: usize,
    tokens: usize,
) -> Vec<f32> {
    let group = shape.n_head() / shape.n_head_kv();
    let projected = shape.projected_kv_elements().unwrap();
    let mut output = vec![0.0; tokens * shape.query_elements().unwrap()];
    for token in 0..tokens {
        let position = start_position + token;
        for head in 0..shape.n_head() {
            let kv_head = head / group;
            let mut scores = Vec::with_capacity(position + 1);
            for key_position in 0..=position {
                let mut score = 0.0_f64;
                for index in 0..shape.head_dim() {
                    let query_index =
                        token * shape.query_elements().unwrap() + head * shape.head_dim() + index;
                    let key_index = key_position * projected + kv_head * shape.head_dim() + index;
                    score += f64::from(query[query_index])
                        * f64::from(f16::from_f32(key[key_index]).to_f32());
                }
                scores.push(score / (shape.head_dim() as f64).sqrt());
            }
            let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let probabilities = scores
                .iter()
                .map(|score| (score - maximum).exp())
                .collect::<Vec<_>>();
            let denominator = probabilities.iter().sum::<f64>();
            for column in 0..shape.head_dim() {
                let result = probabilities
                    .iter()
                    .enumerate()
                    .map(|(key_position, probability)| {
                        let value_index =
                            key_position * projected + kv_head * shape.head_dim() + column;
                        probability * f64::from(f16::from_f32(value[value_index]).to_f32())
                    })
                    .sum::<f64>()
                    / denominator;
                output
                    [token * shape.query_elements().unwrap() + head * shape.head_dim() + column] =
                    result as f32;
            }
        }
    }
    output
}

fn assert_argmax_case(
    metal: &mut MetalBackend,
    cpu: &mut CpuBackend,
    name: &str,
    values: &[f32],
    expected: u32,
) {
    let layout = BufferLayout::f32(values.len()).unwrap();
    let bytes = f32_bytes(values);
    let metal_input = metal.upload(layout, &bytes).unwrap();
    let cpu_input = cpu.upload(layout, &bytes).unwrap();
    let mut metal_output = metal.allocate(BufferLayout::u32(1).unwrap()).unwrap();
    let mut cpu_output = cpu.allocate(BufferLayout::u32(1).unwrap()).unwrap();
    metal.argmax(&metal_input, &mut metal_output).unwrap();
    cpu.argmax(&cpu_input, &mut cpu_output).unwrap();
    let mut metal_index = [0_u32];
    let mut cpu_index = [0_u32];
    metal.read_u32(&metal_output, &mut metal_index).unwrap();
    cpu.read_u32(&cpu_output, &mut cpu_index).unwrap();
    assert_eq!(cpu_index[0], expected, "CPU oracle failed for {name}");
    assert_eq!(metal_index, cpu_index, "Metal argmax failed for {name}");
}

#[test]
fn parallel_argmax_matches_cpu_for_tails_ties_and_non_finite_values() {
    let Ok(mut metal) = MetalBackend::new() else {
        #[cfg(target_os = "macos")]
        panic!("Metal initialization failed on macOS");
        #[cfg(not(target_os = "macos"))]
        return;
    };
    let mut cpu = CpuBackend::new();
    let mut ties_255 = vec![0.0_f32; 255];
    ties_255[3] = 9.0;
    ties_255[254] = 9.0;
    let mut ties_256 = vec![0.0_f32; 256];
    ties_256[0] = 7.0;
    ties_256[255] = 7.0;
    let mut ties_257 = vec![0.0_f32; 257];
    ties_257[7] = 11.0;
    ties_257[256] = 11.0;
    let mut large = vec![-1.0_f32; 1378];
    large[7] = 42.0;
    large[263] = 42.0;
    large[1024] = 42.0;
    let mut model_vocabulary = vec![-1.0_f32; 131_072];
    model_vocabulary[65] = 43.0;
    model_vocabulary[65 + 256 * 300] = 43.0;
    model_vocabulary[131_071] = 42.0;
    for (name, values, expected) in [
        ("length 1", vec![f32::NEG_INFINITY], 0),
        ("length 255", ties_255, 3),
        ("length 256", ties_256, 0),
        ("length 257", ties_257, 7),
        ("large vocabulary", large, 7),
        ("model vocabulary", model_vocabulary, 65),
        ("NaN first", vec![f32::NAN, -1.0, 3.0, f32::NAN], 2),
        ("NaN later", vec![4.0, f32::NAN, 4.0, f32::NAN, 3.0], 0),
        ("all NaN", vec![f32::NAN; 257], 0),
        (
            "infinities",
            vec![
                f32::NEG_INFINITY,
                f32::INFINITY,
                f32::INFINITY,
                f32::NEG_INFINITY,
            ],
            1,
        ),
        ("signed zero", vec![-0.0, 0.0, -0.0, 0.0], 0),
    ] {
        assert_argmax_case(&mut metal, &mut cpu, name, &values, expected);
    }
}

#[test]
fn rope_embedding_argmax_and_device_increment_dispatch() {
    let Ok(mut backend) = MetalBackend::new() else {
        #[cfg(target_os = "macos")]
        panic!("Metal initialization failed on macOS");
        #[cfg(not(target_os = "macos"))]
        return;
    };
    backend
        .configure_rope(4, 10_000.0, None, RopePairing::HalfSplit)
        .unwrap();
    let mut values = backend
        .upload(
            BufferLayout::f32(4).unwrap(),
            &f32_bytes(&[1.0, 2.0, 3.0, 4.0]),
        )
        .unwrap();
    backend
        .rope(&mut values, 0, RopeShape::new(1, 1, 4).unwrap(), 10_000.0)
        .unwrap();
    let mut rope_values = [0.0; 4];
    backend.read_f32(&values, &mut rope_values).unwrap();
    assert_eq!(rope_values, [1.0, 2.0, 3.0, 4.0]);

    let table_shape = QuantMatrix::new(2, 256, QuantFormat::Q4K).unwrap();
    let mut table_bytes = vec![0_u8; table_shape.layout().unwrap().bytes()];
    for block in table_bytes.chunks_exact_mut(144) {
        block[0] = 0;
        block[1] = 0x3c;
        block[2] = 0;
        block[3] = 0x3c;
        block[4..16].fill(1);
        block[16..].fill(0x22);
    }
    let table = backend
        .upload(table_shape.layout().unwrap(), &table_bytes)
        .unwrap();
    let row = backend
        .upload(BufferLayout::u32(1).unwrap(), &1_u32.to_le_bytes())
        .unwrap();
    let mut embedding = backend.allocate(BufferLayout::f32(256).unwrap()).unwrap();
    backend
        .embed_gather(&table, &row, &mut embedding, table_shape)
        .unwrap();
    let mut embedding_values = vec![0.0; 256];
    backend.read_f32(&embedding, &mut embedding_values).unwrap();
    assert!(embedding_values[..128]
        .iter()
        .all(|value| (*value - 1.0).abs() < 0.01));
    assert!(embedding_values[128..]
        .iter()
        .all(|value| (*value - 2.0).abs() < 0.01));
    let invalid_row = backend
        .upload(BufferLayout::u32(1).unwrap(), &2_u32.to_le_bytes())
        .unwrap();
    let error = backend
        .embed_gather(&table, &invalid_row, &mut embedding, table_shape)
        .unwrap_err();
    assert!(matches!(error, leone::BackendError::RowOutOfBounds { .. }));

    let logits = backend
        .upload(
            BufferLayout::f32(4).unwrap(),
            &f32_bytes(&[1.0, 4.0, 4.0, f32::NAN]),
        )
        .unwrap();
    let mut index = backend.allocate(BufferLayout::u32(1).unwrap()).unwrap();
    backend.argmax(&logits, &mut index).unwrap();
    let mut index_value = [0_u32];
    backend.read_u32(&index, &mut index_value).unwrap();
    assert_eq!(index_value, [1]);
    let position = backend
        .upload(BufferLayout::u32(1).unwrap(), &0_u32.to_le_bytes())
        .unwrap();
    let mut position = position;
    backend.increment_u32(&mut position).unwrap();
    backend.read_u32(&position, &mut index_value).unwrap();
    assert_eq!(index_value, [1]);

    let mut max_position = backend
        .upload(BufferLayout::u32(1).unwrap(), &u32::MAX.to_le_bytes())
        .unwrap();
    let error = backend
        .increment_u32(&mut max_position)
        .expect_err("u32 position overflow must be rejected");
    assert_eq!(
        error,
        leone::BackendError::SizeOverflow {
            field: "device position"
        }
    );
    backend.read_u32(&max_position, &mut index_value).unwrap();
    assert_eq!(index_value, [u32::MAX]);

    backend
        .configure_rope(2, 10_000.0, None, RopePairing::HalfSplit)
        .unwrap();
    let mut overflow_rope = backend
        .upload(
            BufferLayout::f32(4).unwrap(),
            &f32_bytes(&[1.0, 2.0, 3.0, 4.0]),
        )
        .unwrap();
    let error = backend
        .rope(
            &mut overflow_rope,
            u32::MAX as usize,
            RopeShape::new(2, 1, 2).unwrap(),
            10_000.0,
        )
        .expect_err("RoPE position overflow must be rejected");
    assert_eq!(
        error,
        leone::BackendError::SizeOverflow {
            field: "RoPE final position"
        }
    );
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_bits().to_le_bytes())
        .collect()
}
