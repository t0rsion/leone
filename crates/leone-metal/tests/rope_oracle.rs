mod oracle_support;

use leone::{
    Backend, BackendError, BufferLayout, CpuBackend, Position, RopePairing, RopeShape, VectorShape,
};

use oracle_support::{assert_close, metal_backend, read_f32, seeded_values, upload_f32};

#[test]
fn rope_pairings_positions_and_frequency_factors_match_cpu_oracle() {
    let Some(mut metal) = metal_backend() else {
        return;
    };
    let mut cpu = CpuBackend::new();
    let theta = 10_000.0;
    let position = 8_193;
    let factors = [1.5, 0.75, 1.25, 2.0];
    let shape = RopeShape::new(3, 2, 8).expect("RoPE shape");
    let values = seeded_values(shape.elements().expect("RoPE elements"), 0x524f_5045);

    for pairing in [RopePairing::HalfSplit, RopePairing::Adjacent] {
        cpu.configure_rope(8, theta, Some(&factors), pairing)
            .expect("CPU RoPE configuration");
        metal
            .configure_rope(8, theta, Some(&factors), pairing)
            .expect("Metal RoPE configuration");
        let mut cpu_values = upload_f32(&mut cpu, &values);
        let mut metal_values = upload_f32(&mut metal, &values);
        cpu.rope(&mut cpu_values, position, shape, theta)
            .expect("CPU RoPE");
        metal
            .rope(&mut metal_values, position, shape, theta)
            .expect("Metal RoPE");
        let cpu_result = read_f32(&mut cpu, &cpu_values, values.len());
        let metal_result = read_f32(&mut metal, &metal_values, values.len());
        let name = match pairing {
            RopePairing::HalfSplit => "RoPE half-split",
            RopePairing::Adjacent => "RoPE adjacent",
        };
        assert_close(name, &metal_result, &cpu_result, 3e-3, 3e-3);
    }
}

#[test]
fn fused_rms_norm_rope_matches_cpu_oracle_at_nonzero_position() {
    let Some(mut metal) = metal_backend() else {
        return;
    };
    let mut cpu = CpuBackend::new();
    let shape = VectorShape::new(3, 8).expect("RMSNorm RoPE shape");
    let values = seeded_values(
        shape.elements().expect("RMSNorm RoPE elements"),
        0x4e4f_524d,
    );
    let weight = seeded_values(shape.columns(), 0x524f_5045)
        .into_iter()
        .map(|value| value.abs() + 0.5)
        .collect::<Vec<_>>();
    let factors = [0.75, 1.25, 1.5, 2.0];
    let theta = 10_000.0;
    let position = 8_193;
    cpu.configure_rope(8, theta, Some(&factors), RopePairing::Adjacent)
        .expect("CPU RoPE configuration");
    metal
        .configure_rope(8, theta, Some(&factors), RopePairing::Adjacent)
        .expect("Metal RoPE configuration");
    let cpu_input = upload_f32(&mut cpu, &values);
    let cpu_weight = upload_f32(&mut cpu, &weight);
    let metal_input = upload_f32(&mut metal, &values);
    let metal_weight = upload_f32(&mut metal, &weight);
    let layout = BufferLayout::f32(values.len()).expect("RMSNorm RoPE output layout");
    let mut cpu_output = cpu.allocate(layout).expect("CPU RMSNorm RoPE output");
    let mut metal_output = metal.allocate(layout).expect("Metal RMSNorm RoPE output");
    cpu.rms_norm_rope(
        &cpu_input,
        &cpu_weight,
        &mut cpu_output,
        shape,
        Position::Host(position),
        1e-5,
        theta,
    )
    .expect("CPU RMSNorm RoPE");
    metal
        .rms_norm_rope(
            &metal_input,
            &metal_weight,
            &mut metal_output,
            shape,
            Position::Host(position),
            1e-5,
            theta,
        )
        .expect("Metal RMSNorm RoPE");
    let cpu_values = read_f32(&mut cpu, &cpu_output, values.len());
    let metal_values = read_f32(&mut metal, &metal_output, values.len());
    assert_close("fused RMSNorm RoPE", &metal_values, &cpu_values, 4e-3, 4e-3);
}

#[test]
fn rope_configuration_and_unconfigured_calls_return_typed_errors() {
    let Some(mut metal) = metal_backend() else {
        return;
    };
    let mut cpu = CpuBackend::new();
    let values = seeded_values(8, 0x4552_524f);
    let mut cpu_values = upload_f32(&mut cpu, &values);
    let mut metal_values = upload_f32(&mut metal, &values);
    let shape = RopeShape::new(1, 1, 8).expect("RoPE shape");
    let cpu_error = cpu
        .rope(&mut cpu_values, 1, shape, 10_000.0)
        .expect_err("CPU must reject unconfigured RoPE");
    let metal_error = metal
        .rope(&mut metal_values, 1, shape, 10_000.0)
        .expect_err("Metal must reject unconfigured RoPE");
    assert!(matches!(
        cpu_error,
        BackendError::SizeMismatch {
            name: "RoPE inverse frequencies",
            ..
        }
    ));
    assert!(matches!(
        metal_error,
        BackendError::SizeMismatch {
            name: "RoPE inverse frequencies",
            ..
        }
    ));

    let factors = [1.0, 2.0, 3.0];
    let error = metal
        .configure_rope(8, 10_000.0, Some(&factors), RopePairing::HalfSplit)
        .expect_err("RoPE factor length must match half dimension");
    assert_eq!(
        error,
        BackendError::SizeMismatch {
            name: "RoPE frequency factors",
            expected: 4,
            actual: 3,
        }
    );
}
