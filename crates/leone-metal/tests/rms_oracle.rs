mod oracle_support;

use leone::{Backend, BackendError, BufferLayout, VectorShape};
use oracle_support::{metal_backend, read_f32, upload_f32};

fn rms_oracle(input: &[f32], weight: &[f32], shape: VectorShape, epsilon: f32) -> Vec<f32> {
    let columns = shape.columns();
    input
        .chunks_exact(columns)
        .flat_map(|row| {
            let sum = row.iter().fold(0.0_f64, |sum, value| {
                let value = f64::from(*value);
                sum + value * value
            });
            let inverse = (sum / columns as f64 + f64::from(epsilon)).sqrt().recip();
            row.iter()
                .zip(weight)
                .map(move |(value, scale)| (f64::from(*value) * f64::from(*scale) * inverse) as f32)
        })
        .collect()
}

fn rms_inputs(rows: usize, columns: usize, case: usize) -> Vec<f32> {
    (0..rows * columns)
        .map(|index| {
            let sign = if index.is_multiple_of(2) { 1.0 } else { -1.0 };
            match case {
                0 => 0.0,
                1 => sign * (1e-8 + (index % 7) as f32 * 1e-9),
                _ => {
                    let exponent = (index % 61) as i32 - 30;
                    sign * 2.0_f32.powi(exponent)
                }
            }
        })
        .collect()
}

fn rms_weights(columns: usize) -> Vec<f32> {
    (0..columns)
        .map(|index| 0.5 + (index % 11) as f32 * 0.125)
        .collect()
}

fn assert_oracle_close(actual: &[f32], expected: &[f32], absolute: f32, relative: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite(),
            "actual RMSNorm value {index} is {actual}"
        );
        assert!(
            expected.is_finite(),
            "oracle RMSNorm value {index} is {expected}"
        );
        let error = (actual - expected).abs();
        let bound = absolute + relative * expected.abs();
        assert!(
            error <= bound,
            "RMSNorm value {index} differs: actual={actual:e}, expected={expected:e}, error={error:e}, bound={bound:e}"
        );
    }
}

#[test]
fn plain_rms_norm_matches_fp64_oracle_at_row_and_column_boundaries() {
    let Some(mut metal) = metal_backend() else {
        return;
    };
    for (rows, columns, case, epsilon) in [(1, 1, 0, 1e-5), (3, 257, 1, 1e-5), (4, 4096, 2, 1e-5)] {
        let shape = VectorShape::new(rows, columns).expect("RMSNorm shape");
        let input = rms_inputs(rows, columns, case);
        let weight = rms_weights(columns);
        let input_buffer = upload_f32(&mut metal, &input);
        let weight_buffer = upload_f32(&mut metal, &weight);
        let mut output = metal
            .allocate(BufferLayout::f32(input.len()).expect("RMSNorm output layout"))
            .expect("RMSNorm output");
        metal
            .prefill_rms_norm(&input_buffer, &weight_buffer, &mut output, shape, epsilon)
            .expect("Metal RMSNorm");
        let actual = read_f32(&mut metal, &output, input.len());
        let expected = rms_oracle(&input, &weight, shape, epsilon);
        assert_oracle_close(&actual, &expected, 2e-5, 2e-4);
    }
}

#[test]
fn plain_rms_norm_reads_the_last_column_at_threadgroup_boundaries() {
    let Some(mut metal) = metal_backend() else {
        return;
    };
    for columns in [255, 256, 257, 511, 512, 513] {
        let shape = VectorShape::new(1, columns).expect("RMSNorm shape");
        let mut input = vec![0.0; columns];
        input[columns - 1] = 1.0;
        let weight = rms_weights(columns);
        let input_buffer = upload_f32(&mut metal, &input);
        let weight_buffer = upload_f32(&mut metal, &weight);
        let mut output = metal
            .allocate(BufferLayout::f32(columns).expect("RMSNorm output layout"))
            .expect("RMSNorm output");
        metal
            .prefill_rms_norm(&input_buffer, &weight_buffer, &mut output, shape, 1e-5)
            .expect("Metal RMSNorm");
        let actual = read_f32(&mut metal, &output, columns);
        let expected = rms_oracle(&input, &weight, shape, 1e-5);
        assert_oracle_close(&actual, &expected, 2e-5, 2e-4);
    }
}

#[test]
fn plain_rms_norm_maps_distinct_single_column_rows() {
    let Some(mut metal) = metal_backend() else {
        return;
    };
    let shape = VectorShape::new(4, 1).expect("RMSNorm shape");
    let input = [1.0, -2.0, 4.0, -8.0];
    let weight = [1.25];
    let input_buffer = upload_f32(&mut metal, &input);
    let weight_buffer = upload_f32(&mut metal, &weight);
    let mut output = metal
        .allocate(BufferLayout::f32(input.len()).expect("RMSNorm output layout"))
        .expect("RMSNorm output");
    metal
        .prefill_rms_norm(&input_buffer, &weight_buffer, &mut output, shape, 1.0)
        .expect("Metal RMSNorm");
    let actual = read_f32(&mut metal, &output, input.len());
    let expected = rms_oracle(&input, &weight, shape, 1.0);
    assert_oracle_close(&actual, &expected, 2e-5, 2e-4);
}

#[test]
fn plain_rms_norm_keeps_signal_below_epsilon() {
    let Some(mut metal) = metal_backend() else {
        return;
    };
    let shape = VectorShape::new(2, 257).expect("RMSNorm shape");
    let input = rms_inputs(2, 257, 1);
    let weight = rms_weights(257);
    let input_buffer = upload_f32(&mut metal, &input);
    let weight_buffer = upload_f32(&mut metal, &weight);
    let mut output = metal
        .allocate(BufferLayout::f32(input.len()).expect("RMSNorm output layout"))
        .expect("RMSNorm output");
    metal
        .prefill_rms_norm(&input_buffer, &weight_buffer, &mut output, shape, 1e-5)
        .expect("Metal RMSNorm");
    let actual = read_f32(&mut metal, &output, input.len());
    let expected = rms_oracle(&input, &weight, shape, 1e-5);
    assert_oracle_close(&actual, &expected, 1e-9, 2e-3);
}

#[test]
fn plain_rms_norm_handles_nonfinite_inputs_without_changing_contract() {
    let Some(mut metal) = metal_backend() else {
        return;
    };
    let shape = VectorShape::new(1, 4).expect("RMSNorm shape");
    let weight = upload_f32(&mut metal, &[1.0; 4]);
    for input in [
        [f32::NAN, 1.0, -2.0, 3.0],
        [1.0, -2.0, 3.0, f32::NAN],
        [f32::INFINITY, 1.0, -2.0, 3.0],
        [1.0, -2.0, 3.0, f32::INFINITY],
        [-f32::INFINITY, 1.0, -2.0, 3.0],
        [1.0, -2.0, 3.0, -f32::INFINITY],
    ] {
        let input_buffer = upload_f32(&mut metal, &input);
        let mut output = metal
            .allocate(BufferLayout::f32(4).expect("RMSNorm output layout"))
            .expect("RMSNorm output");
        metal
            .rms_norm(&input_buffer, &weight, &mut output, shape, 1e-5)
            .expect("Metal RMSNorm");
        let actual = read_f32(&mut metal, &output, 4);
        if input.iter().any(|value| value.is_nan()) {
            assert!(actual.iter().all(|value| value.is_nan()), "{actual:?}");
        } else {
            let infinite = input
                .iter()
                .position(|value| value.is_infinite())
                .expect("infinite input");
            assert!(
                actual[infinite].is_nan(),
                "infinite input must remain nonfinite"
            );
            assert!(
                actual
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| *index != infinite)
                    .all(|(_, value)| *value == 0.0),
                "{actual:?}"
            );
        }
    }
}

#[test]
fn plain_rms_norm_preserves_finite_square_overflow_behavior() {
    let Some(mut metal) = metal_backend() else {
        return;
    };
    let shape = VectorShape::new(1, 4).expect("RMSNorm shape");
    let input = upload_f32(&mut metal, &[1.0e20, 1.0, -2.0, 3.0]);
    let weight = upload_f32(&mut metal, &[1.0; 4]);
    let mut output = metal
        .allocate(BufferLayout::f32(4).expect("RMSNorm output layout"))
        .expect("RMSNorm output");
    metal
        .rms_norm(&input, &weight, &mut output, shape, 1e-5)
        .expect("Metal RMSNorm");
    let actual = read_f32(&mut metal, &output, 4);
    assert!(actual.iter().all(|value| *value == 0.0), "{actual:?}");
}

#[test]
fn plain_rms_norm_keeps_exact_layout_checks() {
    let Some(mut metal) = metal_backend() else {
        return;
    };
    let shape = VectorShape::new(2, 3).expect("RMSNorm shape");
    let input = upload_f32(&mut metal, &[1.0; 6]);
    let weight = upload_f32(&mut metal, &[1.0; 3]);
    let mut output = metal
        .allocate(BufferLayout::f32(5).expect("wrong RMSNorm output layout"))
        .expect("RMSNorm output");
    let error = metal
        .rms_norm(&input, &weight, &mut output, shape, 1e-5)
        .expect_err("RMSNorm must reject a wrong output layout");
    assert!(matches!(
        error,
        BackendError::SizeMismatch {
            name: "RMSNorm output",
            expected: 6,
            actual: 5
        }
    ));
}
