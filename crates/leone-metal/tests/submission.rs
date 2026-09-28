mod oracle_support;

use half::f16;
use leone::{AttentionShape, Backend, BufferLayout, Position, QuantFormat, QuantMatrix};

use oracle_support::{
    assert_close, decode_rows, metal_backend, read_f16, read_f32, upload_f32, upload_u32,
};

#[test]
fn residual_add_pingpong_crosses_pending_dispatch_limit() {
    let Some(mut metal) = metal_backend() else {
        return;
    };
    let left_values = [1.0, 2.0, 4.0, 8.0];
    let increment_values = [1.0; 4];
    let left = upload_f32(&mut metal, &left_values);
    let increment = upload_f32(&mut metal, &increment_values);
    let layout = BufferLayout::f32(left_values.len()).expect("residual layout");
    let mut ping = metal.allocate(layout).expect("ping output");
    let mut pong = metal.allocate(layout).expect("pong output");

    metal
        .residual_add(&left, &increment, &mut ping)
        .expect("first residual add");
    for _ in 0..32 {
        metal
            .residual_add(&ping, &increment, &mut pong)
            .expect("ping to pong residual add");
        metal
            .residual_add(&pong, &increment, &mut ping)
            .expect("pong to ping residual add");
    }
    metal
        .residual_add(&ping, &increment, &mut pong)
        .expect("final residual add");

    let actual = read_f32(&mut metal, &pong, left_values.len());
    assert_eq!(actual, [67.0, 68.0, 70.0, 74.0]);
}

#[test]
fn host_overwrite_waits_for_queued_embedding() {
    let Some(mut metal) = metal_backend() else {
        return;
    };
    let shape = QuantMatrix::new(2, 256, QuantFormat::Q4K).expect("embedding shape");
    let mut table_bytes = vec![0_u8; shape.layout().expect("embedding layout").bytes()];
    for (row, block) in table_bytes.chunks_exact_mut(144).enumerate() {
        block[0] = 0;
        block[1] = 0x3c;
        block[2] = 0;
        block[3] = 0x3c;
        block[4..16].fill(1);
        block[16..].fill(if row == 0 { 0x11 } else { 0x22 });
    }
    let expected_rows = decode_rows(shape.format(), &table_bytes, shape.rows(), shape.columns());
    assert_ne!(
        &expected_rows[..shape.columns()],
        &expected_rows[shape.columns()..2 * shape.columns()]
    );
    let table = metal
        .upload(shape.layout().expect("embedding layout"), &table_bytes)
        .expect("embedding table");
    let mut row = upload_u32(&mut metal, &[1]);
    let mut output = metal
        .allocate(BufferLayout::f32(shape.columns()).expect("embedding output layout"))
        .expect("embedding output");

    metal
        .embed_gather(&table, &row, &mut output, shape)
        .expect("queue embedding row");
    metal
        .write_u32(&mut row, &[0])
        .expect("overwrite embedding row");

    let actual = read_f32(&mut metal, &output, shape.columns());
    assert_close(
        "queued embedding",
        &actual,
        &expected_rows[shape.columns()..2 * shape.columns()],
        0.0,
        0.0,
    );
}

#[test]
fn queued_producer_clone_and_f16_clone_preserve_values() {
    let Some(mut metal) = metal_backend() else {
        return;
    };
    let left_values = [2.0, 4.0, 8.0, 16.0];
    let right_values = [1.0, 2.0, 4.0, 8.0];
    let left = upload_f32(&mut metal, &left_values);
    let right = upload_f32(&mut metal, &right_values);
    let mut produced = metal
        .allocate(BufferLayout::f32(left_values.len()).expect("producer layout"))
        .expect("producer output");
    metal
        .residual_add(&left, &right, &mut produced)
        .expect("queue producer");
    let clone = metal.clone_buffer(&produced).expect("aligned clone");
    assert_eq!(
        read_f32(&mut metal, &clone, left_values.len()),
        [3.0, 6.0, 12.0, 24.0]
    );

    let shape = AttentionShape::new(1, 1, 1, 1).expect("KV shape");
    let key = upload_f32(&mut metal, &[1.5]);
    let value = upload_f32(&mut metal, &[2.5]);
    let cache_layout = BufferLayout::f16(shape.cache_elements().expect("cache elements"))
        .expect("F16 cache layout");
    let mut key_cache = metal.allocate(cache_layout).expect("key cache");
    let mut value_cache = metal.allocate(cache_layout).expect("value cache");
    metal
        .kv_append(
            &key,
            &value,
            &mut key_cache,
            &mut value_cache,
            shape,
            Position::Host(0),
        )
        .expect("queue F16 producer");
    let f16_clone = metal.clone_buffer(&key_cache).expect("unaligned F16 clone");
    let actual = read_f16(&mut metal, &f16_clone, 1);
    assert_eq!(actual, [f32::from(f16::from_f32(1.5))]);
}

#[test]
fn dropping_queued_input_releases_memory_and_keeps_output() {
    let Some(mut metal) = metal_backend() else {
        return;
    };
    let baseline = metal.memory_accounting().live_bytes;
    let input = upload_f32(&mut metal, &[2.0, 4.0]);
    let right = upload_f32(&mut metal, &[3.0, 5.0]);
    let mut output = metal
        .allocate(BufferLayout::f32(2).expect("output layout"))
        .expect("output");
    let with_buffers = metal.memory_accounting().live_bytes;

    metal
        .residual_add(&input, &right, &mut output)
        .expect("queue residual add");
    drop(input);
    assert_eq!(
        metal.memory_accounting().live_bytes,
        with_buffers - BufferLayout::f32(2).expect("input layout").bytes() as u64
    );
    assert_eq!(read_f32(&mut metal, &output, 2), [5.0, 9.0]);

    drop(right);
    drop(output);
    assert_eq!(metal.memory_accounting().live_bytes, baseline);
}

#[test]
fn backend_drop_before_buffers_is_harmless() {
    let Some(mut metal) = metal_backend() else {
        return;
    };
    let input = upload_f32(&mut metal, &[1.0, 2.0]);
    let increment = upload_f32(&mut metal, &[3.0, 4.0]);
    let mut output = metal
        .allocate(BufferLayout::f32(2).expect("output layout"))
        .expect("output");
    metal
        .residual_add(&input, &increment, &mut output)
        .expect("queue residual add");
    drop(metal);
    drop(input);
    drop(increment);
    drop(output);
}
