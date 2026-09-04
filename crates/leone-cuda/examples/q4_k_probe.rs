use leone_cuda::{
    launch_q4_k_apron_pair_probe, launch_q4_k_probe, launch_q4_k_ring_probe, prepare_q4_k_probe,
    Context, Event, GemvScratch, Q4KProbeGeometry, QuantFormat, QuantizedMatrixShape, Stream,
};
use std::error::Error;

const COLUMNS: usize = 4_096;
const RING_TARGET_BYTES: usize = 192 * 1024 * 1024;
const WARMUP_LAUNCHES: usize = 64;
const TIMED_LAUNCHES: usize = 1_000;
const REPS: usize = 5;
const BOUNDARY_WARMUP_CYCLES: usize = 8;
const BOUNDARY_TIMED_CYCLES: usize = 64;
const PERSISTENT_GRID_BLOCKS: usize = 1_536;
const APRON_WARMUP_PAIRS: usize = 64;
const APRON_TIMED_PAIRS: usize = 1_000;
const APRON_BYTES: [usize; 6] = [
    0,
    256 * 1024,
    512 * 1024,
    1024 * 1024,
    2 * 1024 * 1024,
    4 * 1024 * 1024,
];

#[derive(Clone, Copy)]
enum ProbeLayout {
    TensorSplit,
    RowInterleaved,
    Gguf,
}

impl ProbeLayout {
    const fn name(self) -> &'static str {
        match self {
            Self::TensorSplit => "tensor_split",
            Self::RowInterleaved => "row_interleaved",
            Self::Gguf => "gguf",
        }
    }
}

struct ApronState {
    first_weights: leone_cuda::DeviceBuffer<u8>,
    second_weights: leone_cuda::DeviceBuffer<u8>,
    third_weights: leone_cuda::DeviceBuffer<u8>,
    output: leone_cuda::DeviceBuffer<f32>,
    scratch: GemvScratch,
}

struct ApronWeights {
    first_weights: leone_cuda::DeviceBuffer<u8>,
    second_weights: leone_cuda::DeviceBuffer<u8>,
    third_weights: leone_cuda::DeviceBuffer<u8>,
}

struct BoundaryState {
    weights: leone_cuda::DeviceBuffer<u8>,
    output: leone_cuda::DeviceBuffer<f32>,
    scratch: GemvScratch,
}

struct ApronProbeArgs<'a> {
    stream: &'a Stream,
    first_weights: &'a leone_cuda::DeviceBuffer<u8>,
    second_weights: &'a leone_cuda::DeviceBuffer<u8>,
    third_weights: &'a leone_cuda::DeviceBuffer<u8>,
    output: &'a mut leone_cuda::DeviceBuffer<f32>,
    scratch: &'a GemvScratch,
    shape: QuantizedMatrixShape,
    weight_sets: usize,
}

struct BoundaryMeasureArgs<'a> {
    context: &'a Context,
    stream: &'a Stream,
    weights: &'a leone_cuda::DeviceBuffer<u8>,
    output: &'a mut leone_cuda::DeviceBuffer<f32>,
    scratch: &'a GemvScratch,
    shape: QuantizedMatrixShape,
    weight_sets: usize,
}

struct ShapeProbeArgs<'a> {
    context: &'a Context,
    stream: &'a Stream,
    weights: &'a leone_cuda::DeviceBuffer<u8>,
    output: &'a mut leone_cuda::DeviceBuffer<f32>,
    scratch: &'a GemvScratch,
    shape: QuantizedMatrixShape,
    weight_sets: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let context = Context::new(0)?;
    let stream = Stream::new(&context)?;
    let input = (0..COLUMNS)
        .map(|index| (index as f32 * 0.001_953_125).sin())
        .collect::<Vec<_>>();
    let d_input = context.copy_to_device(&input)?;

    match std::env::args().nth(1).as_deref() {
        Some("--boundary") => run_boundary_refutation(&context, &stream, &d_input),
        Some("--apron") => run_apron_probe(&context, &stream, &d_input),
        _ => run_shapes(&context, &stream, &d_input),
    }
}

fn run_shapes(
    context: &Context,
    stream: &Stream,
    input: &leone_cuda::DeviceBuffer<f32>,
) -> Result<(), Box<dyn Error>> {
    println!("Q4_K counter-free GEMV probe");
    println!("columns: {COLUMNS}");
    println!("weight_ring_target_bytes: {RING_TARGET_BYTES}");
    println!("warmup_launches: {WARMUP_LAUNCHES}");
    println!("timed_launches_per_rep: {TIMED_LAUNCHES}");
    println!("reps: {REPS}");
    println!("traffic: logical Q4_K matrix bytes, weight ring exceeds L2");

    run_shape(
        context,
        stream,
        input,
        4_096,
        &[
            (
                "production_w4_r1",
                Q4KProbeGeometry::ProductionFourWarpsOneRow,
            ),
            ("wide_w4_r1", Q4KProbeGeometry::WideFourWarpsOneRow),
            (
                "two_rows_per_warp_group",
                Q4KProbeGeometry::TwoRowsPerWarpGroup,
            ),
            ("split_k_two_ctas", Q4KProbeGeometry::SplitKTwoCtasPerRow),
            (
                "two_block_ilp_w4_r1",
                Q4KProbeGeometry::TwoBlockIlpFourWarpsOneRow,
            ),
            ("full_w1_r4", Q4KProbeGeometry::OneWarpFourRows),
            (
                "loads_only_w4_r1",
                Q4KProbeGeometry::LoadsOnlyFourWarpsOneRow,
            ),
            ("full_w2_r1", Q4KProbeGeometry::TwoWarpsOneRow),
            ("full_w3_r1", Q4KProbeGeometry::ThreeWarpsOneRow),
            ("full_w4_r1", Q4KProbeGeometry::FourWarpsOneRow),
            ("full_w2_r2", Q4KProbeGeometry::TwoWarpsTwoRows),
            ("full_w3_r2", Q4KProbeGeometry::ThreeWarpsTwoRows),
            ("full_w4_r2", Q4KProbeGeometry::FourWarpsTwoRows),
            ("full_w2_r4", Q4KProbeGeometry::TwoWarpsFourRows),
            ("full_w3_r4", Q4KProbeGeometry::ThreeWarpsFourRows),
            ("full_w4_r4", Q4KProbeGeometry::FourWarpsFourRows),
        ],
        ProbeLayout::TensorSplit,
    )?;
    run_shape(
        context,
        stream,
        input,
        12_288,
        &[
            (
                "production_w4_r1",
                Q4KProbeGeometry::ProductionFourWarpsOneRow,
            ),
            ("wide_w4_r1", Q4KProbeGeometry::WideFourWarpsOneRow),
            (
                "two_rows_per_warp_group",
                Q4KProbeGeometry::TwoRowsPerWarpGroup,
            ),
            ("split_k_two_ctas", Q4KProbeGeometry::SplitKTwoCtasPerRow),
            (
                "two_block_ilp_w4_r1",
                Q4KProbeGeometry::TwoBlockIlpFourWarpsOneRow,
            ),
            ("full_w1_r4", Q4KProbeGeometry::OneWarpFourRows),
        ],
        ProbeLayout::TensorSplit,
    )?;
    run_shape(
        context,
        stream,
        input,
        4_096,
        &[(
            "row_interleaved_w4_r1",
            Q4KProbeGeometry::RowInterleavedFourWarpsOneRow,
        )],
        ProbeLayout::RowInterleaved,
    )?;
    run_shape(
        context,
        stream,
        input,
        12_288,
        &[(
            "row_interleaved_w4_r1",
            Q4KProbeGeometry::RowInterleavedFourWarpsOneRow,
        )],
        ProbeLayout::RowInterleaved,
    )?;
    run_shape(
        context,
        stream,
        input,
        4_096,
        &[
            ("gguf_w4_r1", Q4KProbeGeometry::GgufFourWarpsOneRow),
            ("gguf_wide_w4_r1", Q4KProbeGeometry::GgufWideFourWarpsOneRow),
            ("gguf_wide_w1_r4", Q4KProbeGeometry::GgufWideOneWarpFourRows),
        ],
        ProbeLayout::Gguf,
    )?;
    run_shape(
        context,
        stream,
        input,
        12_288,
        &[
            ("gguf_w4_r1", Q4KProbeGeometry::GgufFourWarpsOneRow),
            ("gguf_wide_w4_r1", Q4KProbeGeometry::GgufWideFourWarpsOneRow),
            ("gguf_wide_w1_r4", Q4KProbeGeometry::GgufWideOneWarpFourRows),
        ],
        ProbeLayout::Gguf,
    )?;
    Ok(())
}

fn run_apron_probe(
    context: &Context,
    stream: &Stream,
    input: &leone_cuda::DeviceBuffer<f32>,
) -> Result<(), Box<dyn Error>> {
    let rows = 4_096;
    let shape = QuantizedMatrixShape::new(rows, COLUMNS, QuantFormat::Q4K)?;
    let next_shape = QuantizedMatrixShape::new(12_288, COLUMNS, QuantFormat::Q4K)?;
    let weight_sets = RING_TARGET_BYTES.div_ceil(shape.bytes());
    let ApronState {
        first_weights,
        second_weights,
        third_weights,
        mut output,
        scratch,
    } = prepare_apron_state(context, stream, input, shape, next_shape, weight_sets)?;

    println!("Q4_K tail-filled trip-major L2 apron probe");
    println!("first_shape: output projection rows={rows} columns={COLUMNS} layout=tensor_split");
    println!(
        "second_shape: paired gate and up rows={}+{} columns={COLUMNS} layout=tensor_split",
        next_shape.rows(),
        next_shape.rows()
    );
    println!("first_matrix_bytes: {}", shape.bytes());
    println!("next_matrix_bytes_each: {}", next_shape.bytes());
    println!("weight_sets: {weight_sets}");
    println!("first_ring_bytes: {}", shape.bytes() * weight_sets);
    println!("next_ring_bytes_each: {}", next_shape.bytes() * weight_sets);
    println!("prefetch_ctas: 512 for every nonzero apron");
    println!("prefetch_order: next gate matrix first-trip packets by row");
    println!("prefetch_policy: L2 evict_last");
    println!("weight_policy: evict_first");
    println!("compute_path: unchanged production kernels");
    println!("warmup_pairs_per_size: {APRON_WARMUP_PAIRS}");
    println!("timed_pairs_per_rep: {APRON_TIMED_PAIRS}");
    println!("reps: {REPS}");
    println!("win_criterion: median pair delta versus zero apron is negative");

    let mut probe = ApronProbeArgs {
        stream,
        first_weights: &first_weights,
        second_weights: &second_weights,
        third_weights: &third_weights,
        output: &mut output,
        scratch: &scratch,
        shape,
        weight_sets,
    };
    warm_apron_pairs(&mut probe)?;
    stream.synchronize()?;

    let samples = measure_apron_pairs(context, &mut probe)?;
    report_apron_pairs(&samples);
    Ok(())
}

fn prepare_apron_state(
    context: &Context,
    stream: &Stream,
    input: &leone_cuda::DeviceBuffer<f32>,
    shape: QuantizedMatrixShape,
    next_shape: QuantizedMatrixShape,
    weight_sets: usize,
) -> Result<ApronState, Box<dyn Error>> {
    let ApronWeights {
        first_weights,
        second_weights,
        third_weights,
    } = prepare_apron_weights(context, shape, next_shape, weight_sets)?;
    let output = context.alloc::<f32>(2 * next_shape.rows())?;
    let mut scratch = GemvScratch::new(context, shape)?;
    prepare_q4_k_probe(stream, input, &mut scratch)?;
    stream.synchronize()?;
    Ok(ApronState {
        first_weights,
        second_weights,
        third_weights,
        output,
        scratch,
    })
}

fn prepare_apron_weights(
    context: &Context,
    shape: QuantizedMatrixShape,
    next_shape: QuantizedMatrixShape,
    weight_sets: usize,
) -> Result<ApronWeights, Box<dyn Error>> {
    let first_matrix = probe_matrix(shape, ProbeLayout::TensorSplit);
    let next_matrix = probe_matrix(next_shape, ProbeLayout::TensorSplit);
    let mut first_ring = Vec::with_capacity(shape.bytes() * weight_sets);
    let mut second_ring = Vec::with_capacity(next_shape.bytes() * weight_sets);
    let mut third_ring = Vec::with_capacity(next_shape.bytes() * weight_sets);
    for _ in 0..weight_sets {
        first_ring.extend_from_slice(&first_matrix);
        second_ring.extend_from_slice(&next_matrix);
        third_ring.extend_from_slice(&next_matrix);
    }
    Ok(ApronWeights {
        first_weights: context.copy_to_device(&first_ring)?,
        second_weights: context.copy_to_device(&second_ring)?,
        third_weights: context.copy_to_device(&third_ring)?,
    })
}

fn warm_apron_pairs(probe: &mut ApronProbeArgs<'_>) -> Result<(), Box<dyn Error>> {
    for &apron_bytes in &APRON_BYTES {
        for pair in 0..APRON_WARMUP_PAIRS {
            launch_q4_k_apron_pair_probe(
                probe.stream,
                probe.first_weights,
                probe.second_weights,
                probe.third_weights,
                probe.output,
                probe.scratch,
                probe.shape,
                pair % probe.weight_sets,
                apron_bytes,
            )?;
        }
    }
    Ok(())
}

fn measure_apron_pairs(
    context: &Context,
    probe: &mut ApronProbeArgs<'_>,
) -> Result<Vec<Vec<f64>>, Box<dyn Error>> {
    let mut samples = vec![Vec::with_capacity(REPS); APRON_BYTES.len()];
    for rep in 0..REPS {
        measure_apron_rep(context, probe, rep, &mut samples)?;
    }
    Ok(samples)
}

fn measure_apron_rep(
    context: &Context,
    probe: &mut ApronProbeArgs<'_>,
    rep: usize,
    samples: &mut [Vec<f64>],
) -> Result<(), Box<dyn Error>> {
    let order: Vec<usize> = if rep.is_multiple_of(2) {
        (0..APRON_BYTES.len()).collect()
    } else {
        (0..APRON_BYTES.len()).rev().collect()
    };
    for index in order {
        let apron_bytes = APRON_BYTES[index];
        let us_per_pair = measure_apron_size(context, probe, apron_bytes)?;
        samples[index].push(us_per_pair);
        println!(
            "rep={} apron_bytes={} pair_us={us_per_pair:.6}",
            rep + 1,
            apron_bytes
        );
    }
    Ok(())
}

fn measure_apron_size(
    context: &Context,
    probe: &mut ApronProbeArgs<'_>,
    apron_bytes: usize,
) -> Result<f64, Box<dyn Error>> {
    let mut start = Event::new(context)?;
    let mut end = Event::new(context)?;
    start.record(probe.stream)?;
    for pair in 0..APRON_TIMED_PAIRS {
        launch_q4_k_apron_pair_probe(
            probe.stream,
            probe.first_weights,
            probe.second_weights,
            probe.third_weights,
            probe.output,
            probe.scratch,
            probe.shape,
            pair % probe.weight_sets,
            apron_bytes,
        )?;
    }
    end.record(probe.stream)?;
    end.synchronize()?;
    Ok(f64::from(Event::elapsed_ms(&start, &end)?) * 1_000.0 / APRON_TIMED_PAIRS as f64)
}

fn report_apron_pairs(samples: &[Vec<f64>]) {
    let medians: Vec<_> = samples
        .iter()
        .map(|values| median(&mut values.clone()))
        .collect();
    let baseline = medians[0];
    let mut wins = false;
    for (index, &apron_bytes) in APRON_BYTES.iter().enumerate() {
        let delta = medians[index] - baseline;
        wins |= apron_bytes != 0 && delta < 0.0;
        println!(
            "apron_bytes={apron_bytes} median_pair_us={:.6} pair_delta_us={delta:+.6} pair_us_samples={:?}",
            medians[index], samples[index]
        );
    }
    println!("apron_probe={}", if wins { "wins" } else { "loses" });
}

fn run_boundary_refutation(
    context: &Context,
    stream: &Stream,
    input: &leone_cuda::DeviceBuffer<f32>,
) -> Result<(), Box<dyn Error>> {
    let rows = 4_096;
    let shape = QuantizedMatrixShape::new(rows, COLUMNS, QuantFormat::Q4K)?;
    let weight_sets = RING_TARGET_BYTES.div_ceil(shape.bytes());
    let BoundaryState {
        weights: d_weights,
        output: mut d_output,
        scratch,
    } = prepare_boundary_state(context, stream, input, shape, weight_sets, rows)?;

    println!("Q4_K short-kernel boundary refutation probe");
    println!(
        "threshold: confirm if one-launch >=900.0 GB/s and >=1.10x the matched separate-launch median; refute otherwise"
    );
    println!("columns: {COLUMNS}");
    println!("rows: {rows}");
    println!("matrix_bytes: {}", shape.bytes());
    println!("weight_sets: {weight_sets}");
    println!("ring_bytes: {}", shape.bytes() * weight_sets);
    println!("persistent_grid_blocks: {PERSISTENT_GRID_BLOCKS}");
    println!("warmup_cycles: {BOUNDARY_WARMUP_CYCLES}");
    println!("timed_cycles_per_rep: {BOUNDARY_TIMED_CYCLES}");
    println!("reps: {REPS}");
    println!("traffic: identical logical Q4_K ring bytes on both paths");

    warm_boundary_cycles(
        stream,
        &d_weights,
        &mut d_output,
        &scratch,
        shape,
        weight_sets,
    )?;
    stream.synchronize()?;

    let (separate_samples, persistent_samples) = measure_boundary_cycles(
        context,
        stream,
        &d_weights,
        &mut d_output,
        &scratch,
        shape,
        weight_sets,
    )?;
    report_boundary_cycles(separate_samples, persistent_samples);
    Ok(())
}

fn prepare_boundary_state(
    context: &Context,
    stream: &Stream,
    input: &leone_cuda::DeviceBuffer<f32>,
    shape: QuantizedMatrixShape,
    weight_sets: usize,
    rows: usize,
) -> Result<BoundaryState, Box<dyn Error>> {
    let matrix = probe_matrix(shape, ProbeLayout::TensorSplit);
    let mut ring = Vec::with_capacity(shape.bytes() * weight_sets);
    for _ in 0..weight_sets {
        ring.extend_from_slice(&matrix);
    }
    let d_weights = context.copy_to_device(&ring)?;
    let d_output = context.alloc::<f32>(rows * weight_sets)?;
    let mut scratch = GemvScratch::new(context, shape)?;
    prepare_q4_k_probe(stream, input, &mut scratch)?;
    stream.synchronize()?;
    Ok(BoundaryState {
        weights: d_weights,
        output: d_output,
        scratch,
    })
}

fn warm_boundary_cycles(
    stream: &Stream,
    weights: &leone_cuda::DeviceBuffer<u8>,
    output: &mut leone_cuda::DeviceBuffer<f32>,
    scratch: &GemvScratch,
    shape: QuantizedMatrixShape,
    weight_sets: usize,
) -> Result<(), Box<dyn Error>> {
    for _ in 0..BOUNDARY_WARMUP_CYCLES {
        launch_separate_ring(stream, weights, output, scratch, shape, weight_sets)?;
        launch_q4_k_ring_probe(stream, weights, output, scratch, shape, weight_sets)?;
    }
    Ok(())
}

fn measure_boundary_cycles(
    context: &Context,
    stream: &Stream,
    weights: &leone_cuda::DeviceBuffer<u8>,
    output: &mut leone_cuda::DeviceBuffer<f32>,
    scratch: &GemvScratch,
    shape: QuantizedMatrixShape,
    weight_sets: usize,
) -> Result<(Vec<f64>, Vec<f64>), Box<dyn Error>> {
    let mut separate_samples = Vec::with_capacity(REPS);
    let mut persistent_samples = Vec::with_capacity(REPS);
    let mut measure = BoundaryMeasureArgs {
        context,
        stream,
        weights,
        output,
        scratch,
        shape,
        weight_sets,
    };
    for rep in 0..REPS {
        let (separate, persistent) = measure_boundary_pair(&mut measure, rep)?;
        separate_samples.push(separate);
        persistent_samples.push(persistent);
        println!(
            "pair={} separate_launch_gbs={separate:.3} one_launch_gbs={persistent:.3}",
            rep + 1
        );
    }
    Ok((separate_samples, persistent_samples))
}

fn measure_boundary_pair(
    measure: &mut BoundaryMeasureArgs<'_>,
    rep: usize,
) -> Result<(f64, f64), Box<dyn Error>> {
    if rep.is_multiple_of(2) {
        Ok((
            time_separate_ring(
                measure.context,
                measure.stream,
                measure.weights,
                measure.output,
                measure.scratch,
                measure.shape,
                measure.weight_sets,
            )?,
            time_persistent_ring(
                measure.context,
                measure.stream,
                measure.weights,
                measure.output,
                measure.scratch,
                measure.shape,
                measure.weight_sets,
            )?,
        ))
    } else {
        let persistent = time_persistent_ring(
            measure.context,
            measure.stream,
            measure.weights,
            measure.output,
            measure.scratch,
            measure.shape,
            measure.weight_sets,
        )?;
        let separate = time_separate_ring(
            measure.context,
            measure.stream,
            measure.weights,
            measure.output,
            measure.scratch,
            measure.shape,
            measure.weight_sets,
        )?;
        Ok((separate, persistent))
    }
}

fn report_boundary_cycles(mut separate_samples: Vec<f64>, mut persistent_samples: Vec<f64>) {
    let separate_median = median(&mut separate_samples);
    let persistent_median = median(&mut persistent_samples);
    let ratio = persistent_median / separate_median;
    let confirmed = persistent_median >= 900.0 && ratio >= 1.10;
    println!("separate_launch_median_gbs={separate_median:.3}");
    println!("one_launch_median_gbs={persistent_median:.3}");
    println!("one_over_separate={ratio:.4}");
    println!(
        "boundary_model={}",
        if confirmed { "confirmed" } else { "refuted" }
    );
}

fn launch_separate_ring(
    stream: &Stream,
    weights: &leone_cuda::DeviceBuffer<u8>,
    output: &mut leone_cuda::DeviceBuffer<f32>,
    scratch: &GemvScratch,
    shape: QuantizedMatrixShape,
    weight_sets: usize,
) -> leone_cuda::Result<()> {
    for weight_set in 0..weight_sets {
        launch_q4_k_probe(
            stream,
            weights,
            output,
            scratch,
            shape,
            weight_set,
            Q4KProbeGeometry::ProductionFourWarpsOneRow,
        )?;
    }
    Ok(())
}

fn time_separate_ring(
    context: &Context,
    stream: &Stream,
    weights: &leone_cuda::DeviceBuffer<u8>,
    output: &mut leone_cuda::DeviceBuffer<f32>,
    scratch: &GemvScratch,
    shape: QuantizedMatrixShape,
    weight_sets: usize,
) -> Result<f64, Box<dyn Error>> {
    let mut start = Event::new(context)?;
    let mut end = Event::new(context)?;
    start.record(stream)?;
    for _ in 0..BOUNDARY_TIMED_CYCLES {
        launch_separate_ring(stream, weights, output, scratch, shape, weight_sets)?;
    }
    end.record(stream)?;
    end.synchronize()?;
    ring_gbs(&start, &end, shape, weight_sets)
}

fn time_persistent_ring(
    context: &Context,
    stream: &Stream,
    weights: &leone_cuda::DeviceBuffer<u8>,
    output: &mut leone_cuda::DeviceBuffer<f32>,
    scratch: &GemvScratch,
    shape: QuantizedMatrixShape,
    weight_sets: usize,
) -> Result<f64, Box<dyn Error>> {
    let mut start = Event::new(context)?;
    let mut end = Event::new(context)?;
    start.record(stream)?;
    for _ in 0..BOUNDARY_TIMED_CYCLES {
        launch_q4_k_ring_probe(stream, weights, output, scratch, shape, weight_sets)?;
    }
    end.record(stream)?;
    end.synchronize()?;
    ring_gbs(&start, &end, shape, weight_sets)
}

fn ring_gbs(
    start: &Event,
    end: &Event,
    shape: QuantizedMatrixShape,
    weight_sets: usize,
) -> Result<f64, Box<dyn Error>> {
    let seconds = f64::from(Event::elapsed_ms(start, end)?) / 1_000.0;
    let bytes = shape.bytes() * weight_sets * BOUNDARY_TIMED_CYCLES;
    Ok(bytes as f64 / seconds / 1e9)
}

fn run_shape(
    context: &Context,
    stream: &Stream,
    input: &leone_cuda::DeviceBuffer<f32>,
    rows: usize,
    variants: &[(&str, Q4KProbeGeometry)],
    layout: ProbeLayout,
) -> Result<(), Box<dyn Error>> {
    let shape = QuantizedMatrixShape::new(rows, COLUMNS, QuantFormat::Q4K)?;
    let weight_sets = RING_TARGET_BYTES.div_ceil(shape.bytes());
    let matrix = probe_matrix(shape, layout);
    let mut ring = Vec::with_capacity(shape.bytes() * weight_sets);
    for _ in 0..weight_sets {
        ring.extend_from_slice(&matrix);
    }
    let d_weights = context.copy_to_device(&ring)?;
    let mut d_output = context.alloc::<f32>(rows * 2)?;
    let mut scratch = GemvScratch::new(context, shape)?;
    prepare_q4_k_probe(stream, input, &mut scratch)?;
    stream.synchronize()?;

    println!(
        "shape rows={rows} columns={COLUMNS} matrix_bytes={} weight_sets={} ring_bytes={} layout={}",
        shape.bytes(),
        weight_sets,
        ring.len(),
        layout.name()
    );
    let mut probe = ShapeProbeArgs {
        context,
        stream,
        weights: &d_weights,
        output: &mut d_output,
        scratch: &scratch,
        shape,
        weight_sets,
    };
    for &(name, geometry) in variants {
        probe_variant(&mut probe, name, geometry)?;
    }
    Ok(())
}

fn probe_variant(
    probe: &mut ShapeProbeArgs<'_>,
    name: &str,
    geometry: Q4KProbeGeometry,
) -> Result<(), Box<dyn Error>> {
    for launch in 0..WARMUP_LAUNCHES {
        launch_q4_k_probe(
            probe.stream,
            probe.weights,
            probe.output,
            probe.scratch,
            probe.shape,
            launch % probe.weight_sets,
            geometry,
        )?;
    }
    probe.stream.synchronize()?;
    let mut samples = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        samples.push(time_shape_variant(probe, geometry)?);
    }
    let median_gbs = median(&mut samples);
    println!("variant={name} median_weight_gbs={median_gbs:.3} weight_gbs_samples={samples:?}");
    Ok(())
}

fn time_shape_variant(
    probe: &mut ShapeProbeArgs<'_>,
    geometry: Q4KProbeGeometry,
) -> Result<f64, Box<dyn Error>> {
    let mut start = Event::new(probe.context)?;
    let mut end = Event::new(probe.context)?;
    start.record(probe.stream)?;
    for launch in 0..TIMED_LAUNCHES {
        launch_q4_k_probe(
            probe.stream,
            probe.weights,
            probe.output,
            probe.scratch,
            probe.shape,
            launch % probe.weight_sets,
            geometry,
        )?;
    }
    end.record(probe.stream)?;
    end.synchronize()?;
    let seconds = f64::from(Event::elapsed_ms(&start, &end)?) / 1_000.0;
    Ok(probe.shape.bytes() as f64 * TIMED_LAUNCHES as f64 / seconds / 1e9)
}

fn probe_matrix(shape: QuantizedMatrixShape, layout: ProbeLayout) -> Vec<u8> {
    let blocks = shape.rows() * shape.columns() / 256;
    let mut codes = Vec::with_capacity(blocks * 128);
    for index in 0..blocks * 128 {
        codes.push((index as u8).wrapping_mul(37).wrapping_add(11));
    }
    let mut metadata = Vec::with_capacity(blocks * 16);
    for block in 0..blocks {
        metadata.extend_from_slice(&[
            0x00,
            0x3c,
            0x00,
            0x38,
            0x21_u8.wrapping_add(block as u8),
            0x32,
            0x43,
            0x14,
            0x05,
            0x16,
            0x27,
            0x38,
            0x09,
            0x1a,
            0x2b,
            0x3c,
        ]);
    }
    match layout {
        ProbeLayout::TensorSplit => {
            codes.extend_from_slice(&metadata);
            return codes;
        }
        ProbeLayout::Gguf => {
            let mut matrix = Vec::with_capacity(shape.bytes());
            for block in 0..blocks {
                matrix.extend_from_slice(&metadata[block * 16..(block + 1) * 16]);
                matrix.extend_from_slice(&codes[block * 128..(block + 1) * 128]);
            }
            return matrix;
        }
        ProbeLayout::RowInterleaved => {}
    }
    let blocks_per_row = shape.columns() / 256;
    let mut matrix = Vec::with_capacity(shape.bytes());
    for row in 0..shape.rows() {
        let block_start = row * blocks_per_row;
        matrix.extend_from_slice(&codes[block_start * 128..(block_start + blocks_per_row) * 128]);
        matrix.extend_from_slice(&metadata[block_start * 16..(block_start + blocks_per_row) * 16]);
    }
    matrix
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}
