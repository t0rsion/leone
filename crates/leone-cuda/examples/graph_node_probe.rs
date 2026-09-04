use leone_cuda::{increment_u32_scalar, write_u32_scalar, Context, Event, Graph, Stream};
use std::error::Error;

const NODE_COUNTS: [usize; 2] = [401, 450];
const WARMUP_REPLAYS: usize = 100;
const TIMED_REPLAYS: usize = 2_000;
const SAMPLES: usize = 9;

fn main() -> Result<(), Box<dyn Error>> {
    let context = Context::new(0)?;
    let stream = Stream::new(&context)?;
    let mut scalar = context.alloc::<u32>(1)?;
    write_u32_scalar(&stream, &mut scalar, 0)?;
    stream.synchronize()?;
    run_probe(&context, &stream, &mut scalar)
}

fn run_probe(
    context: &Context,
    stream: &Stream,
    scalar: &mut leone_cuda::DeviceBuffer<u32>,
) -> Result<(), Box<dyn Error>> {
    let mut results = Vec::with_capacity(NODE_COUNTS.len());
    println!("CUDA graph trivial-node replay probe");
    println!("node: one-thread increment of one device u32");
    println!("warmup_replays: {WARMUP_REPLAYS}");
    println!("timed_replays_per_sample: {TIMED_REPLAYS}");
    println!("samples: {SAMPLES}");
    for nodes in NODE_COUNTS {
        let mut ns_per_replay = measure_node_count(context, stream, scalar, nodes)?;
        let median_ns = median(&mut ns_per_replay);
        println!(
            "nodes={nodes} median_ns_per_replay={median_ns:.3} median_ns_per_node={:.3}",
            median_ns / nodes as f64
        );
        println!("nodes={nodes} ns_per_replay_samples={ns_per_replay:?}");
        results.push((nodes, median_ns));
    }

    let node_delta = results[1].0 - results[0].0;
    let replay_delta_ns = results[1].1 - results[0].1;
    println!(
        "delta_nodes={node_delta} delta_ns_per_replay={replay_delta_ns:.3} slope_ns_per_node={:.3}",
        replay_delta_ns / node_delta as f64
    );
    Ok(())
}

fn measure_node_count(
    context: &Context,
    stream: &Stream,
    scalar: &mut leone_cuda::DeviceBuffer<u32>,
    nodes: usize,
) -> Result<Vec<f64>, Box<dyn Error>> {
    let graph = capture_increment_graph(stream, scalar, nodes)?;
    for _ in 0..WARMUP_REPLAYS {
        graph.launch(stream)?;
    }
    stream.synchronize()?;
    let mut samples = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        samples.push(time_node_sample(context, stream, &graph)?);
    }
    Ok(samples)
}

fn time_node_sample(
    context: &Context,
    stream: &Stream,
    graph: &Graph,
) -> Result<f64, Box<dyn Error>> {
    let mut start = Event::new(context)?;
    let mut end = Event::new(context)?;
    start.record(stream)?;
    for _ in 0..TIMED_REPLAYS {
        graph.launch(stream)?;
    }
    end.record(stream)?;
    end.synchronize()?;
    let elapsed_ns = f64::from(Event::elapsed_ms(&start, &end)?) * 1_000_000.0;
    Ok(elapsed_ns / TIMED_REPLAYS as f64)
}

fn capture_increment_graph(
    stream: &Stream,
    scalar: &mut leone_cuda::DeviceBuffer<u32>,
    nodes: usize,
) -> leone_cuda::Result<Graph> {
    stream.begin_graph_capture()?;
    for _ in 0..nodes {
        increment_u32_scalar(stream, scalar)?;
    }
    stream.end_graph_capture()
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}
