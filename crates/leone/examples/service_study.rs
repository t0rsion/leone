use leone::scheduler::run_service_trace_json;
use serde::Serialize;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[derive(Serialize)]
struct ServiceStudyReceipt {
    schema_version: u32,
    generated_unix_ns: u128,
    fixture_path: String,
    execution_ns: u128,
    report: leone::scheduler::ServiceTraceReport,
    limitation: &'static str,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = env::args_os().skip(1);
    let fixture = PathBuf::from(
        arguments
            .next()
            .ok_or("usage: service_study <fixture.json> <receipt.json>")?,
    );
    let output = PathBuf::from(
        arguments
            .next()
            .ok_or("usage: service_study <fixture.json> <receipt.json>")?,
    );
    if arguments.next().is_some() {
        return Err("service_study accepts exactly two arguments".into());
    }
    let bytes = fs::read(&fixture)?;
    let started = Instant::now();
    let report = run_service_trace_json(&bytes)?;
    let receipt = ServiceStudyReceipt {
        schema_version: 1,
        generated_unix_ns: SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos(),
        fixture_path: fixture.display().to_string(),
        execution_ns: started.elapsed().as_nanos(),
        report,
        limitation: "This receipt validates the batch-1 scheduler control plane against isolated token fixtures. It does not measure model throughput.",
    };
    fs::write(output, serde_json::to_vec_pretty(&receipt)?)?;
    Ok(())
}
