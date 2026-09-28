use leone::scheduler::run_service_trace_json;
use serde::Serialize;
use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
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

fn public_fixture_path(path: &Path) -> Result<String, io::Error> {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty() && *name != "." && *name != "..")
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "fixture path has no public filename",
            )
        })?;
    let mut public = PathBuf::from("fixtures");
    public.push(name);
    Ok(public.display().to_string())
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
        fixture_path: public_fixture_path(&fixture)?,
        execution_ns: started.elapsed().as_nanos(),
        report,
        limitation: "This receipt validates the batch-1 scheduler control plane against isolated token fixtures. It does not measure model throughput.",
    };
    fs::write(output, serde_json::to_vec_pretty(&receipt)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::public_fixture_path;
    use std::path::Path;

    #[test]
    fn public_fixture_path_drops_absolute_prefix() {
        assert_eq!(
            public_fixture_path(Path::new("/private/work/fixtures/service.json")).unwrap(),
            "fixtures/service.json"
        );
    }
}
