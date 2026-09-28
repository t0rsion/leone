#![cfg(target_os = "macos")]

use leone::{GenerateOptions, Runtime};
use leone_metal::MetalBackend;
use std::error::Error;
use std::path::PathBuf;

#[test]
#[ignore = "requires a local Q4_K/Q6_K model and an Apple GPU"]
fn metal_runtime_generates_with_chunked_prefill() -> Result<(), Box<dyn Error>> {
    let model = PathBuf::from(
        std::env::var_os("LEONE_METAL_MODEL")
            .expect("LEONE_METAL_MODEL must name the GGUF model for this ignored gate"),
    );
    let mut runtime = Runtime::load(MetalBackend::new()?, model)?;
    let mut options = GenerateOptions::greedy(2);
    options.prefill_chunk_tokens = 4;
    let result = runtime.generate("Hello", options, |_| Ok(()), || false)?;
    assert_eq!(
        result.stats.prefill_method,
        leone::PrefillMethod::ChunkedGpu
    );
    assert_eq!(result.stats.emitted_tokens, 2);
    Ok(())
}
