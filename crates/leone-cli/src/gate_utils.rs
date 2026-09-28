use leone::{DecodeExecution, GenerateOptions, KvCacheDtype};
use std::io;

pub(crate) fn options(tokens: usize) -> GenerateOptions {
    let mut options = GenerateOptions::greedy(tokens);
    options.decode_execution = DecodeExecution::Eager;
    options.kv_cache_dtype = KvCacheDtype::F16;
    options
}

pub(crate) fn alternate_token(token: u32, vocab: usize) -> u32 {
    let token = usize::try_from(token).unwrap_or(0);
    u32::try_from((token + 1) % vocab).unwrap_or(0)
}

pub(crate) fn median(samples: &[u64]) -> u64 {
    let mut samples = samples.to_vec();
    samples.sort_unstable();
    samples[samples.len() / 2]
}

pub(crate) fn nanoseconds(value: u128) -> Result<u64, io::Error> {
    u64::try_from(value).map_err(|_| invalid_data("duration exceeds u64 nanoseconds"))
}

pub(crate) fn value<'a>(arguments: &'a [String], index: &mut usize) -> Result<&'a str, io::Error> {
    *index += 1;
    arguments
        .get(*index)
        .map(String::as_str)
        .ok_or_else(|| invalid_data("an option value is missing"))
}

pub(crate) fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}
