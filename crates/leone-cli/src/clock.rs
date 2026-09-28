use std::sync::OnceLock;
use std::time::Instant;

static ORIGIN: OnceLock<Instant> = OnceLock::new();

/// Returns nanoseconds from the process monotonic clock origin.
pub(crate) fn now_ns() -> u64 {
    let nanos = ORIGIN.get_or_init(Instant::now).elapsed().as_nanos();
    u64::try_from(nanos).unwrap_or(u64::MAX)
}
