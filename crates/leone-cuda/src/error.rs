use thiserror::Error;

/// An error returned by a checked CUDA operation or launch.
#[derive(Debug, Error, PartialEq)]
pub enum Error {
    #[error("CUDA {operation} failed with code {code}: {message}")]
    Runtime {
        operation: &'static str,
        code: i32,
        message: String,
    },
    #[error("{name} has {actual} elements, expected {expected}")]
    SizeMismatch {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("{field} must be nonzero")]
    Zero { field: &'static str },
    #[error("{field} must be divisible by {divisor}, found {value}")]
    NotDivisible {
        field: &'static str,
        value: usize,
        divisor: usize,
    },
    #[error("{field} must not exceed {maximum}, found {value}")]
    TooLarge {
        field: &'static str,
        value: usize,
        maximum: usize,
    },
    #[error("{field} must be finite and greater than zero, found {value}")]
    InvalidPositiveFloat { field: &'static str, value: f32 },
    #[error("{field} overflows the host size")]
    SizeOverflow { field: &'static str },
    #[error("n_head {n_head} is not divisible by n_head_kv {n_head_kv}")]
    InvalidGqa { n_head: usize, n_head_kv: usize },
    #[error("row {row} is outside a matrix with {rows} rows")]
    RowOutOfBounds { row: usize, rows: usize },
    #[error("context length {context_length} is outside 1..={max_context}")]
    ContextLength {
        context_length: usize,
        max_context: usize,
    },
    #[error("CUDA objects belong to different devices")]
    DeviceMismatch,
}

/// Result of a checked CUDA host call.
pub type Result<T> = std::result::Result<T, Error>;
