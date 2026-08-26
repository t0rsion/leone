use std::io;
use thiserror::Error;

/// An error returned while parsing or reading a GGUF file.
#[derive(Debug, Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("GGUF magic is {found:02x?}, expected 47 47 55 46")]
    InvalidMagic { found: [u8; 4] },
    #[error("GGUF version {0} is not version 3")]
    UnsupportedVersion(u32),
    #[error("GGUF {what} count {value} exceeds the limit {limit}")]
    LimitExceeded {
        what: &'static str,
        value: u64,
        limit: u64,
    },
    #[error("GGUF {0} overflows its file offset or host size")]
    IntegerOverflow(&'static str),
    #[error("GGUF {field} is not valid UTF-8: {source}")]
    InvalidUtf8 {
        field: &'static str,
        #[source]
        source: std::string::FromUtf8Error,
    },
    #[error("GGUF metadata value type {0} is not defined")]
    InvalidValueType(u32),
    #[error("GGUF arrays cannot contain arrays")]
    NestedArray,
    #[error("GGUF boolean value {0} is not 0 or 1")]
    InvalidBoolean(u8),
    #[error("GGUF metadata key {0:?} occurs more than once")]
    DuplicateMetadata(String),
    #[error("GGUF tensor name {0:?} occurs more than once")]
    DuplicateTensor(String),
    #[error("GGUF alignment {0} is not a nonzero power of two")]
    InvalidAlignment(u32),
    #[error("GGUF tensor {tensor:?} has {dimensions} dimensions, expected 1 through 4")]
    InvalidDimensions { tensor: String, dimensions: u32 },
    #[error("GGUF tensor {tensor:?} has a zero dimension")]
    ZeroDimension { tensor: String },
    #[error("GGUF tensor {tensor:?} uses removed or unknown ggml type {dtype}")]
    UnsupportedTensorType { tensor: String, dtype: u32 },
    #[error(
        "GGUF tensor {tensor:?} row length {row_elements} is not divisible by the {block_elements}-element {dtype} block"
    )]
    InvalidRowLength {
        tensor: String,
        dtype: String,
        row_elements: u64,
        block_elements: u64,
    },
    #[error("GGUF tensor {tensor:?} offset {offset} is not aligned to {alignment} bytes")]
    MisalignedTensor {
        tensor: String,
        offset: u64,
        alignment: u32,
    },
    #[error("GGUF tensor {tensor:?} byte range is outside the file")]
    TensorOutOfBounds { tensor: String },
    #[error("GGUF tensors {first:?} and {second:?} overlap")]
    OverlappingTensors { first: String, second: String },
    #[error("GGUF has no tensor named {0:?}")]
    TensorNotFound(String),
}

/// A result returned by the GGUF import layer.
pub type Result<T> = std::result::Result<T, Error>;
