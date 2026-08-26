#![deny(unsafe_code)]

//! Parses GGUF v3 files and provides scalar reference dequantizers.
//!
//! The parser streams metadata and reads tensor ranges on demand. It does not
//! map the file. Each tensor read copies bytes into an owned buffer.
//!
//! Build `external/shim` before you run the ignored differential tests:
//!
//! ```text
//! make -C external/shim
//! taskset -c 16-31 cargo +1.92 test -p leone-gguf --release \
//!     --features differential -- --ignored
//! ```

mod container;
mod error;
pub mod model;
mod pread;
pub mod ref_dequant;
mod support;

pub use container::{GgmlType, Gguf, MetadataArray, MetadataValue, TensorInfo, ValueType};
pub use error::{Error, Result};
pub use support::{
    architecture_support, decode_quantized_block, quantization_support, ArchitectureClass,
    ArchitectureSupport, BlockImportError, QuantizationSupport,
};
