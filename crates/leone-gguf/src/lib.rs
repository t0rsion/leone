#![deny(unsafe_code)]

//! Parses GGUF v3 files and provides scalar reference dequantizers.
//!
//! The parser streams metadata and reads tensor ranges on demand. It does not
//! map the file. Each tensor read copies bytes into an owned buffer.
//!
//! Run the pinned llama.cpp differential gate with:
//!
//! ```text
//! scripts/check-quantized-differential.sh
//! ```

mod alloc;
mod container;
mod error;
pub mod model;
mod pread;
pub mod ref_dequant;
mod support;

pub use alloc::{AllocationBudget, AllocationBudgetError, AllocationGuard, AllocationReservation};
pub use container::{GgmlType, Gguf, MetadataArray, MetadataValue, TensorInfo, ValueType};
pub use error::{Error, Result};
pub use support::{
    architecture_support, decode_quantized_block, quantization_support, ArchitectureClass,
    ArchitectureSupport, BlockImportError, QuantizationSupport,
};
