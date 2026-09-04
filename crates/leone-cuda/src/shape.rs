use crate::{Error, Result};

/// The CUDA thread count used by reduction and elementwise kernels.
pub const CUDA_BLOCK_SIZE: usize = 256;

/// The thread count in one decode attention context split.
pub const ATTENTION_BLOCK_SIZE: usize = 128;

/// The target context length covered by one decode attention split.
pub const ATTENTION_KV_TILE: usize = 192;

/// The maximum context split count used by decode attention.
pub const ATTENTION_SPLIT_KV_MAX: usize = 64;

/// The number of contiguous values scanned by each argmax thread.
pub const ARGMAX_ITEMS_PER_THREAD: usize = 4;

const K_BLOCK_ELEMENTS: usize = 256;
const Q4_K_BLOCK_BYTES: usize = 144;
const Q6_K_BLOCK_BYTES: usize = 210;
const ATTENTION_HEAD_DIM_MAX: usize = 128;
const CUDA_GRID_X_MAX: usize = i32::MAX as usize;
const ARGMAX_ELEMENT_MAX: usize = u32::MAX as usize + 1;

/// A K-quant storage format supported by the CUDA kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantFormat {
    Q4K,
    Q6K,
}

impl QuantFormat {
    /// Returns the byte count for one 256-value block.
    pub const fn block_bytes(self) -> usize {
        match self {
            Self::Q4K => Q4_K_BLOCK_BYTES,
            Self::Q6K => Q6_K_BLOCK_BYTES,
        }
    }
}

/// The checked logical and storage shape of a row-major K-quant matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantizedMatrixShape {
    rows: usize,
    columns: usize,
    row_bytes: usize,
    bytes: usize,
    format: QuantFormat,
}

impl QuantizedMatrixShape {
    /// Checks a row-major matrix whose rows contain complete 256-value blocks.
    pub fn new(rows: usize, columns: usize, format: QuantFormat) -> Result<Self> {
        nonzero("rows", rows)?;
        nonzero("columns", columns)?;
        at_most("rows", rows, CUDA_GRID_X_MAX)?;
        divisible("columns", columns, K_BLOCK_ELEMENTS)?;
        let row_bytes = columns
            .checked_div(K_BLOCK_ELEMENTS)
            .and_then(|blocks| blocks.checked_mul(format.block_bytes()))
            .ok_or(Error::SizeOverflow {
                field: "quantized row bytes",
            })?;
        let bytes = rows.checked_mul(row_bytes).ok_or(Error::SizeOverflow {
            field: "quantized matrix bytes",
        })?;
        Ok(Self {
            rows,
            columns,
            row_bytes,
            bytes,
            format,
        })
    }

    pub const fn rows(self) -> usize {
        self.rows
    }

    pub const fn columns(self) -> usize {
        self.columns
    }

    pub const fn row_bytes(self) -> usize {
        self.row_bytes
    }

    pub const fn bytes(self) -> usize {
        self.bytes
    }

    pub const fn format(self) -> QuantFormat {
        self.format
    }

    pub(crate) fn output_elements(self) -> usize {
        self.rows
    }
}

/// The checked shape of one or more row vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VectorShape {
    rows: usize,
    columns: usize,
    elements: usize,
}

impl VectorShape {
    /// Checks a dense row-vector batch.
    pub fn new(rows: usize, columns: usize) -> Result<Self> {
        nonzero("rows", rows)?;
        nonzero("columns", columns)?;
        at_most("rows", rows, CUDA_GRID_X_MAX)?;
        let elements = rows.checked_mul(columns).ok_or(Error::SizeOverflow {
            field: "vector elements",
        })?;
        Ok(Self {
            rows,
            columns,
            elements,
        })
    }

    pub const fn rows(self) -> usize {
        self.rows
    }

    pub const fn columns(self) -> usize {
        self.columns
    }

    pub const fn elements(self) -> usize {
        self.elements
    }
}

/// The checked shape of a GPT-NeoX RoPE input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RopeShape {
    tokens: usize,
    heads: usize,
    head_dim: usize,
    elements: usize,
}

impl RopeShape {
    /// Checks `[token][head][head_dim]` storage for half-pair rotation.
    pub fn new(tokens: usize, heads: usize, head_dim: usize) -> Result<Self> {
        nonzero("tokens", tokens)?;
        nonzero("heads", heads)?;
        nonzero("head_dim", head_dim)?;
        divisible("head_dim", head_dim, 2)?;
        let elements = tokens
            .checked_mul(heads)
            .and_then(|value| value.checked_mul(head_dim))
            .ok_or(Error::SizeOverflow {
                field: "RoPE elements",
            })?;
        let pair_blocks = (elements / 2).div_ceil(CUDA_BLOCK_SIZE);
        at_most("RoPE grid blocks", pair_blocks, CUDA_GRID_X_MAX)?;
        Ok(Self {
            tokens,
            heads,
            head_dim,
            elements,
        })
    }

    pub const fn tokens(self) -> usize {
        self.tokens
    }

    pub const fn heads(self) -> usize {
        self.heads
    }

    pub const fn head_dim(self) -> usize {
        self.head_dim
    }

    pub const fn elements(self) -> usize {
        self.elements
    }
}

/// The checked shape and contiguous KV layout for batch-1 decode attention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionShape {
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    max_context: usize,
    query_elements: usize,
    cache_elements: usize,
}

impl AttentionShape {
    /// Checks Q `[n_head][head_dim]` and KV `[n_head_kv][max_context][head_dim]`.
    pub fn new(
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        max_context: usize,
    ) -> Result<Self> {
        validate_attention_dimensions(n_head, n_head_kv, head_dim, max_context)?;
        let (query_elements, cache_elements) =
            attention_elements(n_head, n_head_kv, head_dim, max_context)?;
        Ok(Self {
            n_head,
            n_head_kv,
            head_dim,
            max_context,
            query_elements,
            cache_elements,
        })
    }

    pub const fn n_head(self) -> usize {
        self.n_head
    }

    pub const fn n_head_kv(self) -> usize {
        self.n_head_kv
    }

    pub const fn head_dim(self) -> usize {
        self.head_dim
    }

    pub const fn max_context(self) -> usize {
        self.max_context
    }

    pub const fn query_elements(self) -> usize {
        self.query_elements
    }

    pub fn projected_kv_elements(self) -> Result<usize> {
        self.n_head_kv
            .checked_mul(self.head_dim)
            .ok_or(Error::SizeOverflow {
                field: "projected KV elements",
            })
    }

    pub const fn cache_elements(self) -> usize {
        self.cache_elements
    }

    pub(crate) fn partial_elements(self) -> Result<usize> {
        let split_capacity = self.split_capacity();
        self.n_head
            .checked_mul(split_capacity)
            .and_then(|value| value.checked_mul(self.head_dim))
            .ok_or(Error::SizeOverflow {
                field: "attention partial elements",
            })
    }

    pub(crate) fn partial_rows(self) -> Result<usize> {
        let split_capacity = self.split_capacity();
        self.n_head
            .checked_mul(split_capacity)
            .ok_or(Error::SizeOverflow {
                field: "attention partial rows",
            })
    }

    const fn split_capacity(self) -> usize {
        let tiles = self.max_context.div_ceil(ATTENTION_KV_TILE);
        if tiles < ATTENTION_SPLIT_KV_MAX {
            tiles
        } else {
            ATTENTION_SPLIT_KV_MAX
        }
    }
}

fn validate_attention_dimensions(
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    max_context: usize,
) -> Result<()> {
    nonzero("n_head", n_head)?;
    nonzero("n_head_kv", n_head_kv)?;
    nonzero("head_dim", head_dim)?;
    nonzero("max_context", max_context)?;
    at_most("n_head", n_head, CUDA_GRID_X_MAX)?;
    if !n_head.is_multiple_of(n_head_kv) {
        return Err(Error::InvalidGqa { n_head, n_head_kv });
    }
    if head_dim > ATTENTION_HEAD_DIM_MAX {
        return Err(Error::TooLarge {
            field: "head_dim",
            value: head_dim,
            maximum: ATTENTION_HEAD_DIM_MAX,
        });
    }
    Ok(())
}

fn attention_elements(
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    max_context: usize,
) -> Result<(usize, usize)> {
    let query_elements = n_head.checked_mul(head_dim).ok_or(Error::SizeOverflow {
        field: "attention query elements",
    })?;
    let cache_elements = n_head_kv
        .checked_mul(max_context)
        .and_then(|value| value.checked_mul(head_dim))
        .ok_or(Error::SizeOverflow {
            field: "KV cache elements",
        })?;
    Ok((query_elements, cache_elements))
}

pub(crate) fn argmax_blocks(elements: usize) -> Result<usize> {
    nonzero("argmax elements", elements)?;
    at_most("argmax elements", elements, ARGMAX_ELEMENT_MAX)?;
    let values_per_block =
        CUDA_BLOCK_SIZE
            .checked_mul(ARGMAX_ITEMS_PER_THREAD)
            .ok_or(Error::SizeOverflow {
                field: "argmax values per block",
            })?;
    let blocks = elements.div_ceil(values_per_block);
    at_most("argmax grid blocks", blocks, CUDA_GRID_X_MAX)?;
    Ok(blocks)
}

pub(crate) fn validate_element_grid(elements: usize, field: &'static str) -> Result<()> {
    nonzero(field, elements)?;
    at_most(field, elements.div_ceil(CUDA_BLOCK_SIZE), CUDA_GRID_X_MAX)
}

fn nonzero(field: &'static str, value: usize) -> Result<()> {
    if value == 0 {
        Err(Error::Zero { field })
    } else {
        Ok(())
    }
}

fn divisible(field: &'static str, value: usize, divisor: usize) -> Result<()> {
    if value.is_multiple_of(divisor) {
        Ok(())
    } else {
        Err(Error::NotDivisible {
            field,
            value,
            divisor,
        })
    }
}

fn at_most(field: &'static str, value: usize, maximum: usize) -> Result<()> {
    if value <= maximum {
        Ok(())
    } else {
        Err(Error::TooLarge {
            field,
            value,
            maximum,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantized_shapes_compute_exact_storage() {
        let q4 = QuantizedMatrixShape::new(4_096, 4_096, QuantFormat::Q4K).unwrap();
        assert_eq!(q4.row_bytes(), 2_304);
        assert_eq!(q4.bytes(), 9_437_184);

        let q6 = QuantizedMatrixShape::new(151_936, 4_096, QuantFormat::Q6K).unwrap();
        assert_eq!(q6.row_bytes(), 3_360);
        assert_eq!(q6.bytes(), 510_504_960);
    }

    #[test]
    fn partial_shapes_are_rejected() {
        assert!(matches!(
            QuantizedMatrixShape::new(1, 255, QuantFormat::Q4K),
            Err(Error::NotDivisible { .. })
        ));
        assert!(matches!(
            RopeShape::new(1, 1, 127),
            Err(Error::NotDivisible { .. })
        ));
        assert!(matches!(
            AttentionShape::new(32, 7, 128, 8_192),
            Err(Error::InvalidGqa { .. })
        ));
        assert!(matches!(
            AttentionShape::new(32, 8, 256, 8_192),
            Err(Error::TooLarge { .. })
        ));
    }

    #[test]
    fn qwen_attention_layout_has_checked_size() {
        let shape = AttentionShape::new(32, 8, 128, 8_192).unwrap();
        assert_eq!(shape.query_elements(), 4_096);
        assert_eq!(shape.cache_elements(), 8_388_608);
        assert_eq!(shape.partial_rows().unwrap(), 1_376);
        assert_eq!(shape.partial_elements().unwrap(), 176_128);
    }

    #[test]
    fn argmax_grid_covers_the_qwen_vocabulary() {
        assert_eq!(argmax_blocks(151_936).unwrap(), 149);
    }
}
