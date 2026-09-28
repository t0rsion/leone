use std::num::NonZeroUsize;

use super::{AttentionShape, BackendError, Position};

/// Borrows one independent query and output for batched decode attention.
///
/// Immutable KV allocations may be shared between rows. Query values, causal
/// positions, softmax state, and output values remain row-specific. Each cache
/// uses its spans' physical strides and its own logical attention shape.
/// Backends must not retain these references or the host span slices.
pub struct AttentionDecodeRow<'a, T> {
    pub query: &'a T,
    pub cache: KvReadView<'a, T>,
    pub output: &'a mut T,
    pub shape: AttentionShape,
    pub position: Position<'a, T>,
}

/// Describes one contiguous immutable KV cache segment.
///
/// The physical row stride is `capacity_tokens`. `tokens` bounds the mapped
/// prefix. A final span may map its full capacity for graph replay, even when
/// later rows are not initialized. Attention operations must keep their causal
/// read end within the initialized prefix supplied by preceding work.
pub struct KvReadSpan<'a, T> {
    key: &'a T,
    value: &'a T,
    logical_start: usize,
    tokens: NonZeroUsize,
    capacity_tokens: NonZeroUsize,
}

impl<T> Copy for KvReadSpan<'_, T> {}

impl<T> Clone for KvReadSpan<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T> KvReadSpan<'a, T> {
    /// Creates a checked immutable KV cache segment descriptor.
    pub fn new(
        key: &'a T,
        value: &'a T,
        logical_start: usize,
        tokens: usize,
        capacity_tokens: usize,
    ) -> Result<Self, BackendError> {
        let tokens = NonZeroUsize::new(tokens).ok_or(BackendError::Zero {
            field: "KV read span tokens",
        })?;
        let capacity_tokens = NonZeroUsize::new(capacity_tokens).ok_or(BackendError::Zero {
            field: "KV read span capacity tokens",
        })?;
        if tokens > capacity_tokens {
            return Err(BackendError::SizeMismatch {
                name: "KV read span token capacity",
                expected: capacity_tokens.get(),
                actual: tokens.get(),
            });
        }
        logical_start
            .checked_add(capacity_tokens.get())
            .ok_or(BackendError::SizeOverflow {
                field: "KV read span end",
            })?;
        Ok(Self {
            key,
            value,
            logical_start,
            tokens,
            capacity_tokens,
        })
    }

    /// Returns the immutable key buffer.
    pub fn key(&self) -> &'a T {
        self.key
    }

    /// Returns the immutable value buffer.
    pub fn value(&self) -> &'a T {
        self.value
    }

    /// Returns the first logical token address in the segment.
    pub const fn logical_start(&self) -> usize {
        self.logical_start
    }

    /// Returns the mapped token count.
    pub const fn tokens(&self) -> NonZeroUsize {
        self.tokens
    }

    /// Returns the mapped token count as `usize`.
    pub const fn token_count(&self) -> usize {
        self.tokens.get()
    }

    /// Returns the physical token capacity.
    pub const fn capacity_tokens(&self) -> NonZeroUsize {
        self.capacity_tokens
    }

    /// Returns the physical token capacity as `usize`.
    pub const fn capacity_token_count(&self) -> usize {
        self.capacity_tokens.get()
    }

    /// Returns the exclusive logical end of the mapped prefix.
    pub fn mapped_end(&self) -> usize {
        self.logical_start + self.tokens.get()
    }

    /// Returns the exclusive physical end of the segment.
    pub fn capacity_end(&self) -> usize {
        self.logical_start + self.capacity_tokens.get()
    }
}

/// Provides ordered immutable KV cache segments for attention.
///
/// The view checks logical coverage and derives `mapped_tokens` as an address
/// bound. It does not inspect backend buffers or claim that mapped tail bytes
/// are initialized. The attention shape keeps the logical context and
/// reduction bound; each span supplies only its physical row stride.
pub struct KvReadView<'a, T> {
    spans: &'a [KvReadSpan<'a, T>],
    mapped_tokens: usize,
}

impl<T> Copy for KvReadView<'_, T> {}

impl<T> Clone for KvReadView<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T> KvReadView<'a, T> {
    /// Checks ordered, gap-free mapped prefixes and derives their address bound.
    pub fn new(spans: &'a [KvReadSpan<'a, T>]) -> Result<Self, BackendError> {
        let mut mapped_tokens = 0;
        for span in spans {
            if span.logical_start != mapped_tokens {
                return Err(BackendError::operation(
                    "construct KV read view",
                    "spans must provide ordered gap-free coverage from zero",
                ));
            }
            mapped_tokens = span.logical_start.checked_add(span.tokens.get()).ok_or(
                BackendError::SizeOverflow {
                    field: "KV read view mapped end",
                },
            )?;
        }
        Ok(Self {
            spans,
            mapped_tokens,
        })
    }

    /// Returns the borrowed segment descriptors.
    pub fn spans(&self) -> &'a [KvReadSpan<'a, T>] {
        self.spans
    }

    /// Returns the exclusive mapped logical address bound.
    ///
    /// The caller must initialize every causal read range before dispatch.
    pub const fn mapped_tokens(&self) -> usize {
        self.mapped_tokens
    }

    /// Finds the mapped segment containing one logical token address.
    pub fn span_for_position(&self, position: usize) -> Option<(&'a KvReadSpan<'a, T>, usize)> {
        self.spans.iter().find_map(|span| {
            let local = position.checked_sub(span.logical_start)?;
            (local < span.tokens.get()).then_some((span, local))
        })
    }
}

/// Describes one mutable KV cache segment append target.
///
/// The physical row stride is `capacity_tokens`. Buffer dtype and element
/// counts remain backend checks because this type does not inspect `T`.
pub struct KvWriteSpan<'a, T> {
    key: &'a mut T,
    value: &'a mut T,
    logical_start: usize,
    capacity_tokens: NonZeroUsize,
}

impl<'a, T> KvWriteSpan<'a, T> {
    /// Creates a checked mutable KV cache segment descriptor.
    pub fn new(
        key: &'a mut T,
        value: &'a mut T,
        logical_start: usize,
        capacity_tokens: usize,
    ) -> Result<Self, BackendError> {
        let capacity_tokens = NonZeroUsize::new(capacity_tokens).ok_or(BackendError::Zero {
            field: "KV write span capacity tokens",
        })?;
        logical_start
            .checked_add(capacity_tokens.get())
            .ok_or(BackendError::SizeOverflow {
                field: "KV write span end",
            })?;
        Ok(Self {
            key,
            value,
            logical_start,
            capacity_tokens,
        })
    }

    /// Returns the first logical token address in the segment.
    pub const fn logical_start(&self) -> usize {
        self.logical_start
    }

    /// Returns the physical token capacity.
    pub const fn capacity_tokens(&self) -> NonZeroUsize {
        self.capacity_tokens
    }

    /// Returns the physical token capacity as `usize`.
    pub const fn capacity_token_count(&self) -> usize {
        self.capacity_tokens.get()
    }

    /// Converts an absolute position to a checked local position.
    pub fn local_position(&self, absolute_position: usize) -> Result<usize, BackendError> {
        let local = absolute_position
            .checked_sub(self.logical_start)
            .ok_or_else(|| {
                BackendError::operation("address KV write span", "position precedes span start")
            })?;
        if local >= self.capacity_tokens.get() {
            return Err(BackendError::operation(
                "address KV write span",
                "position exceeds span capacity",
            ));
        }
        Ok(local)
    }

    /// Converts an absolute append range to a checked local range.
    pub fn local_range(
        &self,
        start: usize,
        tokens: usize,
    ) -> Result<std::ops::Range<usize>, BackendError> {
        let local_start = start.checked_sub(self.logical_start).ok_or_else(|| {
            BackendError::operation("address KV write span", "range precedes span start")
        })?;
        let local_end = local_start
            .checked_add(tokens)
            .ok_or(BackendError::SizeOverflow {
                field: "KV write span range end",
            })?;
        if local_end > self.capacity_tokens.get() {
            return Err(BackendError::operation(
                "address KV write span",
                "range exceeds span capacity",
            ));
        }
        Ok(local_start..local_end)
    }

    /// Splits the descriptor into its mutable buffers and physical address data.
    pub fn into_parts(self) -> (&'a mut T, &'a mut T, usize, NonZeroUsize) {
        (
            self.key,
            self.value,
            self.logical_start,
            self.capacity_tokens,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_span_checks_sizes_and_end() {
        let key = ();
        let value = ();
        assert!(matches!(
            KvReadSpan::new(&key, &value, 0, 0, 1),
            Err(BackendError::Zero { .. })
        ));
        assert!(matches!(
            KvReadSpan::new(&key, &value, 0, 2, 1),
            Err(BackendError::SizeMismatch { .. })
        ));
        assert!(matches!(
            KvReadSpan::new(&key, &value, usize::MAX, 1, 1),
            Err(BackendError::SizeOverflow { .. })
        ));
    }

    #[test]
    fn read_view_requires_gap_free_prefixes() {
        let key = ();
        let value = ();
        let first = KvReadSpan::new(&key, &value, 0, 2, 4).unwrap();
        let second = KvReadSpan::new(&key, &value, 3, 1, 2).unwrap();
        assert!(matches!(
            KvReadView::new(&[first, second]),
            Err(BackendError::Operation { .. })
        ));
        let second = KvReadSpan::new(&key, &value, 2, 1, 2).unwrap();
        let spans = [first, second];
        let view = KvReadView::new(&spans).unwrap();
        assert_eq!(view.mapped_tokens(), 3);
        assert_eq!(view.span_for_position(2).map(|(_, local)| local), Some(0));
    }

    #[test]
    fn write_span_checks_local_ranges() {
        let mut key = ();
        let mut value = ();
        let span = KvWriteSpan::new(&mut key, &mut value, 4, 3).unwrap();
        assert_eq!(span.local_position(5).unwrap(), 1);
        assert_eq!(span.local_range(5, 2).unwrap(), 1..3);
        assert!(span.local_position(3).is_err());
        assert!(span.local_range(6, 2).is_err());
    }
}
