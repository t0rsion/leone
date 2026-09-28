use crate::KvCacheDtype;
use crate::{AttentionShape, Backend, BackendError, BufferLayout, BufferSnapshot, MemoryClass};
use crate::{KvReadSpan, KvWriteSpan};
use std::num::NonZeroUsize;
use std::rc::Rc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct KvGraphRevision {
    pub(crate) layout: u64,
    pub(crate) generation: u64,
}

#[derive(Debug)]
pub(crate) struct KvLayer<B: Backend> {
    pub(crate) key: B::Buffer,
    pub(crate) value: B::Buffer,
}

#[derive(Debug)]
struct KvStorage<B: Backend> {
    capacity_tokens: NonZeroUsize,
    bytes: u64,
    layers: Box<[KvLayer<B>]>,
}

#[derive(Debug)]
struct KvSegment<B: Backend> {
    logical_start: usize,
    committed_tokens: usize,
    storage: Rc<KvStorage<B>>,
}

struct KvAppendRollback<B: Backend> {
    original_position: usize,
    original_shape: AttentionShape,
    original_revision: u64,
    original_graph_revision: Option<KvGraphRevision>,
    original_pending: Option<KvAppendRange>,
    original_len: usize,
    original_committed: Vec<usize>,
    removed: Vec<KvSegment<B>>,
}

impl<B: Backend> Clone for KvSegment<B> {
    fn clone(&self) -> Self {
        Self {
            logical_start: self.logical_start,
            committed_tokens: self.committed_tokens,
            storage: Rc::clone(&self.storage),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct KvAppendRange {
    pub(crate) start: usize,
    pub(crate) tokens: NonZeroUsize,
    pub(crate) segment_index: usize,
}

#[derive(Debug)]
pub(crate) struct KvState<B: Backend> {
    segments: Vec<KvSegment<B>>,
    pub(crate) position: usize,
    pub(crate) device_position: B::Buffer,
    pub(crate) shape: AttentionShape,
    pub(crate) dtype: KvCacheDtype,
    revision: u64,
    pub(crate) graph_revision: Option<KvGraphRevision>,
    pending: Option<KvAppendRange>,
}

#[derive(Debug)]
pub(crate) struct KvSnapshot {
    pub(crate) segments: Vec<KvSnapshotSegment>,
    pub(crate) position: usize,
    pub(crate) device_position: BufferSnapshot,
    pub(crate) shape: AttentionShape,
    pub(crate) dtype: KvCacheDtype,
}

#[derive(Debug)]
pub(crate) struct KvSnapshotSegment {
    pub(crate) logical_start: usize,
    pub(crate) committed_tokens: usize,
    pub(crate) capacity_tokens: NonZeroUsize,
    pub(crate) layers: Vec<(BufferSnapshot, BufferSnapshot)>,
}

impl<B: Backend> KvStorage<B> {
    fn physical_bytes(&self) -> u64 {
        self.bytes
    }
}

fn reserve_vec<T>(
    values: &mut Vec<T>,
    additional: usize,
    operation: &'static str,
) -> Result<(), BackendError> {
    values
        .try_reserve_exact(additional)
        .map_err(|error| BackendError::operation(operation, error))
}

impl<B: Backend> KvState<B> {
    pub(crate) fn metadata_bound_bytes(
        context_tokens: usize,
        layers: usize,
    ) -> Result<u64, BackendError> {
        let segments = context_tokens
            .checked_add(1)
            .ok_or(BackendError::SizeOverflow {
                field: "KV metadata segment count",
            })?;
        let state_segment_capacity = metadata_vec_capacity(segments)?;
        let layer_count = segments
            .checked_mul(layers)
            .ok_or(BackendError::SizeOverflow {
                field: "KV metadata layer count",
            })?;
        kv_metadata_bound_bytes::<B>(state_segment_capacity, segments, layer_count)
    }

    pub(crate) fn new(
        backend: &mut B,
        _layers: usize,
        shape: AttentionShape,
        dtype: KvCacheDtype,
    ) -> Result<Self, BackendError> {
        Ok(Self {
            segments: Vec::new(),
            position: 0,
            device_position: backend
                .allocate_classified(BufferLayout::u32(1)?, MemoryClass::KvCache)?,
            shape,
            dtype,
            revision: 0,
            graph_revision: None,
            pending: None,
        })
    }

    pub(crate) fn revision(&self) -> u64 {
        self.revision
    }

    /// Returns whether the segmented cache can keep its prefix at `shape`.
    pub(crate) fn can_grow_to(&self, shape: AttentionShape) -> bool {
        self.has_compatible_layout(shape) && shape.max_context() >= self.shape.max_context()
    }

    /// Returns whether the segmented cache can retain its prefix at `shape`.
    pub(crate) fn can_retain_at(&self, shape: AttentionShape) -> bool {
        self.has_compatible_layout(shape) && shape.max_context() >= self.position
    }

    fn has_compatible_layout(&self, shape: AttentionShape) -> bool {
        self.shape.n_head() == shape.n_head()
            && self.shape.n_head_kv() == shape.n_head_kv()
            && self.shape.head_dim() == shape.head_dim()
    }

    /// Grows the logical context and stages its first append as one transaction.
    pub(crate) fn prepare_shape_transition(
        &mut self,
        backend: &mut B,
        layers: usize,
        shape: AttentionShape,
        position: usize,
        first_append: Option<(usize, usize)>,
    ) -> Result<(), BackendError> {
        let rollback = self.capture_append_rollback()?;
        if let Err(error) = self.stage_shape_growth(shape) {
            self.restore_append_position(rollback);
            return Err(error);
        }
        let result = match first_append {
            Some((tokens, growth_tokens)) => self
                .prepare_append_at(backend, layers, position, tokens, growth_tokens)
                .map(|_| ()),
            None => self.truncate(position),
        };
        if let Err(error) = result {
            self.restore_append_position(rollback);
            return Err(error);
        }
        Ok(())
    }

    fn stage_shape_growth(&mut self, shape: AttentionShape) -> Result<(), BackendError> {
        if shape == self.shape {
            return Ok(());
        }
        if !self.can_grow_to(shape) {
            return Err(BackendError::operation(
                "grow KV state",
                "the requested shape changes the KV layout or shrinks its context",
            ));
        }
        let revision = self
            .revision
            .checked_add(1)
            .ok_or(BackendError::SizeOverflow {
                field: "KV layout revision",
            })?;
        self.revision = revision;
        self.shape = shape;
        self.graph_revision = None;
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn physical_bytes(&self) -> u64 {
        self.segments
            .iter()
            .map(|segment| segment.storage.physical_bytes())
            .sum::<u64>()
            .saturating_add(4)
    }

    #[cfg(test)]
    pub(crate) fn has_pending_append(&self) -> bool {
        self.pending.is_some()
    }

    #[cfg(test)]
    pub(crate) fn pending_range(&self) -> Option<KvAppendRange> {
        self.pending
    }

    pub(crate) fn snapshot_payload_bytes(&self) -> Result<u64, BackendError> {
        let position_bytes = u64::try_from(BufferLayout::u32(1)?.bytes()).map_err(|_| {
            BackendError::SizeOverflow {
                field: "KV device position bytes",
            }
        })?;
        self.segments
            .iter()
            .filter(|segment| segment.committed_tokens != 0)
            .try_fold(position_bytes, |bytes: u64, segment| {
                bytes.checked_add(segment.storage.physical_bytes()).ok_or(
                    BackendError::SizeOverflow {
                        field: "KV snapshot bytes",
                    },
                )
            })
    }

    pub(crate) fn snapshot_container_bytes(&self) -> Result<u64, BackendError> {
        let (segments, layers) = self
            .segments
            .iter()
            .filter(|segment| segment.committed_tokens != 0)
            .try_fold((0_usize, 0_usize), |(segments, layers), segment| {
                let segments = segments.checked_add(1).ok_or(BackendError::SizeOverflow {
                    field: "KV snapshot segment count",
                })?;
                let layers = layers.checked_add(segment.storage.layers.len()).ok_or(
                    BackendError::SizeOverflow {
                        field: "KV snapshot layer count",
                    },
                )?;
                Ok::<_, BackendError>((segments, layers))
            })?;
        let segment_bytes =
            checked_container_bytes::<KvSnapshotSegment>(segments, "KV snapshot segment metadata")?;
        let layer_bytes = checked_container_bytes::<(BufferSnapshot, BufferSnapshot)>(
            layers,
            "KV snapshot layer metadata",
        )?;
        segment_bytes
            .checked_add(layer_bytes)
            .ok_or(BackendError::SizeOverflow {
                field: "KV snapshot container metadata",
            })
    }

    pub(crate) fn container_bytes(&self) -> Result<u64, BackendError> {
        let segment_bytes = checked_container_bytes::<KvSegment<B>>(
            self.segments.capacity(),
            "KV state segment metadata",
        )?;
        let storage_bytes = checked_container_bytes::<KvStorage<B>>(
            self.segments.len(),
            "KV state storage metadata",
        )?;
        let layers = self.segments.iter().try_fold(0_usize, |layers, segment| {
            layers
                .checked_add(segment.storage.layers.len())
                .ok_or(BackendError::SizeOverflow {
                    field: "KV state layer metadata",
                })
        })?;
        let layer_bytes = checked_container_bytes::<KvLayer<B>>(layers, "KV state layer metadata")?;
        segment_bytes
            .checked_add(storage_bytes)
            .ok_or(BackendError::SizeOverflow {
                field: "KV state container metadata",
            })?
            .checked_add(layer_bytes)
            .ok_or(BackendError::SizeOverflow {
                field: "KV state container metadata",
            })
    }

    pub(crate) fn fork(backend: &mut B, source: &Self) -> Result<Self, BackendError> {
        let segments = source
            .segments
            .iter()
            .filter(|segment| segment.committed_tokens != 0)
            .cloned()
            .collect();
        Ok(Self {
            segments,
            position: source.position,
            device_position: backend.clone_buffer(&source.device_position)?,
            shape: source.shape,
            dtype: source.dtype,
            revision: source.revision,
            graph_revision: None,
            pending: None,
        })
    }

    pub(crate) fn prepare_append(
        &mut self,
        backend: &mut B,
        layers: usize,
        tokens: usize,
        growth_tokens: usize,
    ) -> Result<KvAppendRange, BackendError> {
        let tokens = NonZeroUsize::new(tokens).ok_or(BackendError::Zero {
            field: "KV append tokens",
        })?;
        self.check_append_end(tokens.get())?;
        let mut rollback = self.capture_append_rollback()?;
        let result =
            self.prepare_append_staged(backend, layers, tokens, growth_tokens, &mut rollback);
        match result {
            Ok(range) => Ok(range),
            Err(error) => {
                self.restore_append_position(rollback);
                Err(error)
            }
        }
    }

    fn prepare_append_staged(
        &mut self,
        backend: &mut B,
        layers: usize,
        tokens: NonZeroUsize,
        growth_tokens: usize,
        rollback: &mut KvAppendRollback<B>,
    ) -> Result<KvAppendRange, BackendError> {
        if let Some(range) = self.pending_append(tokens, rollback)? {
            return Ok(range);
        }
        if let Some((segment_index, _capacity)) = self.reusable_tail(tokens.get()) {
            let range = KvAppendRange {
                start: self.position,
                tokens,
                segment_index,
            };
            self.pending = Some(range);
            return Ok(range);
        }
        self.allocate_append(backend, layers, tokens, growth_tokens)
    }

    pub(crate) fn prepare_append_at(
        &mut self,
        backend: &mut B,
        layers: usize,
        position: usize,
        tokens: usize,
        growth_tokens: usize,
    ) -> Result<KvAppendRange, BackendError> {
        let rollback = self.stage_append_position(position, tokens)?;
        match self.prepare_append(backend, layers, tokens, growth_tokens) {
            Ok(range) => Ok(range),
            Err(error) => {
                self.restore_append_position(rollback);
                Err(error)
            }
        }
    }

    fn stage_append_position(
        &mut self,
        position: usize,
        tokens: usize,
    ) -> Result<KvAppendRollback<B>, BackendError> {
        if position > self.position {
            return Err(BackendError::PositionOutOfBounds {
                position,
                max_context: self.position,
            });
        }
        let mut rollback = self.capture_append_rollback()?;
        let keep_pending = self.pending_reusable_at(position, tokens);
        let pending_removed = if keep_pending {
            false
        } else {
            self.detach_pending(&mut rollback)
        };
        let removed_segments = if keep_pending {
            false
        } else {
            self.remove_segments_at_or_after(position, &mut rollback)
        };
        let shortened_tail = if keep_pending {
            false
        } else {
            self.shorten_tail(position)
        };
        let changed = pending_removed || removed_segments || shortened_tail;
        self.position = position;
        let increments = u64::from(pending_removed)
            .checked_add(u64::from(changed || rollback.original_position == position))
            .ok_or(BackendError::SizeOverflow {
                field: "KV layout revision",
            })?;
        self.finish_staged_revision(rollback, increments)
    }

    fn capture_append_rollback(&self) -> Result<KvAppendRollback<B>, BackendError> {
        let mut original_committed = Vec::new();
        reserve_vec(
            &mut original_committed,
            self.segments.len(),
            "allocate KV rollback tokens",
        )?;
        original_committed.extend(self.segments.iter().map(|segment| segment.committed_tokens));
        let mut removed = Vec::new();
        reserve_vec(
            &mut removed,
            self.segments.len(),
            "allocate KV rollback segments",
        )?;
        Ok(KvAppendRollback {
            original_position: self.position,
            original_shape: self.shape,
            original_revision: self.revision,
            original_graph_revision: self.graph_revision,
            original_pending: self.pending,
            original_len: self.segments.len(),
            original_committed,
            removed,
        })
    }

    fn pending_reusable_at(&mut self, position: usize, tokens: usize) -> bool {
        let Some(range) = self.pending else {
            return false;
        };
        position == self.position
            && range.start == position
            && range.tokens.get() == tokens
            && self.pending_target_writable(range)
    }

    fn detach_pending(&mut self, rollback: &mut KvAppendRollback<B>) -> bool {
        let Some(range) = self.pending.take() else {
            return false;
        };
        let is_empty_tail = range.segment_index == self.segments.len().saturating_sub(1)
            && self
                .segments
                .last()
                .is_some_and(|segment| segment.committed_tokens == 0);
        if is_empty_tail {
            if let Some(segment) = self.segments.pop() {
                rollback.removed.push(segment);
            }
        }
        true
    }

    fn remove_segments_at_or_after(
        &mut self,
        position: usize,
        rollback: &mut KvAppendRollback<B>,
    ) -> bool {
        let mut removed = false;
        while self
            .segments
            .last()
            .is_some_and(|segment| segment.logical_start >= position)
        {
            let Some(segment) = self.segments.pop() else {
                break;
            };
            rollback.removed.push(segment);
            removed = true;
        }
        removed
    }

    fn finish_staged_revision(
        &mut self,
        rollback: KvAppendRollback<B>,
        increments: u64,
    ) -> Result<KvAppendRollback<B>, BackendError> {
        if increments == 0 {
            return Ok(rollback);
        }
        let Some(revision) = self.revision.checked_add(increments) else {
            self.restore_append_position(rollback);
            return Err(BackendError::SizeOverflow {
                field: "KV layout revision",
            });
        };
        self.revision = revision;
        Ok(rollback)
    }

    fn shorten_tail(&mut self, position: usize) -> bool {
        let Some(segment) = self.segments.last_mut() else {
            return false;
        };
        let visible = position.saturating_sub(segment.logical_start);
        if visible < segment.committed_tokens {
            segment.committed_tokens = visible;
            true
        } else {
            false
        }
    }

    fn restore_append_position(&mut self, rollback: KvAppendRollback<B>) {
        let survivor_len = rollback.original_len.saturating_sub(rollback.removed.len());
        self.segments.truncate(survivor_len);
        for (segment, committed) in self
            .segments
            .iter_mut()
            .zip(rollback.original_committed.iter().copied())
        {
            segment.committed_tokens = committed;
        }
        self.segments.extend(rollback.removed.into_iter().rev());
        self.position = rollback.original_position;
        self.shape = rollback.original_shape;
        self.revision = rollback.original_revision;
        self.graph_revision = rollback.original_graph_revision;
        self.pending = rollback.original_pending;
    }

    fn check_append_end(&self, tokens: usize) -> Result<(), BackendError> {
        let end = self
            .position
            .checked_add(tokens)
            .ok_or(BackendError::SizeOverflow {
                field: "KV append end",
            })?;
        if end > self.shape.max_context() {
            return Err(BackendError::PositionOutOfBounds {
                position: end,
                max_context: self.shape.max_context(),
            });
        }
        Ok(())
    }

    fn pending_append(
        &mut self,
        tokens: NonZeroUsize,
        rollback: &mut KvAppendRollback<B>,
    ) -> Result<Option<KvAppendRange>, BackendError> {
        let Some(pending) = self.pending else {
            return Ok(None);
        };
        if pending.start == self.position {
            if pending.tokens == tokens && self.pending_target_writable(pending) {
                return Ok(Some(pending));
            }
            if pending.tokens != tokens {
                if let Some(range) = self.retarget_pending(pending, tokens)? {
                    return Ok(Some(range));
                }
            }
            self.discard_pending_with_rollback(rollback)?;
        }
        if self.pending.is_some() {
            return Err(BackendError::operation(
                "prepare KV append",
                "another append target is pending",
            ));
        }
        Ok(None)
    }

    fn discard_pending_with_rollback(
        &mut self,
        rollback: &mut KvAppendRollback<B>,
    ) -> Result<(), BackendError> {
        if self.pending.is_none() {
            return Ok(());
        }
        let revision = self
            .revision
            .checked_add(1)
            .ok_or(BackendError::SizeOverflow {
                field: "KV layout revision",
            })?;
        self.detach_pending(rollback);
        self.pending = None;
        self.revision = revision;
        Ok(())
    }

    fn retarget_pending(
        &mut self,
        pending: KvAppendRange,
        tokens: NonZeroUsize,
    ) -> Result<Option<KvAppendRange>, BackendError> {
        let range = KvAppendRange {
            start: pending.start,
            tokens,
            segment_index: pending.segment_index,
        };
        if !self.pending_target_writable(range) {
            return Ok(None);
        }
        let revision = self
            .revision
            .checked_add(1)
            .ok_or(BackendError::SizeOverflow {
                field: "KV layout revision",
            })?;
        self.pending = Some(range);
        self.revision = revision;
        Ok(Some(range))
    }

    fn allocate_append(
        &mut self,
        backend: &mut B,
        layers: usize,
        tokens: NonZeroUsize,
        growth_tokens: usize,
    ) -> Result<KvAppendRange, BackendError> {
        let revision = self
            .revision
            .checked_add(1)
            .ok_or(BackendError::SizeOverflow {
                field: "KV layout revision",
            })?;
        let remaining = self.shape.max_context().checked_sub(self.position).ok_or(
            BackendError::SizeOverflow {
                field: "KV remaining context",
            },
        )?;
        let requested_capacity = growth_tokens.max(tokens.get()).max(1);
        let capacity = requested_capacity.min(remaining);
        let capacity = self.dtype_capacity(capacity, remaining)?;
        reserve_vec(&mut self.segments, 1, "allocate KV segments")?;
        let storage = self.allocate_storage(backend, layers, capacity)?;
        let segment_index = self.segments.len();
        self.segments.push(KvSegment {
            logical_start: self.position,
            committed_tokens: 0,
            storage: Rc::new(storage),
        });
        let range = KvAppendRange {
            start: self.position,
            tokens,
            segment_index,
        };
        self.pending = Some(range);
        self.revision = revision;
        Ok(range)
    }

    fn reusable_tail(&mut self, tokens: usize) -> Option<(usize, usize)> {
        let index = self.segments.len().checked_sub(1)?;
        let (logical_start, committed_tokens) = {
            let segment = self.segments.get(index)?;
            (segment.logical_start, segment.committed_tokens)
        };
        if logical_start + committed_tokens != self.position {
            return None;
        }
        let storage = Rc::get_mut(&mut self.segments[index].storage)?;
        let remaining = storage
            .capacity_tokens
            .get()
            .checked_sub(committed_tokens)?;
        (remaining >= tokens).then_some((index, remaining))
    }

    fn pending_target_writable(&mut self, range: KvAppendRange) -> bool {
        self.segments
            .get_mut(range.segment_index)
            .and_then(|segment| {
                let local_start = range.start.checked_sub(segment.logical_start)?;
                let storage = Rc::get_mut(&mut segment.storage)?;
                local_start
                    .checked_add(range.tokens.get())
                    .filter(|end| *end <= storage.capacity_tokens.get())
                    .map(|_| storage)
            })
            .is_some()
    }

    fn dtype_capacity(&self, requested: usize, remaining: usize) -> Result<usize, BackendError> {
        let capacity = requested;
        if self.dtype == KvCacheDtype::Q8 && !self.shape.head_dim().is_multiple_of(32) {
            return Err(BackendError::NotDivisible {
                field: "Q8 KV head dimension",
                value: self.shape.head_dim(),
                divisor: 32,
            });
        }
        if capacity > remaining {
            return Err(BackendError::operation(
                "allocate KV segment",
                "the physical capacity cannot hold complete KV blocks within context",
            ));
        }
        NonZeroUsize::new(capacity)
            .ok_or(BackendError::Zero {
                field: "KV segment capacity",
            })
            .map(NonZeroUsize::get)
    }

    fn allocate_storage(
        &self,
        backend: &mut B,
        layers: usize,
        capacity: usize,
    ) -> Result<KvStorage<B>, BackendError> {
        let (layout, bytes) = self.storage_layout(layers, capacity)?;
        let layers = Self::allocate_layers(backend, layout, layers)?;
        Ok(KvStorage {
            capacity_tokens: NonZeroUsize::new(capacity).ok_or(BackendError::Zero {
                field: "KV segment capacity",
            })?,
            bytes,
            layers,
        })
    }

    fn storage_layout(
        &self,
        layers: usize,
        capacity: usize,
    ) -> Result<(BufferLayout, u64), BackendError> {
        let elements = self
            .shape
            .n_head_kv()
            .checked_mul(capacity)
            .and_then(|value| value.checked_mul(self.shape.head_dim()))
            .ok_or(BackendError::SizeOverflow {
                field: "KV segment elements",
            })?;
        let layout = self.dtype.layout(elements)?;
        let bytes_per_layer =
            u64::try_from(layout.bytes()).map_err(|_| BackendError::SizeOverflow {
                field: "KV segment bytes",
            })?;
        let layer_count = u64::try_from(layers).map_err(|_| BackendError::SizeOverflow {
            field: "KV layer count",
        })?;
        let bytes = bytes_per_layer
            .checked_mul(layer_count)
            .and_then(|value| value.checked_mul(2))
            .ok_or(BackendError::SizeOverflow {
                field: "KV segment bytes",
            })?;
        Ok((layout, bytes))
    }

    fn allocate_layers(
        backend: &mut B,
        layout: BufferLayout,
        layers: usize,
    ) -> Result<Box<[KvLayer<B>]>, BackendError> {
        let mut storage = Vec::new();
        reserve_vec(&mut storage, layers, "allocate KV layers")?;
        for _ in 0..layers {
            let key = backend.allocate_classified(layout, MemoryClass::KvCache)?;
            let value = backend.allocate_classified(layout, MemoryClass::KvCache)?;
            storage.push(KvLayer { key, value });
        }
        Ok(storage.into_boxed_slice())
    }

    pub(crate) fn write_span(
        &mut self,
        layer: usize,
        range: KvAppendRange,
    ) -> Result<KvWriteSpan<'_, B::Buffer>, BackendError> {
        if self.pending != Some(range) {
            return Err(BackendError::operation(
                "write KV span",
                "the append target is not pending",
            ));
        }
        let segment = self
            .segments
            .get_mut(range.segment_index)
            .ok_or(BackendError::operation(
                "write KV span",
                "segment is missing",
            ))?;
        let storage = Rc::get_mut(&mut segment.storage).ok_or(BackendError::operation(
            "write KV span",
            "the append segment is shared",
        ))?;
        let layer = storage
            .layers
            .get_mut(layer)
            .ok_or(BackendError::operation("write KV span", "layer is missing"))?;
        KvWriteSpan::new(
            &mut layer.key,
            &mut layer.value,
            segment.logical_start,
            storage.capacity_tokens.get(),
        )
    }

    pub(crate) fn read_spans(
        &self,
        layer: usize,
    ) -> Result<Vec<KvReadSpan<'_, B::Buffer>>, BackendError> {
        let mut spans = Vec::new();
        reserve_vec(&mut spans, self.segments.len(), "allocate KV read spans")?;
        for (index, segment) in self.segments.iter().enumerate() {
            let layer = segment
                .storage
                .layers
                .get(layer)
                .ok_or(BackendError::operation("read KV spans", "layer is missing"))?;
            let tokens = if self.pending.map(|range| range.segment_index) == Some(index) {
                segment.storage.capacity_tokens.get()
            } else {
                segment.committed_tokens
            };
            if tokens == 0 {
                continue;
            }
            spans.push(KvReadSpan::new(
                &layer.key,
                &layer.value,
                segment.logical_start,
                tokens,
                segment.storage.capacity_tokens.get(),
            )?);
        }
        Ok(spans)
    }

    pub(crate) fn commit_append(&mut self, range: KvAppendRange) -> Result<(), BackendError> {
        if self.pending != Some(range) {
            return Err(BackendError::operation(
                "commit KV append",
                "the append target is not pending",
            ));
        }
        let segment = self
            .segments
            .get_mut(range.segment_index)
            .ok_or(BackendError::operation(
                "commit KV append",
                "segment is missing",
            ))?;
        let local_start =
            range
                .start
                .checked_sub(segment.logical_start)
                .ok_or(BackendError::SizeOverflow {
                    field: "KV local append start",
                })?;
        if segment.committed_tokens != local_start {
            return Err(BackendError::operation(
                "commit KV append",
                "the append range does not follow the committed prefix",
            ));
        }
        let committed =
            local_start
                .checked_add(range.tokens.get())
                .ok_or(BackendError::SizeOverflow {
                    field: "KV committed tokens",
                })?;
        if committed > segment.storage.capacity_tokens.get() {
            return Err(BackendError::PositionOutOfBounds {
                position: committed,
                max_context: segment.storage.capacity_tokens.get(),
            });
        }
        segment.committed_tokens = committed;
        self.position =
            range
                .start
                .checked_add(range.tokens.get())
                .ok_or(BackendError::SizeOverflow {
                    field: "KV position",
                })?;
        self.pending = None;
        Ok(())
    }

    pub(crate) fn discard_pending(&mut self) -> Result<(), BackendError> {
        if self.pending.is_none() {
            return Ok(());
        }
        let revision = self
            .revision
            .checked_add(1)
            .ok_or(BackendError::SizeOverflow {
                field: "KV layout revision",
            })?;
        self.pending = None;
        self.segments
            .retain(|segment| segment.committed_tokens != 0);
        if let Some(last) = self.segments.last() {
            self.position = last.logical_start + last.committed_tokens;
        } else {
            self.position = 0;
        }
        self.revision = revision;
        Ok(())
    }

    pub(crate) fn truncate(&mut self, position: usize) -> Result<(), BackendError> {
        if position > self.position {
            return Err(BackendError::PositionOutOfBounds {
                position,
                max_context: self.position,
            });
        }
        let old_position = self.position;
        let mut rollback = self.capture_append_rollback()?;
        let pending_removed = self.detach_pending(&mut rollback);
        let removed_segments = self.remove_segments_at_or_after(position, &mut rollback);
        let shortened_tail = self.shorten_tail(position);
        let changed = pending_removed || removed_segments || shortened_tail;
        self.position = position;
        let increments = u64::from(pending_removed)
            .checked_add(u64::from(changed || old_position == position))
            .ok_or(BackendError::SizeOverflow {
                field: "KV layout revision",
            })?;
        self.finish_staged_revision(rollback, increments)
            .map(|_| ())
    }

    pub(crate) fn snapshot(&self, backend: &mut B) -> Result<KvSnapshot, BackendError> {
        let committed_segments = self
            .segments
            .iter()
            .filter(|segment| segment.committed_tokens != 0)
            .count();
        let mut segments = Vec::new();
        segments
            .try_reserve_exact(committed_segments)
            .map_err(|error| BackendError::operation("allocate KV snapshot segments", error))?;
        for segment in self
            .segments
            .iter()
            .filter(|segment| segment.committed_tokens != 0)
        {
            segments.push(Self::snapshot_segment(backend, segment)?);
        }
        Ok(KvSnapshot {
            segments,
            position: self.position,
            device_position: backend.download_buffer(&self.device_position)?,
            shape: self.shape,
            dtype: self.dtype,
        })
    }

    fn snapshot_segment(
        backend: &mut B,
        segment: &KvSegment<B>,
    ) -> Result<KvSnapshotSegment, BackendError> {
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(segment.storage.layers.len())
            .map_err(|error| BackendError::operation("allocate KV snapshot layers", error))?;
        for layer in &segment.storage.layers {
            layers.push((
                backend.download_buffer(&layer.key)?,
                backend.download_buffer(&layer.value)?,
            ));
        }
        Ok(KvSnapshotSegment {
            logical_start: segment.logical_start,
            committed_tokens: segment.committed_tokens,
            capacity_tokens: segment.storage.capacity_tokens,
            layers,
        })
    }

    pub(crate) fn snapshot_bytes(snapshot: &KvSnapshot) -> Result<u64, BackendError> {
        let position_bytes =
            u64::try_from(snapshot.device_position.layout().bytes()).map_err(|_| {
                BackendError::SizeOverflow {
                    field: "KV device position bytes",
                }
            })?;
        snapshot
            .segments
            .iter()
            .try_fold(position_bytes, |bytes, segment| {
                bytes
                    .checked_add(Self::snapshot_segment_bytes(segment)?)
                    .ok_or(BackendError::SizeOverflow {
                        field: "KV snapshot bytes",
                    })
            })
    }

    pub(crate) fn restore(backend: &mut B, snapshot: &KvSnapshot) -> Result<Self, BackendError> {
        let mut segments = Vec::new();
        segments
            .try_reserve_exact(snapshot.segments.len())
            .map_err(|error| BackendError::operation("allocate restored KV segments", error))?;
        for source in snapshot
            .segments
            .iter()
            .filter(|source| source.committed_tokens != 0)
        {
            segments.push(Self::restore_segment(backend, source)?);
        }
        Ok(Self {
            segments,
            position: snapshot.position,
            device_position: backend
                .restore_buffer_classified(&snapshot.device_position, MemoryClass::KvCache)?,
            shape: snapshot.shape,
            dtype: snapshot.dtype,
            revision: 0,
            graph_revision: None,
            pending: None,
        })
    }

    fn restore_segment(
        backend: &mut B,
        source: &KvSnapshotSegment,
    ) -> Result<KvSegment<B>, BackendError> {
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(source.layers.len())
            .map_err(|error| BackendError::operation("allocate restored KV layers", error))?;
        for (key, value) in &source.layers {
            layers.push(KvLayer {
                key: backend.restore_buffer_classified(key, MemoryClass::KvCache)?,
                value: backend.restore_buffer_classified(value, MemoryClass::KvCache)?,
            });
        }
        Ok(KvSegment {
            logical_start: source.logical_start,
            committed_tokens: source.committed_tokens,
            storage: Rc::new(KvStorage {
                capacity_tokens: source.capacity_tokens,
                bytes: Self::snapshot_segment_bytes(source)?,
                layers: layers.into_boxed_slice(),
            }),
        })
    }

    fn snapshot_segment_bytes(source: &KvSnapshotSegment) -> Result<u64, BackendError> {
        source
            .layers
            .first()
            .map_or(0, |(key, _)| key.layout().bytes() as u64)
            .checked_mul(u64::try_from(source.layers.len()).map_err(|_| {
                BackendError::SizeOverflow {
                    field: "KV layer count",
                }
            })?)
            .and_then(|value| value.checked_mul(2))
            .ok_or(BackendError::SizeOverflow {
                field: "KV segment bytes",
            })
    }
}

fn kv_metadata_bound_bytes<B: Backend>(
    state_segments: usize,
    snapshot_segments: usize,
    layer_count: usize,
) -> Result<u64, BackendError> {
    let state_segments =
        checked_container_bytes::<KvSegment<B>>(state_segments, "KV state segment metadata")?;
    let state_storage =
        checked_container_bytes::<KvStorage<B>>(snapshot_segments, "KV state storage metadata")?;
    let state_layers =
        checked_container_bytes::<KvLayer<B>>(layer_count, "KV state layer metadata")?;
    let snapshot_segments = checked_container_bytes::<KvSnapshotSegment>(
        snapshot_segments,
        "KV snapshot segment metadata",
    )?;
    let snapshot_layers = checked_container_bytes::<(BufferSnapshot, BufferSnapshot)>(
        layer_count,
        "KV snapshot layer metadata",
    )?;
    state_segments
        .checked_add(state_storage)
        .and_then(|bytes| bytes.checked_add(state_layers))
        .and_then(|bytes| bytes.checked_add(snapshot_segments))
        .and_then(|bytes| bytes.checked_add(snapshot_layers))
        .ok_or(BackendError::SizeOverflow {
            field: "KV metadata bound",
        })
}

fn metadata_vec_capacity(len: usize) -> Result<usize, BackendError> {
    len.checked_mul(2)
        .and_then(|capacity| capacity.checked_add(4))
        .ok_or(BackendError::SizeOverflow {
            field: "KV metadata vector capacity",
        })
}

impl KvSnapshot {
    pub(crate) fn restore_container_bytes<B: Backend>(&self) -> Result<u64, BackendError> {
        let segment_bytes = checked_container_bytes::<KvSegment<B>>(
            self.segments.len(),
            "restored KV segment metadata",
        )?;
        let storage_bytes = checked_container_bytes::<KvStorage<B>>(
            self.segments.len(),
            "restored KV storage metadata",
        )?;
        let layers = self.segments.iter().try_fold(0_usize, |layers, segment| {
            layers
                .checked_add(segment.layers.len())
                .ok_or(BackendError::SizeOverflow {
                    field: "restored KV layer metadata",
                })
        })?;
        let layer_bytes =
            checked_container_bytes::<KvLayer<B>>(layers, "restored KV layer metadata")?;
        segment_bytes
            .checked_add(storage_bytes)
            .ok_or(BackendError::SizeOverflow {
                field: "restored KV container metadata",
            })?
            .checked_add(layer_bytes)
            .ok_or(BackendError::SizeOverflow {
                field: "restored KV container metadata",
            })
    }
}

fn checked_container_bytes<T>(count: usize, field: &'static str) -> Result<u64, BackendError> {
    let count = u64::try_from(count).map_err(|_| BackendError::SizeOverflow { field })?;
    let element_bytes = u64::try_from(std::mem::size_of::<T>())
        .map_err(|_| BackendError::SizeOverflow { field })?;
    count
        .checked_mul(element_bytes)
        .ok_or(BackendError::SizeOverflow { field })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::KvReadView;
    use crate::{AttentionShape, CpuBackend, MemoryClass};

    fn test_shape() -> AttentionShape {
        AttentionShape::new(4, 2, 32, 64).expect("test shape is valid")
    }

    #[test]
    fn checked_kv_host_reservation_reports_capacity_overflow() {
        let mut values = Vec::<u8>::new();
        let error = reserve_vec(&mut values, usize::MAX, "allocate KV rollback tokens")
            .expect_err("capacity overflow must be returned");
        assert!(matches!(
            error,
            BackendError::Operation {
                operation: "allocate KV rollback tokens",
                ..
            }
        ));
        assert!(values.is_empty());
    }

    #[test]
    fn layer_reservation_rejects_overflow_before_backend_allocations() {
        let mut backend = CpuBackend::new();
        let layout = BufferLayout::u32(1).expect("test layout is valid");
        let error = KvState::<CpuBackend>::allocate_layers(&mut backend, layout, usize::MAX)
            .expect_err("layer metadata overflow must be returned");
        assert!(matches!(
            error,
            BackendError::Operation {
                operation: "allocate KV layers",
                ..
            }
        ));
        assert_eq!(backend.memory_accounting().live_allocations, 0);
    }

    fn segmented_state(backend: &mut CpuBackend) -> KvState<CpuBackend> {
        let mut state = KvState::new(backend, 2, test_shape(), KvCacheDtype::F16)
            .expect("state allocation succeeds");
        for _ in 0..3 {
            let range = state
                .prepare_append(backend, 2, 4, 4)
                .expect("segment allocation succeeds");
            state.commit_append(range).expect("segment commits");
        }
        state
    }

    fn assert_gap_free_spans(state: &KvState<CpuBackend>) {
        let spans = state.read_spans(0).expect("spans are readable");
        let view = KvReadView::new(&spans).expect("spans are gap free");
        let mut end = 0;
        for span in &spans {
            assert_eq!(span.logical_start(), end);
            end = span.mapped_end();
        }
        assert_eq!(view.mapped_tokens(), end);
    }

    fn stage_cut(
        backend: &mut CpuBackend,
        state: &mut KvState<CpuBackend>,
        cut: usize,
    ) -> KvAppendRange {
        state
            .prepare_append_at(backend, 2, cut, 1, 4)
            .expect("cut stages")
    }

    #[test]
    fn fork_shares_kv_storage_and_copies_only_position_state() {
        let mut backend = CpuBackend::new();
        let mut source = KvState::new(&mut backend, 2, test_shape(), KvCacheDtype::F16)
            .expect("state allocation succeeds");
        let range = source
            .prepare_append(&mut backend, 2, 4, 8)
            .expect("append allocation succeeds");
        source.commit_append(range).expect("append commits");
        let before = backend.memory_accounting().class(MemoryClass::KvCache);
        let child = KvState::fork(&mut backend, &source).expect("fork succeeds");
        let after = backend.memory_accounting().class(MemoryClass::KvCache);
        assert_eq!(after.live_bytes - before.live_bytes, 4);
        assert_eq!(source.physical_bytes() - 4, child.physical_bytes() - 4);
        assert_eq!(source.position, child.position);
    }

    #[test]
    fn source_append_after_fork_allocates_a_private_tail() {
        let mut backend = CpuBackend::new();
        let mut source = KvState::new(&mut backend, 1, test_shape(), KvCacheDtype::F16)
            .expect("state allocation succeeds");
        let range = source
            .prepare_append(&mut backend, 1, 2, 4)
            .expect("append allocation succeeds");
        source.commit_append(range).expect("append commits");
        let child = KvState::fork(&mut backend, &source).expect("fork succeeds");
        let before = backend.memory_accounting().class(MemoryClass::KvCache);
        let range = source
            .prepare_append(&mut backend, 1, 1, 4)
            .expect("copy-on-write allocation succeeds");
        let after = backend.memory_accounting().class(MemoryClass::KvCache);
        assert!(after.live_bytes > before.live_bytes);
        source.commit_append(range).expect("append commits");
        assert_eq!(child.position, 2);
        assert_eq!(source.position, 3);
    }

    #[test]
    fn truncate_drops_private_segments_without_rewriting_shared_prefix() {
        let mut backend = CpuBackend::new();
        let mut source = KvState::new(&mut backend, 1, test_shape(), KvCacheDtype::F16)
            .expect("state allocation succeeds");
        let first = source
            .prepare_append(&mut backend, 1, 2, 2)
            .expect("first append allocation succeeds");
        source.commit_append(first).expect("first append commits");
        let child = KvState::fork(&mut backend, &source).expect("fork succeeds");
        let second = source
            .prepare_append(&mut backend, 1, 2, 2)
            .expect("second append allocation succeeds");
        source.commit_append(second).expect("second append commits");
        let revision = source.revision();
        source.truncate(2).expect("truncate succeeds");
        assert!(source.revision() > revision);
        assert_eq!(source.position, child.position);
        assert_eq!(
            backend
                .memory_accounting()
                .class(MemoryClass::KvCache)
                .live_allocations,
            4
        );
    }

    #[test]
    fn shared_pending_tail_is_restaged_and_invalidates_the_old_plan() {
        let mut backend = CpuBackend::new();
        let mut source = KvState::new(&mut backend, 1, test_shape(), KvCacheDtype::F16)
            .expect("state allocation succeeds");
        let first = source
            .prepare_append(&mut backend, 1, 2, 4)
            .expect("first append allocation succeeds");
        source.commit_append(first).expect("first append commits");
        let pending = source
            .prepare_append(&mut backend, 1, 1, 4)
            .expect("pending append stages");
        let child = KvState::fork(&mut backend, &source).expect("fork succeeds");
        let revision = source.revision();
        let restaged = source
            .prepare_append(&mut backend, 1, 1, 4)
            .expect("shared pending append restages");
        assert_ne!(restaged.segment_index, pending.segment_index);
        assert!(source.revision() > revision);
        assert_eq!(child.position, 2);
    }

    #[test]
    fn truncating_at_the_committed_position_invalidates_graph_state() {
        let mut backend = CpuBackend::new();
        let mut state = KvState::new(&mut backend, 1, test_shape(), KvCacheDtype::F16)
            .expect("state allocation succeeds");
        let range = state
            .prepare_append(&mut backend, 1, 2, 2)
            .expect("append allocation succeeds");
        state.commit_append(range).expect("append commits");
        let revision = state.revision();
        state.truncate(state.position).expect("truncate succeeds");
        assert!(state.revision() > revision);
    }

    #[test]
    fn failed_tail_allocation_preserves_committed_state_and_releases_partials() {
        let mut backend = CpuBackend::new();
        let mut state = KvState::new(&mut backend, 1, test_shape(), KvCacheDtype::F16)
            .expect("state allocation succeeds");
        let first = state
            .prepare_append(&mut backend, 1, 2, 2)
            .expect("first append allocation succeeds");
        state.commit_append(first).expect("append commits");
        let before = backend.memory_accounting();
        backend.fail_allocations_after(1);
        let result = state.prepare_append(&mut backend, 2, 1, 4);
        assert!(result.is_err());
        assert_eq!(state.position, 2);
        assert_eq!(state.segments.len(), 1);
        let after = backend.memory_accounting();
        assert_eq!(after.live_bytes, before.live_bytes);
        assert_eq!(after.live_allocations, before.live_allocations);
    }

    #[test]
    fn failed_staged_branch_append_restores_the_warm_layout() {
        let mut backend = CpuBackend::new();
        let mut state = KvState::new(&mut backend, 1, test_shape(), KvCacheDtype::F16)
            .expect("state allocation succeeds");
        let first = state
            .prepare_append(&mut backend, 1, 2, 2)
            .expect("first append allocation succeeds");
        state.commit_append(first).expect("first append commits");
        let second = state
            .prepare_append(&mut backend, 1, 2, 2)
            .expect("second append allocation succeeds");
        state.commit_append(second).expect("second append commits");
        let before = backend.memory_accounting();
        let revision = state.revision();
        backend.fail_allocations_after(0);
        let result = state.prepare_append_at(&mut backend, 1, 2, 1, 4);
        assert!(result.is_err());
        assert_eq!(state.position, 4);
        assert_eq!(state.revision(), revision);
        assert_eq!(state.segments.len(), 2);
        assert_eq!(state.segments[0].committed_tokens, 2);
        assert_eq!(state.segments[1].committed_tokens, 2);
        let after = backend.memory_accounting();
        assert_eq!(after.live_bytes, before.live_bytes);
        assert_eq!(after.live_allocations, before.live_allocations);
    }

    #[test]
    fn staged_branch_append_commits_only_after_the_target_is_ready() {
        let mut backend = CpuBackend::new();
        let mut state = KvState::new(&mut backend, 1, test_shape(), KvCacheDtype::F16)
            .expect("state allocation succeeds");
        let first = state
            .prepare_append(&mut backend, 1, 2, 2)
            .expect("first append allocation succeeds");
        state.commit_append(first).expect("first append commits");
        let second = state
            .prepare_append(&mut backend, 1, 2, 2)
            .expect("second append allocation succeeds");
        state.commit_append(second).expect("second append commits");
        let range = state
            .prepare_append_at(&mut backend, 1, 2, 1, 4)
            .expect("branch append stages");
        assert_eq!(range.start, 2);
        assert_eq!(state.position, 2);
        state.commit_append(range).expect("branch append commits");
        assert_eq!(state.position, 3);
    }

    #[test]
    fn interior_branch_shortens_the_retained_tail_after_removing_segments() {
        let mut backend = CpuBackend::new();
        let mut state = segmented_state(&mut backend);
        let range = state
            .prepare_append_at(&mut backend, 2, 6, 1, 4)
            .expect("interior branch stages");
        assert_eq!(range.start, 6);
        assert_eq!(state.segments.len(), 2);
        assert_eq!(state.segments[0].committed_tokens, 4);
        assert_eq!(state.segments[1].committed_tokens, 2);
        assert_gap_free_spans(&state);
        state.commit_append(range).expect("branch append commits");
        assert_eq!(state.position, 7);
        assert_gap_free_spans(&state);
    }

    #[test]
    fn staged_branch_cuts_keep_contiguous_spans_for_shared_and_private_prefixes() {
        let cuts = [0, 2, 4, 6, 8, 10, 12];
        for shared in [false, true] {
            for cut in cuts {
                let mut backend = CpuBackend::new();
                let mut source = segmented_state(&mut backend);
                if shared {
                    let mut branch = KvState::fork(&mut backend, &source).expect("fork succeeds");
                    let range = stage_cut(&mut backend, &mut branch, cut);
                    assert_gap_free_spans(&branch);
                    branch.commit_append(range).expect("cut commits");
                    assert_eq!(branch.position, cut + 1);
                    assert_eq!(source.position, 12);
                    assert_gap_free_spans(&source);
                } else {
                    let range = stage_cut(&mut backend, &mut source, cut);
                    assert_gap_free_spans(&source);
                    source.commit_append(range).expect("cut commits");
                    assert_eq!(source.position, cut + 1);
                }
            }
        }
    }

    #[test]
    fn allocation_failure_restores_every_partition_cut() {
        for cut in [0, 2, 4, 6, 8, 10, 12] {
            let mut backend = CpuBackend::new();
            let mut state = segmented_state(&mut backend);
            let before = backend.memory_accounting();
            let revision = state.revision();
            backend.fail_allocations_after(0);
            let result = state.prepare_append_at(&mut backend, 2, cut, 3, 4);
            assert!(result.is_err());
            assert_eq!(state.position, 12);
            assert_eq!(state.revision(), revision);
            assert_eq!(state.segments.len(), 3);
            assert!(state
                .segments
                .iter()
                .all(|segment| segment.committed_tokens == 4));
            assert_gap_free_spans(&state);
            let after = backend.memory_accounting();
            assert_eq!(after.live_bytes, before.live_bytes);
            assert_eq!(after.live_allocations, before.live_allocations);
        }
    }

    #[test]
    fn matching_pending_append_survives_transactional_staging() {
        let mut backend = CpuBackend::new();
        let mut state = KvState::new(&mut backend, 1, test_shape(), KvCacheDtype::F16)
            .expect("state allocation succeeds");
        let first = state
            .prepare_append(&mut backend, 1, 2, 4)
            .expect("first append stages");
        state.commit_append(first).expect("first append commits");
        let pending = state
            .prepare_append(&mut backend, 1, 1, 4)
            .expect("pending append stages");
        let restaged = state
            .prepare_append_at(&mut backend, 1, 2, 1, 4)
            .expect("matching pending append remains valid");
        assert_eq!(restaged, pending);
        state
            .commit_append(restaged)
            .expect("pending append commits");
        assert_eq!(state.position, 3);
    }

    #[test]
    fn shape_growth_reuses_pending_capture_tail_without_allocation() {
        let mut backend = CpuBackend::new();
        let mut state = KvState::new(&mut backend, 1, test_shape(), KvCacheDtype::F16)
            .expect("state allocation succeeds");
        let committed = state
            .prepare_append(&mut backend, 1, 4, 4)
            .expect("committed prefix allocates");
        state
            .commit_append(committed)
            .expect("committed prefix commits");
        let pending = state
            .prepare_append(&mut backend, 1, 1, 4)
            .expect("capture tail stages");
        let before = backend.memory_accounting();
        let grown_shape = AttentionShape::new(4, 2, 32, 128).expect("grown shape is valid");
        backend.fail_allocations_after(0);
        state
            .prepare_shape_transition(&mut backend, 1, grown_shape, 4, Some((1, 4)))
            .expect("matching pending tail is reused");
        assert_eq!(state.pending, Some(pending));
        assert_eq!(state.shape, grown_shape);
        assert_eq!(backend.memory_accounting(), before);
        state
            .commit_append(pending)
            .expect("reused capture tail commits");
        assert_eq!(state.position, 5);
    }

    #[test]
    fn failed_shape_growth_restores_pending_capture_tail() {
        let mut backend = CpuBackend::new();
        let mut state = KvState::new(&mut backend, 1, test_shape(), KvCacheDtype::F16)
            .expect("state allocation succeeds");
        let committed = state
            .prepare_append(&mut backend, 1, 4, 4)
            .expect("committed prefix allocates");
        state
            .commit_append(committed)
            .expect("committed prefix commits");
        let pending = state
            .prepare_append(&mut backend, 1, 1, 4)
            .expect("capture tail stages");
        let before = backend.memory_accounting();
        let revision = state.revision();
        let shape = state.shape;
        let grown_shape = AttentionShape::new(4, 2, 32, 128).expect("grown shape is valid");
        backend.fail_allocations_after(0);
        let error = state
            .prepare_shape_transition(&mut backend, 1, grown_shape, 4, Some((5, 8)))
            .expect_err("new tail allocation is denied");
        assert!(error.to_string().contains("allocate CPU buffer"));
        assert_eq!(state.pending, Some(pending));
        assert_eq!(state.shape, shape);
        assert_eq!(state.revision(), revision);
        assert_eq!(state.position, 4);
        assert_eq!(backend.memory_accounting(), before);
    }

    #[test]
    fn pending_capture_range_retargets_for_a_wider_verifier_append() {
        let mut backend = CpuBackend::new();
        let mut state = KvState::new(&mut backend, 1, test_shape(), KvCacheDtype::F16)
            .expect("state allocation succeeds");
        let committed = state
            .prepare_append(&mut backend, 1, 4, 4)
            .expect("committed prefix allocates");
        state
            .commit_append(committed)
            .expect("committed prefix commits");
        let captured = state
            .prepare_append(&mut backend, 1, 1, 32)
            .expect("capture target stages");
        let revision = state.revision();

        let verifier = state
            .prepare_append(&mut backend, 1, 4, 4)
            .expect("verifier target retargets");

        assert_eq!(verifier.segment_index, captured.segment_index);
        assert_eq!(verifier.start, captured.start);
        assert_eq!(verifier.tokens.get(), 4);
        assert!(state.revision() > revision);
        state
            .commit_append(verifier)
            .expect("verifier target commits");
        assert_eq!(state.position, 8);
    }

    #[test]
    fn mismatched_pending_append_restores_on_allocation_failure() {
        let mut backend = CpuBackend::new();
        let mut state = KvState::new(&mut backend, 1, test_shape(), KvCacheDtype::F16)
            .expect("state allocation succeeds");
        let first = state
            .prepare_append(&mut backend, 1, 4, 4)
            .expect("first append stages");
        state.commit_append(first).expect("first append commits");
        let pending = state
            .prepare_append(&mut backend, 1, 1, 4)
            .expect("pending append stages");
        let before = backend.memory_accounting();
        let revision = state.revision();
        backend.fail_allocations_after(0);
        let result = state.prepare_append_at(&mut backend, 1, 4, 2, 4);
        assert!(result.is_err());
        assert_eq!(state.pending, Some(pending));
        assert_eq!(state.position, 4);
        assert_eq!(state.revision(), revision);
        let after = backend.memory_accounting();
        assert_eq!(after.live_bytes, before.live_bytes);
        assert_eq!(after.live_allocations, before.live_allocations);
    }

    #[test]
    fn failed_wider_verifier_append_restores_the_captured_pending_range() {
        let mut backend = CpuBackend::new();
        let mut state = KvState::new(&mut backend, 1, test_shape(), KvCacheDtype::F16)
            .expect("state allocation succeeds");
        let committed = state
            .prepare_append(&mut backend, 1, 4, 4)
            .expect("committed prefix allocates");
        state
            .commit_append(committed)
            .expect("committed prefix commits");
        let captured = state
            .prepare_append(&mut backend, 1, 1, 1)
            .expect("capture target stages");
        let before = backend.memory_accounting();
        let revision = state.revision();
        backend.fail_allocations_after(0);

        let result = state.prepare_append(&mut backend, 1, 2, 2);

        assert!(result.is_err());
        assert_eq!(state.pending, Some(captured));
        assert_eq!(state.position, 4);
        assert_eq!(state.revision(), revision);
        assert_eq!(state.segments.len(), 2);
        let after = backend.memory_accounting();
        assert_eq!(after.live_bytes, before.live_bytes);
        assert_eq!(after.live_allocations, before.live_allocations);
    }

    #[test]
    fn snapshot_omits_empty_pending_tails_and_reports_captured_bytes() {
        let mut backend = CpuBackend::new();
        let mut state = KvState::new(&mut backend, 1, test_shape(), KvCacheDtype::F16)
            .expect("state allocation succeeds");
        let committed = state
            .prepare_append(&mut backend, 1, 2, 2)
            .expect("committed segment allocates");
        state
            .commit_append(committed)
            .expect("committed segment commits");
        state
            .prepare_append(&mut backend, 1, 1, 8)
            .expect("pending tail allocates");
        let snapshot = state.snapshot(&mut backend).expect("snapshot succeeds");

        assert_eq!(snapshot.segments.len(), 1);
        assert_eq!(snapshot.segments[0].committed_tokens, 2);
        let snapshot_bytes =
            KvState::<CpuBackend>::snapshot_bytes(&snapshot).expect("snapshot bytes are checked");
        assert!(snapshot_bytes < state.physical_bytes());
        let restored = KvState::restore(&mut backend, &snapshot).expect("restore succeeds");
        assert_eq!(restored.physical_bytes(), snapshot_bytes);
        assert_eq!(restored.pending, None);
    }

    #[test]
    fn revision_overflow_restores_staged_and_truncated_layouts() {
        let mut backend = CpuBackend::new();
        let mut state = KvState::new(&mut backend, 1, test_shape(), KvCacheDtype::F16)
            .expect("state allocation succeeds");
        let range = state
            .prepare_append(&mut backend, 1, 4, 4)
            .expect("append stages");
        state.commit_append(range).expect("append commits");
        state.revision = u64::MAX;
        let result = state.prepare_append_at(&mut backend, 1, 2, 1, 4);
        assert!(result.is_err());
        assert_eq!(state.position, 4);
        assert_eq!(state.revision(), u64::MAX);
        assert_eq!(state.segments[0].committed_tokens, 4);
        assert!(state.truncate(4).is_err());
        assert_eq!(state.position, 4);
        assert_eq!(state.revision(), u64::MAX);
        assert_eq!(state.segments[0].committed_tokens, 4);
    }

    #[test]
    fn revision_overflow_rejects_direct_append_before_allocation() {
        let mut backend = CpuBackend::new();
        let mut state = KvState::new(&mut backend, 1, test_shape(), KvCacheDtype::F16)
            .expect("state allocation succeeds");
        state.revision = u64::MAX;
        let before = backend.memory_accounting();
        let result = state.prepare_append(&mut backend, 1, 1, 4);
        assert!(result.is_err());
        assert!(state.segments.is_empty());
        assert_eq!(state.pending, None);
        assert_eq!(state.revision(), u64::MAX);
        let after = backend.memory_accounting();
        assert_eq!(after.live_bytes, before.live_bytes);
        assert_eq!(after.live_allocations, before.live_allocations);
    }
}
