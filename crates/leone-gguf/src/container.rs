use crate::alloc::{
    AllocationBudget, AllocationGuard, AllocationReservation, UnlimitedAllocationBudget,
};
use crate::pread::ReadOnlyFile;
use crate::{Error, Result};
use std::borrow::Borrow;
use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::io::{BufReader, Read};
use std::path::Path;

const GGUF_MAGIC: [u8; 4] = *b"GGUF";
const GGUF_VERSION: u32 = 3;
const DEFAULT_ALIGNMENT: u32 = 32;
const MAX_COLLECTION_LEN: u64 = 16_777_216;
const MAX_STRING_LEN: u64 = 1_073_741_824;
const MIN_TENSOR_DESCRIPTOR_BYTES: u64 = 32;
const PARSER_BUFFER_BYTES: u64 = 8 * 1024;
const PARSER_RESERVATION_CAPACITY: usize = 64;
const PARSER_MAX_RETAINED_RESERVATION_WRAPPERS: usize = PARSER_RESERVATION_CAPACITY - 1;
const PARSER_RESERVATION_METADATA_BYTES: u64 = 8 * 1024;
const PARSER_RESERVATION_WRAPPER_BOUND_BYTES: u64 = 128;
const _: () = assert!(
    PARSER_RESERVATION_METADATA_BYTES
        >= (PARSER_RESERVATION_CAPACITY as u64) * PARSER_RESERVATION_WRAPPER_BOUND_BYTES
);
// The pinned hash table uses power-of-two buckets, one control byte per bucket,
// and a small alignment tail. This bound deliberately overestimates that layout.
const HASH_BUCKET_FACTOR: u64 = 2;
const HASH_CONTROL_AND_ALIGNMENT_BYTES: u64 = 64;
// The ordered metadata tree reserves a node-sized factor per entry. The fixed
// term covers links, alignment, and node bookkeeping; it is a reservation bound.
const ORDERED_NODE_ENTRY_FACTOR: u64 = 8;
const ORDERED_NODE_OVERHEAD_BYTES: u64 = 128;

/// A GGUF metadata value type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl TryFrom<u32> for ValueType {
    type Error = Error;

    fn try_from(value: u32) -> Result<Self> {
        VALUE_TYPES
            .get(value as usize)
            .copied()
            .ok_or(Error::InvalidValueType(value))
    }
}

const VALUE_TYPES: [ValueType; 13] = [
    ValueType::Uint8,
    ValueType::Int8,
    ValueType::Uint16,
    ValueType::Int16,
    ValueType::Uint32,
    ValueType::Int32,
    ValueType::Float32,
    ValueType::Bool,
    ValueType::String,
    ValueType::Array,
    ValueType::Uint64,
    ValueType::Int64,
    ValueType::Float64,
];

/// A typed GGUF metadata array.
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataArray {
    Uint8(Vec<u8>),
    Int8(Vec<i8>),
    Uint16(Vec<u16>),
    Int16(Vec<i16>),
    Uint32(Vec<u32>),
    Int32(Vec<i32>),
    Float32(Vec<f32>),
    Bool(Vec<bool>),
    String(Vec<String>),
    Uint64(Vec<u64>),
    Int64(Vec<i64>),
    Float64(Vec<f64>),
}

impl MetadataArray {
    /// Returns the number of values in the array.
    pub fn len(&self) -> usize {
        match self {
            Self::Uint8(values) => values.len(),
            Self::Int8(values) => values.len(),
            Self::Uint16(values) => values.len(),
            Self::Int16(values) => values.len(),
            Self::Uint32(values) => values.len(),
            Self::Int32(values) => values.len(),
            Self::Float32(values) => values.len(),
            _ => self.len_remaining(),
        }
    }

    fn len_remaining(&self) -> usize {
        match self {
            Self::Bool(values) => values.len(),
            Self::String(values) => values.len(),
            Self::Uint64(values) => values.len(),
            Self::Int64(values) => values.len(),
            Self::Float64(values) => values.len(),
            _ => unreachable!(),
        }
    }

    /// Returns true when the array has no values.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the scalar type stored in the array.
    pub const fn element_type(&self) -> ValueType {
        match self {
            Self::Uint8(_) => ValueType::Uint8,
            Self::Int8(_) => ValueType::Int8,
            Self::Uint16(_) => ValueType::Uint16,
            Self::Int16(_) => ValueType::Int16,
            Self::Uint32(_) => ValueType::Uint32,
            Self::Int32(_) => ValueType::Int32,
            Self::Float32(_) => ValueType::Float32,
            _ => self.element_type_remaining(),
        }
    }

    const fn element_type_remaining(&self) -> ValueType {
        match self {
            Self::Bool(_) => ValueType::Bool,
            Self::String(_) => ValueType::String,
            Self::Uint64(_) => ValueType::Uint64,
            Self::Int64(_) => ValueType::Int64,
            Self::Float64(_) => ValueType::Float64,
            _ => unreachable!(),
        }
    }
}

/// A typed GGUF metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(MetadataArray),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl MetadataValue {
    /// Returns the value as an unsigned integer when its type is unsigned.
    pub const fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Uint8(value) => Some(*value as u64),
            Self::Uint16(value) => Some(*value as u64),
            Self::Uint32(value) => Some(*value as u64),
            Self::Uint64(value) => Some(*value),
            _ => None,
        }
    }

    /// Returns the value as an `f64` when its type is floating point.
    pub const fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Float32(value) => Some(*value as f64),
            Self::Float64(value) => Some(*value),
            _ => None,
        }
    }

    /// Returns the string when this is a string value.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }

    /// Returns the array when this is an array value.
    pub const fn as_array(&self) -> Option<&MetadataArray> {
        match self {
            Self::Array(value) => Some(value),
            _ => None,
        }
    }
}

/// A ggml tensor storage type from the GGUF tensor table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GgmlType(u32);

impl GgmlType {
    pub const F32: Self = Self(0);
    pub const F16: Self = Self(1);
    pub const Q4_0: Self = Self(2);
    pub const Q4_1: Self = Self(3);
    pub const Q5_0: Self = Self(6);
    pub const Q5_1: Self = Self(7);
    pub const Q8_0: Self = Self(8);
    pub const Q8_1: Self = Self(9);
    pub const Q2_K: Self = Self(10);
    pub const Q3_K: Self = Self(11);
    pub const Q4_K: Self = Self(12);
    pub const Q5_K: Self = Self(13);
    pub const Q6_K: Self = Self(14);
    pub const Q8_K: Self = Self(15);
    pub const BF16: Self = Self(30);

    /// Returns the GGUF numeric type code.
    pub const fn code(self) -> u32 {
        self.0
    }

    /// Returns the number of values and bytes in one storage block.
    pub const fn block_layout(self) -> Option<(u64, u64)> {
        let index = self.0 as usize;
        if index < BLOCK_LAYOUTS.len() {
            BLOCK_LAYOUTS[index]
        } else {
            None
        }
    }

    /// Returns the ggml type name used by llama.cpp.
    pub const fn name(self) -> &'static str {
        let index = self.0 as usize;
        if index < GGML_TYPE_NAMES.len() {
            GGML_TYPE_NAMES[index]
        } else {
            "UNKNOWN"
        }
    }
}

const BLOCK_LAYOUTS: [Option<(u64, u64)>; 43] = [
    Some((1, 4)),
    Some((1, 2)),
    Some((32, 18)),
    Some((32, 20)),
    None,
    None,
    Some((32, 22)),
    Some((32, 24)),
    Some((32, 34)),
    Some((32, 36)),
    Some((256, 84)),
    Some((256, 110)),
    Some((256, 144)),
    Some((256, 176)),
    Some((256, 210)),
    Some((256, 292)),
    Some((256, 66)),
    Some((256, 74)),
    Some((256, 98)),
    Some((256, 50)),
    Some((32, 18)),
    Some((256, 110)),
    Some((256, 82)),
    Some((256, 136)),
    Some((1, 1)),
    Some((1, 2)),
    Some((1, 4)),
    Some((1, 8)),
    Some((1, 8)),
    Some((256, 56)),
    Some((1, 2)),
    None,
    None,
    None,
    Some((256, 54)),
    Some((256, 66)),
    None,
    None,
    None,
    Some((32, 17)),
    Some((64, 36)),
    Some((128, 18)),
    Some((64, 18)),
];

const GGML_TYPE_NAMES: [&str; 43] = [
    "F32", "F16", "Q4_0", "Q4_1", "REMOVED", "REMOVED", "Q5_0", "Q5_1", "Q8_0", "Q8_1", "Q2_K",
    "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K", "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ1_S", "IQ4_NL",
    "IQ3_S", "IQ2_S", "IQ4_XS", "I8", "I16", "I32", "I64", "F64", "IQ1_M", "BF16", "REMOVED",
    "REMOVED", "REMOVED", "TQ1_0", "TQ2_0", "REMOVED", "REMOVED", "REMOVED", "MXFP4", "NVFP4",
    "Q1_0", "Q2_0",
];

impl fmt::Display for GgmlType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

/// One tensor descriptor from a GGUF v3 tensor table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: GgmlType,
    /// The byte offset relative to the GGUF data section.
    pub offset: u64,
    pub n_bytes: u64,
}

impl TensorInfo {
    /// Returns the number of logical tensor elements.
    pub fn n_elements(&self) -> Result<u64> {
        checked_product(&self.shape, "tensor element count")
    }
}

/// A parsed GGUF v3 file with lazy tensor reads.
#[derive(Debug)]
pub struct Gguf {
    version: u32,
    alignment: u32,
    data_offset: u64,
    metadata: BTreeMap<String, MetadataValue>,
    tensors: Vec<TensorInfo>,
    source: ReadOnlyFile,
    _allocations: Vec<Box<dyn AllocationGuard>>,
    _reservations: Vec<Box<dyn AllocationReservation>>,
}

impl Gguf {
    /// Opens and validates a GGUF v3 file.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let budget = UnlimitedAllocationBudget;
        Self::open_with_budget(path, &budget)
    }

    /// Opens and validates a GGUF file with checked parser allocations.
    pub fn open_with_budget(path: impl AsRef<Path>, budget: &dyn AllocationBudget) -> Result<Self> {
        let source = ReadOnlyFile::open(path.as_ref())?;
        let reservations = prepare_reservation_storage(budget)?;
        let buffer_reservation = budget
            .reserve(PARSER_BUFFER_BYTES, "GGUF parser buffer")
            .map_err(Error::from)?;
        let parsed = parse_with_prepared_budget(
            BufReader::with_capacity(
                usize::try_from(PARSER_BUFFER_BYTES)
                    .map_err(|_| Error::IntegerOverflow("parser buffer"))?,
                source.try_clone()?,
            ),
            source.len(),
            budget,
            Some(buffer_reservation),
            reservations,
        )?;
        Ok(Self {
            version: parsed.version,
            alignment: parsed.alignment,
            data_offset: parsed.data_offset,
            metadata: parsed.metadata,
            tensors: parsed.tensors,
            source,
            _allocations: parsed.allocations,
            _reservations: parsed.reservations,
        })
    }

    /// Returns the GGUF version. A parsed file always returns 3.
    pub const fn version(&self) -> u32 {
        self.version
    }

    /// Returns the tensor data alignment in bytes.
    pub const fn alignment(&self) -> u32 {
        self.alignment
    }

    /// Returns the absolute byte offset of the tensor data section.
    pub const fn data_offset(&self) -> u64 {
        self.data_offset
    }

    /// Returns all metadata in key order.
    pub const fn metadata(&self) -> &BTreeMap<String, MetadataValue> {
        &self.metadata
    }

    /// Returns all tensor descriptors in file order.
    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    /// Finds a tensor by its exact GGUF name.
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|tensor| tensor.name == name)
    }

    /// Reads one complete tensor into an owned byte buffer.
    pub fn tensor_data(&self, name: &str) -> Result<Vec<u8>> {
        let tensor = self
            .tensor(name)
            .ok_or_else(|| Error::TensorNotFound(name.to_owned()))?;
        let offset = self
            .data_offset
            .checked_add(tensor.offset)
            .ok_or(Error::IntegerOverflow("tensor absolute offset"))?;
        self.source.read(offset, tensor.n_bytes)
    }

    /// Reads a byte range relative to the start of a tensor.
    pub fn tensor_range(&self, name: &str, offset: u64, len: u64) -> Result<Vec<u8>> {
        let tensor = self
            .tensor(name)
            .ok_or_else(|| Error::TensorNotFound(name.to_owned()))?;
        let end = offset
            .checked_add(len)
            .ok_or(Error::IntegerOverflow("tensor subrange"))?;
        if end > tensor.n_bytes {
            return Err(Error::TensorOutOfBounds {
                tensor: tensor.name.clone(),
            });
        }
        let absolute = self
            .data_offset
            .checked_add(tensor.offset)
            .and_then(|value| value.checked_add(offset))
            .ok_or(Error::IntegerOverflow("tensor subrange offset"))?;
        self.source.read(absolute, len)
    }
}

struct Parsed {
    version: u32,
    alignment: u32,
    data_offset: u64,
    metadata: BTreeMap<String, MetadataValue>,
    tensors: Vec<TensorInfo>,
    allocations: ParserAllocations,
    reservations: ParserReservations,
}

type ParserAllocations = Vec<Box<dyn AllocationGuard>>;
type ParserReservations = Vec<Box<dyn AllocationReservation>>;

fn prepare_reservation_storage(budget: &dyn AllocationBudget) -> Result<ParserReservations> {
    let wrapper_bytes = budget.reservation_metadata_bytes();
    let metadata_reservation = if wrapper_bytes == 0 {
        None
    } else {
        let bytes = (PARSER_RESERVATION_CAPACITY as u64)
            .checked_mul(wrapper_bytes)
            .ok_or(Error::IntegerOverflow("parser reservation metadata bytes"))?;
        // The tracker charge precedes construction of this reservation wrapper.
        Some(
            budget
                .reserve(bytes, "parser reservation metadata")
                .map_err(Error::from)?,
        )
    };
    let mut reservations = Vec::new();
    reservations
        .try_reserve_exact(PARSER_RESERVATION_CAPACITY)
        .map_err(|_| Error::Allocation {
            what: "parser reservation guards",
            count: PARSER_RESERVATION_CAPACITY,
        })?;
    if let Some(reservation) = metadata_reservation {
        reservations.push(reservation);
    }
    Ok(reservations)
}

struct Input<'a, R> {
    reader: R,
    position: u64,
    file_len: u64,
    budget: &'a dyn AllocationBudget,
    allocations: Vec<Box<dyn AllocationGuard>>,
    buffer_reservation: Option<Box<dyn AllocationReservation>>,
    reservations: Vec<Box<dyn AllocationReservation>>,
    guard_metadata_capacity: u64,
    guard_metadata_used: u64,
}

impl<'a, R: Read> Input<'a, R> {
    fn new(
        reader: R,
        file_len: u64,
        budget: &'a dyn AllocationBudget,
        buffer_reservation: Option<Box<dyn AllocationReservation>>,
        reservations: ParserReservations,
    ) -> Self {
        Self {
            reader,
            position: 0,
            file_len,
            budget,
            allocations: Vec::new(),
            buffer_reservation,
            reservations,
            guard_metadata_capacity: 0,
            guard_metadata_used: 0,
        }
    }

    fn bytes<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut bytes = [0; N];
        self.reader.read_exact(&mut bytes)?;
        self.position = self
            .position
            .checked_add(N as u64)
            .ok_or(Error::IntegerOverflow("parser position"))?;
        Ok(bytes)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.bytes::<1>()?[0])
    }

    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.bytes()?))
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.bytes()?))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.bytes()?))
    }

    fn reserve(&self, bytes: u64, what: &'static str) -> Result<Box<dyn AllocationReservation>> {
        self.budget.reserve(bytes, what).map_err(Error::from)
    }

    fn commit_retained(&mut self, reservation: Box<dyn AllocationReservation>) -> Result<()> {
        self.reserve_guard_metadata(true)?;
        self.allocations.push(reservation.commit()?);
        Ok(())
    }

    fn retain_reservation(&mut self, reservation: Box<dyn AllocationReservation>) -> Result<()> {
        if self.reservations.len() >= PARSER_MAX_RETAINED_RESERVATION_WRAPPERS {
            return Err(Error::Allocation {
                what: "parser reservation guards",
                count: self.reservations.len().saturating_add(1),
            });
        }
        self.reservations.push(reservation);
        Ok(())
    }

    fn reserve_guard_metadata(&mut self, store_in_vector: bool) -> Result<()> {
        let required = self
            .guard_metadata_used
            .checked_add(1)
            .ok_or(Error::IntegerOverflow("parser guard metadata count"))?;
        if required > self.guard_metadata_capacity {
            self.grow_guard_metadata(required, store_in_vector)?;
        } else if store_in_vector {
            self.ensure_guard_storage_capacity()?;
        }
        self.guard_metadata_used = required;
        Ok(())
    }

    fn grow_guard_metadata(&mut self, required: u64, store_in_vector: bool) -> Result<()> {
        let capacity = geometric_capacity_u64(
            required,
            self.guard_metadata_capacity,
            "parser guard metadata count",
        )?;
        let additional = capacity - self.guard_metadata_capacity;
        let guard_bytes = self.budget.guard_metadata_bytes();
        if guard_bytes != 0 {
            let bytes = additional
                .checked_mul(guard_bytes)
                .ok_or(Error::IntegerOverflow("parser guard metadata bytes"))?;
            let reservation = self.reserve(bytes, "parser allocation guard metadata")?;
            self.retain_reservation(reservation)?;
        }
        if store_in_vector {
            self.ensure_guard_storage_capacity()?;
        }
        self.guard_metadata_capacity = capacity;
        Ok(())
    }

    fn ensure_guard_storage_capacity(&mut self) -> Result<()> {
        let required = self
            .allocations
            .len()
            .checked_add(1)
            .ok_or(Error::IntegerOverflow("parser allocation guard count"))?;
        let old_capacity = self.allocations.capacity();
        if required <= old_capacity {
            return Ok(());
        }
        let target =
            geometric_capacity_usize(required, old_capacity, "parser allocation guard capacity")?;
        let old_bytes = guard_storage_bytes(old_capacity)?;
        let resize_reservation = self.reserve_guard_resize(old_bytes)?;
        self.allocations
            .try_reserve_exact(target - self.allocations.len())
            .map_err(|_| Error::Allocation {
                what: "parser allocation guards",
                count: target,
            })?;
        drop(resize_reservation);
        Ok(())
    }

    fn reserve_guard_resize(
        &self,
        old_bytes: u64,
    ) -> Result<Option<Box<dyn AllocationReservation>>> {
        if old_bytes == 0 {
            return Ok(None);
        }
        self.reserve(old_bytes, "parser allocation guard vector resize")
            .map(Some)
    }

    fn allocate_vec<T>(&mut self, count: usize, what: &'static str) -> Result<Vec<T>> {
        if count == 0 {
            return Ok(Vec::new());
        }
        let bytes = vector_bytes::<T>(count, what)?;
        let reservation = self.reserve(bytes, what)?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|_| Error::Allocation { what, count })?;
        self.commit_retained(reservation)?;
        Ok(values)
    }

    fn reserve_hash_collection<T>(
        &self,
        count: u64,
        what: &'static str,
    ) -> Result<Option<Box<dyn AllocationReservation>>> {
        if count == 0 {
            return Ok(None);
        }
        let bytes = hash_collection_bytes::<T>(count, what)?;
        self.reserve(bytes, what).map(Some)
    }

    fn reserve_ordered_collection<T>(
        &self,
        count: u64,
        what: &'static str,
    ) -> Result<Option<Box<dyn AllocationReservation>>> {
        if count == 0 {
            return Ok(None);
        }
        let bytes = ordered_collection_bytes::<T>(count, what)?;
        self.reserve(bytes, what).map(Some)
    }

    fn string(&mut self, field: &'static str) -> Result<String> {
        let Some(len) = self.string_length(field)? else {
            return Ok(String::new());
        };
        self.read_string_body(len, field)
    }

    fn string_length(&mut self, field: &'static str) -> Result<Option<usize>> {
        let len = self.u64()?;
        check_limit(field, len, MAX_STRING_LEN)?;
        let remaining = self.file_len.saturating_sub(self.position);
        if len > remaining {
            return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof).into());
        }
        let len = usize::try_from(len).map_err(|_| Error::IntegerOverflow(field))?;
        if len == 0 {
            return Ok(None);
        }
        Ok(Some(len))
    }

    fn read_string_body(&mut self, len: usize, field: &'static str) -> Result<String> {
        let length = u64::try_from(len).map_err(|_| Error::IntegerOverflow(field))?;
        let reservation = self.reserve(length, field)?;
        let mut bytes = allocate_bytes(len, field)?;
        bytes.resize(len, 0);
        self.reader.read_exact(&mut bytes)?;
        self.position = self
            .position
            .checked_add(length)
            .ok_or(Error::IntegerOverflow("parser position"))?;
        let value =
            String::from_utf8(bytes).map_err(|source| Error::InvalidUtf8 { field, source })?;
        self.commit_retained(reservation)?;
        Ok(value)
    }

    fn clone_string(
        &mut self,
        value: &str,
        field: &'static str,
    ) -> Result<(String, Option<Box<dyn AllocationGuard>>)> {
        if value.is_empty() {
            return Ok((String::new(), None));
        }
        let length = u64::try_from(value.len()).map_err(|_| Error::IntegerOverflow(field))?;
        let reservation = self.reserve(length, field)?;
        self.reserve_guard_metadata(false)?;
        let mut clone = String::new();
        clone
            .try_reserve_exact(value.len())
            .map_err(|_| Error::Allocation {
                what: field,
                count: value.len(),
            })?;
        clone.push_str(value);
        let allocation = reservation.commit()?;
        Ok((clone, Some(allocation)))
    }

    fn take_allocations(self) -> (ParserAllocations, ParserReservations) {
        let Self {
            reader,
            allocations,
            buffer_reservation,
            reservations,
            ..
        } = self;
        drop(reader);
        drop(buffer_reservation);
        (allocations, reservations)
    }

    fn bool(&mut self) -> Result<bool> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            value => Err(Error::InvalidBoolean(value)),
        }
    }
}

fn geometric_capacity_u64(required: u64, current: u64, what: &'static str) -> Result<u64> {
    if required <= current {
        return Ok(current);
    }
    if current == 0 {
        return Ok(required);
    }
    let doubled = current.checked_mul(2).ok_or(Error::IntegerOverflow(what))?;
    Ok(required.max(doubled))
}

fn geometric_capacity_usize(required: usize, current: usize, what: &'static str) -> Result<usize> {
    if required <= current {
        return Ok(current);
    }
    if current == 0 {
        return Ok(required);
    }
    let doubled = current.checked_mul(2).ok_or(Error::IntegerOverflow(what))?;
    Ok(required.max(doubled))
}

fn guard_storage_bytes(capacity: usize) -> Result<u64> {
    let capacity = u64::try_from(capacity)
        .map_err(|_| Error::IntegerOverflow("parser allocation guard slots"))?;
    let slot_bytes = u64::try_from(std::mem::size_of::<Box<dyn AllocationGuard>>())
        .map_err(|_| Error::IntegerOverflow("parser allocation guard slots"))?;
    capacity
        .checked_mul(slot_bytes)
        .ok_or(Error::IntegerOverflow(
            "parser allocation guard resize bytes",
        ))
}

fn vector_bytes<T>(count: usize, what: &'static str) -> Result<u64> {
    let count = u64::try_from(count).map_err(|_| Error::IntegerOverflow(what))?;
    let size = u64::try_from(std::mem::size_of::<T>()).map_err(|_| Error::IntegerOverflow(what))?;
    let payload = count
        .checked_mul(size)
        .ok_or(Error::IntegerOverflow(what))?;
    let header =
        u64::try_from(std::mem::size_of::<Vec<T>>()).map_err(|_| Error::IntegerOverflow(what))?;
    payload
        .checked_add(header)
        .ok_or(Error::IntegerOverflow(what))
}

fn hash_collection_bytes<T>(count: u64, what: &'static str) -> Result<u64> {
    if count == 0 {
        return Ok(0);
    }
    let minimum_buckets = count
        .max(4)
        .checked_mul(HASH_BUCKET_FACTOR)
        .ok_or(Error::IntegerOverflow(what))?;
    let buckets = minimum_buckets
        .checked_next_power_of_two()
        .ok_or(Error::IntegerOverflow(what))?;
    let entry_bytes =
        u64::try_from(std::mem::size_of::<T>()).map_err(|_| Error::IntegerOverflow(what))?;
    buckets
        .checked_mul(
            entry_bytes
                .checked_add(1)
                .ok_or(Error::IntegerOverflow(what))?,
        )
        .and_then(|bytes| bytes.checked_add(HASH_CONTROL_AND_ALIGNMENT_BYTES))
        .ok_or(Error::IntegerOverflow(what))
}

fn ordered_collection_bytes<T>(count: u64, what: &'static str) -> Result<u64> {
    let entry_bytes =
        u64::try_from(std::mem::size_of::<T>()).map_err(|_| Error::IntegerOverflow(what))?;
    let entry_bytes = entry_bytes
        .checked_mul(ORDERED_NODE_ENTRY_FACTOR)
        .and_then(|bytes| bytes.checked_add(ORDERED_NODE_OVERHEAD_BYTES))
        .ok_or(Error::IntegerOverflow(what))?;
    count
        .checked_mul(entry_bytes)
        .ok_or(Error::IntegerOverflow(what))
}

fn reserve_vector<T>(
    budget: &dyn AllocationBudget,
    count: usize,
    what: &'static str,
) -> Result<Option<Box<dyn AllocationReservation>>> {
    if count == 0 {
        return Ok(None);
    }
    let bytes = vector_bytes::<T>(count, what)?;
    budget.reserve(bytes, what).map(Some).map_err(Error::from)
}

fn allocate_bytes(len: usize, what: &'static str) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(len)
        .map_err(|_| Error::Allocation { what, count: len })?;
    Ok(bytes)
}

#[cfg(test)]
fn parse(reader: impl Read, file_len: u64) -> Result<Parsed> {
    let budget = UnlimitedAllocationBudget;
    parse_with_budget(reader, file_len, &budget)
}

#[cfg(test)]
fn parse_with_budget(
    reader: impl Read,
    file_len: u64,
    budget: &dyn AllocationBudget,
) -> Result<Parsed> {
    let reservations = prepare_reservation_storage(budget)?;
    parse_with_prepared_budget(reader, file_len, budget, None, reservations)
}

fn parse_with_prepared_budget(
    reader: impl Read,
    file_len: u64,
    budget: &dyn AllocationBudget,
    buffer_reservation: Option<Box<dyn AllocationReservation>>,
    reservations: ParserReservations,
) -> Result<Parsed> {
    let mut input = Input::new(reader, file_len, budget, buffer_reservation, reservations);
    let (tensor_count, metadata_count) = parse_header(&mut input)?;
    let metadata = parse_metadata(&mut input, metadata_count)?;
    let alignment = parse_alignment(&metadata)?;
    let tensors = parse_tensors(&mut input, tensor_count, alignment)?;
    let data_offset = if tensor_count == 0 {
        input.position
    } else {
        align_up(input.position, u64::from(alignment))?
    };
    validate_tensor_ranges(&tensors, data_offset, file_len, budget)?;
    let (allocations, reservations) = input.take_allocations();
    Ok(Parsed {
        version: GGUF_VERSION,
        alignment,
        data_offset,
        metadata,
        tensors,
        allocations,
        reservations,
    })
}

fn parse_header<R: Read>(input: &mut Input<R>) -> Result<(u64, u64)> {
    let magic = input.bytes()?;
    if magic != GGUF_MAGIC {
        return Err(Error::InvalidMagic { found: magic });
    }
    let version = input.u32()?;
    if version != GGUF_VERSION {
        return Err(Error::UnsupportedVersion(version));
    }
    let tensor_count = input.u64()?;
    let metadata_count = input.u64()?;
    check_limit("tensor", tensor_count, MAX_COLLECTION_LEN)?;
    check_limit("metadata", metadata_count, MAX_COLLECTION_LEN)?;
    Ok((tensor_count, metadata_count))
}

fn parse_metadata<R: Read>(
    input: &mut Input<R>,
    metadata_count: u64,
) -> Result<BTreeMap<String, MetadataValue>> {
    let reservation = input
        .reserve_ordered_collection::<(String, MetadataValue)>(metadata_count, "metadata map")?;
    let mut metadata = BTreeMap::new();
    for _ in 0..metadata_count {
        let key = input.string("metadata key")?;
        let value_type = ValueType::try_from(input.u32()?)?;
        let value = parse_value(input, value_type)?;
        if metadata.contains_key(&key) {
            return Err(Error::DuplicateMetadata(key));
        }
        metadata.insert(key, value);
    }
    if let Some(reservation) = reservation {
        input.retain_reservation(reservation)?;
    }
    Ok(metadata)
}

fn parse_alignment(metadata: &BTreeMap<String, MetadataValue>) -> Result<u32> {
    Ok(match metadata.get("general.alignment") {
        None => DEFAULT_ALIGNMENT,
        Some(MetadataValue::Uint32(value)) if value.is_power_of_two() => *value,
        Some(MetadataValue::Uint32(value)) => return Err(Error::InvalidAlignment(*value)),
        Some(_) => return Err(Error::InvalidAlignment(0)),
    })
}

fn parse_tensors<R: Read>(
    input: &mut Input<R>,
    tensor_count: u64,
    alignment: u32,
) -> Result<Vec<TensorInfo>> {
    ensure_tensor_descriptors_fit(input, tensor_count)?;
    let tensor_count =
        usize::try_from(tensor_count).map_err(|_| Error::IntegerOverflow("tensor count"))?;
    let mut tensors = input.allocate_vec(tensor_count, "tensor descriptors")?;
    let mut tensor_names = TensorNames::new(input, tensor_count)?;
    for _ in 0..tensor_count {
        tensors.push(parse_tensor(input, &mut tensor_names.values, alignment)?);
    }
    Ok(tensors)
}

fn ensure_tensor_descriptors_fit<R: Read>(input: &Input<R>, tensor_count: u64) -> Result<()> {
    let minimum_bytes = tensor_count
        .checked_mul(MIN_TENSOR_DESCRIPTOR_BYTES)
        .ok_or(Error::IntegerOverflow("tensor descriptor bytes"))?;
    if minimum_bytes > input.file_len.saturating_sub(input.position) {
        return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof).into());
    }
    Ok(())
}

fn parse_tensor<R: Read>(
    input: &mut Input<R>,
    tensor_names: &mut HashSet<GuardedString>,
    alignment: u32,
) -> Result<TensorInfo> {
    let name = input.string("tensor name")?;
    if tensor_names.contains(name.as_str()) {
        return Err(Error::DuplicateTensor(name));
    }
    let (name_copy, name_allocation) = input.clone_string(&name, "tensor name index")?;
    tensor_names.insert(GuardedString {
        value: name_copy,
        _allocation: name_allocation,
    });
    let shape = parse_shape(input, &name)?;
    let dtype = GgmlType(input.u32()?);
    let offset = input.u64()?;
    if offset % u64::from(alignment) != 0 {
        return Err(Error::MisalignedTensor {
            tensor: name,
            offset,
            alignment,
        });
    }
    let n_bytes = tensor_bytes(&name, &shape, dtype)?;
    Ok(TensorInfo {
        name,
        shape,
        dtype,
        offset,
        n_bytes,
    })
}

fn parse_shape<R: Read>(input: &mut Input<R>, tensor: &str) -> Result<Vec<u64>> {
    let dimensions = input.u32()?;
    if !(1..=4).contains(&dimensions) {
        return Err(Error::InvalidDimensions {
            tensor: tensor.to_owned(),
            dimensions,
        });
    }
    let dimensions = dimensions as usize;
    let mut shape = input.allocate_vec(dimensions, "tensor shape")?;
    for _ in 0..dimensions {
        let dimension = input.u64()?;
        if dimension == 0 {
            return Err(Error::ZeroDimension {
                tensor: tensor.to_owned(),
            });
        }
        shape.push(dimension);
    }
    Ok(shape)
}

fn parse_value<R: Read>(input: &mut Input<R>, value_type: ValueType) -> Result<MetadataValue> {
    match value_type {
        ValueType::Uint8
        | ValueType::Int8
        | ValueType::Uint16
        | ValueType::Int16
        | ValueType::Uint32
        | ValueType::Int32
        | ValueType::Float32
        | ValueType::Bool => parse_narrow_value(input, value_type),
        ValueType::String => Ok(MetadataValue::String(input.string("metadata string")?)),
        ValueType::Array => Ok(MetadataValue::Array(parse_array(input)?)),
        ValueType::Uint64 | ValueType::Int64 | ValueType::Float64 => {
            parse_wide_value(input, value_type)
        }
    }
}

fn parse_narrow_value<R: Read>(
    input: &mut Input<R>,
    value_type: ValueType,
) -> Result<MetadataValue> {
    match value_type {
        ValueType::Uint8 | ValueType::Uint16 | ValueType::Uint32 => {
            parse_unsigned_value(input, value_type)
        }
        ValueType::Int8 | ValueType::Int16 | ValueType::Int32 => {
            parse_signed_value(input, value_type)
        }
        ValueType::Float32 | ValueType::Bool => parse_float_bool_value(input, value_type),
        _ => unreachable!("parse_narrow_value receives a narrow value type"),
    }
}

fn parse_unsigned_value<R: Read>(
    input: &mut Input<R>,
    value_type: ValueType,
) -> Result<MetadataValue> {
    Ok(match value_type {
        ValueType::Uint8 => MetadataValue::Uint8(input.u8()?),
        ValueType::Uint16 => MetadataValue::Uint16(input.u16()?),
        ValueType::Uint32 => MetadataValue::Uint32(input.u32()?),
        _ => unreachable!("parse_unsigned_value receives an unsigned value type"),
    })
}

fn parse_signed_value<R: Read>(
    input: &mut Input<R>,
    value_type: ValueType,
) -> Result<MetadataValue> {
    Ok(match value_type {
        ValueType::Int8 => MetadataValue::Int8(input.u8()? as i8),
        ValueType::Int16 => MetadataValue::Int16(input.u16()? as i16),
        ValueType::Int32 => MetadataValue::Int32(input.u32()? as i32),
        _ => unreachable!("parse_signed_value receives a signed value type"),
    })
}

fn parse_float_bool_value<R: Read>(
    input: &mut Input<R>,
    value_type: ValueType,
) -> Result<MetadataValue> {
    Ok(match value_type {
        ValueType::Float32 => MetadataValue::Float32(f32::from_bits(input.u32()?)),
        ValueType::Bool => MetadataValue::Bool(input.bool()?),
        _ => unreachable!("parse_float_bool_value receives a float or bool value type"),
    })
}

fn parse_wide_value<R: Read>(input: &mut Input<R>, value_type: ValueType) -> Result<MetadataValue> {
    Ok(match value_type {
        ValueType::Uint64 => MetadataValue::Uint64(input.u64()?),
        ValueType::Int64 => MetadataValue::Int64(input.u64()? as i64),
        ValueType::Float64 => MetadataValue::Float64(f64::from_bits(input.u64()?)),
        _ => unreachable!("parse_wide_value receives a wide value type"),
    })
}

fn parse_array<R: Read>(input: &mut Input<R>) -> Result<MetadataArray> {
    let element_type = ValueType::try_from(input.u32()?)?;
    if element_type == ValueType::Array {
        return Err(Error::NestedArray);
    }
    let len = input.u64()?;
    check_limit("array", len, MAX_COLLECTION_LEN)?;
    let minimum_bytes = ARRAY_ELEMENT_BYTES[element_type as usize];
    let required = len
        .checked_mul(minimum_bytes)
        .ok_or(Error::IntegerOverflow("array minimum byte count"))?;
    if required > input.file_len.saturating_sub(input.position) {
        return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof).into());
    }
    let len = usize::try_from(len).map_err(|_| Error::IntegerOverflow("array length"))?;
    parse_array_values(input, element_type, len)
}

const ARRAY_ELEMENT_BYTES: [u64; 13] = [1, 1, 2, 2, 4, 4, 4, 1, 8, 0, 8, 8, 8];

fn parse_array_values<R: Read>(
    input: &mut Input<R>,
    element_type: ValueType,
    len: usize,
) -> Result<MetadataArray> {
    match element_type {
        ValueType::Uint8
        | ValueType::Int8
        | ValueType::Uint16
        | ValueType::Int16
        | ValueType::Uint32
        | ValueType::Int32
        | ValueType::Float32
        | ValueType::Bool => parse_narrow_array(input, element_type, len),
        ValueType::String | ValueType::Uint64 | ValueType::Int64 | ValueType::Float64 => {
            parse_wide_array(input, element_type, len)
        }
        ValueType::Array => Err(Error::NestedArray),
    }
}

fn parse_narrow_array<R: Read>(
    input: &mut Input<R>,
    element_type: ValueType,
    len: usize,
) -> Result<MetadataArray> {
    match element_type {
        ValueType::Uint8 | ValueType::Uint16 | ValueType::Uint32 => {
            parse_unsigned_array(input, element_type, len)
        }
        ValueType::Int8 | ValueType::Int16 | ValueType::Int32 => {
            parse_signed_array(input, element_type, len)
        }
        ValueType::Float32 | ValueType::Bool => parse_float_bool_array(input, element_type, len),
        _ => unreachable!("parse_narrow_array receives a narrow array type"),
    }
}

fn parse_unsigned_array<R: Read>(
    input: &mut Input<R>,
    element_type: ValueType,
    len: usize,
) -> Result<MetadataArray> {
    Ok(match element_type {
        ValueType::Uint8 => MetadataArray::Uint8(read_array_values(input, len, Input::u8)?),
        ValueType::Uint16 => MetadataArray::Uint16(read_array_values(input, len, Input::u16)?),
        ValueType::Uint32 => MetadataArray::Uint32(read_array_values(input, len, Input::u32)?),
        _ => unreachable!("parse_unsigned_array receives an unsigned array type"),
    })
}

fn parse_signed_array<R: Read>(
    input: &mut Input<R>,
    element_type: ValueType,
    len: usize,
) -> Result<MetadataArray> {
    match element_type {
        ValueType::Int8 => parse_int8_array(input, len),
        ValueType::Int16 | ValueType::Int32 => parse_wide_signed_array(input, element_type, len),
        _ => unreachable!("parse_signed_array receives a signed array type"),
    }
}

fn parse_int8_array<R: Read>(input: &mut Input<R>, len: usize) -> Result<MetadataArray> {
    Ok(MetadataArray::Int8(read_array_values_mapped(
        input,
        len,
        Input::u8,
        |value| value as i8,
    )?))
}

fn parse_wide_signed_array<R: Read>(
    input: &mut Input<R>,
    element_type: ValueType,
    len: usize,
) -> Result<MetadataArray> {
    Ok(match element_type {
        ValueType::Int16 => {
            MetadataArray::Int16(read_array_values_mapped(input, len, Input::u16, |value| {
                value as i16
            })?)
        }
        ValueType::Int32 => {
            MetadataArray::Int32(read_array_values_mapped(input, len, Input::u32, |value| {
                value as i32
            })?)
        }
        _ => unreachable!("parse_wide_signed_array receives a wide signed array type"),
    })
}

fn parse_float_bool_array<R: Read>(
    input: &mut Input<R>,
    element_type: ValueType,
    len: usize,
) -> Result<MetadataArray> {
    Ok(match element_type {
        ValueType::Float32 => MetadataArray::Float32(read_array_values_mapped(
            input,
            len,
            Input::u32,
            f32::from_bits,
        )?),
        ValueType::Bool => MetadataArray::Bool(read_array_values(input, len, Input::bool)?),
        _ => unreachable!("parse_float_bool_array receives a float or bool array type"),
    })
}

fn parse_wide_array<R: Read>(
    input: &mut Input<R>,
    element_type: ValueType,
    len: usize,
) -> Result<MetadataArray> {
    match element_type {
        ValueType::String => parse_string_array(input, len),
        ValueType::Uint64 | ValueType::Int64 | ValueType::Float64 => {
            parse_numeric_array(input, element_type, len)
        }
        _ => unreachable!("parse_wide_array receives a wide array type"),
    }
}

fn parse_string_array<R: Read>(input: &mut Input<R>, len: usize) -> Result<MetadataArray> {
    Ok(MetadataArray::String(read_array_values(
        input,
        len,
        |input| input.string("metadata array string"),
    )?))
}

fn parse_numeric_array<R: Read>(
    input: &mut Input<R>,
    element_type: ValueType,
    len: usize,
) -> Result<MetadataArray> {
    Ok(match element_type {
        ValueType::Uint64 => MetadataArray::Uint64(read_array_values(input, len, Input::u64)?),
        ValueType::Int64 => {
            MetadataArray::Int64(read_array_values_mapped(input, len, Input::u64, |value| {
                value as i64
            })?)
        }
        ValueType::Float64 => MetadataArray::Float64(read_array_values_mapped(
            input,
            len,
            Input::u64,
            f64::from_bits,
        )?),
        _ => unreachable!("parse_numeric_array receives a numeric array type"),
    })
}

fn read_array_values<'a, R: Read, T>(
    input: &mut Input<'a, R>,
    len: usize,
    read: impl FnMut(&mut Input<'a, R>) -> Result<T>,
) -> Result<Vec<T>> {
    read_array_values_mapped(input, len, read, |value| value)
}

fn read_array_values_mapped<'a, R: Read, T, U>(
    input: &mut Input<'a, R>,
    len: usize,
    mut read: impl FnMut(&mut Input<'a, R>) -> Result<T>,
    mut map: impl FnMut(T) -> U,
) -> Result<Vec<U>> {
    let mut values = input.allocate_vec(len, "metadata array")?;
    for _ in 0..len {
        values.push(map(read(input)?));
    }
    Ok(values)
}

fn tensor_bytes(name: &str, shape: &[u64], dtype: GgmlType) -> Result<u64> {
    let (block_elements, block_bytes) =
        dtype
            .block_layout()
            .ok_or_else(|| Error::UnsupportedTensorType {
                tensor: name.to_owned(),
                dtype: dtype.code(),
            })?;
    let row_elements = shape[0];
    if !row_elements.is_multiple_of(block_elements) {
        return Err(Error::InvalidRowLength {
            tensor: name.to_owned(),
            dtype: dtype.to_string(),
            row_elements,
            block_elements,
        });
    }
    let row_bytes = row_elements
        .checked_div(block_elements)
        .and_then(|blocks| blocks.checked_mul(block_bytes))
        .ok_or(Error::IntegerOverflow("tensor row byte count"))?;
    let rows = checked_product(&shape[1..], "tensor row count")?;
    row_bytes
        .checked_mul(rows)
        .ok_or(Error::IntegerOverflow("tensor byte count"))
}

fn validate_tensor_ranges(
    tensors: &[TensorInfo],
    data_offset: u64,
    file_len: u64,
    budget: &dyn AllocationBudget,
) -> Result<()> {
    let ranges = collect_tensor_ranges(tensors, data_offset, file_len, budget)?;
    check_tensor_overlap(ranges)
}

fn collect_tensor_ranges<'a>(
    tensors: &'a [TensorInfo],
    data_offset: u64,
    file_len: u64,
    budget: &dyn AllocationBudget,
) -> Result<GuardedRanges<'a>> {
    let reservation = reserve_vector::<(u64, u64, &str)>(budget, tensors.len(), "tensor ranges")?;
    let mut ranges = Vec::new();
    ranges
        .try_reserve_exact(tensors.len())
        .map_err(|_| Error::Allocation {
            what: "tensor ranges",
            count: tensors.len(),
        })?;
    for tensor in tensors {
        let start = data_offset
            .checked_add(tensor.offset)
            .ok_or(Error::IntegerOverflow("tensor absolute offset"))?;
        let end = start
            .checked_add(tensor.n_bytes)
            .ok_or(Error::IntegerOverflow("tensor end offset"))?;
        if end > file_len {
            return Err(Error::TensorOutOfBounds {
                tensor: tensor.name.clone(),
            });
        }
        ranges.push((start, end, tensor.name.as_str()));
    }
    Ok(GuardedRanges {
        values: ranges,
        _reservation: reservation,
    })
}

fn check_tensor_overlap(mut ranges: GuardedRanges<'_>) -> Result<()> {
    ranges.values.sort_unstable_by_key(|range| range.0);
    for pair in ranges.values.windows(2) {
        if pair[0].1 > pair[1].0 {
            return Err(Error::OverlappingTensors {
                first: pair[0].2.to_owned(),
                second: pair[1].2.to_owned(),
            });
        }
    }
    Ok(())
}

struct GuardedRanges<'a> {
    values: Vec<(u64, u64, &'a str)>,
    _reservation: Option<Box<dyn AllocationReservation>>,
}

#[derive(Debug)]
struct GuardedString {
    value: String,
    _allocation: Option<Box<dyn AllocationGuard>>,
}

struct TensorNames {
    values: HashSet<GuardedString>,
    _reservation: Option<Box<dyn AllocationReservation>>,
}

impl TensorNames {
    fn new<R: Read>(input: &Input<R>, count: usize) -> Result<Self> {
        let count_u64 = u64::try_from(count).map_err(|_| Error::IntegerOverflow("tensor count"))?;
        let reservation =
            input.reserve_hash_collection::<GuardedString>(count_u64, "tensor names")?;
        let mut values = HashSet::new();
        values.try_reserve(count).map_err(|_| Error::Allocation {
            what: "tensor names",
            count,
        })?;
        Ok(Self {
            values,
            _reservation: reservation,
        })
    }
}

impl Borrow<str> for GuardedString {
    fn borrow(&self) -> &str {
        &self.value
    }
}

impl Hash for GuardedString {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}

impl PartialEq for GuardedString {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl Eq for GuardedString {}

fn checked_product(values: &[u64], what: &'static str) -> Result<u64> {
    values.iter().try_fold(1_u64, |product, value| {
        product
            .checked_mul(*value)
            .ok_or(Error::IntegerOverflow(what))
    })
}

fn check_limit(what: &'static str, value: u64, limit: u64) -> Result<()> {
    if value > limit {
        return Err(Error::LimitExceeded { what, value, limit });
    }
    Ok(())
}

fn align_up(value: u64, alignment: u64) -> Result<u64> {
    let mask = alignment - 1;
    value
        .checked_add(mask)
        .map(|sum| sum & !mask)
        .ok_or(Error::IntegerOverflow("aligned data offset"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AllocationBudgetError;
    use proptest::prelude::*;
    use std::io::Cursor;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Debug, Default)]
    struct DenyBudget {
        requests: AtomicUsize,
    }

    impl AllocationBudget for DenyBudget {
        fn guard_metadata_bytes(&self) -> u64 {
            0
        }

        fn reservation_metadata_bytes(&self) -> u64 {
            0
        }

        fn reserve(
            &self,
            bytes: u64,
            what: &'static str,
        ) -> std::result::Result<Box<dyn AllocationReservation>, AllocationBudgetError> {
            self.requests.fetch_add(1, Ordering::Relaxed);
            Err(AllocationBudgetError::new(
                what,
                bytes,
                std::io::Error::other("allocation denied"),
            ))
        }
    }

    #[derive(Debug, Default)]
    struct GuardMetadataDenyBudget;

    impl AllocationBudget for GuardMetadataDenyBudget {
        fn guard_metadata_bytes(&self) -> u64 {
            64
        }

        fn reservation_metadata_bytes(&self) -> u64 {
            0
        }

        fn reserve(
            &self,
            bytes: u64,
            what: &'static str,
        ) -> std::result::Result<Box<dyn AllocationReservation>, AllocationBudgetError> {
            if what == "parser allocation guard metadata" {
                return Err(AllocationBudgetError::new(
                    what,
                    bytes,
                    std::io::Error::other("guard metadata denied"),
                ));
            }
            UnlimitedAllocationBudget.reserve(bytes, what)
        }
    }

    #[derive(Debug, Default)]
    struct ReservationMetadataDenyBudget;

    impl AllocationBudget for ReservationMetadataDenyBudget {
        fn guard_metadata_bytes(&self) -> u64 {
            0
        }

        fn reservation_metadata_bytes(&self) -> u64 {
            PARSER_RESERVATION_WRAPPER_BOUND_BYTES
        }

        fn reserve(
            &self,
            bytes: u64,
            what: &'static str,
        ) -> std::result::Result<Box<dyn AllocationReservation>, AllocationBudgetError> {
            if what == "parser reservation metadata" {
                return Err(AllocationBudgetError::new(
                    what,
                    bytes,
                    std::io::Error::other("reservation metadata denied"),
                ));
            }
            UnlimitedAllocationBudget.reserve(bytes, what)
        }
    }

    fn minimal_file() -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        bytes
    }

    #[test]
    fn parses_every_metadata_scalar_and_array_type() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        bytes.extend_from_slice(&13_u64.to_le_bytes());
        for type_code in 0_u32..=12 {
            put_string(&mut bytes, &format!("k{type_code}"));
            bytes.extend_from_slice(&type_code.to_le_bytes());
            match type_code {
                0 | 1 | 7 => bytes.push((type_code == 7).into()),
                2 | 3 => bytes.extend_from_slice(&1_u16.to_le_bytes()),
                4..=6 => bytes.extend_from_slice(&1_u32.to_le_bytes()),
                8 => put_string(&mut bytes, "value"),
                9 => {
                    bytes.extend_from_slice(&(ValueType::Uint16 as u32).to_le_bytes());
                    bytes.extend_from_slice(&2_u64.to_le_bytes());
                    bytes.extend_from_slice(&1_u16.to_le_bytes());
                    bytes.extend_from_slice(&2_u16.to_le_bytes());
                }
                10..=12 => bytes.extend_from_slice(&1_u64.to_le_bytes()),
                _ => unreachable!(),
            }
        }
        let parsed = parse(Cursor::new(&bytes), bytes.len() as u64).unwrap();
        assert_eq!(parsed.metadata.len(), 13);
        assert_eq!(
            parsed.metadata["k9"],
            MetadataValue::Array(MetadataArray::Uint16(vec![1, 2]))
        );
    }

    #[test]
    fn computes_aligned_tensor_data_offset_and_size() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        put_string(&mut bytes, "weight");
        bytes.extend_from_slice(&2_u32.to_le_bytes());
        bytes.extend_from_slice(&256_u64.to_le_bytes());
        bytes.extend_from_slice(&2_u64.to_le_bytes());
        bytes.extend_from_slice(&GgmlType::Q4_K.code().to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        bytes.resize(align_up(bytes.len() as u64, 32).unwrap() as usize, 0);
        bytes.resize(bytes.len() + 288, 0);
        let parsed = parse(Cursor::new(&bytes), bytes.len() as u64).unwrap();
        assert_eq!(parsed.data_offset % 32, 0);
        assert_eq!(parsed.tensors[0].n_bytes, 288);
    }

    #[test]
    fn rejects_bad_magic_and_version() {
        let mut bytes = minimal_file();
        bytes[0] = 0;
        assert!(matches!(
            parse(Cursor::new(&bytes), bytes.len() as u64),
            Err(Error::InvalidMagic { .. })
        ));
        let mut bytes = minimal_file();
        bytes[4..8].copy_from_slice(&2_u32.to_le_bytes());
        assert!(matches!(
            parse(Cursor::new(&bytes), bytes.len() as u64),
            Err(Error::UnsupportedVersion(2))
        ));
    }

    #[test]
    fn rejects_impossible_tensor_count_before_reserving_descriptors() {
        let mut bytes = minimal_file();
        bytes[8..16].copy_from_slice(&MAX_COLLECTION_LEN.to_le_bytes());
        let budget = DenyBudget::default();
        assert!(matches!(
            parse_with_budget(Cursor::new(&bytes), bytes.len() as u64, &budget),
            Err(Error::Io(error)) if error.kind() == std::io::ErrorKind::UnexpectedEof
        ));
        assert_eq!(budget.requests.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn tiny_budget_rejects_before_metadata_storage() {
        let mut bytes = minimal_file();
        bytes[16..24].copy_from_slice(&1_u64.to_le_bytes());
        put_string(&mut bytes, "general.alignment");
        bytes.extend_from_slice(&(ValueType::Uint32 as u32).to_le_bytes());
        bytes.extend_from_slice(&32_u32.to_le_bytes());
        let budget = DenyBudget::default();
        assert!(matches!(
            parse_with_budget(Cursor::new(&bytes), bytes.len() as u64, &budget),
            Err(Error::AllocationBudget(error)) if error.what() == "metadata map"
        ));
        assert_eq!(budget.requests.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn guard_metadata_denial_precedes_guard_commit() {
        let mut bytes = minimal_file();
        bytes[16..24].copy_from_slice(&1_u64.to_le_bytes());
        put_string(&mut bytes, "key");
        bytes.extend_from_slice(&(ValueType::String as u32).to_le_bytes());
        put_string(&mut bytes, "value");
        let budget = GuardMetadataDenyBudget;
        let error = parse_with_budget(Cursor::new(&bytes), bytes.len() as u64, &budget)
            .err()
            .expect("guard metadata denial");
        assert!(
            matches!(
                &error,
                Error::AllocationBudget(error)
                    if error.what() == "parser allocation guard metadata"
                        && error.bytes() == 64
            ),
            "unexpected error: {error:?}"
        );
    }

    #[test]
    fn reservation_metadata_denial_precedes_parser_storage() {
        let bytes = minimal_file();
        let budget = ReservationMetadataDenyBudget;
        let error = parse_with_budget(Cursor::new(&bytes), bytes.len() as u64, &budget)
            .err()
            .expect("reservation metadata denial");
        assert!(matches!(
            error,
            Error::AllocationBudget(error)
                if error.what() == "parser reservation metadata"
                    && error.bytes() == PARSER_RESERVATION_METADATA_BYTES
        ));
    }

    #[test]
    fn parser_guard_storage_grows_at_reserved_boundaries() {
        let budget = UnlimitedAllocationBudget;
        let reservations = prepare_reservation_storage(&budget).unwrap();
        let mut input = Input::new(Cursor::new([]), 0, &budget, None, reservations);
        for index in 0usize..1024 {
            let reservation = input.reserve(0, "test guard").unwrap();
            input.commit_retained(reservation).unwrap();
            assert_eq!(
                input.allocations.capacity(),
                (index + 1).next_power_of_two()
            );
        }
    }

    proptest! {
        #[test]
        fn corrupted_headers_return_without_panicking(
            changes in prop::collection::vec((0_usize..24, any::<u8>()), 1..24)
        ) {
            let mut bytes = minimal_file();
            for (index, value) in changes {
                bytes[index] = value;
            }
            let _ = parse(Cursor::new(&bytes), bytes.len() as u64);
        }

        #[test]
        fn arbitrary_short_headers_return_without_panicking(bytes in prop::collection::vec(any::<u8>(), 0..128)) {
            let _ = parse(Cursor::new(&bytes), bytes.len() as u64);
        }
    }

    fn put_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }
}
