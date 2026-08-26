use crate::pread::ReadOnlyFile;
use crate::{Error, Result};
use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::io::{BufReader, Read};
use std::path::Path;

const GGUF_MAGIC: [u8; 4] = *b"GGUF";
const GGUF_VERSION: u32 = 3;
const DEFAULT_ALIGNMENT: u32 = 32;
const MAX_COLLECTION_LEN: u64 = 16_777_216;
const MAX_STRING_LEN: u64 = 1_073_741_824;

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
        match value {
            0 => Ok(Self::Uint8),
            1 => Ok(Self::Int8),
            2 => Ok(Self::Uint16),
            3 => Ok(Self::Int16),
            4 => Ok(Self::Uint32),
            5 => Ok(Self::Int32),
            6 => Ok(Self::Float32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::Uint64),
            11 => Ok(Self::Int64),
            12 => Ok(Self::Float64),
            other => Err(Error::InvalidValueType(other)),
        }
    }
}

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
            Self::Bool(values) => values.len(),
            Self::String(values) => values.len(),
            Self::Uint64(values) => values.len(),
            Self::Int64(values) => values.len(),
            Self::Float64(values) => values.len(),
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
            Self::Bool(_) => ValueType::Bool,
            Self::String(_) => ValueType::String,
            Self::Uint64(_) => ValueType::Uint64,
            Self::Int64(_) => ValueType::Int64,
            Self::Float64(_) => ValueType::Float64,
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
        match self.0 {
            0 => Some((1, 4)),
            1 => Some((1, 2)),
            2 => Some((32, 18)),
            3 => Some((32, 20)),
            6 => Some((32, 22)),
            7 => Some((32, 24)),
            8 => Some((32, 34)),
            9 => Some((32, 36)),
            10 => Some((256, 84)),
            11 => Some((256, 110)),
            12 => Some((256, 144)),
            13 => Some((256, 176)),
            14 => Some((256, 210)),
            15 => Some((256, 292)),
            16 => Some((256, 66)),
            17 => Some((256, 74)),
            18 => Some((256, 98)),
            19 => Some((256, 50)),
            20 => Some((32, 18)),
            21 => Some((256, 110)),
            22 => Some((256, 82)),
            23 => Some((256, 136)),
            24 => Some((1, 1)),
            25 => Some((1, 2)),
            26 => Some((1, 4)),
            27 => Some((1, 8)),
            28 => Some((1, 8)),
            29 => Some((256, 56)),
            30 => Some((1, 2)),
            34 => Some((256, 54)),
            35 => Some((256, 66)),
            39 => Some((32, 17)),
            40 => Some((64, 36)),
            41 => Some((128, 18)),
            42 => Some((64, 18)),
            _ => None,
        }
    }

    /// Returns the ggml type name used by llama.cpp.
    pub const fn name(self) -> &'static str {
        match self.0 {
            0 => "F32",
            1 => "F16",
            2 => "Q4_0",
            3 => "Q4_1",
            4 | 5 | 31..=33 | 36..=38 => "REMOVED",
            6 => "Q5_0",
            7 => "Q5_1",
            8 => "Q8_0",
            9 => "Q8_1",
            10 => "Q2_K",
            11 => "Q3_K",
            12 => "Q4_K",
            13 => "Q5_K",
            14 => "Q6_K",
            15 => "Q8_K",
            16 => "IQ2_XXS",
            17 => "IQ2_XS",
            18 => "IQ3_XXS",
            19 => "IQ1_S",
            20 => "IQ4_NL",
            21 => "IQ3_S",
            22 => "IQ2_S",
            23 => "IQ4_XS",
            24 => "I8",
            25 => "I16",
            26 => "I32",
            27 => "I64",
            28 => "F64",
            29 => "IQ1_M",
            30 => "BF16",
            34 => "TQ1_0",
            35 => "TQ2_0",
            39 => "MXFP4",
            40 => "NVFP4",
            41 => "Q1_0",
            42 => "Q2_0",
            _ => "UNKNOWN",
        }
    }
}

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
}

impl Gguf {
    /// Opens and validates a GGUF v3 file.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let source = ReadOnlyFile::open(path.as_ref())?;
        let parsed = parse(BufReader::new(source.try_clone()?), source.len())?;
        Ok(Self {
            version: parsed.version,
            alignment: parsed.alignment,
            data_offset: parsed.data_offset,
            metadata: parsed.metadata,
            tensors: parsed.tensors,
            source,
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
}

struct Input<R> {
    reader: R,
    position: u64,
    file_len: u64,
}

impl<R: Read> Input<R> {
    fn new(reader: R, file_len: u64) -> Self {
        Self {
            reader,
            position: 0,
            file_len,
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

    fn string(&mut self, field: &'static str) -> Result<String> {
        let len = self.u64()?;
        check_limit(field, len, MAX_STRING_LEN)?;
        let remaining = self.file_len.saturating_sub(self.position);
        if len > remaining {
            return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof).into());
        }
        let len = usize::try_from(len).map_err(|_| Error::IntegerOverflow(field))?;
        let mut bytes = vec![0; len];
        self.reader.read_exact(&mut bytes)?;
        self.position = self
            .position
            .checked_add(len as u64)
            .ok_or(Error::IntegerOverflow("parser position"))?;
        String::from_utf8(bytes).map_err(|source| Error::InvalidUtf8 { field, source })
    }

    fn bool(&mut self) -> Result<bool> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            value => Err(Error::InvalidBoolean(value)),
        }
    }
}

fn parse(reader: impl Read, file_len: u64) -> Result<Parsed> {
    let mut input = Input::new(reader, file_len);
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

    let mut metadata = BTreeMap::new();
    for _ in 0..metadata_count {
        let key = input.string("metadata key")?;
        let value_type = ValueType::try_from(input.u32()?)?;
        let value = parse_value(&mut input, value_type)?;
        if metadata.insert(key.clone(), value).is_some() {
            return Err(Error::DuplicateMetadata(key));
        }
    }

    let alignment = match metadata.get("general.alignment") {
        None => DEFAULT_ALIGNMENT,
        Some(MetadataValue::Uint32(value)) if value.is_power_of_two() => *value,
        Some(MetadataValue::Uint32(value)) => return Err(Error::InvalidAlignment(*value)),
        Some(_) => return Err(Error::InvalidAlignment(0)),
    };

    let mut tensors = Vec::new();
    let mut tensor_names = HashSet::new();
    for _ in 0..tensor_count {
        let name = input.string("tensor name")?;
        if !tensor_names.insert(name.clone()) {
            return Err(Error::DuplicateTensor(name));
        }
        let dimensions = input.u32()?;
        if !(1..=4).contains(&dimensions) {
            return Err(Error::InvalidDimensions {
                tensor: name,
                dimensions,
            });
        }
        let mut shape = Vec::with_capacity(dimensions as usize);
        for _ in 0..dimensions {
            let dimension = input.u64()?;
            if dimension == 0 {
                return Err(Error::ZeroDimension {
                    tensor: name.clone(),
                });
            }
            shape.push(dimension);
        }
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
        tensors.push(TensorInfo {
            name,
            shape,
            dtype,
            offset,
            n_bytes,
        });
    }

    let data_offset = if tensor_count == 0 {
        input.position
    } else {
        align_up(input.position, u64::from(alignment))?
    };
    validate_tensor_ranges(&tensors, data_offset, file_len)?;
    Ok(Parsed {
        version,
        alignment,
        data_offset,
        metadata,
        tensors,
    })
}

fn parse_value<R: Read>(input: &mut Input<R>, value_type: ValueType) -> Result<MetadataValue> {
    Ok(match value_type {
        ValueType::Uint8 => MetadataValue::Uint8(input.u8()?),
        ValueType::Int8 => MetadataValue::Int8(input.u8()? as i8),
        ValueType::Uint16 => MetadataValue::Uint16(input.u16()?),
        ValueType::Int16 => MetadataValue::Int16(input.u16()? as i16),
        ValueType::Uint32 => MetadataValue::Uint32(input.u32()?),
        ValueType::Int32 => MetadataValue::Int32(input.u32()? as i32),
        ValueType::Float32 => MetadataValue::Float32(f32::from_bits(input.u32()?)),
        ValueType::Bool => MetadataValue::Bool(input.bool()?),
        ValueType::String => MetadataValue::String(input.string("metadata string")?),
        ValueType::Array => MetadataValue::Array(parse_array(input)?),
        ValueType::Uint64 => MetadataValue::Uint64(input.u64()?),
        ValueType::Int64 => MetadataValue::Int64(input.u64()? as i64),
        ValueType::Float64 => MetadataValue::Float64(f64::from_bits(input.u64()?)),
    })
}

fn parse_array<R: Read>(input: &mut Input<R>) -> Result<MetadataArray> {
    let element_type = ValueType::try_from(input.u32()?)?;
    if element_type == ValueType::Array {
        return Err(Error::NestedArray);
    }
    let len = input.u64()?;
    check_limit("array", len, MAX_COLLECTION_LEN)?;
    let minimum_bytes = match element_type {
        ValueType::Uint8 | ValueType::Int8 | ValueType::Bool => 1,
        ValueType::Uint16 | ValueType::Int16 => 2,
        ValueType::Uint32 | ValueType::Int32 | ValueType::Float32 => 4,
        ValueType::String | ValueType::Uint64 | ValueType::Int64 | ValueType::Float64 => 8,
        ValueType::Array => return Err(Error::NestedArray),
    };
    let required = len
        .checked_mul(minimum_bytes)
        .ok_or(Error::IntegerOverflow("array minimum byte count"))?;
    if required > input.file_len.saturating_sub(input.position) {
        return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof).into());
    }
    let len = usize::try_from(len).map_err(|_| Error::IntegerOverflow("array length"))?;

    macro_rules! values {
        ($read:expr) => {{
            let mut values = Vec::with_capacity(len);
            for _ in 0..len {
                values.push($read?);
            }
            values
        }};
    }

    Ok(match element_type {
        ValueType::Uint8 => MetadataArray::Uint8(values!(input.u8())),
        ValueType::Int8 => {
            MetadataArray::Int8(values!(input.u8()).into_iter().map(|v| v as i8).collect())
        }
        ValueType::Uint16 => MetadataArray::Uint16(values!(input.u16())),
        ValueType::Int16 => {
            MetadataArray::Int16(values!(input.u16()).into_iter().map(|v| v as i16).collect())
        }
        ValueType::Uint32 => MetadataArray::Uint32(values!(input.u32())),
        ValueType::Int32 => {
            MetadataArray::Int32(values!(input.u32()).into_iter().map(|v| v as i32).collect())
        }
        ValueType::Float32 => MetadataArray::Float32(
            values!(input.u32())
                .into_iter()
                .map(f32::from_bits)
                .collect(),
        ),
        ValueType::Bool => MetadataArray::Bool(values!(input.bool())),
        ValueType::String => MetadataArray::String(values!(input.string("metadata array string"))),
        ValueType::Uint64 => MetadataArray::Uint64(values!(input.u64())),
        ValueType::Int64 => {
            MetadataArray::Int64(values!(input.u64()).into_iter().map(|v| v as i64).collect())
        }
        ValueType::Float64 => MetadataArray::Float64(
            values!(input.u64())
                .into_iter()
                .map(f64::from_bits)
                .collect(),
        ),
        ValueType::Array => return Err(Error::NestedArray),
    })
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

fn validate_tensor_ranges(tensors: &[TensorInfo], data_offset: u64, file_len: u64) -> Result<()> {
    let mut ranges = Vec::with_capacity(tensors.len());
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
        ranges.push((start, end, &tensor.name));
    }
    ranges.sort_unstable_by_key(|range| range.0);
    for pair in ranges.windows(2) {
        if pair[0].1 > pair[1].0 {
            return Err(Error::OverlappingTensors {
                first: pair[0].2.clone(),
                second: pair[1].2.clone(),
            });
        }
    }
    Ok(())
}

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
    use proptest::prelude::*;
    use std::io::Cursor;

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
