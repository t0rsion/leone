use crate::{
    Backend, BackendError, BufferLayout, MemoryCapacity, QuantFormat, QuantMatrix, Tokenizer,
    TokenizerError,
};
use leone_gguf::model::{ModelConfig as GgufModelConfig, ModelError};
use leone_gguf::{GgmlType, Gguf, TensorInfo};
use leone_receipt::TensorClass;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use thiserror::Error;

/// A dense decoder architecture with an explicit execution contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelArchitecture {
    /// Qwen3 with per-head query and key RMS normalization.
    Qwen3,
    /// Llama with direct query and key RoPE.
    Llama,
}

impl ModelArchitecture {
    fn from_gguf(value: &str) -> Result<Self, ModelLoadError> {
        if value == "qwen3" {
            Ok(Self::Qwen3)
        } else if value.starts_with("llama") {
            Ok(Self::Llama)
        } else {
            Err(ModelError::UnsupportedArchitecture(value.to_owned()).into())
        }
    }

    /// Returns the stable GGUF architecture family name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Qwen3 => "qwen3",
            Self::Llama => "llama",
        }
    }

    /// Returns true when each attention head has learned Q/K norm weights.
    pub const fn uses_qk_norm(self) -> bool {
        matches!(self, Self::Qwen3)
    }

    /// Returns true when this architecture supports a decode graph.
    pub const fn decode_graph_supported(self) -> bool {
        true
    }
}

/// The checked dense-decoder dimensions used by the runtime.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    pub architecture: ModelArchitecture,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub context_length: usize,
    pub rope_theta: f32,
    pub rope_frequency_factors: Option<Vec<f32>>,
    pub rms_epsilon: f32,
}

/// An error returned while checking or uploading one model.
#[derive(Debug, Error)]
pub enum ModelLoadError {
    #[error(transparent)]
    Gguf(#[from] leone_gguf::Error),
    #[error(transparent)]
    Metadata(#[from] ModelError),
    #[error(transparent)]
    Tokenizer(#[from] TokenizerError),
    #[error(transparent)]
    Backend(#[from] BackendError),
    #[error("model dimension {field} does not fit the host size")]
    Dimension { field: &'static str },
    #[error("model dimensions do not satisfy {constraint}")]
    DimensionRelation { constraint: &'static str },
    #[error("model field {field} is not supported")]
    UnsupportedField { field: &'static str },
    #[error("model float {field} is outside the positive f32 range")]
    FloatRange { field: &'static str },
    #[error("RoPE frequency factor {index} must be finite and positive, got {value}")]
    InvalidRopeFrequencyFactor { index: usize, value: f32 },
    #[error("tensor {0:?} is missing")]
    MissingTensor(String),
    #[error("tensor {name:?} has shape {actual:?}, expected {expected:?}")]
    TensorShape {
        name: String,
        expected: Vec<u64>,
        actual: Vec<u64>,
    },
    #[error("tensor {name:?} has type {actual}, expected {expected}")]
    TensorType {
        name: String,
        expected: &'static str,
        actual: GgmlType,
    },
    #[error("tensor {name:?} uses unsupported type {dtype}")]
    UnsupportedTensorType { name: String, dtype: GgmlType },
    #[error("model contains unexpected tensor {0:?}")]
    UnexpectedTensor(String),
    #[error("model tensors need {required_bytes} bytes but the backend has {available_bytes} bytes free")]
    WeightBudget {
        required_bytes: u64,
        available_bytes: u64,
    },
    #[error("tensor byte accounting overflowed")]
    ByteAccounting,
}

/// A dense decoder model whose weights use one backend's opaque buffers.
#[derive(Debug)]
pub struct LoadedModel<B: Backend> {
    pub(crate) config: ModelConfig,
    pub(crate) tokenizer: Tokenizer,
    pub(crate) weights: DenseWeights<B>,
    path: PathBuf,
    file_bytes: u64,
    tensor_bytes: u64,
    weights_resident_bytes_by_class: BTreeMap<TensorClass, u64>,
    decode_weight_bytes_by_class: BTreeMap<TensorClass, u64>,
}

impl<B: Backend> LoadedModel<B> {
    /// Opens, validates, accounts, and uploads one supported dense GGUF file.
    pub fn load(backend: &mut B, path: impl AsRef<Path>) -> Result<Self, ModelLoadError> {
        let path = path.as_ref();
        let (gguf, architecture, config) = open_model_source(path)?;
        configure_model_backend(backend, &config, architecture)?;
        let tokenizer = Tokenizer::from_metadata(gguf.metadata())?;
        let accounting = account_model_tensors(&gguf, &config)?;
        check_weight_budget(backend, accounting.tensor_bytes)?;
        let (weights, loaded) = load_weights(backend, &gguf, &config, architecture)?;
        validate_loaded_tensors(&gguf, architecture, &config, &loaded)?;
        let file_bytes = std::fs::metadata(path)
            .map_err(leone_gguf::Error::from)?
            .len();
        Ok(Self {
            config,
            tokenizer,
            weights,
            path: path.to_owned(),
            file_bytes,
            tensor_bytes: accounting.tensor_bytes,
            weights_resident_bytes_by_class: accounting.resident,
            decode_weight_bytes_by_class: accounting.decode,
        })
    }

    pub const fn config(&self) -> &ModelConfig {
        &self.config
    }

    pub const fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub const fn file_bytes(&self) -> u64 {
        self.file_bytes
    }

    pub const fn tensor_bytes(&self) -> u64 {
        self.tensor_bytes
    }

    pub const fn weights_resident_bytes_by_class(&self) -> &BTreeMap<TensorClass, u64> {
        &self.weights_resident_bytes_by_class
    }

    /// Returns weight bytes read by one decode evaluation, before KV traffic.
    pub const fn decode_weight_bytes_by_class(&self) -> &BTreeMap<TensorClass, u64> {
        &self.decode_weight_bytes_by_class
    }

    pub const fn output_is_tied(&self) -> bool {
        matches!(self.weights.output, OutputWeight::Tied)
    }
}

struct TensorAccounting {
    tensor_bytes: u64,
    resident: BTreeMap<TensorClass, u64>,
    decode: BTreeMap<TensorClass, u64>,
}

struct AttentionWeights<B: Backend> {
    query: QuantWeight<B>,
    key: QuantWeight<B>,
    value: QuantWeight<B>,
}

struct FfnWeights<B: Backend> {
    norm: B::Buffer,
    gate: QuantWeight<B>,
    up: QuantWeight<B>,
    down: QuantWeight<B>,
}

fn open_model_source(
    path: &Path,
) -> Result<(Gguf, ModelArchitecture, ModelConfig), ModelLoadError> {
    let gguf = Gguf::open(path)?;
    let source_config = GgufModelConfig::from_metadata(gguf.metadata())?;
    let architecture = ModelArchitecture::from_gguf(&source_config.architecture)?;
    if source_config.rope_scaling.is_some() {
        return Err(ModelLoadError::UnsupportedField {
            field: "RoPE scaling",
        });
    }
    let mut config = ModelConfig::from_gguf(&source_config, architecture)?;
    if architecture == ModelArchitecture::Llama {
        config.rope_frequency_factors = read_llama_rope_frequency_factors(&gguf, &config)?;
    }
    Ok((gguf, architecture, config))
}

fn configure_model_backend<B: Backend>(
    backend: &mut B,
    config: &ModelConfig,
    architecture: ModelArchitecture,
) -> Result<(), ModelLoadError> {
    backend.configure_rope(
        config.head_dim,
        config.rope_theta,
        config.rope_frequency_factors.as_deref(),
        rope_pairing(architecture),
    )?;
    Ok(())
}

const fn rope_pairing(architecture: ModelArchitecture) -> crate::RopePairing {
    match architecture {
        ModelArchitecture::Qwen3 => crate::RopePairing::HalfSplit,
        ModelArchitecture::Llama => crate::RopePairing::Adjacent,
    }
}

fn account_model_tensors(
    gguf: &Gguf,
    config: &ModelConfig,
) -> Result<TensorAccounting, ModelLoadError> {
    let (tensor_bytes, resident) = account_tensor_bytes(gguf)?;
    let token_embedding = require_tensor(gguf, "token_embd.weight")?;
    let embedding_row_bytes = embedding_row_bytes(token_embedding, config.vocab_size)?;
    let mut decode = resident.clone();
    decode.insert(TensorClass::Embed, embedding_row_bytes);
    if gguf.tensor("output.weight").is_none() {
        decode.insert(TensorClass::Head, token_embedding.n_bytes);
    }
    Ok(TensorAccounting {
        tensor_bytes,
        resident,
        decode,
    })
}

fn account_tensor_bytes(gguf: &Gguf) -> Result<(u64, BTreeMap<TensorClass, u64>), ModelLoadError> {
    let mut resident = TensorClass::zero_map();
    let mut tensor_bytes = 0_u64;
    for tensor in gguf.tensors() {
        tensor_bytes = tensor_bytes
            .checked_add(tensor.n_bytes)
            .ok_or(ModelLoadError::ByteAccounting)?;
        let class = TensorClass::from_gguf_name(&tensor.name);
        let class_bytes = resident[&class]
            .checked_add(tensor.n_bytes)
            .ok_or(ModelLoadError::ByteAccounting)?;
        resident.insert(class, class_bytes);
    }
    Ok((tensor_bytes, resident))
}

fn embedding_row_bytes(tensor: &TensorInfo, vocab_size: usize) -> Result<u64, ModelLoadError> {
    let vocab_size = u64::try_from(vocab_size).map_err(|_| ModelLoadError::ByteAccounting)?;
    tensor
        .n_bytes
        .checked_div(vocab_size)
        .filter(|_| tensor.n_bytes.is_multiple_of(vocab_size))
        .ok_or(ModelLoadError::ByteAccounting)
}

fn check_weight_budget<B: Backend>(
    backend: &mut B,
    tensor_bytes: u64,
) -> Result<(), ModelLoadError> {
    if let MemoryCapacity::Limited {
        available_bytes, ..
    } = backend.memory_capacity()?
    {
        if tensor_bytes > available_bytes {
            return Err(ModelLoadError::WeightBudget {
                required_bytes: tensor_bytes,
                available_bytes,
            });
        }
    }
    Ok(())
}

fn load_weights<B: Backend>(
    backend: &mut B,
    gguf: &Gguf,
    config: &ModelConfig,
    architecture: ModelArchitecture,
) -> Result<(DenseWeights<B>, BTreeSet<String>), ModelLoadError> {
    let mut loaded = BTreeSet::new();
    let token_embedding = load_quant(
        backend,
        gguf,
        "token_embd.weight",
        config.vocab_size,
        config.n_embd,
        &mut loaded,
    )?;
    let output_norm = load_f32(
        backend,
        gguf,
        "output_norm.weight",
        config.n_embd,
        &mut loaded,
    )?;
    let output = load_output_weight(backend, gguf, config, &mut loaded)?;
    let layers = load_layers(backend, gguf, config, architecture, &mut loaded)?;
    Ok((
        DenseWeights {
            token_embedding,
            output_norm,
            output,
            layers,
        },
        loaded,
    ))
}

fn load_output_weight<B: Backend>(
    backend: &mut B,
    gguf: &Gguf,
    config: &ModelConfig,
    loaded: &mut BTreeSet<String>,
) -> Result<OutputWeight<B>, ModelLoadError> {
    match gguf.tensor("output.weight") {
        Some(_) => Ok(OutputWeight::Separate(load_quant(
            backend,
            gguf,
            "output.weight",
            config.vocab_size,
            config.n_embd,
            loaded,
        )?)),
        None => Ok(OutputWeight::Tied),
    }
}

fn load_layers<B: Backend>(
    backend: &mut B,
    gguf: &Gguf,
    config: &ModelConfig,
    architecture: ModelArchitecture,
    loaded: &mut BTreeSet<String>,
) -> Result<Vec<DenseLayer<B>>, ModelLoadError> {
    let mut layers = Vec::with_capacity(config.n_layer);
    for layer in 0..config.n_layer {
        layers.push(load_layer(
            backend,
            gguf,
            config,
            architecture,
            layer,
            loaded,
        )?);
    }
    Ok(layers)
}

#[allow(clippy::too_many_arguments)]
fn load_layer<B: Backend>(
    backend: &mut B,
    gguf: &Gguf,
    config: &ModelConfig,
    architecture: ModelArchitecture,
    layer: usize,
    loaded: &mut BTreeSet<String>,
) -> Result<DenseLayer<B>, ModelLoadError> {
    let attention_norm = load_f32(
        backend,
        gguf,
        &name(layer, "attn_norm"),
        config.n_embd,
        loaded,
    )?;
    let AttentionWeights { query, key, value } =
        load_attention_weights(backend, gguf, config, layer, loaded)?;
    let qk_norm = load_qk_norm(backend, gguf, config, architecture, layer, loaded)?;
    let attention_output = load_quant(
        backend,
        gguf,
        &name(layer, "attn_output"),
        config.n_embd,
        config.n_embd,
        loaded,
    )?;
    let FfnWeights {
        norm: ffn_norm,
        gate: ffn_gate,
        up: ffn_up,
        down: ffn_down,
    } = load_ffn_weights(backend, gguf, config, layer, loaded)?;
    Ok(DenseLayer {
        attention_norm,
        query,
        key,
        value,
        qk_norm,
        attention_output,
        ffn_norm,
        ffn_gate,
        ffn_up,
        ffn_down,
    })
}

fn load_attention_weights<B: Backend>(
    backend: &mut B,
    gguf: &Gguf,
    config: &ModelConfig,
    layer: usize,
    loaded: &mut BTreeSet<String>,
) -> Result<AttentionWeights<B>, ModelLoadError> {
    let query = load_quant(
        backend,
        gguf,
        &name(layer, "attn_q"),
        config.n_embd,
        config.n_embd,
        loaded,
    )?;
    let key = load_quant(
        backend,
        gguf,
        &name(layer, "attn_k"),
        config.n_head_kv * config.head_dim,
        config.n_embd,
        loaded,
    )?;
    let value = load_quant(
        backend,
        gguf,
        &name(layer, "attn_v"),
        config.n_head_kv * config.head_dim,
        config.n_embd,
        loaded,
    )?;
    Ok(AttentionWeights { query, key, value })
}

#[allow(clippy::too_many_arguments)]
fn load_qk_norm<B: Backend>(
    backend: &mut B,
    gguf: &Gguf,
    config: &ModelConfig,
    architecture: ModelArchitecture,
    layer: usize,
    loaded: &mut BTreeSet<String>,
) -> Result<QkNorm<B>, ModelLoadError> {
    match architecture {
        ModelArchitecture::Qwen3 => Ok(QkNorm::Rms {
            query: load_f32(
                backend,
                gguf,
                &name(layer, "attn_q_norm"),
                config.head_dim,
                loaded,
            )?,
            key: load_f32(
                backend,
                gguf,
                &name(layer, "attn_k_norm"),
                config.head_dim,
                loaded,
            )?,
        }),
        ModelArchitecture::Llama => Ok(QkNorm::Identity),
    }
}

fn load_ffn_weights<B: Backend>(
    backend: &mut B,
    gguf: &Gguf,
    config: &ModelConfig,
    layer: usize,
    loaded: &mut BTreeSet<String>,
) -> Result<FfnWeights<B>, ModelLoadError> {
    let ffn_norm = load_f32(
        backend,
        gguf,
        &name(layer, "ffn_norm"),
        config.n_embd,
        loaded,
    )?;
    let ffn_gate = load_quant(
        backend,
        gguf,
        &name(layer, "ffn_gate"),
        config.n_ff,
        config.n_embd,
        loaded,
    )?;
    let ffn_up = load_quant(
        backend,
        gguf,
        &name(layer, "ffn_up"),
        config.n_ff,
        config.n_embd,
        loaded,
    )?;
    let ffn_down = load_quant(
        backend,
        gguf,
        &name(layer, "ffn_down"),
        config.n_embd,
        config.n_ff,
        loaded,
    )?;
    Ok(FfnWeights {
        norm: ffn_norm,
        gate: ffn_gate,
        up: ffn_up,
        down: ffn_down,
    })
}

fn validate_loaded_tensors(
    gguf: &Gguf,
    architecture: ModelArchitecture,
    config: &ModelConfig,
    loaded: &BTreeSet<String>,
) -> Result<(), ModelLoadError> {
    if architecture == ModelArchitecture::Llama && config.rope_frequency_factors.is_some() {
        // The frequency tensor is represented by metadata and is not uploaded.
        return validate_unexpected_tensors(gguf, loaded, Some("rope_freqs.weight"));
    }
    validate_unexpected_tensors(gguf, loaded, None)
}

fn validate_unexpected_tensors(
    gguf: &Gguf,
    loaded: &BTreeSet<String>,
    metadata_tensor: Option<&str>,
) -> Result<(), ModelLoadError> {
    let unexpected = gguf.tensors().iter().find(|tensor| {
        Some(tensor.name.as_str()) != metadata_tensor && !loaded.contains(&tensor.name)
    });
    if let Some(tensor) = unexpected {
        return Err(ModelLoadError::UnexpectedTensor(tensor.name.clone()));
    }
    Ok(())
}

impl ModelConfig {
    fn from_gguf(
        config: &GgufModelConfig,
        architecture: ModelArchitecture,
    ) -> Result<Self, ModelLoadError> {
        let dimensions = ModelDimensions::from_gguf(config)?;
        let head_dim = dimensions.head_dim(config.head_dim)?;
        dimensions.validate_head_dim(head_dim)?;
        let rope_theta = checked_f32(config.rope_theta, "rope_theta")?;
        let rms_epsilon = checked_f32(config.rms_eps, "rms_epsilon")?;
        Ok(Self {
            architecture,
            n_layer: dimensions.n_layer,
            n_head: dimensions.n_head,
            n_head_kv: dimensions.n_head_kv,
            n_embd: dimensions.n_embd,
            n_ff: dimensions.n_ff,
            head_dim,
            vocab_size: dimensions.vocab_size,
            context_length: dimensions.context_length,
            rope_theta,
            rope_frequency_factors: None,
            rms_epsilon,
        })
    }
}

struct ModelDimensions {
    n_layer: usize,
    n_head: usize,
    n_head_kv: usize,
    n_embd: usize,
    n_ff: usize,
    vocab_size: usize,
    context_length: usize,
}

impl ModelDimensions {
    fn from_gguf(config: &GgufModelConfig) -> Result<Self, ModelLoadError> {
        Ok(Self {
            n_layer: dimension(config.n_layer, "n_layer")?,
            n_head: dimension(config.n_head, "n_head")?,
            n_head_kv: dimension(config.n_head_kv, "n_head_kv")?,
            n_embd: dimension(config.n_embd, "n_embd")?,
            n_ff: dimension(config.n_ff, "n_ff")?,
            vocab_size: dimension(config.vocab_size, "vocab_size")?,
            context_length: dimension(config.context_length, "context_length")?,
        })
    }

    fn head_dim(&self, configured: Option<u64>) -> Result<usize, ModelLoadError> {
        match configured {
            Some(value) => dimension(value, "head_dim"),
            None => Ok(self.n_embd / self.n_head),
        }
    }

    fn validate_head_dim(&self, head_dim: usize) -> Result<(), ModelLoadError> {
        if head_dim != self.n_embd / self.n_head {
            return Err(ModelLoadError::DimensionRelation {
                constraint: "head_dim = n_embd / n_head",
            });
        }
        Ok(())
    }
}

#[derive(Debug)]
pub(crate) struct DenseWeights<B: Backend> {
    pub(crate) token_embedding: QuantWeight<B>,
    pub(crate) output_norm: B::Buffer,
    pub(crate) output: OutputWeight<B>,
    pub(crate) layers: Vec<DenseLayer<B>>,
}

#[derive(Debug)]
pub(crate) struct DenseLayer<B: Backend> {
    pub(crate) attention_norm: B::Buffer,
    pub(crate) query: QuantWeight<B>,
    pub(crate) key: QuantWeight<B>,
    pub(crate) value: QuantWeight<B>,
    pub(crate) qk_norm: QkNorm<B>,
    pub(crate) attention_output: QuantWeight<B>,
    pub(crate) ffn_norm: B::Buffer,
    pub(crate) ffn_gate: QuantWeight<B>,
    pub(crate) ffn_up: QuantWeight<B>,
    pub(crate) ffn_down: QuantWeight<B>,
}

#[derive(Debug)]
pub(crate) enum QkNorm<B: Backend> {
    Rms { query: B::Buffer, key: B::Buffer },
    Identity,
}

#[derive(Debug)]
pub(crate) struct QuantWeight<B: Backend> {
    pub(crate) buffer: B::Buffer,
    pub(crate) shape: QuantMatrix,
}

#[derive(Debug)]
pub(crate) enum OutputWeight<B: Backend> {
    Separate(QuantWeight<B>),
    Tied,
}

fn load_quant<B: Backend>(
    backend: &mut B,
    gguf: &Gguf,
    name: &str,
    rows: usize,
    columns: usize,
    loaded: &mut BTreeSet<String>,
) -> Result<QuantWeight<B>, ModelLoadError> {
    let tensor = require_tensor(gguf, name)?;
    let format = quant_format(tensor.dtype, name)?;
    let (shape, layout) = validate_quant_tensor(tensor, name, rows, columns, format)?;
    let data = gguf.tensor_data(name)?;
    let buffer = backend.upload(layout, &data)?;
    loaded.insert(name.to_owned());
    Ok(QuantWeight { buffer, shape })
}

fn quant_format(dtype: GgmlType, name: &str) -> Result<QuantFormat, ModelLoadError> {
    match dtype {
        GgmlType::Q4_K => Ok(QuantFormat::Q4K),
        GgmlType::Q6_K => Ok(QuantFormat::Q6K),
        dtype => Err(ModelLoadError::UnsupportedTensorType {
            name: name.to_owned(),
            dtype,
        }),
    }
}

fn validate_quant_tensor(
    tensor: &TensorInfo,
    name: &str,
    rows: usize,
    columns: usize,
    format: QuantFormat,
) -> Result<(QuantMatrix, BufferLayout), ModelLoadError> {
    let expected_columns = host_u64(columns, "tensor columns")?;
    let expected_rows = host_u64(rows, "tensor rows")?;
    check_shape(tensor, &[expected_columns, expected_rows])?;
    let shape = QuantMatrix::new(rows, columns, format)?;
    let layout = shape.layout()?;
    let expected_bytes = host_u64(layout.bytes(), "tensor bytes")?;
    if tensor.n_bytes != expected_bytes {
        return Err(ModelLoadError::TensorShape {
            name: name.to_owned(),
            expected: vec![expected_bytes],
            actual: vec![tensor.n_bytes],
        });
    }
    Ok((shape, layout))
}

fn load_f32<B: Backend>(
    backend: &mut B,
    gguf: &Gguf,
    name: &str,
    elements: usize,
    loaded: &mut BTreeSet<String>,
) -> Result<B::Buffer, ModelLoadError> {
    let tensor = require_tensor(gguf, name)?;
    check_shape(tensor, &[host_u64(elements, "tensor elements")?])?;
    if tensor.dtype != GgmlType::F32 {
        return Err(ModelLoadError::TensorType {
            name: name.to_owned(),
            expected: "F32",
            actual: tensor.dtype,
        });
    }
    let layout = BufferLayout::f32(elements)?;
    let data = gguf.tensor_data(name)?;
    let buffer = backend.upload(layout, &data)?;
    loaded.insert(name.to_owned());
    Ok(buffer)
}

fn read_llama_rope_frequency_factors(
    gguf: &Gguf,
    config: &ModelConfig,
) -> Result<Option<Vec<f32>>, ModelLoadError> {
    const NAME: &str = "rope_freqs.weight";
    let Some(tensor) = gguf.tensor(NAME) else {
        return Ok(None);
    };
    let frequencies = config.head_dim / 2;
    check_shape(tensor, &[host_u64(frequencies, "RoPE frequencies")?])?;
    if tensor.dtype != GgmlType::F32 {
        return Err(ModelLoadError::TensorType {
            name: NAME.to_owned(),
            expected: "F32",
            actual: tensor.dtype,
        });
    }
    let bytes = gguf.tensor_data(NAME)?;
    let mut factors = Vec::with_capacity(frequencies);
    for (index, chunk) in bytes.chunks_exact(4).enumerate() {
        let value = f32::from_le_bytes(
            chunk
                .try_into()
                .expect("chunks_exact returns four-byte frequency values"),
        );
        if !value.is_finite() || value <= 0.0 {
            return Err(ModelLoadError::InvalidRopeFrequencyFactor { index, value });
        }
        factors.push(value);
    }
    Ok(Some(factors))
}

fn require_tensor<'a>(gguf: &'a Gguf, name: &str) -> Result<&'a TensorInfo, ModelLoadError> {
    gguf.tensor(name)
        .ok_or_else(|| ModelLoadError::MissingTensor(name.to_owned()))
}

fn check_shape(tensor: &TensorInfo, expected: &[u64]) -> Result<(), ModelLoadError> {
    if tensor.shape == expected {
        Ok(())
    } else {
        Err(ModelLoadError::TensorShape {
            name: tensor.name.clone(),
            expected: expected.to_vec(),
            actual: tensor.shape.clone(),
        })
    }
}

fn name(layer: usize, suffix: &str) -> String {
    format!("blk.{layer}.{suffix}.weight")
}

fn dimension(value: u64, field: &'static str) -> Result<usize, ModelLoadError> {
    usize::try_from(value).map_err(|_| ModelLoadError::Dimension { field })
}

fn host_u64(value: usize, field: &'static str) -> Result<u64, ModelLoadError> {
    u64::try_from(value).map_err(|_| ModelLoadError::Dimension { field })
}

fn checked_f32(value: f64, field: &'static str) -> Result<f32, ModelLoadError> {
    let narrowed = value as f32;
    if narrowed.is_finite() && narrowed > 0.0 {
        Ok(narrowed)
    } else {
        Err(ModelLoadError::FloatRange { field })
    }
}
