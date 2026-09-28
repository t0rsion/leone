use crate::{
    Backend, BackendError, BufferLayout, HostStaging, MemoryAllocation, MemoryCapacity,
    QuantFormat, QuantMatrix, Tokenizer, TokenizerError,
};
use leone_gguf::model::{ModelConfig as GgufModelConfig, ModelError};
use leone_gguf::{GgmlType, Gguf, TensorInfo};
use leone_receipt::TensorClass;
use std::collections::BTreeMap;
use std::ops::Deref;
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
    _host_allocations: Vec<MemoryAllocation>,
}

impl<B: Backend> LoadedModel<B> {
    #[cfg(test)]
    pub(crate) fn from_test_parts(
        config: ModelConfig,
        tokenizer: Tokenizer,
        weights: DenseWeights<B>,
    ) -> Self {
        Self {
            config,
            tokenizer,
            weights,
            path: PathBuf::new(),
            file_bytes: 0,
            tensor_bytes: 0,
            weights_resident_bytes_by_class: BTreeMap::new(),
            decode_weight_bytes_by_class: BTreeMap::new(),
            _host_allocations: Vec::new(),
        }
    }

    /// Opens, validates, accounts, and uploads one supported dense GGUF file.
    pub fn load(backend: &mut B, path: impl AsRef<Path>) -> Result<Self, ModelLoadError> {
        Self::load_with_host_staging(backend, path, &HostStaging::unlimited())
    }

    /// Opens, validates, and uploads a model with a checked host staging ledger.
    pub fn load_with_host_staging(
        backend: &mut B,
        path: impl AsRef<Path>,
        staging: &HostStaging,
    ) -> Result<Self, ModelLoadError> {
        let path = path.as_ref();
        let (gguf, architecture, staged_config) = open_model_source(path, staging)?;
        configure_model_backend(backend, &staged_config.config, architecture, staging)?;
        let tokenizer = Tokenizer::from_metadata_with_host_staging(gguf.metadata(), staging)?;
        let accounting = account_model_tensors(&gguf, &staged_config.config)?;
        check_weight_budget(backend, accounting.tensor_bytes)?;
        let (weights, loaded) =
            load_weights(backend, &gguf, &staged_config.config, architecture, staging)?;
        validate_loaded_tensors(&gguf, architecture, &staged_config.config, &loaded)?;
        let file_bytes = std::fs::metadata(path)
            .map_err(leone_gguf::Error::from)?
            .len();
        let StagedModelConfig {
            config,
            host_allocations: _host_allocations,
        } = staged_config;
        Ok(Self {
            config,
            tokenizer,
            weights,
            path: path.to_owned(),
            file_bytes,
            tensor_bytes: accounting.tensor_bytes,
            weights_resident_bytes_by_class: accounting.resident,
            decode_weight_bytes_by_class: accounting.decode,
            _host_allocations,
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

    /// Returns source file size for artifact accounting, not host residency.
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
    staging: &HostStaging,
) -> Result<(Gguf, ModelArchitecture, StagedModelConfig), ModelLoadError> {
    let gguf = Gguf::open_with_budget(path, staging)?;
    let source_config = GgufModelConfig::from_metadata(gguf.metadata())?;
    let architecture = ModelArchitecture::from_gguf(&source_config.architecture)?;
    if source_config.rope_scaling.is_some() {
        return Err(ModelLoadError::UnsupportedField {
            field: "RoPE scaling",
        });
    }
    let mut config = ModelConfig::from_gguf(&source_config, architecture)?;
    validate_layer_tensor_count(&gguf, &config, architecture)?;
    let mut host_allocations = Vec::new();
    if architecture == ModelArchitecture::Llama {
        if let Some((factors, allocation)) =
            read_llama_rope_frequency_factors(&gguf, &config, staging)?
        {
            config.rope_frequency_factors = Some(factors);
            host_allocations.push(allocation);
        }
    }
    Ok((
        gguf,
        architecture,
        StagedModelConfig {
            config,
            host_allocations,
        },
    ))
}

struct StagedModelConfig {
    config: ModelConfig,
    host_allocations: Vec<MemoryAllocation>,
}

fn configure_model_backend<B: Backend>(
    backend: &mut B,
    config: &ModelConfig,
    architecture: ModelArchitecture,
    staging: &HostStaging,
) -> Result<(), ModelLoadError> {
    backend.configure_rope_with_host_staging(
        config.head_dim,
        config.rope_theta,
        config.rope_frequency_factors.as_deref(),
        rope_pairing(architecture),
        staging,
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

fn validate_layer_tensor_count(
    gguf: &Gguf,
    config: &ModelConfig,
    architecture: ModelArchitecture,
) -> Result<(), ModelLoadError> {
    let per_layer = match architecture {
        ModelArchitecture::Qwen3 => 11,
        ModelArchitecture::Llama => 9,
    };
    let required = config
        .n_layer
        .checked_mul(per_layer)
        .and_then(|count| count.checked_add(2))
        .ok_or(ModelLoadError::ByteAccounting)?;
    if required > gguf.tensors().len() {
        return Err(ModelLoadError::DimensionRelation {
            constraint: "model layer tensor count fits GGUF",
        });
    }
    Ok(())
}

fn mark_loaded(
    gguf: &Gguf,
    tensor: &TensorInfo,
    loaded: &mut LoadedTensorSet,
) -> Result<(), ModelLoadError> {
    let index = gguf
        .tensors()
        .iter()
        .position(|candidate| candidate.name == tensor.name)
        .ok_or_else(|| ModelLoadError::MissingTensor(tensor.name.clone()))?;
    loaded.insert(index);
    Ok(())
}

fn staged_vector_bytes<T>(count: usize) -> Result<u64, ModelLoadError> {
    let count = u64::try_from(count).map_err(|_| ModelLoadError::ByteAccounting)?;
    let element_bytes =
        u64::try_from(std::mem::size_of::<T>()).map_err(|_| ModelLoadError::ByteAccounting)?;
    let payload = count
        .checked_mul(element_bytes)
        .ok_or(ModelLoadError::ByteAccounting)?;
    Ok(payload)
}

fn load_weights<B: Backend>(
    backend: &mut B,
    gguf: &Gguf,
    config: &ModelConfig,
    architecture: ModelArchitecture,
    staging: &HostStaging,
) -> Result<(DenseWeights<B>, LoadedTensorSet), ModelLoadError> {
    let mut loaded = LoadedTensorSet::new(gguf.tensors().len(), staging)?;
    let token_embedding = load_quant(
        backend,
        gguf,
        "token_embd.weight",
        config.vocab_size,
        config.n_embd,
        &mut loaded,
        staging,
    )?;
    let output_norm = load_f32(
        backend,
        gguf,
        "output_norm.weight",
        config.n_embd,
        &mut loaded,
        staging,
    )?;
    let output = load_output_weight(backend, gguf, config, &mut loaded, staging)?;
    let layers = load_layers(backend, gguf, config, architecture, &mut loaded, staging)?;
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
    loaded: &mut LoadedTensorSet,
    staging: &HostStaging,
) -> Result<OutputWeight<B>, ModelLoadError> {
    match gguf.tensor("output.weight") {
        Some(_) => Ok(OutputWeight::Separate(load_quant(
            backend,
            gguf,
            "output.weight",
            config.vocab_size,
            config.n_embd,
            loaded,
            staging,
        )?)),
        None => Ok(OutputWeight::Tied),
    }
}

fn load_layers<B: Backend>(
    backend: &mut B,
    gguf: &Gguf,
    config: &ModelConfig,
    architecture: ModelArchitecture,
    loaded: &mut LoadedTensorSet,
    staging: &HostStaging,
) -> Result<StagedVec<DenseLayer<B>>, ModelLoadError> {
    let bytes = staged_vector_bytes::<DenseLayer<B>>(config.n_layer)?;
    let reservation = if bytes == 0 {
        None
    } else {
        Some(staging.reserve(bytes).map_err(BackendError::from)?)
    };
    let mut layers = Vec::new();
    layers
        .try_reserve_exact(config.n_layer)
        .map_err(|error| BackendError::operation("allocate model layers", error))?;
    let mut layers = StagedVec {
        values: layers,
        _allocation: reservation
            .map(|reservation| reservation.commit())
            .transpose()
            .map_err(BackendError::from)?,
    };
    for layer in 0..config.n_layer {
        layers.values.push(load_layer(
            backend,
            gguf,
            config,
            architecture,
            layer,
            loaded,
            staging,
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
    loaded: &mut LoadedTensorSet,
    staging: &HostStaging,
) -> Result<DenseLayer<B>, ModelLoadError> {
    let attention_norm = load_f32(
        backend,
        gguf,
        &name(layer, "attn_norm"),
        config.n_embd,
        loaded,
        staging,
    )?;
    let AttentionWeights { query, key, value } =
        load_attention_weights(backend, gguf, config, layer, loaded, staging)?;
    let qk_norm = load_qk_norm(backend, gguf, config, architecture, layer, loaded, staging)?;
    let attention_output = load_quant(
        backend,
        gguf,
        &name(layer, "attn_output"),
        config.n_embd,
        config.n_embd,
        loaded,
        staging,
    )?;
    let FfnWeights {
        norm: ffn_norm,
        gate: ffn_gate,
        up: ffn_up,
        down: ffn_down,
    } = load_ffn_weights(backend, gguf, config, layer, loaded, staging)?;
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
    loaded: &mut LoadedTensorSet,
    staging: &HostStaging,
) -> Result<AttentionWeights<B>, ModelLoadError> {
    let query = load_quant(
        backend,
        gguf,
        &name(layer, "attn_q"),
        config.n_embd,
        config.n_embd,
        loaded,
        staging,
    )?;
    let key = load_quant(
        backend,
        gguf,
        &name(layer, "attn_k"),
        config.n_head_kv * config.head_dim,
        config.n_embd,
        loaded,
        staging,
    )?;
    let value = load_quant(
        backend,
        gguf,
        &name(layer, "attn_v"),
        config.n_head_kv * config.head_dim,
        config.n_embd,
        loaded,
        staging,
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
    loaded: &mut LoadedTensorSet,
    staging: &HostStaging,
) -> Result<QkNorm<B>, ModelLoadError> {
    match architecture {
        ModelArchitecture::Qwen3 => Ok(QkNorm::Rms {
            query: load_f32(
                backend,
                gguf,
                &name(layer, "attn_q_norm"),
                config.head_dim,
                loaded,
                staging,
            )?,
            key: load_f32(
                backend,
                gguf,
                &name(layer, "attn_k_norm"),
                config.head_dim,
                loaded,
                staging,
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
    loaded: &mut LoadedTensorSet,
    staging: &HostStaging,
) -> Result<FfnWeights<B>, ModelLoadError> {
    let ffn_norm = load_f32(
        backend,
        gguf,
        &name(layer, "ffn_norm"),
        config.n_embd,
        loaded,
        staging,
    )?;
    let ffn_gate = load_quant(
        backend,
        gguf,
        &name(layer, "ffn_gate"),
        config.n_ff,
        config.n_embd,
        loaded,
        staging,
    )?;
    let ffn_up = load_quant(
        backend,
        gguf,
        &name(layer, "ffn_up"),
        config.n_ff,
        config.n_embd,
        loaded,
        staging,
    )?;
    let ffn_down = load_quant(
        backend,
        gguf,
        &name(layer, "ffn_down"),
        config.n_embd,
        config.n_ff,
        loaded,
        staging,
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
    loaded: &LoadedTensorSet,
) -> Result<(), ModelLoadError> {
    if architecture == ModelArchitecture::Llama && config.rope_frequency_factors.is_some() {
        // The frequency tensor is represented by metadata and is not uploaded.
        return validate_unexpected_tensors(gguf, loaded, Some("rope_freqs.weight"));
    }
    validate_unexpected_tensors(gguf, loaded, None)
}

fn validate_unexpected_tensors(
    gguf: &Gguf,
    loaded: &LoadedTensorSet,
    metadata_tensor: Option<&str>,
) -> Result<(), ModelLoadError> {
    let unexpected = gguf
        .tensors()
        .iter()
        .enumerate()
        .find_map(|(index, tensor)| {
            (Some(tensor.name.as_str()) != metadata_tensor && !loaded.contains(index))
                .then_some(tensor)
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
    pub(crate) layers: StagedVec<DenseLayer<B>>,
}

#[derive(Debug)]
pub(crate) struct StagedVec<T> {
    values: Vec<T>,
    _allocation: Option<MemoryAllocation>,
}

#[cfg(test)]
impl<T> StagedVec<T> {
    pub(crate) fn empty() -> Self {
        Self::from_test_values(Vec::new())
    }

    pub(crate) fn from_test_values(values: Vec<T>) -> Self {
        Self {
            values,
            _allocation: None,
        }
    }
}

impl<T> Deref for StagedVec<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

#[derive(Debug)]
struct LoadedTensorSet {
    values: Vec<u8>,
    _allocation: Option<MemoryAllocation>,
}

impl LoadedTensorSet {
    fn new(count: usize, staging: &HostStaging) -> Result<Self, ModelLoadError> {
        let bytes = u64::try_from(count).map_err(|_| ModelLoadError::ByteAccounting)?;
        let reservation = if bytes == 0 {
            None
        } else {
            Some(staging.reserve(bytes).map_err(BackendError::from)?)
        };
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|error| BackendError::operation("allocate loaded tensor set", error))?;
        let allocation = reservation
            .map(|reservation| reservation.commit())
            .transpose()
            .map_err(BackendError::from)?;
        values.resize(count, 0);
        Ok(Self {
            values,
            _allocation: allocation,
        })
    }

    fn insert(&mut self, index: usize) {
        self.values[index] = 1;
    }

    fn contains(&self, index: usize) -> bool {
        self.values[index] != 0
    }
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
    loaded: &mut LoadedTensorSet,
    staging: &HostStaging,
) -> Result<QuantWeight<B>, ModelLoadError> {
    let tensor = require_tensor(gguf, name)?;
    let format = quant_format(tensor.dtype, name)?;
    let (shape, layout) = validate_quant_tensor(tensor, name, rows, columns, format)?;
    let staged = staged_tensor_data(gguf, tensor, staging)?;
    let buffer = backend.upload_with_host_staging(layout, staged.as_slice(), staging)?;
    mark_loaded(gguf, tensor, loaded)?;
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
    loaded: &mut LoadedTensorSet,
    staging: &HostStaging,
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
    let staged = staged_tensor_data(gguf, tensor, staging)?;
    let buffer = backend.upload_with_host_staging(layout, staged.as_slice(), staging)?;
    mark_loaded(gguf, tensor, loaded)?;
    Ok(buffer)
}

fn read_llama_rope_frequency_factors(
    gguf: &Gguf,
    config: &ModelConfig,
    staging: &HostStaging,
) -> Result<Option<(Vec<f32>, MemoryAllocation)>, ModelLoadError> {
    const NAME: &str = "rope_freqs.weight";
    let Some(tensor) = gguf.tensor(NAME) else {
        return Ok(None);
    };
    let frequencies = config.head_dim / 2;
    validate_rope_tensor(tensor, frequencies, NAME)?;
    let factor_bytes = frequencies
        .checked_mul(std::mem::size_of::<f32>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(ModelLoadError::ByteAccounting)?;
    let reservation = staging.reserve(factor_bytes).map_err(BackendError::from)?;
    let staged = staged_tensor_data(gguf, tensor, staging)?;
    let factors = parse_rope_factors(staged.as_slice(), frequencies)?;
    let allocation = reservation.commit().map_err(BackendError::from)?;
    Ok(Some((factors, allocation)))
}

fn validate_rope_tensor(
    tensor: &TensorInfo,
    frequencies: usize,
    name: &str,
) -> Result<(), ModelLoadError> {
    check_shape(tensor, &[host_u64(frequencies, "RoPE frequencies")?])?;
    if tensor.dtype != GgmlType::F32 {
        return Err(ModelLoadError::TensorType {
            name: name.to_owned(),
            expected: "F32",
            actual: tensor.dtype,
        });
    }
    Ok(())
}

fn parse_rope_factors(bytes: &[u8], frequencies: usize) -> Result<Vec<f32>, ModelLoadError> {
    let mut factors = Vec::new();
    factors
        .try_reserve_exact(frequencies)
        .map_err(|error| BackendError::operation("allocate RoPE frequencies", error))?;
    for (index, chunk) in bytes.chunks_exact(4).enumerate() {
        let value = f32::from_le_bytes(
            chunk
                .try_into()
                .expect("chunks_exact returns four-byte frequency values"),
        );
        validate_rope_factor(index, value)?;
        factors.push(value);
    }
    Ok(factors)
}

fn validate_rope_factor(index: usize, value: f32) -> Result<(), ModelLoadError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(ModelLoadError::InvalidRopeFrequencyFactor { index, value });
    }
    Ok(())
}

fn staged_tensor_data(
    gguf: &Gguf,
    tensor: &TensorInfo,
    staging: &HostStaging,
) -> Result<StagedBytes, ModelLoadError> {
    let reservation = staging
        .reserve(tensor.n_bytes)
        .map_err(BackendError::from)?;
    let data = gguf.tensor_data(&tensor.name)?;
    let allocation = match reservation.commit() {
        Ok(allocation) => allocation,
        Err(error) => {
            drop(data);
            return Err(BackendError::from(error).into());
        }
    };
    Ok(StagedBytes {
        data,
        _allocation: allocation,
    })
}

struct StagedBytes {
    data: Vec<u8>,
    _allocation: MemoryAllocation,
}

impl StagedBytes {
    fn as_slice(&self) -> &[u8] {
        &self.data
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MemoryBudget, MemoryError};

    #[test]
    fn loaded_tensor_set_holds_exact_marker_storage() {
        let staging = HostStaging::new(MemoryBudget::limited(1).unwrap());
        let loaded = LoadedTensorSet::new(1, &staging).unwrap();
        let snapshot = staging.snapshot();
        assert_eq!(snapshot.live_bytes, 1);
        assert_eq!(snapshot.reserved_bytes, 0);
        drop(loaded);
        assert_eq!(staging.snapshot().live_bytes, 0);
    }

    #[test]
    fn loaded_tensor_set_denial_precedes_marker_storage() {
        let staging = HostStaging::new(MemoryBudget::limited(1).unwrap());
        assert!(matches!(
            LoadedTensorSet::new(2, &staging),
            Err(ModelLoadError::Backend(BackendError::Memory(
                MemoryError::BudgetExceeded { .. }
            )))
        ));
        assert_eq!(staging.snapshot().reserved_bytes, 0);
        assert_eq!(staging.snapshot().live_bytes, 0);
    }

    #[test]
    fn staged_vector_releases_storage_after_values_drop() {
        let staging = HostStaging::new(MemoryBudget::limited(8).unwrap());
        let reservation = staging.reserve(8).unwrap();
        let values = vec![0_u8; 8];
        let staged = StagedVec {
            values,
            _allocation: Some(reservation.commit().unwrap()),
        };
        assert_eq!(staging.snapshot().live_bytes, 8);
        drop(staged);
        assert_eq!(staging.snapshot().live_bytes, 0);
    }
}
