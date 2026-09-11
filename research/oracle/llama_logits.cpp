#include "llama.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    size_t window_tokens;
    int threads;
    uint32_t n_batch;
    uint32_t n_ubatch;
    bool cuda;
    llama_flash_attn_type flash_attn;
};

struct ModelInfo {
    int32_t vocab_size;
    int32_t model_context;
};

struct ContextInfo {
    uint32_t capacity;
    uint32_t n_batch;
    uint32_t n_ubatch;
};

struct BackendGuard {
    BackendGuard() {
        llama_backend_init();
    }

    ~BackendGuard() {
        llama_backend_free();
    }
};

struct BatchGuard {
    explicit BatchGuard(int32_t n_tokens) : batch(llama_batch_init(n_tokens, 0, 1)) {
        if (batch.token == nullptr || batch.pos == nullptr || batch.n_seq_id == nullptr ||
            batch.seq_id == nullptr || batch.logits == nullptr) {
            llama_batch_free(batch);
            throw std::runtime_error("llama.cpp could not allocate the input batch");
        }
    }

    ~BatchGuard() {
        llama_batch_free(batch);
    }

    llama_batch batch;
};

using ModelPtr = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using ContextPtr = std::unique_ptr<llama_context, decltype(&llama_free)>;

std::vector<uint8_t> read_token_bytes(const char * path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error(std::string("cannot open token file: ") + path);
    }
    const auto size = input.tellg();
    if (size < 0 || size % static_cast<std::streamoff>(sizeof(uint32_t)) != 0) {
        throw std::runtime_error("token file size is not a multiple of four bytes");
    }
    if (size > std::numeric_limits<std::streamsize>::max()) {
        throw std::runtime_error("token file is too large for the input stream");
    }
    const auto count = static_cast<uint64_t>(size) / sizeof(uint32_t);
    if (count > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("token file is too large");
    }
    const auto byte_count = static_cast<size_t>(size);
    input.seekg(0);
    std::vector<uint8_t> bytes(byte_count);
    input.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(byte_count));
    if (!input || input.gcount() != static_cast<std::streamsize>(byte_count)) {
        throw std::runtime_error(std::string("failed while reading token file: ") + path);
    }
    return bytes;
}

llama_token decode_token(const uint8_t * bytes) {
    const uint32_t raw = static_cast<uint32_t>(bytes[0])
        | (static_cast<uint32_t>(bytes[1]) << 8)
        | (static_cast<uint32_t>(bytes[2]) << 16)
        | (static_cast<uint32_t>(bytes[3]) << 24);
    if (raw > static_cast<uint32_t>(std::numeric_limits<llama_token>::max())) {
        throw std::runtime_error("token ID exceeds the oracle token range");
    }
    return static_cast<llama_token>(raw);
}

std::vector<llama_token> read_tokens(const char * path) {
    const auto bytes = read_token_bytes(path);
    const size_t count = bytes.size() / sizeof(uint32_t);
    std::vector<llama_token> tokens(static_cast<size_t>(count));
    for (size_t index = 0; index < tokens.size(); ++index) {
        tokens[index] = decode_token(bytes.data() + index * sizeof(uint32_t));
    }
    if (tokens.size() < 2) {
        throw std::runtime_error("token file must contain at least two tokens");
    }
    return tokens;
}

long parse_integer(const char * value, const char * field, long minimum, long maximum) {
    errno = 0;
    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed < minimum || parsed > maximum) {
        throw std::runtime_error(std::string(field) + " must be an integer in the requested range");
    }
    return parsed;
}

size_t parse_size(const char * value, const char * field) {
    return static_cast<size_t>(parse_integer(
        value,
        field,
        2,
        static_cast<long>(std::numeric_limits<uint32_t>::max())));
}

int parse_threads(const char * value) {
    return static_cast<int>(parse_integer(value, "thread count", 1, std::numeric_limits<int32_t>::max()));
}

uint32_t parse_batch_size(const char * value, const char * field) {
    return static_cast<uint32_t>(parse_integer(
        value,
        field,
        1,
        static_cast<long>(std::numeric_limits<int32_t>::max())));
}

bool parse_cuda(const char * value) {
    if (std::string(value) == "cpu") {
        return false;
    }
    if (std::string(value) == "cuda") {
        return true;
    }
    throw std::runtime_error("device must be cpu or cuda");
}

llama_flash_attn_type parse_flash_attn(const char * value) {
    if (std::string(value) == "auto") {
        return LLAMA_FLASH_ATTN_TYPE_AUTO;
    }
    if (std::string(value) == "on") {
        return LLAMA_FLASH_ATTN_TYPE_ENABLED;
    }
    if (std::string(value) == "off") {
        return LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }
    throw std::runtime_error("flash attention must be auto, on, or off");
}

void validate_batch_sizes(size_t window_tokens, uint32_t n_batch, uint32_t n_ubatch) {
    if (n_batch < window_tokens) {
        throw std::runtime_error("n_batch must be at least the evaluation window");
    }
    if (n_ubatch > n_batch) {
        throw std::runtime_error("n_ubatch must not exceed n_batch");
    }
}

Options parse_options(int argc, char ** argv, const std::vector<llama_token> & tokens) {
    if (argc < 4 || argc > 10) {
        throw std::runtime_error(
            std::string("usage: ") + argv[0] +
            " MODEL TOKEN_U32LE OUTPUT_F32 [WINDOW] [THREADS] [DEVICE] [N_BATCH] [N_UBATCH] [FLASH_ATTN]");
    }
    const size_t window_tokens = argc >= 5 ? parse_size(argv[4], "window") : tokens.size();
    const int threads = argc >= 6 ? parse_threads(argv[5]) : 16;
    const bool cuda = argc >= 7 ? parse_cuda(argv[6]) : false;
    const uint32_t n_batch = argc >= 8 ? parse_batch_size(argv[7], "n_batch") : static_cast<uint32_t>(window_tokens);
    const uint32_t n_ubatch = argc >= 9 ? parse_batch_size(argv[8], "n_ubatch") : std::min<uint32_t>(512, n_batch);
    const llama_flash_attn_type flash_attn =
        argc >= 10 ? parse_flash_attn(argv[9]) : LLAMA_FLASH_ATTN_TYPE_AUTO;
    validate_batch_sizes(window_tokens, n_batch, n_ubatch);
    return Options{window_tokens, threads, n_batch, n_ubatch, cuda, flash_attn};
}

ggml_backend_dev_t select_device(bool cuda) {
    const auto type = cuda ? GGML_BACKEND_DEVICE_TYPE_GPU : GGML_BACKEND_DEVICE_TYPE_CPU;
    ggml_backend_dev_t device = ggml_backend_dev_by_type(type);
    if (device == nullptr) {
        throw std::runtime_error(
            cuda ? "llama.cpp could not find a CUDA device" : "llama.cpp could not find a CPU device");
    }
    return device;
}

ModelPtr load_model(const char * path, bool cuda) {
    auto model_params = llama_model_default_params();
    ggml_backend_dev_t devices[] = {select_device(cuda), nullptr};
    model_params.devices = devices;
    model_params.n_gpu_layers = cuda ? -1 : 0;
    model_params.split_mode = LLAMA_SPLIT_MODE_NONE;
    model_params.main_gpu = 0;
    ModelPtr model(llama_model_load_from_file(path, model_params), llama_model_free);
    if (!model) {
        throw std::runtime_error("llama.cpp could not load the model");
    }
    return model;
}

void validate_tokens(const std::vector<llama_token> & tokens, const llama_vocab * vocab) {
    const int32_t vocab_size = llama_vocab_n_tokens(vocab);
    for (const auto token : tokens) {
        if (token < 0 || token >= vocab_size) {
            throw std::runtime_error("token ID exceeds the oracle vocabulary");
        }
    }
}

ModelInfo validate_model(
    const llama_model * model,
    const std::vector<llama_token> & tokens,
    const Options & options) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    if (vocab == nullptr) {
        throw std::runtime_error("oracle model has no vocabulary metadata");
    }
    const int32_t vocab_size = llama_vocab_n_tokens(vocab);
    if (vocab_size <= 0) {
        throw std::runtime_error("oracle model has no vocabulary");
    }
    validate_tokens(tokens, vocab);
    const int32_t model_context = llama_model_n_ctx_train(model);
    if (model_context <= 0 || options.window_tokens > static_cast<size_t>(model_context)) {
        throw std::runtime_error("evaluation window exceeds the oracle model context");
    }
    return ModelInfo{vocab_size, model_context};
}

ContextPtr create_context(llama_model * model, const Options & options) {
    auto context_params = llama_context_default_params();
    context_params.n_ctx = static_cast<uint32_t>(options.window_tokens);
    context_params.n_batch = options.n_batch;
    context_params.n_ubatch = options.n_ubatch;
    context_params.n_threads = options.threads;
    context_params.n_threads_batch = options.threads;
    context_params.n_outputs_max = static_cast<uint32_t>(options.window_tokens - 1);
    context_params.flash_attn_type = options.flash_attn;
    context_params.offload_kqv = options.cuda;
    context_params.op_offload = options.cuda;
    ContextPtr context(llama_init_from_model(model, context_params), llama_free);
    if (!context) {
        throw std::runtime_error("llama.cpp could not create the oracle context");
    }
    return context;
}

void check_context_configuration(
    llama_context * context,
    ContextInfo & context_info) {
    const uint32_t actual_context_capacity = llama_n_ctx(context);
    const uint32_t actual_n_batch = llama_n_batch(context);
    const uint32_t actual_n_ubatch = llama_n_ubatch(context);
    if (context_info.capacity == 0) {
        context_info.capacity = actual_context_capacity;
        context_info.n_batch = actual_n_batch;
        context_info.n_ubatch = actual_n_ubatch;
    } else if (context_info.capacity != actual_context_capacity || context_info.n_batch != actual_n_batch ||
               context_info.n_ubatch != actual_n_ubatch) {
        throw std::runtime_error("llama.cpp changed the effective context configuration between windows");
    }
}

void write_logits(
    llama_context * context,
    std::ofstream & output,
    int32_t vocab_size,
    size_t window_size) {
    const auto row_bytes = static_cast<std::streamsize>(vocab_size) * sizeof(float);
    for (size_t index = 0; index + 1 < window_size; ++index) {
        const float * logits = llama_get_logits_ith(context, static_cast<int32_t>(index));
        if (logits == nullptr) {
            throw std::runtime_error("llama.cpp did not retain a requested logit row");
        }
        output.write(reinterpret_cast<const char *>(logits), row_bytes);
    }
    if (!output) {
        throw std::runtime_error("failed while writing oracle logits");
    }
}

size_t decode_window(
    llama_model * model,
    const std::vector<llama_token> & tokens,
    size_t start,
    const Options & options,
    std::ofstream & output,
    int32_t vocab_size,
    ContextInfo & context_info) {
    const size_t end = std::min(start + options.window_tokens, tokens.size());
    const size_t window_size = end - start;
    if (window_size < 2) {
        throw std::runtime_error("evaluation window produced fewer than two tokens");
    }
    auto context = create_context(model, options);
    check_context_configuration(context.get(), context_info);
    BatchGuard batch(static_cast<int32_t>(options.n_batch));
    batch.batch.n_tokens = static_cast<int32_t>(window_size);
    for (size_t index = 0; index < window_size; ++index) {
        batch.batch.token[index] = tokens[start + index];
        batch.batch.pos[index] = static_cast<llama_pos>(index);
        batch.batch.n_seq_id[index] = 1;
        batch.batch.seq_id[index][0] = 0;
        batch.batch.logits[index] = index + 1 < window_size ? 1 : 0;
    }
    const int32_t decode_status = llama_decode(context.get(), batch.batch);
    if (decode_status != 0) {
        throw std::runtime_error("llama.cpp decode failed with status " + std::to_string(decode_status));
    }
    write_logits(context.get(), output, vocab_size, window_size);
    return window_size - 1;
}

std::string json_escape(const std::string & value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (const unsigned char character : value) {
        switch (character) {
            case '"': escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default:
                if (character < 0x20) {
                    escaped += "?";
                } else {
                    escaped += static_cast<char>(character);
                }
                break;
        }
    }
    return escaped;
}

std::string model_metadata(const llama_model * model, const char * key) {
    char value[256] = {};
    if (llama_model_meta_val_str(model, key, value, sizeof(value)) < 0) {
        return {};
    }
    return value;
}

size_t decode_tokens(
    llama_model * model,
    const std::vector<llama_token> & tokens,
    const Options & options,
    std::ofstream & output,
    int32_t vocab_size,
    ContextInfo & context_info) {
    const size_t stride = options.window_tokens - 1;
    size_t rows = 0;
    for (size_t start = 0; start < tokens.size() - 1; start += stride) {
        rows += decode_window(
            model,
            tokens,
            start,
            options,
            output,
            vocab_size,
            context_info);
    }
    return rows;
}

void write_metadata(
    const llama_model * model,
    const Options & options,
    size_t token_count,
    size_t rows,
    const ModelInfo & model_info,
    const ContextInfo & context_info) {
    const size_t stride = options.window_tokens - 1;
    const auto model_ftype = llama_model_ftype(model);
    auto * device = select_device(options.cuda);
    std::cout << "{\"tokens\":" << token_count
              << ",\"rows\":" << rows
              << ",\"vocab\":" << model_info.vocab_size
              << ",\"window\":" << options.window_tokens
              << ",\"stride\":" << stride
              << ",\"threads\":" << options.threads
              << ",\"n_batch\":" << options.n_batch
              << ",\"n_ubatch\":" << options.n_ubatch
              << ",\"context_capacity\":" << context_info.capacity
              << ",\"effective_n_batch\":" << context_info.n_batch
              << ",\"effective_n_ubatch\":" << context_info.n_ubatch
              << ",\"device\":\"" << (options.cuda ? "cuda" : "cpu") << "\""
              << ",\"device_name\":\"" << json_escape(ggml_backend_dev_name(device)) << "\""
              << ",\"flash_attn\":\"" << llama_flash_attn_type_name(options.flash_attn) << "\""
              << ",\"flash_attn_effective\":\"implementation-selected\""
              << ",\"offload_kqv\":" << (options.cuda ? "true" : "false")
              << ",\"op_offload\":" << (options.cuda ? "true" : "false")
              << ",\"model_context\":" << model_info.model_context
              << ",\"model_size\":" << llama_model_size(model)
              << ",\"model_ftype\":" << static_cast<int>(model_ftype)
              << ",\"model_ftype_name\":\"" << json_escape(llama_ftype_name(model_ftype)) << "\""
              << ",\"architecture\":\"" << json_escape(model_metadata(model, "general.architecture")) << "\""
              << ",\"tokenizer\":\"" << json_escape(model_metadata(model, "tokenizer.ggml.model")) << "\"}\n";
}

}

int main(int argc, char ** argv) {
    if (argc < 4 || argc > 10) {
        std::cerr << "usage: " << argv[0]
                  << " MODEL TOKEN_U32LE OUTPUT_F32 [WINDOW] [THREADS] [DEVICE] [N_BATCH] [N_UBATCH] [FLASH_ATTN]\n";
        return 2;
    }

    try {
        const auto tokens = read_tokens(argv[2]);
        const Options options = parse_options(argc, argv, tokens);
        BackendGuard backend;
        auto model = load_model(argv[1], options.cuda);
        const ModelInfo model_info = validate_model(model.get(), tokens, options);

        std::ofstream output(argv[3], std::ios::binary | std::ios::trunc);
        if (!output) {
            throw std::runtime_error(std::string("cannot open output file: ") + argv[3]);
        }
        ContextInfo context_info{};
        const size_t rows = decode_tokens(
            model.get(),
            tokens,
            options,
            output,
            model_info.vocab_size,
            context_info);
        output.flush();
        if (!output) {
            throw std::runtime_error("failed while flushing oracle logits");
        }

        write_metadata(
            model.get(),
            options,
            tokens.size(),
            rows,
            model_info,
            context_info);
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
