#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
#    include <dlfcn.h>
#endif

namespace {

constexpr char PLAN_MAGIC[] = "LCMPPLN1";
constexpr uint32_t PLAN_VERSION = 2;
constexpr uint32_t MAX_CASES = 64;
constexpr uint32_t MAX_EVENTS = 1024;
constexpr uint32_t MAX_STREAMS = 64;
constexpr uint32_t MAX_NAME_BYTES = 128;
constexpr uint32_t MAX_TOKENS = 1U << 20;

struct Options {
    std::string model_path;
    std::string plan_path;
    std::string logits_path;
    std::string backend;
    uint32_t n_ctx;
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    int32_t threads;
    ggml_type kv_type;
    llama_flash_attn_type flash_attention;
    bool kv_unified;
    bool swa_full;
};

struct Stream {
    int32_t sequence;
    int32_t position;
    std::vector<llama_token> tokens;
    std::vector<uint8_t> outputs;
};

struct Event {
    uint8_t kind;
    std::string name;
    std::vector<Stream> streams;
    int32_t sequence = -1;
    int32_t source = -1;
    int32_t target = -1;
    int32_t p0 = -1;
    int32_t p1 = -1;
};

struct Case {
    std::string name;
    std::vector<Event> events;
};

struct Plan {
    std::vector<Case> cases;
};

struct DeviceInfo {
    ggml_backend_dev_t device;
    size_t free_before;
    size_t total_before;
    std::string module_path;
    bool module_dynamic;
};

struct LogitRow {
    size_t stream;
    size_t token;
    size_t batch;
    int32_t sequence;
    int32_t input_position;
    uint64_t offset;
};

struct EventResult {
    uint64_t duration_ns;
    size_t submitted_tokens;
    std::vector<LogitRow> rows;
};

class PlanReader {
public:
    explicit PlanReader(const std::string & path) : input_(path, std::ios::binary) {
        if (!input_) {
            throw std::runtime_error("cannot open the comparator plan");
        }
    }

    void expect_magic() {
        char actual[sizeof(PLAN_MAGIC) - 1] = {};
        read_bytes(actual, sizeof(actual));
        if (std::memcmp(actual, PLAN_MAGIC, sizeof(actual)) != 0) {
            throw std::runtime_error("comparator plan magic differs");
        }
    }

    uint8_t read_u8() {
        uint8_t value = 0;
        read_bytes(&value, sizeof(value));
        return value;
    }

    uint32_t read_u32() {
        uint8_t bytes[4] = {};
        read_bytes(bytes, sizeof(bytes));
        return static_cast<uint32_t>(bytes[0]) |
            (static_cast<uint32_t>(bytes[1]) << 8) |
            (static_cast<uint32_t>(bytes[2]) << 16) |
            (static_cast<uint32_t>(bytes[3]) << 24);
    }

    int32_t read_i32() {
        const uint32_t value = read_u32();
        int32_t result = 0;
        std::memcpy(&result, &value, sizeof(result));
        return result;
    }

    std::string read_string() {
        const uint32_t length = read_bounded(MAX_NAME_BYTES, "plan string");
        if (length == 0) {
            throw std::runtime_error("comparator plan contains an empty name");
        }
        std::string value(length, '\0');
        read_bytes(value.data(), length);
        return value;
    }

    uint32_t read_bounded(uint32_t maximum, const char * field) {
        const uint32_t value = read_u32();
        if (value > maximum) {
            throw std::runtime_error(std::string(field) + " exceeds its bound");
        }
        return value;
    }

    void expect_end() {
        char byte = 0;
        input_.read(&byte, 1);
        if (!input_.eof()) {
            throw std::runtime_error("comparator plan has trailing bytes");
        }
    }

private:
    void read_bytes(void * output, size_t length) {
        if (length > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
            throw std::runtime_error("comparator plan read is too large");
        }
        input_.read(static_cast<char *>(output), static_cast<std::streamsize>(length));
        if (!input_) {
            throw std::runtime_error("comparator plan is truncated");
        }
    }

    std::ifstream input_;
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
    explicit BatchGuard(int32_t capacity) : batch(llama_batch_init(capacity, 0, 1)) {
        if (batch.token == nullptr || batch.pos == nullptr || batch.n_seq_id == nullptr ||
            batch.seq_id == nullptr || batch.logits == nullptr) {
            llama_batch_free(batch);
            throw std::runtime_error("llama.cpp could not allocate the comparator batch");
        }
    }

    ~BatchGuard() {
        llama_batch_free(batch);
    }

    llama_batch batch;
};

using ModelPtr = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using ContextPtr = std::unique_ptr<llama_context, decltype(&llama_free)>;

uint64_t elapsed_ns(std::chrono::steady_clock::time_point start) {
    const auto elapsed = std::chrono::steady_clock::now() - start;
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
}

long parse_integer(const char * value, const char * field, long minimum, long maximum) {
    char * end = nullptr;
    errno = 0;
    const long parsed = std::strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed < minimum || parsed > maximum) {
        throw std::runtime_error(std::string(field) + " is outside its accepted integer range");
    }
    return parsed;
}

bool parse_bool(const char * value, const char * field) {
    if (std::string(value) == "true") {
        return true;
    }
    if (std::string(value) == "false") {
        return false;
    }
    throw std::runtime_error(std::string(field) + " must be true or false");
}

ggml_type parse_kv_type(const char * value) {
    if (std::string(value) == "f16") {
        return GGML_TYPE_F16;
    }
    if (std::string(value) == "q8_0") {
        return GGML_TYPE_Q8_0;
    }
    throw std::runtime_error("KV type must be f16 or q8_0");
}

llama_flash_attn_type parse_flash_attention(const char * value) {
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

Options parse_options(int argc, char ** argv) {
    if (argc != 14) {
        throw std::runtime_error(
            "usage: llama-cached-comparator MODEL PLAN LOGITS BACKEND N_CTX N_BATCH N_UBATCH "
            "N_SEQ_MAX THREADS KV_TYPE FLASH_ATTN KV_UNIFIED SWA_FULL");
    }
    const std::string backend = argv[4];
    if (backend != "cpu" && backend != "cuda" && backend != "metal") {
        throw std::runtime_error("backend must be cpu, cuda, or metal");
    }
    return Options{
        argv[1], argv[2], argv[3], backend,
        static_cast<uint32_t>(parse_integer(argv[5], "n_ctx", 2, MAX_TOKENS)),
        static_cast<uint32_t>(parse_integer(argv[6], "n_batch", 1, MAX_TOKENS)),
        static_cast<uint32_t>(parse_integer(argv[7], "n_ubatch", 1, MAX_TOKENS)),
        static_cast<uint32_t>(parse_integer(argv[8], "n_seq_max", 1, MAX_STREAMS)),
        static_cast<int32_t>(parse_integer(argv[9], "threads", 1, 1024)),
        parse_kv_type(argv[10]), parse_flash_attention(argv[11]),
        parse_bool(argv[12], "kv_unified"), parse_bool(argv[13], "swa_full")};
}

Stream read_stream(PlanReader & reader) {
    Stream stream;
    stream.sequence = reader.read_i32();
    stream.position = reader.read_i32();
    const uint32_t count = reader.read_bounded(MAX_TOKENS, "plan stream token count");
    if (count == 0) {
        throw std::runtime_error("comparator plan contains an empty stream");
    }
    stream.tokens.reserve(count);
    stream.outputs.reserve(count);
    for (uint32_t index = 0; index < count; ++index) {
        stream.tokens.push_back(reader.read_i32());
        const uint8_t output = reader.read_u8();
        if (output > 1) {
            throw std::runtime_error("comparator plan output flag is not boolean");
        }
        stream.outputs.push_back(output);
    }
    return stream;
}

Event read_decode_event(PlanReader & reader, std::string name) {
    Event event;
    event.kind = 1;
    event.name = std::move(name);
    const uint32_t count = reader.read_bounded(MAX_STREAMS, "plan stream count");
    if (count == 0) {
        throw std::runtime_error("comparator plan contains an empty decode event");
    }
    event.streams.reserve(count);
    for (uint32_t index = 0; index < count; ++index) {
        event.streams.push_back(read_stream(reader));
    }
    return event;
}

Event read_copy_event(PlanReader & reader, std::string name) {
    Event event;
    event.kind = 2;
    event.name = std::move(name);
    event.source = reader.read_i32();
    event.target = reader.read_i32();
    event.p0 = reader.read_i32();
    event.p1 = reader.read_i32();
    return event;
}

Event read_remove_event(PlanReader & reader, std::string name) {
    Event event;
    event.kind = 3;
    event.name = std::move(name);
    event.sequence = reader.read_i32();
    event.p0 = reader.read_i32();
    event.p1 = reader.read_i32();
    return event;
}

Event read_event(PlanReader & reader) {
    const uint8_t kind = reader.read_u8();
    std::string name = reader.read_string();
    if (kind == 1) {
        return read_decode_event(reader, std::move(name));
    }
    if (kind == 2) {
        return read_copy_event(reader, std::move(name));
    }
    if (kind == 3) {
        return read_remove_event(reader, std::move(name));
    }
    throw std::runtime_error("comparator plan contains an unknown event");
}

Case read_case(PlanReader & reader) {
    Case value;
    value.name = reader.read_string();
    const uint32_t count = reader.read_bounded(MAX_EVENTS, "plan event count");
    if (count == 0) {
        throw std::runtime_error("comparator plan contains an empty case");
    }
    value.events.reserve(count);
    for (uint32_t index = 0; index < count; ++index) {
        value.events.push_back(read_event(reader));
    }
    return value;
}

Plan read_plan(const std::string & path) {
    PlanReader reader(path);
    reader.expect_magic();
    if (reader.read_u32() != PLAN_VERSION) {
        throw std::runtime_error("comparator plan version differs");
    }
    const uint32_t count = reader.read_bounded(MAX_CASES, "plan case count");
    if (count == 0) {
        throw std::runtime_error("comparator plan contains no cases");
    }
    Plan plan;
    plan.cases.reserve(count);
    for (uint32_t index = 0; index < count; ++index) {
        plan.cases.push_back(read_case(reader));
    }
    reader.expect_end();
    return plan;
}

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

bool device_matches(ggml_backend_dev_t device, const std::string & backend) {
    const std::string registry = lower(ggml_backend_reg_name(ggml_backend_dev_backend_reg(device)));
    if (backend == "cpu") {
        return ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU && registry.find("cpu") != std::string::npos;
    }
    return registry.find(backend) != std::string::npos;
}

std::string device_backend(ggml_backend_dev_t device) {
    const std::string registry = lower(ggml_backend_reg_name(ggml_backend_dev_backend_reg(device)));
    if (registry.find("cuda") != std::string::npos) {
        return "cuda";
    }
    if (registry.find("metal") != std::string::npos) {
        return "metal";
    }
    if (ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU && registry.find("cpu") != std::string::npos) {
        return "cpu";
    }
    return "unknown";
}

std::string backend_module_path(ggml_backend_dev_t device) {
    const auto registry = ggml_backend_dev_backend_reg(device);
    const void * probe = ggml_backend_reg_get_proc_address(registry, "ggml_backend_get_features");
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
    if (probe != nullptr) {
        Dl_info info = {};
        if (dladdr(probe, &info) != 0 && info.dli_fname != nullptr) {
            return info.dli_fname;
        }
    }
#else
    static_cast<void>(probe);
#endif
    return {};
}

DeviceInfo select_device(const std::string & backend) {
    for (size_t index = 0; index < ggml_backend_dev_count(); ++index) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        if (device_matches(device, backend)) {
            size_t free = 0;
            size_t total = 0;
            ggml_backend_dev_memory(device, &free, &total);
            std::string module_path = backend_module_path(device);
            const bool module_dynamic = module_path.find("libggml-") != std::string::npos;
            return DeviceInfo{device, free, total, std::move(module_path), module_dynamic};
        }
    }
    throw std::runtime_error("llama.cpp could not find the requested backend device");
}

ModelPtr load_model(const Options & options, ggml_backend_dev_t device) {
    auto parameters = llama_model_default_params();
    ggml_backend_dev_t devices[] = {device, nullptr};
    parameters.devices = devices;
    parameters.n_gpu_layers = options.backend == "cpu" ? 0 : -1;
    parameters.split_mode = LLAMA_SPLIT_MODE_NONE;
    parameters.main_gpu = 0;
    ModelPtr model(llama_model_load_from_file(options.model_path.c_str(), parameters), llama_model_free);
    if (!model) {
        throw std::runtime_error("llama.cpp could not load the comparator model");
    }
    return model;
}

size_t maximum_output_rows(const Plan & plan) {
    size_t maximum = 1;
    for (const auto & test_case : plan.cases) {
        for (const auto & event : test_case.events) {
            size_t outputs = 0;
            for (const auto & stream : event.streams) {
                outputs += static_cast<size_t>(std::count(stream.outputs.begin(), stream.outputs.end(), 1));
            }
            maximum = std::max(maximum, outputs);
        }
    }
    return maximum;
}

ContextPtr create_context(llama_model * model, const Options & options, const Plan & plan) {
    auto parameters = llama_context_default_params();
    parameters.n_ctx = options.n_ctx;
    parameters.n_batch = options.n_batch;
    parameters.n_ubatch = options.n_ubatch;
    parameters.n_seq_max = options.n_seq_max;
    parameters.n_outputs_max = static_cast<uint32_t>(std::max<size_t>(options.n_seq_max, maximum_output_rows(plan)));
    parameters.n_outputs_max_per_seq = 1;
    parameters.n_threads = options.threads;
    parameters.n_threads_batch = options.threads;
    parameters.type_k = options.kv_type;
    parameters.type_v = options.kv_type;
    parameters.flash_attn_type = options.flash_attention;
    parameters.offload_kqv = options.backend != "cpu";
    parameters.op_offload = options.backend != "cpu";
    parameters.no_perf = true;
    parameters.swa_full = options.swa_full;
    parameters.kv_unified = options.kv_unified;
    ContextPtr context(llama_init_from_model(model, parameters), llama_free);
    if (!context) {
        throw std::runtime_error("llama.cpp could not create the comparator context");
    }
    return context;
}

void validate_token(llama_token token, int32_t vocabulary) {
    if (token < 0 || token >= vocabulary) {
        throw std::runtime_error("comparator plan token exceeds the model vocabulary");
    }
}

using SequenceState = std::vector<std::set<int32_t>>;

void validate_sequence(int32_t sequence, const Options & options, const char * operation) {
    if (sequence < 0 || static_cast<uint32_t>(sequence) >= options.n_seq_max) {
        throw std::runtime_error(std::string("comparator ") + operation + " sequence exceeds n_seq_max");
    }
}

void validate_decode_stream(
    const Stream & stream,
    const Options & options,
    int32_t vocabulary,
    SequenceState & state,
    std::vector<bool> & sequences) {
    validate_sequence(stream.sequence, options, "decode");
    const size_t sequence = static_cast<size_t>(stream.sequence);
    if (sequences[sequence]) {
        throw std::runtime_error("comparator decode event repeats a sequence");
    }
    sequences[sequence] = true;
    if (stream.position < 0 || static_cast<uint64_t>(stream.position) + stream.tokens.size() > options.n_ctx) {
        throw std::runtime_error("comparator decode event exceeds n_ctx");
    }
    for (llama_token token : stream.tokens) {
        validate_token(token, vocabulary);
    }
    std::set<int32_t> positions;
    for (size_t index = 0; index < stream.tokens.size(); ++index) {
        positions.insert(stream.position + static_cast<int32_t>(index));
    }
    const auto & current = state[sequence];
    for (int32_t position : positions) {
        if (current.count(position) != 0) {
            throw std::runtime_error("comparator decode overwrites a sequence position");
        }
    }
    const int32_t expected = current.empty() ? 0 : *current.rbegin() + 1;
    if (stream.position != expected) {
        throw std::runtime_error("comparator decode sequence position has a gap");
    }
    state[sequence].insert(positions.begin(), positions.end());
}

size_t validate_decode_event(
    const Event & event,
    const Options & options,
    int32_t vocabulary,
    SequenceState & state) {
    size_t submitted = 0;
    std::vector<bool> sequences(options.n_seq_max, false);
    for (const auto & stream : event.streams) {
        validate_decode_stream(stream, options, vocabulary, state, sequences);
        submitted += stream.tokens.size();
    }
    return submitted;
}

void validate_position_range(const Event & event, const Options & options) {
    if (event.p0 < 0) {
        throw std::runtime_error("comparator position range start exceeds n_ctx");
    }
    if (event.p0 >= static_cast<int32_t>(options.n_ctx)) {
        throw std::runtime_error("comparator position range start exceeds n_ctx");
    }
    if (event.p1 == -1) {
        return;
    }
    if (event.p1 <= event.p0) {
        throw std::runtime_error("comparator position range end is invalid");
    }
    if (event.p1 > static_cast<int32_t>(options.n_ctx)) {
        throw std::runtime_error("comparator position range end is invalid");
    }
}

void validate_copy_event(const Event & event, const Options & options, SequenceState & state) {
    validate_sequence(event.source, options, "copy source");
    validate_sequence(event.target, options, "copy target");
    if (event.source == event.target) {
        throw std::runtime_error("comparator copy source and target must differ");
    }
    validate_position_range(event, options);
    const auto & source = state[static_cast<size_t>(event.source)];
    auto & target = state[static_cast<size_t>(event.target)];
    if (!target.empty()) {
        throw std::runtime_error("comparator copy target sequence is not empty");
    }
    const int32_t upper = event.p1 < 0 ? std::numeric_limits<int32_t>::max() : event.p1;
    std::set<int32_t> copied;
    for (int32_t position : source) {
        if (position >= event.p0 && position < upper) {
            copied.insert(position);
        }
    }
    if (copied.empty()) {
        throw std::runtime_error("comparator copy range contains no positions");
    }
    int32_t expected = *copied.begin();
    for (int32_t position : copied) {
        if (position != expected) {
            throw std::runtime_error("comparator copy source range is not contiguous");
        }
        ++expected;
    }
    target.insert(copied.begin(), copied.end());
}

void validate_remove_event(const Event & event, const Options & options, SequenceState & state) {
    validate_sequence(event.sequence, options, "remove");
    validate_position_range(event, options);
    auto & sequence = state[static_cast<size_t>(event.sequence)];
    const int32_t upper = event.p1 < 0 ? std::numeric_limits<int32_t>::max() : event.p1;
    size_t removed = 0;
    for (auto iterator = sequence.begin(); iterator != sequence.end();) {
        if (*iterator >= event.p0 && *iterator < upper) {
            iterator = sequence.erase(iterator);
            ++removed;
        } else {
            ++iterator;
        }
    }
    if (removed == 0) {
        throw std::runtime_error("comparator remove range contains no positions");
    }
}

void validate_plan_for_model(const Plan & plan, const Options & options, int32_t vocabulary) {
    for (const auto & test_case : plan.cases) {
        SequenceState state(options.n_seq_max);
        size_t submitted = 0;
        for (const auto & event : test_case.events) {
            if (event.kind == 1) {
                const size_t event_submitted =
                    validate_decode_event(event, options, vocabulary, state);
                if (event_submitted > options.n_batch) {
                    throw std::runtime_error("comparator decode event exceeds n_batch");
                }
                submitted += event_submitted;
            } else if (event.kind == 2) {
                validate_copy_event(event, options, state);
            } else {
                validate_remove_event(event, options, state);
            }
        }
        if (submitted > options.n_ctx) {
            throw std::runtime_error("comparator case exceeds n_ctx");
        }
    }
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
            default: escaped += character < 0x20 ? '?' : static_cast<char>(character); break;
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

const char * kv_type_name(ggml_type type) {
    return type == GGML_TYPE_F16 ? "f16" : "q8_0";
}

const char * flash_attention_name(llama_flash_attn_type value) {
    if (value == LLAMA_FLASH_ATTN_TYPE_ENABLED) {
        return "on";
    }
    if (value == LLAMA_FLASH_ATTN_TYPE_DISABLED) {
        return "off";
    }
    return "auto";
}

void write_requested_context(const Options & options) {
    std::cout << "{\"n_ctx\":" << options.n_ctx
              << ",\"n_batch\":" << options.n_batch
              << ",\"n_ubatch\":" << options.n_ubatch
              << ",\"n_seq_max\":" << options.n_seq_max
              << ",\"threads\":" << options.threads
              << ",\"kv_type\":\"" << kv_type_name(options.kv_type) << "\""
              << ",\"flash_attention\":\"" << flash_attention_name(options.flash_attention) << "\""
              << ",\"kv_unified\":" << (options.kv_unified ? "true" : "false")
              << ",\"swa_full\":" << (options.swa_full ? "true" : "false") << '}';
}

void write_device(const DeviceInfo & selected, size_t free_after, size_t total_after) {
    auto registry = ggml_backend_dev_backend_reg(selected.device);
    std::cout << "{\"backend_module_path\":\"" << json_escape(selected.module_path)
              << "\",\"backend_module_dynamic\":" << (selected.module_dynamic ? "true" : "false")
              << ",\"backend\":\"" << device_backend(selected.device)
              << "\",\"registry\":\"" << json_escape(ggml_backend_reg_name(registry))
              << "\",\"name\":\"" << json_escape(ggml_backend_dev_name(selected.device))
              << "\",\"description\":\"" << json_escape(ggml_backend_dev_description(selected.device))
              << "\",\"type\":" << static_cast<int>(ggml_backend_dev_type(selected.device))
              << ",\"memory_free_before_model\":" << selected.free_before
              << ",\"memory_total_before_model\":" << selected.total_before
              << ",\"memory_free_after_context\":" << free_after
              << ",\"memory_total_after_context\":" << total_after << '}';
}

void write_model(const llama_model * model, int32_t vocabulary) {
    const auto ftype = llama_model_ftype(model);
    std::cout << "{\"vocab\":" << vocabulary
              << ",\"context_train\":" << llama_model_n_ctx_train(model)
              << ",\"size_bytes\":" << llama_model_size(model)
              << ",\"ftype\":" << static_cast<int>(ftype)
              << ",\"ftype_name\":\"" << json_escape(llama_ftype_name(ftype))
              << "\",\"architecture\":\"" << json_escape(model_metadata(model, "general.architecture"))
              << "\",\"tokenizer\":\"" << json_escape(model_metadata(model, "tokenizer.ggml.model")) << "\"}";
}

void write_header(
    const Options & options,
    const DeviceInfo & selected,
    const llama_model * model,
    const llama_context * context,
    int32_t vocabulary,
    uint64_t load_ns,
    uint64_t context_ns) {
    size_t free_after = 0;
    size_t total_after = 0;
    ggml_backend_dev_memory(selected.device, &free_after, &total_after);
    std::cout << "{\"kind\":\"header\",\"protocol\":\"leone.llama-cached-engine.v2\",\"requested_backend\":\""
              << options.backend << "\",\"device\":";
    write_device(selected, free_after, total_after);
    std::cout << ",\"model\":";
    write_model(model, vocabulary);
    std::cout << ",\"context\":{\"requested\":";
    write_requested_context(options);
    std::cout << ",\"effective\":{\"n_ctx\":" << llama_n_ctx(context)
              << ",\"n_ctx_seq\":" << llama_n_ctx_seq(context)
              << ",\"n_batch\":" << llama_n_batch(context)
              << ",\"n_ubatch\":" << llama_n_ubatch(context)
              << ",\"n_seq_max\":" << llama_n_seq_max(context)
              << "}},\"model_load_ns\":" << load_ns
              << ",\"context_create_ns\":" << context_ns << "}\n";
}

size_t fill_batch(llama_batch & batch, const Event & event, int32_t vocabulary) {
    size_t index = 0;
    for (const auto & stream : event.streams) {
        for (size_t token_index = 0; token_index < stream.tokens.size(); ++token_index) {
            validate_token(stream.tokens[token_index], vocabulary);
            batch.token[index] = stream.tokens[token_index];
            batch.pos[index] = stream.position + static_cast<int32_t>(token_index);
            batch.n_seq_id[index] = 1;
            batch.seq_id[index][0] = stream.sequence;
            batch.logits[index] = static_cast<int8_t>(stream.outputs[token_index]);
            ++index;
        }
    }
    batch.n_tokens = static_cast<int32_t>(index);
    return index;
}

std::vector<LogitRow> copy_logits(
    llama_context * context,
    const Event & event,
    std::ofstream & output,
    int32_t vocabulary,
    uint64_t & offset) {
    std::vector<LogitRow> rows;
    size_t batch_index = 0;
    const auto bytes = static_cast<std::streamsize>(vocabulary) * static_cast<std::streamsize>(sizeof(float));
    for (size_t stream_index = 0; stream_index < event.streams.size(); ++stream_index) {
        const auto & stream = event.streams[stream_index];
        for (size_t token_index = 0; token_index < stream.tokens.size(); ++token_index) {
            if (stream.outputs[token_index] != 0) {
                const float * logits = llama_get_logits_ith(context, static_cast<int32_t>(batch_index));
                if (logits == nullptr) {
                    throw std::runtime_error("llama.cpp did not retain a requested logit row");
                }
                output.write(reinterpret_cast<const char *>(logits), bytes);
                rows.push_back(LogitRow{stream_index, token_index, batch_index, stream.sequence,
                                        stream.position + static_cast<int32_t>(token_index), offset});
                offset += static_cast<uint64_t>(vocabulary) * sizeof(float);
            }
            ++batch_index;
        }
    }
    if (!output) {
        throw std::runtime_error("failed while writing comparator logits");
    }
    return rows;
}

EventResult run_decode(
    llama_context * context,
    llama_batch & batch,
    const Event & event,
    std::ofstream & output,
    int32_t vocabulary,
    uint64_t & offset) {
    const auto start = std::chrono::steady_clock::now();
    const size_t submitted = fill_batch(batch, event, vocabulary);
    const int32_t status = llama_decode(context, batch);
    if (status != 0) {
        throw std::runtime_error("llama.cpp decode failed with status " + std::to_string(status));
    }
    llama_synchronize(context);
    auto rows = copy_logits(context, event, output, vocabulary, offset);
    return EventResult{elapsed_ns(start), submitted, std::move(rows)};
}

EventResult run_copy(llama_context * context, const Event & event) {
    const auto start = std::chrono::steady_clock::now();
    llama_memory_seq_cp(llama_get_memory(context), event.source, event.target, event.p0, event.p1);
    llama_synchronize(context);
    return EventResult{elapsed_ns(start), 0, {}};
}

EventResult run_remove(llama_context * context, const Event & event) {
    const auto start = std::chrono::steady_clock::now();
    const bool removed = llama_memory_seq_rm(
        llama_get_memory(context), event.sequence, event.p0, event.p1);
    if (!removed) {
        throw std::runtime_error("llama.cpp rejected a sequence removal");
    }
    llama_synchronize(context);
    return EventResult{elapsed_ns(start), 0, {}};
}

const char * event_kind_name(uint8_t kind) {
    if (kind == 1) {
        return "decode";
    }
    if (kind == 2) {
        return "copy";
    }
    return "remove";
}

void write_ranges(llama_context * context, uint32_t n_seq_max) {
    llama_memory_t memory = llama_get_memory(context);
    std::cout << '[';
    for (uint32_t sequence = 0; sequence < n_seq_max; ++sequence) {
        if (sequence != 0) {
            std::cout << ',';
        }
        std::cout << "{\"seq\":" << sequence
                  << ",\"min\":" << llama_memory_seq_pos_min(memory, static_cast<int32_t>(sequence))
                  << ",\"max\":" << llama_memory_seq_pos_max(memory, static_cast<int32_t>(sequence)) << '}';
    }
    std::cout << ']';
}

void write_event_record(
    llama_context * context,
    const Options & options,
    const Event & event,
    const EventResult & result,
    size_t case_index,
    size_t event_index) {
    std::cout << "{\"kind\":\"event\",\"case\":" << case_index
              << ",\"event\":" << event_index
              << ",\"name\":\"" << json_escape(event.name)
              << "\",\"operation\":\"" << event_kind_name(event.kind)
              << "\",\"duration_ns\":" << result.duration_ns
              << ",\"submitted_tokens\":" << result.submitted_tokens
              << ",\"output_rows\":" << result.rows.size() << ",\"ranges\":";
    write_ranges(context, options.n_seq_max);
    std::cout << "}\n";
}

void write_logit_records(
    const std::vector<LogitRow> & rows,
    size_t case_index,
    size_t event_index,
    int32_t vocabulary) {
    for (const auto & row : rows) {
        std::cout << "{\"kind\":\"logit\",\"case\":" << case_index
                  << ",\"event\":" << event_index
                  << ",\"stream\":" << row.stream
                  << ",\"token_index\":" << row.token
                  << ",\"batch_index\":" << row.batch
                  << ",\"seq\":" << row.sequence
                  << ",\"input_position\":" << row.input_position
                  << ",\"predicted_position\":" << row.input_position + 1
                  << ",\"offset_bytes\":" << row.offset
                  << ",\"float_count\":" << vocabulary << "}\n";
    }
}

void run_cases(
    llama_context * context,
    const Plan & plan,
    const Options & options,
    std::ofstream & output,
    int32_t vocabulary) {
    BatchGuard batch(static_cast<int32_t>(options.n_batch));
    uint64_t offset = 0;
    for (size_t case_index = 0; case_index < plan.cases.size(); ++case_index) {
        llama_memory_clear(llama_get_memory(context), false);
        const auto & test_case = plan.cases[case_index];
        for (size_t event_index = 0; event_index < test_case.events.size(); ++event_index) {
            const auto & event = test_case.events[event_index];
            EventResult result{0, 0, {}};
            if (event.kind == 1) {
                result = run_decode(context, batch.batch, event, output, vocabulary, offset);
            } else if (event.kind == 2) {
                result = run_copy(context, event);
            } else {
                result = run_remove(context, event);
            }
            write_event_record(context, options, event, result, case_index, event_index);
            write_logit_records(result.rows, case_index, event_index, vocabulary);
        }
    }
}

void validate_platform() {
    static_assert(sizeof(float) == 4, "comparator requires 32-bit float");
    static_assert(std::numeric_limits<float>::is_iec559, "comparator requires IEEE-754 float");
    const uint32_t one = 1;
    if (*reinterpret_cast<const uint8_t *>(&one) != 1) {
        throw std::runtime_error("comparator requires a little-endian host");
    }
}

void llama_log_callback(ggml_log_level level, const char * message, void *) {
    if (level == GGML_LOG_LEVEL_ERROR) {
        std::cerr << message;
    }
}

int run(int argc, char ** argv) {
    validate_platform();
    if (argc == 3 && std::string(argv[1]) == "--validate-plan") {
        const Plan plan = read_plan(argv[2]);
        std::cout << "{\"plan_version\":" << PLAN_VERSION << ",\"cases\":" << plan.cases.size()
                  << ",\"validation\":\"structure-only\",\"history_validation\":\"runner-required\"}\n";
        return 0;
    }
    const Options options = parse_options(argc, argv);
    if (options.n_ubatch > options.n_batch) {
        throw std::runtime_error("n_ubatch must not exceed n_batch");
    }
    const Plan plan = read_plan(options.plan_path);
    llama_log_set(llama_log_callback, nullptr);
    BackendGuard backend;
    const DeviceInfo selected = select_device(options.backend);
    const auto load_start = std::chrono::steady_clock::now();
    ModelPtr model = load_model(options, selected.device);
    const uint64_t load_ns = elapsed_ns(load_start);
    const llama_vocab * vocab = llama_model_get_vocab(model.get());
    if (vocab == nullptr || llama_vocab_n_tokens(vocab) <= 0) {
        throw std::runtime_error("comparator model has no vocabulary");
    }
    const int32_t vocabulary = llama_vocab_n_tokens(vocab);
    validate_plan_for_model(plan, options, vocabulary);
    const auto context_start = std::chrono::steady_clock::now();
    ContextPtr context = create_context(model.get(), options, plan);
    const uint64_t context_ns = elapsed_ns(context_start);
    std::ofstream output(options.logits_path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot create the comparator logit sidecar");
    }
    write_header(options, selected, model.get(), context.get(), vocabulary, load_ns, context_ns);
    run_cases(context.get(), plan, options, output, vocabulary);
    output.flush();
    if (!output) {
        throw std::runtime_error("failed while flushing comparator logits");
    }
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
