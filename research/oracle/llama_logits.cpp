#include "llama.h"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<llama_token> read_tokens(const char * path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error(std::string("cannot open token file: ") + path);
    }
    const auto size = input.tellg();
    if (size < 0 || size % static_cast<std::streamoff>(sizeof(uint32_t)) != 0) {
        throw std::runtime_error("token file size is not a multiple of four bytes");
    }
    input.seekg(0);
    std::vector<llama_token> tokens(static_cast<size_t>(size) / sizeof(uint32_t));
    input.read(reinterpret_cast<char *>(tokens.data()), size);
    if (!input || tokens.size() < 2) {
        throw std::runtime_error("token file must contain at least two tokens");
    }
    return tokens;
}

int parse_threads(const char * value) {
    const long parsed = std::strtol(value, nullptr, 10);
    if (parsed <= 0 || parsed > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error("thread count must be a positive integer");
    }
    return static_cast<int>(parsed);
}

}

int main(int argc, char ** argv) {
    if (argc < 4 || argc > 5) {
        std::cerr << "usage: " << argv[0] << " MODEL TOKEN_U32LE OUTPUT_F32 [THREADS]\n";
        return 2;
    }

    try {
        const int threads = argc == 5 ? parse_threads(argv[4]) : 16;
        const auto tokens = read_tokens(argv[2]);

        llama_backend_init();
        auto model_params = llama_model_default_params();
        ggml_backend_dev_t no_devices[] = {nullptr};
        model_params.devices = no_devices;
        model_params.n_gpu_layers = 0;
        llama_model * model = llama_model_load_from_file(argv[1], model_params);
        if (model == nullptr) {
            throw std::runtime_error("llama.cpp could not load the model");
        }

        const llama_vocab * vocab = llama_model_get_vocab(model);
        const int32_t vocab_size = llama_vocab_n_tokens(vocab);
        for (const auto token : tokens) {
            if (token < 0 || token >= vocab_size) {
                llama_model_free(model);
                throw std::runtime_error("token ID exceeds the oracle vocabulary");
            }
        }

        auto context_params = llama_context_default_params();
        context_params.n_ctx = static_cast<uint32_t>(tokens.size());
        context_params.n_batch = static_cast<uint32_t>(tokens.size());
        context_params.n_ubatch = 512;
        context_params.n_threads = threads;
        context_params.n_threads_batch = threads;
        context_params.n_outputs_max = static_cast<uint32_t>(tokens.size() - 1);
        context_params.offload_kqv = false;
        llama_context * context = llama_init_from_model(model, context_params);
        if (context == nullptr) {
            llama_model_free(model);
            throw std::runtime_error("llama.cpp could not create the oracle context");
        }

        llama_batch batch = llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1);
        batch.n_tokens = static_cast<int32_t>(tokens.size());
        for (size_t index = 0; index < tokens.size(); ++index) {
            batch.token[index] = tokens[index];
            batch.pos[index] = static_cast<llama_pos>(index);
            batch.n_seq_id[index] = 1;
            batch.seq_id[index][0] = 0;
            batch.logits[index] = index + 1 < tokens.size() ? 1 : 0;
        }
        const int32_t decode_status = llama_decode(context, batch);
        if (decode_status != 0) {
            llama_batch_free(batch);
            llama_free(context);
            llama_model_free(model);
            throw std::runtime_error("llama.cpp decode failed with status " + std::to_string(decode_status));
        }

        std::ofstream output(argv[3], std::ios::binary | std::ios::trunc);
        if (!output) {
            llama_batch_free(batch);
            llama_free(context);
            llama_model_free(model);
            throw std::runtime_error(std::string("cannot open output file: ") + argv[3]);
        }
        const auto row_bytes = static_cast<std::streamsize>(vocab_size) * sizeof(float);
        for (size_t index = 0; index + 1 < tokens.size(); ++index) {
            const float * logits = llama_get_logits_ith(context, static_cast<int32_t>(index));
            if (logits == nullptr) {
                llama_batch_free(batch);
                llama_free(context);
                llama_model_free(model);
                throw std::runtime_error("llama.cpp did not retain a requested logit row");
            }
            output.write(reinterpret_cast<const char *>(logits), row_bytes);
        }
        if (!output) {
            llama_batch_free(batch);
            llama_free(context);
            llama_model_free(model);
            throw std::runtime_error("failed while writing oracle logits");
        }

        llama_batch_free(batch);
        llama_free(context);
        llama_model_free(model);
        llama_backend_free();
        std::cout << "tokens: " << tokens.size() << "\n"
                  << "rows: " << tokens.size() - 1 << "\n"
                  << "vocab: " << vocab_size << "\n"
                  << "logits: " << argv[3] << "\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
