#include "llama.h"

#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    DEFAULT_EVAL_WINDOW = 512,
};

static int read_window(const char *text, uint32_t *window) {
    char *end = NULL;
    errno = 0;
    const unsigned long value = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value < 2 || value > INT32_MAX) {
        fprintf(stderr, "window must be between 2 and INT32_MAX\n");
        return 0;
    }
    *window = (uint32_t) value;
    return 1;
}

static int prepare_token_file(FILE *file, const char *path, size_t *token_count) {
    if (fseek(file, 0, SEEK_END) != 0) {
        fprintf(stderr, "failed to seek token file %s\n", path);
        return 0;
    }
    const long bytes = ftell(file);
    if (bytes < 0 || bytes % 4 != 0 || fseek(file, 0, SEEK_SET) != 0) {
        fprintf(stderr, "token file size must be a multiple of four bytes\n");
        return 0;
    }
    *token_count = (size_t) bytes / 4;
    if (*token_count < 2 || *token_count > INT32_MAX) {
        fprintf(stderr, "token file must contain between 2 and INT32_MAX tokens\n");
        return 0;
    }
    return 1;
}

static int read_token(FILE *file, const char *path, size_t index, llama_token *token) {
    unsigned char encoded[4];
    if (fread(encoded, 1, sizeof(encoded), file) != sizeof(encoded)) {
        fprintf(stderr, "failed to read token file %s\n", path);
        return 0;
    }
    const uint32_t value = (uint32_t) encoded[0]
            | (uint32_t) encoded[1] << 8
            | (uint32_t) encoded[2] << 16
            | (uint32_t) encoded[3] << 24;
    if (value > INT32_MAX) {
        fprintf(stderr, "token %zu is outside the llama.cpp token range\n", index);
        return 0;
    }
    *token = (llama_token) value;
    return 1;
}

static int read_tokens(const char *path, llama_token **tokens, size_t *count) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "failed to open token file %s: %s\n", path, strerror(errno));
        return 0;
    }
    size_t token_count;
    if (!prepare_token_file(file, path, &token_count)) {
        fclose(file);
        return 0;
    }
    llama_token *values = malloc(token_count * sizeof(*values));
    if (values == NULL) {
        fprintf(stderr, "failed to allocate the token buffer\n");
        fclose(file);
        return 0;
    }
    for (size_t index = 0; index < token_count; ++index) {
        if (!read_token(file, path, index, &values[index])) {
            free(values);
            fclose(file);
            return 0;
        }
    }
    fclose(file);
    *tokens = values;
    *count = token_count;
    return 1;
}

static int dump_window(
        struct llama_context *context,
        struct llama_batch batch,
        const llama_token *tokens,
        size_t count,
        int32_t vocab_size,
        FILE *output) {
    llama_memory_clear(llama_get_memory(context), true);
    batch.n_tokens = (int32_t) count;
    for (size_t index = 0; index < count; ++index) {
        batch.token[index] = tokens[index];
        batch.pos[index] = (llama_pos) index;
        batch.n_seq_id[index] = 1;
        batch.seq_id[index][0] = 0;
        batch.logits[index] = index + 1 < count;
    }
    const int32_t decode_result = llama_decode(context, batch);
    if (decode_result != 0) {
        fprintf(stderr, "llama_decode failed with code %d\n", decode_result);
        return 0;
    }
    for (size_t index = 0; index + 1 < count; ++index) {
        const float *logits = llama_get_logits_ith(context, (int32_t) index);
        if (logits == NULL || fwrite(logits, sizeof(float), (size_t) vocab_size, output) != (size_t) vocab_size) {
            fprintf(stderr, "failed to write logits for window position %zu\n", index);
            return 0;
        }
    }
    return 1;
}

static void cleanup_oracle(
        struct llama_batch *batch,
        struct llama_context *context,
        struct llama_model *model) {
    if (batch != NULL) {
        llama_batch_free(*batch);
    }
    if (context != NULL) {
        llama_free(context);
    }
    if (model != NULL) {
        llama_model_free(model);
    }
    llama_backend_free();
}

static int decode_windows(
        struct llama_context *context,
        struct llama_batch batch,
        const llama_token *tokens,
        size_t token_count,
        int32_t vocab_size,
        uint32_t window,
        FILE *output,
        const char *output_path) {
    size_t window_count = 0;
    const size_t stride = (size_t) window - 1;
    int success = 1;
    for (size_t start = 0; start + 1 < token_count; start += stride) {
        const size_t remaining = token_count - start;
        const size_t count = remaining < window ? remaining : window;
        if (!dump_window(context, batch, tokens + start, count, vocab_size, output)) {
            success = 0;
            break;
        }
        ++window_count;
    }
    if (!success || fflush(output) != 0) {
        if (success) {
            fprintf(stderr, "failed to flush output file %s\n", output_path);
        }
        return 0;
    }
    fprintf(stderr, "wrote %zu positions across %zu windows with vocab size %d\n", token_count - 1, window_count, vocab_size);
    return 1;
}

static int run_oracle(
        const char *model_path,
        llama_token *tokens,
        size_t token_count,
        uint32_t window,
        const char *output_path,
        FILE *output) {
    llama_backend_init();
    struct llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = -1;
    struct llama_model *model = llama_model_load_from_file(model_path, model_params);
    if (model == NULL) {
        fprintf(stderr, "failed to load model %s\n", model_path);
        cleanup_oracle(NULL, NULL, NULL);
        return 0;
    }

    struct llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = window;
    context_params.n_batch = window;
    context_params.n_ubatch = window;
    context_params.n_outputs_max = window;
    context_params.no_perf = true;
    struct llama_context *context = llama_init_from_model(model, context_params);
    if (context == NULL) {
        fprintf(stderr, "failed to create the llama.cpp context\n");
        cleanup_oracle(NULL, NULL, model);
        return 0;
    }

    const int32_t vocab_size = llama_vocab_n_tokens(llama_model_get_vocab(model));
    struct llama_batch batch = llama_batch_init((int32_t) window, 0, 1);
    const int success = decode_windows(
        context,
        batch,
        tokens,
        token_count,
        vocab_size,
        window,
        output,
        output_path);
    cleanup_oracle(&batch, context, model);
    return success;
}

int main(int argc, char **argv) {
    if (argc != 4 && argc != 5) {
        fprintf(stderr, "usage: %s <model.gguf> <tokens.bin> <out.f32> [window]\n", argv[0]);
        return 2;
    }
    uint32_t window = DEFAULT_EVAL_WINDOW;
    if (argc == 5 && !read_window(argv[4], &window)) {
        return 2;
    }

    llama_token *tokens = NULL;
    size_t token_count = 0;
    if (!read_tokens(argv[2], &tokens, &token_count)) {
        return 1;
    }
    FILE *output = fopen(argv[3], "wb");
    if (output == NULL) {
        fprintf(stderr, "failed to open output file %s: %s\n", argv[3], strerror(errno));
        free(tokens);
        return 1;
    }

    const int success = run_oracle(argv[1], tokens, token_count, window, argv[3], output);
    fclose(output);
    free(tokens);
    return success ? 0 : 1;
}
