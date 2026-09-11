#include "ggml.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct format {
    size_t block_elements;
    size_t block_bytes;
    ggml_to_float_t to_float;
};

static const enum ggml_type formats[] = {
    GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0,
    GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,
};

static void copy_f32(const void *input, float *output, int64_t count) {
    memcpy(output, input, (size_t) count * sizeof(float));
}

static int parse_count(const char *text, int64_t *value) {
    char *end = NULL;
    errno = 0;
    const long long parsed = strtoll(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed <= 0) {
        return 0;
    }
    *value = (int64_t) parsed;
    return 1;
}

static int get_format(const char *name, struct format *format) {
    for (size_t index = 0; index < sizeof(formats) / sizeof(formats[0]); ++index) {
        const struct ggml_type_traits *traits = ggml_get_type_traits(formats[index]);
        if (strcmp(name, traits->type_name) == 0) {
            *format = (struct format) {
                (size_t) traits->blck_size,
                traits->type_size,
                formats[index] == GGML_TYPE_F32 ? copy_f32 : traits->to_float,
            };
            return format->to_float != NULL;
        }
    }
    return 0;
}

static int validate_size(const struct format *format, int64_t count, size_t *input_bytes) {
    if ((uint64_t) count % format->block_elements != 0) {
        fprintf(stderr, "element count is not a whole number of blocks\n");
        return 0;
    }

    const size_t block_count = (size_t) count / format->block_elements;
    if (block_count > SIZE_MAX / format->block_bytes || (size_t) count > SIZE_MAX / sizeof(float)) {
        fprintf(stderr, "input size overflows the host size\n");
        return 0;
    }
    *input_bytes = block_count * format->block_bytes;
    return 1;
}

static int process_row(const struct format *format, size_t input_bytes, int64_t count) {
    unsigned char *input = malloc(input_bytes);
    float *output = malloc((size_t) count * sizeof(float));
    if (input == NULL || output == NULL) {
        fprintf(stderr, "allocation failed\n");
        free(input);
        free(output);
        return 0;
    }
    if (fread(input, 1, input_bytes, stdin) != input_bytes || fgetc(stdin) != EOF) {
        fprintf(stderr, "stdin length does not match the requested row\n");
        free(input);
        free(output);
        return 0;
    }
    format->to_float(input, output, count);
    if (fwrite(output, sizeof(float), (size_t) count, stdout) != (size_t) count) {
        fprintf(stderr, "failed to write output\n");
        free(input);
        free(output);
        return 0;
    }
    free(input);
    free(output);
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <f32|f16|q5_0|q8_0|q4_K|q5_K|q6_K> <element-count>\n", argv[0]);
        return 2;
    }
    struct format format;
    int64_t count;
    size_t input_bytes;
    if (!get_format(argv[1], &format)) {
        fprintf(stderr, "unsupported format: %s\n", argv[1]);
        return 2;
    }
    if (!parse_count(argv[2], &count)) {
        fprintf(stderr, "invalid element count: %s\n", argv[2]);
        return 2;
    }
    if (!validate_size(&format, count, &input_bytes)) {
        return 2;
    }

    return process_row(&format, input_bytes, count) ? 0 : 1;
}
