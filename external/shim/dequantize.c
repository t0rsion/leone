#include "ggml-quants.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct format {
    size_t block_elements;
    size_t block_bytes;
};

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
    if (strcmp(name, "f32") == 0) {
        *format = (struct format) { 1, 4 };
    } else if (strcmp(name, "f16") == 0) {
        *format = (struct format) { 1, 2 };
    } else if (strcmp(name, "q5_0") == 0) {
        *format = (struct format) { QK5_0, sizeof(block_q5_0) };
    } else if (strcmp(name, "q8_0") == 0) {
        *format = (struct format) { QK8_0, sizeof(block_q8_0) };
    } else if (strcmp(name, "q4_K") == 0) {
        *format = (struct format) { QK_K, sizeof(block_q4_K) };
    } else if (strcmp(name, "q5_K") == 0) {
        *format = (struct format) { QK_K, sizeof(block_q5_K) };
    } else if (strcmp(name, "q6_K") == 0) {
        *format = (struct format) { QK_K, sizeof(block_q6_K) };
    } else {
        return 0;
    }
    return 1;
}

static void dequantize(const char *name, const void *input, float *output, int64_t count) {
    if (strcmp(name, "f32") == 0) {
        memcpy(output, input, (size_t) count * sizeof(float));
    } else if (strcmp(name, "f16") == 0) {
        ggml_fp16_to_fp32_row(input, output, count);
    } else if (strcmp(name, "q5_0") == 0) {
        dequantize_row_q5_0(input, output, count);
    } else if (strcmp(name, "q8_0") == 0) {
        dequantize_row_q8_0(input, output, count);
    } else if (strcmp(name, "q4_K") == 0) {
        dequantize_row_q4_K(input, output, count);
    } else if (strcmp(name, "q5_K") == 0) {
        dequantize_row_q5_K(input, output, count);
    } else if (strcmp(name, "q6_K") == 0) {
        dequantize_row_q6_K(input, output, count);
    }
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <f32|f16|q5_0|q8_0|q4_K|q5_K|q6_K> <element-count>\n", argv[0]);
        return 2;
    }

    struct format format;
    int64_t count;
    if (!get_format(argv[1], &format)) {
        fprintf(stderr, "unsupported format: %s\n", argv[1]);
        return 2;
    }
    if (!parse_count(argv[2], &count)) {
        fprintf(stderr, "invalid element count: %s\n", argv[2]);
        return 2;
    }
    if ((uint64_t) count % format.block_elements != 0) {
        fprintf(stderr, "element count is not a whole number of blocks\n");
        return 2;
    }

    const size_t block_count = (size_t) count / format.block_elements;
    if (block_count > SIZE_MAX / format.block_bytes || (size_t) count > SIZE_MAX / sizeof(float)) {
        fprintf(stderr, "input size overflows the host size\n");
        return 2;
    }
    const size_t input_bytes = block_count * format.block_bytes;
    unsigned char *input = malloc(input_bytes);
    float *output = malloc((size_t) count * sizeof(float));
    if (input == NULL || output == NULL) {
        fprintf(stderr, "allocation failed\n");
        free(input);
        free(output);
        return 1;
    }
    if (fread(input, 1, input_bytes, stdin) != input_bytes || fgetc(stdin) != EOF) {
        fprintf(stderr, "stdin length does not match the requested row\n");
        free(input);
        free(output);
        return 1;
    }

    dequantize(argv[1], input, output, count);
    if (fwrite(output, sizeof(float), (size_t) count, stdout) != (size_t) count) {
        fprintf(stderr, "failed to write output\n");
        free(input);
        free(output);
        return 1;
    }
    free(input);
    free(output);
    return 0;
}
