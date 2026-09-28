#include <stddef.h>
#include <stdint.h>

#define LEONE_METAL_TEXT_BYTES 128
#define LEONE_METAL_HASH_BYTES 65

typedef struct LeoneMetalDeviceInfo {
    uint64_t max_buffer_length;
    uint64_t recommended_working_set;
    uint64_t current_allocated;
    uint8_t supports_simdgroup_matrix;
} LeoneMetalDeviceInfo;

typedef struct LeoneMetalHostMemoryInfo {
    uint64_t process_resident_bytes;
    uint64_t process_virtual_bytes;
    uint64_t system_total_bytes;
    uint64_t system_page_size;
    uint64_t system_free_pages;
    uint64_t system_file_backed_pages;
    uint64_t system_speculative_pages;
    uint8_t available;
} LeoneMetalHostMemoryInfo;

typedef struct LeoneMetalDeviceMetadata {
    uint64_t registry_id;
    char device_name[LEONE_METAL_TEXT_BYTES];
    char architecture_name[LEONE_METAL_TEXT_BYTES];
    char os_version[LEONE_METAL_TEXT_BYTES];
    char compiler_version[LEONE_METAL_TEXT_BYTES];
    char shader_source_hash[LEONE_METAL_HASH_BYTES];
    uint8_t fast_math_enabled;
} LeoneMetalDeviceMetadata;

typedef struct LeoneMetalDispatchArgs {
    uint32_t op;
    uint32_t format;
    uint32_t rows;
    uint32_t columns;
    uint32_t tokens;
    uint32_t position;
    uint32_t start_position;
    uint32_t max_context;
    uint32_t gather_stride;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t head_dim;
    uint32_t table_rows;
    uint32_t row;
    uint32_t pairing;
    float epsilon;
    float theta;
    uint32_t threads;
    uint32_t batch_rows;
    uint32_t tile_tokens;
    uint32_t tile_count;
    uint32_t group_count;
    uint32_t threadgroup_bytes;
    uint32_t prefill_tile_rows;
    uint32_t prefill_k_tile;
} LeoneMetalDispatchArgs;

typedef struct LeoneMetalCopyRegion {
    void *source;
    void *destination;
    size_t source_offset;
    size_t destination_offset;
    size_t bytes;
} LeoneMetalCopyRegion;

static void unsupported(char *error, size_t length) {
    const char *message = "Metal backend requires macOS";
    if (error == NULL || length == 0) {
        return;
    }
    size_t index = 0;
    while (message[index] != '\0' && index + 1 < length) {
        error[index] = message[index];
        index += 1;
    }
    error[index] = '\0';
}

void *leone_metal_context_new(const char *source, size_t source_length,
                              LeoneMetalDeviceInfo *info, char *error,
                              size_t error_length) {
    (void)source;
    (void)source_length;
    (void)info;
    unsupported(error, error_length);
    return NULL;
}

void leone_metal_context_free(void *context) { (void)context; }

int leone_metal_device_info(void *context, LeoneMetalDeviceInfo *info,
                            char *error, size_t error_length) {
    (void)context;
    (void)info;
    unsupported(error, error_length);
    return -1;
}

int leone_metal_device_metadata(void *context, LeoneMetalDeviceMetadata *metadata,
                                char *error, size_t error_length) {
    (void)context;
    (void)metadata;
    unsupported(error, error_length);
    return -1;
}

int leone_metal_host_memory_snapshot(LeoneMetalHostMemoryInfo *info,
                                     char *error, size_t error_length) {
    (void)info;
    unsupported(error, error_length);
    return -1;
}

int leone_metal_alloc(void *context, size_t bytes, void **buffer, char *error,
                      size_t error_length) {
    (void)context;
    (void)bytes;
    (void)buffer;
    unsupported(error, error_length);
    return -1;
}

void leone_metal_buffer_free(void *buffer) { (void)buffer; }

int leone_metal_buffer_write(void *buffer, const void *source, size_t bytes,
                             char *error, size_t error_length) {
    (void)buffer;
    (void)source;
    (void)bytes;
    unsupported(error, error_length);
    return -1;
}

int leone_metal_buffer_read(void *buffer, void *destination, size_t bytes,
                            char *error, size_t error_length) {
    (void)buffer;
    (void)destination;
    (void)bytes;
    unsupported(error, error_length);
    return -1;
}

int leone_metal_buffer_copy(void *context, void *source, void *destination,
                            size_t bytes, char *error, size_t error_length) {
    (void)context;
    (void)source;
    (void)destination;
    (void)bytes;
    unsupported(error, error_length);
    return -1;
}

int leone_metal_buffer_copy_regions(void *context, const LeoneMetalCopyRegion *regions,
                                    size_t region_count, char *error,
                                    size_t error_length) {
    (void)context;
    (void)regions;
    (void)region_count;
    unsupported(error, error_length);
    return -1;
}

int leone_metal_dispatch(void *context, const LeoneMetalDispatchArgs *args,
                         void *const *buffers, size_t buffer_count, char *error,
                         size_t error_length) {
    (void)context;
    (void)args;
    (void)buffers;
    (void)buffer_count;
    unsupported(error, error_length);
    return -1;
}

int leone_metal_dispatch_sequence(void *context,
                                  const LeoneMetalDispatchArgs *args,
                                  size_t command_count,
                                  void *const *buffers,
                                  size_t buffer_count,
                                  char *error,
                                  size_t error_length) {
    (void)context;
    (void)args;
    (void)command_count;
    (void)buffers;
    (void)buffer_count;
    unsupported(error, error_length);
    return -1;
}

int leone_metal_sync(void *context, char *error, size_t error_length) {
    (void)context;
    unsupported(error, error_length);
    return -1;
}
