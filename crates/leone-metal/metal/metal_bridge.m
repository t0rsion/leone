#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <CommonCrypto/CommonDigest.h>
#include <mach/host_info.h>
#include <mach/mach.h>
#include <mach/task_info.h>
#include <mach/vm_statistics.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/sysctl.h>

#define LEONE_METAL_TEXT_BYTES 128
#define LEONE_METAL_HASH_BYTES 65
#define LEONE_METAL_MAX_PENDING_DISPATCHES 64

typedef struct LeoneMetalContext {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> pipelines[28];
    id<MTLBuffer> dummy;
    id<MTLCommandBuffer> pending;
    size_t pending_dispatches;
    uint64_t max_buffer_length;
    char shader_source_hash[LEONE_METAL_HASH_BYTES];
} LeoneMetalContext;

typedef struct LeoneMetalBuffer {
    id<MTLBuffer> buffer;
} LeoneMetalBuffer;

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

#define LEONE_METAL_HOST_PROCESS_RESIDENT (1u << 0)
#define LEONE_METAL_HOST_PROCESS_VIRTUAL (1u << 1)
#define LEONE_METAL_HOST_SYSTEM_TOTAL (1u << 2)
#define LEONE_METAL_HOST_SYSTEM_PAGE_SIZE (1u << 3)
#define LEONE_METAL_HOST_SYSTEM_FREE_PAGES (1u << 4)
#define LEONE_METAL_HOST_SYSTEM_FILE_BACKED_PAGES (1u << 5)
#define LEONE_METAL_HOST_SYSTEM_SPECULATIVE_PAGES (1u << 6)

typedef struct LeoneMetalDeviceMetadata {
    uint64_t registry_id;
    char device_name[LEONE_METAL_TEXT_BYTES];
    char architecture_name[LEONE_METAL_TEXT_BYTES];
    char os_version[LEONE_METAL_TEXT_BYTES];
    char compiler_version[LEONE_METAL_TEXT_BYTES];
    char shader_source_hash[LEONE_METAL_HASH_BYTES];
    uint8_t fast_math_enabled;
} LeoneMetalDeviceMetadata;

static void clear_pipelines(LeoneMetalContext *context) {
    for (uint32_t op = 1; op <= 27; ++op) {
        context->pipelines[op] = nil;
    }
}

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

static int copy_region_valid(const LeoneMetalCopyRegion *region) {
    LeoneMetalBuffer *source = region->source;
    LeoneMetalBuffer *destination = region->destination;
    if (source == NULL || destination == NULL) {
        return 0;
    }
    if (region->source_offset > source->buffer.length
        || region->destination_offset > destination->buffer.length) {
        return 0;
    }
    return region->bytes <= source->buffer.length - region->source_offset
        && region->bytes <= destination->buffer.length - region->destination_offset;
}

static int copy_regions_valid(
    LeoneMetalContext *context,
    const LeoneMetalCopyRegion *regions,
    size_t region_count
) {
    if (context == NULL || regions == NULL || region_count == 0 || region_count > 65536) {
        return 0;
    }
    for (size_t index = 0; index < region_count; ++index) {
        if (!copy_region_valid(&regions[index])) {
            return 0;
        }
    }
    return 1;
}

static int copy_region_aligned4(const LeoneMetalCopyRegion *region) {
    return region->source_offset % 4 == 0
        && region->destination_offset % 4 == 0
        && region->bytes % 4 == 0;
}

static int dispatch_arguments_valid(
    LeoneMetalContext *context,
    const LeoneMetalDispatchArgs *args,
    void *const *opaque_buffers
) {
    if (context == NULL || args == NULL || opaque_buffers == NULL) {
        return 0;
    }
    if (args->threads == 0 || args->op == 0 || args->op > 27) {
        return 0;
    }
    return 1;
}

static int dispatch_buffer_count_valid(
    const LeoneMetalDispatchArgs *args,
    size_t buffer_count
) {
    size_t expected_buffers = args->op == 21 || args->op == 22 ? 13 : 12;
    return buffer_count == expected_buffers;
}

static int dispatch_valid(
    LeoneMetalContext *context,
    const LeoneMetalDispatchArgs *args,
    void *const *opaque_buffers,
    size_t buffer_count
) {
    if (!dispatch_arguments_valid(context, args, opaque_buffers)) {
        return 0;
    }
    if (!dispatch_buffer_count_valid(args, buffer_count)) {
        return 0;
    }
    return context->pipelines[args->op] != nil;
}

static void set_error(char *error, size_t length, NSString *message) {
    if (error == NULL || length == 0) {
        return;
    }
    const char *text = message.UTF8String;
    if (text == NULL) {
        text = "Metal operation failed";
    }
    snprintf(error, length, "%s", text);
}

static void set_text(char *destination, size_t length, NSString *value) {
    if (destination == NULL || length == 0) {
        return;
    }
    const char *text = value.UTF8String;
    if (text == NULL) {
        destination[0] = '\0';
        return;
    }
    snprintf(destination, length, "%s", text);
}

static void set_shader_hash(
    char *destination,
    size_t length,
    const char *source,
    size_t source_length
) {
    if (destination == NULL || length < LEONE_METAL_HASH_BYTES || source == NULL
        || source_length > UINT32_MAX) {
        return;
    }
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(source, (CC_LONG)source_length, digest);
    for (size_t index = 0; index < CC_SHA256_DIGEST_LENGTH; ++index) {
        snprintf(destination + index * 2, length - index * 2, "%02x", digest[index]);
    }
}

static void fill_device_metadata(
    id<MTLDevice> device,
    const char *source,
    size_t source_length,
    LeoneMetalDeviceMetadata *metadata
) {
    if (metadata == NULL) {
        return;
    }
    memset(metadata, 0, sizeof(*metadata));
    metadata->registry_id = (uint64_t)device.registryID;
    set_text(metadata->device_name, sizeof(metadata->device_name), device.name);
    if (device.architecture != nil) {
        set_text(metadata->architecture_name,
                 sizeof(metadata->architecture_name),
                 device.architecture.name);
    }
    set_text(metadata->os_version,
             sizeof(metadata->os_version),
             NSProcessInfo.processInfo.operatingSystemVersionString);
    set_shader_hash(metadata->shader_source_hash,
                    sizeof(metadata->shader_source_hash),
                    source,
                    source_length);
    metadata->fast_math_enabled = 0;
}

static int finish_command(id<MTLCommandBuffer> command, char *error, size_t error_length) {
    if (command == nil) {
        set_error(error, error_length, @"Metal command buffer creation failed");
        return -1;
    }
    [command waitUntilCompleted];
    if (command.status == MTLCommandBufferStatusCompleted) {
        return 0;
    }
    set_error(error, error_length, command.error.localizedDescription);
    return -1;
}

static int flush_commands(LeoneMetalContext *context, char *error, size_t error_length) {
    if (context == NULL) {
        set_error(error, error_length, @"invalid Metal context");
        return -1;
    }
    if (context->pending == nil) {
        return 0;
    }
    id<MTLCommandBuffer> command = context->pending;
    [command commit];
    int result = finish_command(command, error, error_length);
    context->pending = nil;
    context->pending_dispatches = 0;
    return result;
}

static int reserve_dispatches(
    LeoneMetalContext *context,
    size_t count,
    char *error,
    size_t error_length
) {
    if (context->pending_dispatches > LEONE_METAL_MAX_PENDING_DISPATCHES - count
        && flush_commands(context, error, error_length) != 0) {
        return -1;
    }
    if (context->pending == nil) {
        context->pending = [context->queue commandBuffer];
    }
    if (context->pending == nil) {
        set_error(error, error_length, @"Metal command buffer creation failed");
        return -1;
    }
    return 0;
}

static id<MTLComputePipelineState> compile_pipeline(
    id<MTLDevice> device,
    id<MTLLibrary> library,
    const char *name,
    char *error,
    size_t error_length
) {
    NSString *function_name = [[NSString alloc] initWithUTF8String:name];
    id<MTLFunction> function = [library newFunctionWithName:function_name];
    if (function == nil) {
        set_error(error, error_length, [NSString stringWithFormat:@"Metal shader function %s is missing", name]);
        return nil;
    }
    NSError *pipeline_error = nil;
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&pipeline_error];
    if (pipeline == nil) {
        set_error(error, error_length, pipeline_error.localizedDescription);
    }
    return pipeline;
}

static int fixed_dispatch_width_supported(LeoneMetalContext *context) {
    for (uint32_t op = 1; op <= 27; ++op) {
        if (context->pipelines[op].maxTotalThreadsPerThreadgroup < 256) {
            return 0;
        }
    }
    return 1;
}

static uint8_t simdgroup_matrix_supported(LeoneMetalContext *context) {
    // Apple7 provides simdgroup matrices. The kernel requires eight 32-lane groups.
    return [context->device supportsFamily:MTLGPUFamilyApple7]
        && context->pipelines[3].threadExecutionWidth == 32
        && context->pipelines[3].maxTotalThreadsPerThreadgroup >= 256;
}

static void bind_dispatch_buffers(
    id<MTLComputeCommandEncoder> encoder,
    LeoneMetalContext *context,
    void *const *opaque_buffers,
    size_t buffer_count
) {
    for (NSUInteger index = 0; index < buffer_count; ++index) {
        LeoneMetalBuffer *buffer = opaque_buffers[index];
        [encoder setBuffer:buffer == NULL ? context->dummy : buffer->buffer offset:0 atIndex:index];
    }
}

void *leone_metal_context_new(
    const char *shader_source,
    size_t shader_length,
    LeoneMetalDeviceInfo *info,
    char *error,
    size_t error_length
) {
    @autoreleasepool {
        if (shader_source == NULL || shader_length == 0) {
            set_error(error, error_length, @"Metal shader source is empty");
            return NULL;
        }
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            set_error(error, error_length, @"Metal is unavailable on this device");
            return NULL;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) {
            set_error(error, error_length, @"Metal command queue creation failed");
            return NULL;
        }
        NSString *source = [[NSString alloc] initWithBytes:shader_source length:shader_length encoding:NSUTF8StringEncoding];
        if (source == nil) {
            set_error(error, error_length, @"Metal shader source is invalid UTF-8");
            return NULL;
        }
        NSError *compile_error = nil;
        MTLCompileOptions *compile_options = [MTLCompileOptions new];
        compile_options.fastMathEnabled = NO;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:compile_options error:&compile_error];
        if (library == nil) {
            set_error(error, error_length, compile_error.localizedDescription);
            return NULL;
        }
        const char *pipeline_names[] = {
            "",
            "leone_gemv",
            "leone_gemv_residual",
            "leone_prefill_gemm",
            "leone_rms_norm",
            "leone_rms_norm_rope",
            "leone_rms_norm_residual",
            "leone_rms_norm_residual_store",
            "leone_rope",
            "leone_swiglu",
            "leone_residual_add",
            "leone_kv_append",
            "leone_kv_append_chunk",
            "leone_attention_decode",
            "leone_attention_prefill",
            "leone_embed_gather",
            "leone_embed_gather_batch",
            "leone_copy_row",
            "leone_write_row",
            "leone_argmax",
            "leone_increment",
            "leone_attention_decode_spans",
            "leone_attention_prefill_spans",
            "leone_kv_copy_span",
            "leone_attention_batch_fixed_tile",
            "leone_attention_batch_shared",
            "leone_attention_batch_fixed_shared",
            "leone_prefill_rms_norm_rope",
        };
        LeoneMetalContext *context = calloc(1, sizeof(*context));
        if (context == NULL) {
            set_error(error, error_length, @"Metal context allocation failed");
            return NULL;
        }
        for (uint32_t op = 1; op <= 27; ++op) {
            context->pipelines[op] = compile_pipeline(device, library, pipeline_names[op], error, error_length);
            if (context->pipelines[op] == nil) {
                clear_pipelines(context);
                free(context);
                return NULL;
            }
        }
        if (!fixed_dispatch_width_supported(context)) {
            clear_pipelines(context);
            free(context);
            set_error(error, error_length, @"Metal device cannot run 256 lane dispatches");
            return NULL;
        }
        id<MTLBuffer> dummy = [device newBufferWithLength:4 options:MTLResourceStorageModeShared];
        if (dummy == nil) {
            clear_pipelines(context);
            free(context);
            set_error(error, error_length, @"Metal dummy buffer allocation failed");
            return NULL;
        }
        context->device = device;
        context->queue = queue;
        context->dummy = dummy;
        context->max_buffer_length = (uint64_t)device.maxBufferLength;
        LeoneMetalDeviceMetadata metadata;
        fill_device_metadata(device, shader_source, shader_length, &metadata);
        memcpy(context->shader_source_hash,
               metadata.shader_source_hash,
               sizeof(context->shader_source_hash));
        if (info != NULL) {
            info->max_buffer_length = (uint64_t)device.maxBufferLength;
            info->recommended_working_set = (uint64_t)device.recommendedMaxWorkingSetSize;
            info->current_allocated = (uint64_t)device.currentAllocatedSize;
            info->supports_simdgroup_matrix = simdgroup_matrix_supported(context);
        }
        return context;
    }
}

void leone_metal_context_free(void *opaque_context) {
    @autoreleasepool {
        LeoneMetalContext *context = opaque_context;
        if (context != NULL) {
            char error[1024];
            flush_commands(context, error, sizeof(error));
            clear_pipelines(context);
            context->queue = nil;
            context->device = nil;
            context->dummy = nil;
            free(context);
        }
    }
}

int leone_metal_device_info(
    void *opaque_context,
    LeoneMetalDeviceInfo *info,
    char *error,
    size_t error_length
) {
    @autoreleasepool {
        LeoneMetalContext *context = opaque_context;
        if (context == NULL || info == NULL) {
            set_error(error, error_length, @"invalid Metal device handle");
            return -1;
        }
        info->max_buffer_length = (uint64_t)context->device.maxBufferLength;
        info->recommended_working_set = (uint64_t)context->device.recommendedMaxWorkingSetSize;
        info->current_allocated = (uint64_t)context->device.currentAllocatedSize;
        info->supports_simdgroup_matrix = simdgroup_matrix_supported(context);
        return 0;
    }
}

int leone_metal_device_metadata(
    void *opaque_context,
    LeoneMetalDeviceMetadata *metadata,
    char *error,
    size_t error_length
) {
    @autoreleasepool {
        LeoneMetalContext *context = opaque_context;
        if (context == NULL || metadata == NULL) {
            set_error(error, error_length, @"invalid Metal device handle");
            return -1;
        }
        fill_device_metadata(context->device, NULL, 0, metadata);
        memcpy(metadata->shader_source_hash,
               context->shader_source_hash,
               sizeof(metadata->shader_source_hash));
        return 0;
    }
}

int leone_metal_alloc(
    void *opaque_context,
    size_t bytes,
    void **opaque_buffer,
    char *error,
    size_t error_length
) {
    @autoreleasepool {
        LeoneMetalContext *context = opaque_context;
        if (context == NULL || opaque_buffer == NULL) {
            set_error(error, error_length, @"invalid Metal allocation handle");
            return -1;
        }
        if (bytes == 0 || (uint64_t)bytes > context->max_buffer_length) {
            set_error(error, error_length, @"allocation exceeds MTLDevice maxBufferLength");
            return -1;
        }
        id<MTLBuffer> buffer = [context->device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (buffer == nil) {
            set_error(error, error_length, @"MTLBuffer allocation failed");
            return -1;
        }
        LeoneMetalBuffer *owned = calloc(1, sizeof(*owned));
        if (owned == NULL) {
            set_error(error, error_length, @"Metal buffer handle allocation failed");
            return -1;
        }
        owned->buffer = buffer;
        *opaque_buffer = owned;
        return 0;
    }
}

void leone_metal_buffer_free(void *opaque_buffer) {
    @autoreleasepool {
        LeoneMetalBuffer *buffer = opaque_buffer;
        if (buffer != NULL) {
            buffer->buffer = nil;
            free(buffer);
        }
    }
}

int leone_metal_buffer_write(
    void *opaque_buffer,
    const void *source,
    size_t bytes,
    char *error,
    size_t error_length
) {
    @autoreleasepool {
        LeoneMetalBuffer *buffer = opaque_buffer;
        if (buffer == NULL || source == NULL || bytes > buffer->buffer.length) {
            set_error(error, error_length, @"invalid Metal buffer write");
            return -1;
        }
        memcpy(buffer->buffer.contents, source, bytes);
        return 0;
    }
}

int leone_metal_buffer_read(
    void *opaque_buffer,
    void *destination,
    size_t bytes,
    char *error,
    size_t error_length
) {
    @autoreleasepool {
        LeoneMetalBuffer *buffer = opaque_buffer;
        if (buffer == NULL || destination == NULL || bytes > buffer->buffer.length) {
            set_error(error, error_length, @"invalid Metal buffer read");
            return -1;
        }
        memcpy(destination, buffer->buffer.contents, bytes);
        return 0;
    }
}

int leone_metal_buffer_copy(
    void *opaque_context,
    void *opaque_source,
    void *opaque_destination,
    size_t bytes,
    char *error,
    size_t error_length
) {
    @autoreleasepool {
        LeoneMetalContext *context = opaque_context;
        LeoneMetalBuffer *source = opaque_source;
        LeoneMetalBuffer *destination = opaque_destination;
        if (context == NULL || source == NULL || destination == NULL
            || bytes > source->buffer.length || bytes > destination->buffer.length) {
            set_error(error, error_length, @"invalid Metal buffer copy");
            return -1;
        }
        if (bytes % 4 != 0) {
            // Backend allocations use shared storage, so host copying handles
            // the 2-byte F16 sizes that Metal blit commands reject.
            memcpy(destination->buffer.contents, source->buffer.contents, bytes);
            return 0;
        }
        id<MTLCommandBuffer> command = [context->queue commandBuffer];
        if (command == nil) {
            set_error(error, error_length, @"Metal command buffer creation failed");
            return -1;
        }
        id<MTLBlitCommandEncoder> encoder = [command blitCommandEncoder];
        if (encoder == nil) {
            set_error(error, error_length, @"Metal blit encoder creation failed");
            return -1;
        }
        [encoder copyFromBuffer:source->buffer sourceOffset:0 toBuffer:destination->buffer destinationOffset:0 size:bytes];
        [encoder endEncoding];
        [command commit];
        return finish_command(command, error, error_length);
    }
}

int leone_metal_buffer_copy_regions(
    void *opaque_context,
    const LeoneMetalCopyRegion *regions,
    size_t region_count,
    char *error,
    size_t error_length
) {
    @autoreleasepool {
        LeoneMetalContext *context = opaque_context;
        if (!copy_regions_valid(context, regions, region_count)) {
            set_error(error, error_length, @"invalid Metal copy regions");
            return -1;
        }
        for (size_t index = 0; index < region_count; ++index) {
            if (!copy_region_aligned4(&regions[index])) {
                // Backend allocations use shared storage, so host copying handles
                // unaligned F16 span offsets and sizes.
                for (size_t fallback = 0; fallback < region_count; ++fallback) {
                    LeoneMetalCopyRegion region = regions[fallback];
                    LeoneMetalBuffer *source = region.source;
                    LeoneMetalBuffer *destination = region.destination;
                    memcpy((char *)destination->buffer.contents + region.destination_offset,
                           (const char *)source->buffer.contents + region.source_offset,
                           region.bytes);
                }
                return 0;
            }
        }
        id<MTLCommandBuffer> command = [context->queue commandBuffer];
        if (command == nil) {
            set_error(error, error_length, @"Metal command buffer creation failed");
            return -1;
        }
        id<MTLBlitCommandEncoder> encoder = [command blitCommandEncoder];
        if (encoder == nil) {
            set_error(error, error_length, @"Metal blit encoder creation failed");
            return -1;
        }
        for (size_t index = 0; index < region_count; ++index) {
            LeoneMetalCopyRegion region = regions[index];
            LeoneMetalBuffer *source = region.source;
            LeoneMetalBuffer *destination = region.destination;
            [encoder copyFromBuffer:source->buffer
                        sourceOffset:region.source_offset
                            toBuffer:destination->buffer
                   destinationOffset:region.destination_offset
                                size:region.bytes];
        }
        [encoder endEncoding];
        [command commit];
        return finish_command(command, error, error_length);
    }
}

static int encode_dispatch(
    LeoneMetalContext *context,
    id<MTLCommandBuffer> command,
    const LeoneMetalDispatchArgs *args,
    void *const *opaque_buffers,
    size_t buffer_count,
    char *error,
    size_t error_length
) {
    if (!dispatch_valid(context, args, opaque_buffers, buffer_count)) {
        set_error(error, error_length, @"invalid Metal dispatch arguments");
        return -1;
    }
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (encoder == nil) {
        set_error(error, error_length, @"Metal compute encoder creation failed");
        return -1;
    }
    id<MTLComputePipelineState> pipeline = context->pipelines[args->op];
    [encoder setComputePipelineState:pipeline];
    bind_dispatch_buffers(encoder, context, opaque_buffers, buffer_count);
    NSUInteger args_index = args->op == 21 || args->op == 22 ? 13 : 12;
    [encoder setBytes:args length:sizeof(*args) atIndex:args_index];
    if (args->threadgroup_bytes > 0) {
        [encoder setThreadgroupMemoryLength:args->threadgroup_bytes atIndex:0];
    }
    NSUInteger width = MIN((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup, 256U);
    MTLSize grid = MTLSizeMake(args->threads, 1, 1);
    MTLSize group = MTLSizeMake(width, 1, 1);
    [encoder dispatchThreads:grid threadsPerThreadgroup:group];
    [encoder endEncoding];
    return 0;
}

int leone_metal_dispatch(
    void *opaque_context,
    const LeoneMetalDispatchArgs *args,
    void *const *opaque_buffers,
    size_t buffer_count,
    char *error,
    size_t error_length
) {
    @autoreleasepool {
        LeoneMetalContext *context = opaque_context;
        if (context == NULL || args == NULL || opaque_buffers == NULL) {
            set_error(error, error_length, @"invalid Metal dispatch arguments");
            return -1;
        }
        if (!dispatch_valid(context, args, opaque_buffers, buffer_count)) {
            set_error(error, error_length, @"invalid Metal dispatch arguments");
            return -1;
        }
        if (reserve_dispatches(context, 1, error, error_length) != 0) {
            return -1;
        }
        if (encode_dispatch(
                context,
                context->pending,
                args,
                opaque_buffers,
                buffer_count,
                error,
                error_length
        ) != 0) {
            return -1;
        }
        context->pending_dispatches += 1;
        return 0;
    }
}

static int validate_dispatch_sequence(
    LeoneMetalContext *context,
    const LeoneMetalDispatchArgs *args,
    size_t command_count,
    void *const *opaque_buffers,
    size_t buffer_count
) {
    if (context == NULL || args == NULL || opaque_buffers == NULL
        || command_count == 0 || command_count > LEONE_METAL_MAX_PENDING_DISPATCHES
        || buffer_count != 12) {
        return 0;
    }
    for (size_t index = 0; index < command_count; ++index) {
        if (!dispatch_valid(context, &args[index], opaque_buffers + index * buffer_count, buffer_count)) {
            return 0;
        }
    }
    return 1;
}

int leone_metal_dispatch_sequence(
    void *opaque_context,
    const LeoneMetalDispatchArgs *args,
    size_t command_count,
    void *const *opaque_buffers,
    size_t buffer_count,
    char *error,
    size_t error_length
) {
    @autoreleasepool {
        LeoneMetalContext *context = opaque_context;
        if (!validate_dispatch_sequence(context, args, command_count, opaque_buffers, buffer_count)) {
            set_error(error, error_length, @"invalid Metal dispatch sequence arguments");
            return -1;
        }
        if (reserve_dispatches(context, command_count, error, error_length) != 0) {
            return -1;
        }
        for (size_t index = 0; index < command_count; ++index) {
            if (encode_dispatch(
                    context,
                    context->pending,
                    &args[index],
                    opaque_buffers + index * buffer_count,
                    buffer_count,
                    error,
                    error_length
                ) != 0) {
                return -1;
            }
            context->pending_dispatches += 1;
        }
        return 0;
    }
}

int leone_metal_sync(void *opaque_context, char *error, size_t error_length) {
    @autoreleasepool {
        return flush_commands(opaque_context, error, error_length);
    }
}

static void leone_metal_fill_process_memory(LeoneMetalHostMemoryInfo *info) {
    mach_task_basic_info_data_t task = {0};
    mach_msg_type_number_t task_count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(
            mach_task_self(),
            MACH_TASK_BASIC_INFO,
            (task_info_t)&task,
            &task_count
        ) == KERN_SUCCESS
        && task_count >= MACH_TASK_BASIC_INFO_COUNT) {
        info->process_resident_bytes = (uint64_t)task.resident_size;
        info->process_virtual_bytes = (uint64_t)task.virtual_size;
        info->available |= LEONE_METAL_HOST_PROCESS_RESIDENT;
        info->available |= LEONE_METAL_HOST_PROCESS_VIRTUAL;
    }
}

static void leone_metal_fill_total_memory(LeoneMetalHostMemoryInfo *info) {
    uint64_t total_bytes = 0;
    size_t total_length = sizeof(total_bytes);
    if (sysctlbyname("hw.memsize", &total_bytes, &total_length, NULL, 0) == 0
        && total_length == sizeof(total_bytes)
        && total_bytes > 0) {
        info->system_total_bytes = total_bytes;
        info->available |= LEONE_METAL_HOST_SYSTEM_TOTAL;
    }
}

static void leone_metal_fill_page_memory(
    LeoneMetalHostMemoryInfo *info,
    mach_port_t host
) {
    vm_size_t page_size = 0;
    if (host_page_size(host, &page_size) == KERN_SUCCESS
        && page_size > 0) {
        info->system_page_size = (uint64_t)page_size;
        info->available |= LEONE_METAL_HOST_SYSTEM_PAGE_SIZE;
    }
    vm_statistics64_data_t statistics = {0};
    /* Rev1 includes external and speculative page counts without later fields. */
    mach_msg_type_number_t statistics_count = HOST_VM_INFO64_REV1_COUNT;
    if (host_statistics64(
            host,
            HOST_VM_INFO64,
            (host_info64_t)&statistics,
            &statistics_count
        ) == KERN_SUCCESS
        && statistics_count >= HOST_VM_INFO64_REV1_COUNT) {
        info->system_free_pages = (uint64_t)statistics.free_count;
        info->system_file_backed_pages = (uint64_t)statistics.external_page_count;
        info->system_speculative_pages = (uint64_t)statistics.speculative_count;
        info->available |= LEONE_METAL_HOST_SYSTEM_FREE_PAGES;
        info->available |= LEONE_METAL_HOST_SYSTEM_FILE_BACKED_PAGES;
        info->available |= LEONE_METAL_HOST_SYSTEM_SPECULATIVE_PAGES;
    }
}

int leone_metal_host_memory_snapshot(
    LeoneMetalHostMemoryInfo *info,
    char *error,
    size_t error_length
) {
    if (info == NULL) {
        set_error(error, error_length, @"invalid host memory snapshot output");
        return -1;
    }
    memset(info, 0, sizeof(*info));
    leone_metal_fill_process_memory(info);
    leone_metal_fill_total_memory(info);
    mach_port_t host = mach_host_self();
    if (host != MACH_PORT_NULL) {
        leone_metal_fill_page_memory(info, host);
        mach_port_deallocate(mach_task_self(), host);
    }
    return 0;
}
