use std::cell::RefCell;
use std::ffi::{c_char, c_void, CStr};
use std::mem::size_of;
use std::rc::Rc;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct DeviceInfo {
    pub max_buffer_length: u64,
    pub recommended_working_set: u64,
    pub current_allocated: u64,
    pub supports_simdgroup_matrix: u8,
}

pub(crate) const HOST_PROCESS_RESIDENT: u8 = 1 << 0;
pub(crate) const HOST_PROCESS_VIRTUAL: u8 = 1 << 1;
pub(crate) const HOST_SYSTEM_TOTAL: u8 = 1 << 2;
pub(crate) const HOST_SYSTEM_PAGE_SIZE: u8 = 1 << 3;
pub(crate) const HOST_SYSTEM_FREE_PAGES: u8 = 1 << 4;
pub(crate) const HOST_SYSTEM_FILE_BACKED_PAGES: u8 = 1 << 5;
pub(crate) const HOST_SYSTEM_SPECULATIVE_PAGES: u8 = 1 << 6;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct HostMemoryInfo {
    pub process_resident_bytes: u64,
    pub process_virtual_bytes: u64,
    pub system_total_bytes: u64,
    pub system_page_size: u64,
    pub system_free_pages: u64,
    pub system_file_backed_pages: u64,
    pub system_speculative_pages: u64,
    pub available: u8,
}

impl HostMemoryInfo {
    pub(crate) fn has(self, field: u8) -> bool {
        self.available & field != 0
    }
}

pub(crate) const DEVICE_TEXT_BYTES: usize = 128;
pub(crate) const DEVICE_HASH_BYTES: usize = 65;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct DeviceMetadata {
    pub registry_id: u64,
    pub device_name: [u8; DEVICE_TEXT_BYTES],
    pub architecture_name: [u8; DEVICE_TEXT_BYTES],
    pub os_version: [u8; DEVICE_TEXT_BYTES],
    pub compiler_version: [u8; DEVICE_TEXT_BYTES],
    pub shader_source_hash: [u8; DEVICE_HASH_BYTES],
    pub fast_math_enabled: u8,
}

impl Default for DeviceMetadata {
    fn default() -> Self {
        Self {
            registry_id: 0,
            device_name: [0; DEVICE_TEXT_BYTES],
            architecture_name: [0; DEVICE_TEXT_BYTES],
            os_version: [0; DEVICE_TEXT_BYTES],
            compiler_version: [0; DEVICE_TEXT_BYTES],
            shader_source_hash: [0; DEVICE_HASH_BYTES],
            fast_math_enabled: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct DispatchArgs {
    pub op: u32,
    pub format: u32,
    pub rows: u32,
    pub columns: u32,
    pub tokens: u32,
    pub position: u32,
    pub start_position: u32,
    pub max_context: u32,
    pub gather_stride: u32,
    pub n_head: u32,
    pub n_head_kv: u32,
    pub head_dim: u32,
    pub table_rows: u32,
    pub row: u32,
    pub pairing: u32,
    pub epsilon: f32,
    pub theta: f32,
    pub threads: u32,
    pub batch_rows: u32,
    pub tile_tokens: u32,
    pub tile_count: u32,
    pub group_count: u32,
    pub threadgroup_bytes: u32,
    pub prefill_tile_rows: u32,
    pub prefill_k_tile: u32,
}

#[repr(C)]
pub(crate) struct ContextHandle(c_void);

#[repr(C)]
pub(crate) struct BufferHandle(c_void);

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CopyRegion {
    source: *mut BufferHandle,
    destination: *mut BufferHandle,
    source_offset: usize,
    destination_offset: usize,
    bytes: usize,
}

impl CopyRegion {
    pub(crate) fn new(
        source: *mut BufferHandle,
        destination: *mut BufferHandle,
        source_offset: usize,
        destination_offset: usize,
        bytes: usize,
    ) -> Self {
        Self {
            source,
            destination,
            source_offset,
            destination_offset,
            bytes,
        }
    }
}

unsafe extern "C" {
    fn leone_metal_context_new(
        shader_source: *const c_char,
        shader_length: usize,
        info: *mut DeviceInfo,
        error: *mut c_char,
        error_length: usize,
    ) -> *mut ContextHandle;
    fn leone_metal_context_free(context: *mut ContextHandle);
    fn leone_metal_device_info(
        context: *mut ContextHandle,
        info: *mut DeviceInfo,
        error: *mut c_char,
        error_length: usize,
    ) -> i32;
    fn leone_metal_device_metadata(
        context: *mut ContextHandle,
        metadata: *mut DeviceMetadata,
        error: *mut c_char,
        error_length: usize,
    ) -> i32;
    fn leone_metal_host_memory_snapshot(
        info: *mut HostMemoryInfo,
        error: *mut c_char,
        error_length: usize,
    ) -> i32;
    fn leone_metal_alloc(
        context: *mut ContextHandle,
        bytes: usize,
        buffer: *mut *mut BufferHandle,
        error: *mut c_char,
        error_length: usize,
    ) -> i32;
    fn leone_metal_buffer_free(buffer: *mut BufferHandle);
    fn leone_metal_buffer_write(
        buffer: *mut BufferHandle,
        source: *const c_void,
        bytes: usize,
        error: *mut c_char,
        error_length: usize,
    ) -> i32;
    fn leone_metal_buffer_read(
        buffer: *mut BufferHandle,
        destination: *mut c_void,
        bytes: usize,
        error: *mut c_char,
        error_length: usize,
    ) -> i32;
    fn leone_metal_buffer_copy(
        context: *mut ContextHandle,
        source: *mut BufferHandle,
        destination: *mut BufferHandle,
        bytes: usize,
        error: *mut c_char,
        error_length: usize,
    ) -> i32;
    fn leone_metal_buffer_copy_regions(
        context: *mut ContextHandle,
        regions: *const CopyRegion,
        region_count: usize,
        error: *mut c_char,
        error_length: usize,
    ) -> i32;
    fn leone_metal_dispatch(
        context: *mut ContextHandle,
        args: *const DispatchArgs,
        buffers: *const *mut BufferHandle,
        buffer_count: usize,
        error: *mut c_char,
        error_length: usize,
    ) -> i32;
    #[allow(dead_code)]
    fn leone_metal_dispatch_sequence(
        context: *mut ContextHandle,
        args: *const DispatchArgs,
        command_count: usize,
        buffers: *const *mut BufferHandle,
        buffer_count: usize,
        error: *mut c_char,
        error_length: usize,
    ) -> i32;
    fn leone_metal_sync(
        context: *mut ContextHandle,
        error: *mut c_char,
        error_length: usize,
    ) -> i32;
}

const ERROR_BYTES: usize = 1024;

struct ContextInner {
    raw: *mut ContextHandle,
    deferred_error: RefCell<Option<String>>,
}

#[derive(Clone)]
pub(crate) struct Context {
    inner: Rc<ContextInner>,
}

impl Context {
    pub(crate) fn new(shader: &CStr) -> Result<(Self, DeviceInfo), String> {
        let mut info = DeviceInfo::default();
        let mut error = [0 as c_char; ERROR_BYTES];
        // SAFETY: The C bridge reads the immutable shader bytes for this call.
        // It returns an owned context or null and writes a NUL-terminated error.
        let raw = unsafe {
            leone_metal_context_new(
                shader.as_ptr(),
                shader.to_bytes().len(),
                &mut info,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if raw.is_null() {
            Err(error_string(&error))
        } else {
            Ok((
                Self {
                    inner: Rc::new(ContextInner {
                        raw,
                        deferred_error: RefCell::new(None),
                    }),
                },
                info,
            ))
        }
    }

    pub(crate) fn alloc(&self, bytes: usize) -> Result<Buffer, String> {
        self.check_deferred_error()?;
        let mut raw = std::ptr::null_mut();
        let mut error = [0 as c_char; ERROR_BYTES];
        // SAFETY: `self.inner.raw` is live for the call. The bridge initializes one
        // output pointer and copies no Rust-owned memory.
        let code = unsafe {
            leone_metal_alloc(
                self.inner.raw,
                bytes,
                &mut raw,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if code == 0 && !raw.is_null() {
            Ok(Buffer {
                raw,
                context: self.clone(),
            })
        } else {
            Err(error_string(&error))
        }
    }

    pub(crate) fn device_info(&self) -> Result<DeviceInfo, String> {
        let mut info = DeviceInfo::default();
        call_status(|error| {
            // SAFETY: `self.inner.raw` and the output structure are live for the call.
            unsafe {
                leone_metal_device_info(self.inner.raw, &mut info, error.as_mut_ptr(), error.len())
            }
        })?;
        Ok(info)
    }

    pub(crate) fn device_metadata(&self) -> Result<DeviceMetadata, String> {
        let mut metadata = DeviceMetadata::default();
        call_status(|error| {
            // SAFETY: `self.inner.raw` and the output structure are live for the call.
            unsafe {
                leone_metal_device_metadata(
                    self.inner.raw,
                    &mut metadata,
                    error.as_mut_ptr(),
                    error.len(),
                )
            }
        })?;
        Ok(metadata)
    }

    pub(crate) fn copy(
        &self,
        source: &Buffer,
        destination: &Buffer,
        bytes: usize,
    ) -> Result<(), String> {
        self.synchronize()?;
        call_status(|error| {
            // SAFETY: Handles are owned by `Context` and `Buffer` for this call.
            unsafe {
                leone_metal_buffer_copy(
                    self.inner.raw,
                    source.raw,
                    destination.raw,
                    bytes,
                    error.as_mut_ptr(),
                    error.len(),
                )
            }
        })
    }

    pub(crate) fn copy_regions(&self, regions: &[CopyRegion]) -> Result<(), String> {
        if regions.is_empty() {
            return Err(String::from("Metal copy region list is empty"));
        }
        self.synchronize()?;
        call_status(|error| {
            // SAFETY: Every region points to a live buffer owned by the caller.
            // The bridge reads only this immutable region array during the call.
            unsafe {
                leone_metal_buffer_copy_regions(
                    self.inner.raw,
                    regions.as_ptr(),
                    regions.len(),
                    error.as_mut_ptr(),
                    error.len(),
                )
            }
        })
    }

    pub(crate) fn dispatch(
        &self,
        args: &DispatchArgs,
        buffers: &[Option<&Buffer>],
    ) -> Result<(), String> {
        self.check_deferred_error()?;
        let handles = buffers
            .iter()
            .map(|buffer| buffer.map_or(std::ptr::null_mut(), |buffer| buffer.raw))
            .collect::<Vec<_>>();
        call_status(|error| {
            // SAFETY: The handle array and arguments remain live for the call.
            unsafe {
                leone_metal_dispatch(
                    self.inner.raw,
                    args,
                    handles.as_ptr(),
                    handles.len(),
                    error.as_mut_ptr(),
                    error.len(),
                )
            }
        })
    }

    #[allow(dead_code)]
    pub(crate) fn dispatch_sequence(
        &self,
        args: &[DispatchArgs],
        buffers: &[[Option<&Buffer>; 12]],
    ) -> Result<(), String> {
        self.check_deferred_error()?;
        if args.is_empty() || args.len() != buffers.len() {
            return Err("Metal dispatch sequence has mismatched commands".to_string());
        }
        let handles = buffers
            .iter()
            .flat_map(|command| {
                command
                    .iter()
                    .map(|buffer| buffer.map_or(std::ptr::null_mut(), |buffer| buffer.raw))
            })
            .collect::<Vec<_>>();
        call_status(|error| {
            // SAFETY: Encoding copies the arguments and binds live Metal buffers.
            // Buffer destruction fences pending work before freeing its handle.
            unsafe {
                leone_metal_dispatch_sequence(
                    self.inner.raw,
                    args.as_ptr(),
                    args.len(),
                    handles.as_ptr(),
                    12,
                    error.as_mut_ptr(),
                    error.len(),
                )
            }
        })
    }

    pub(crate) fn synchronize(&self) -> Result<(), String> {
        let deferred = self.inner.deferred_error.borrow_mut().take();
        let result = call_status(|error| {
            // SAFETY: `self.inner.raw` is live for the call.
            unsafe { leone_metal_sync(self.inner.raw, error.as_mut_ptr(), error.len()) }
        });
        deferred.map_or(result, Err)
    }

    fn check_deferred_error(&self) -> Result<(), String> {
        self.inner
            .deferred_error
            .borrow()
            .clone()
            .map_or(Ok(()), Err)
    }
}

impl Drop for ContextInner {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // SAFETY: This is the unique owner of the context pointer.
            unsafe { leone_metal_context_free(self.raw) };
        }
    }
}

pub(crate) struct Buffer {
    raw: *mut BufferHandle,
    context: Context,
}

impl Buffer {
    pub(crate) fn raw(&self) -> *mut BufferHandle {
        self.raw
    }

    pub(crate) fn write(&self, bytes: &[u8]) -> Result<(), String> {
        self.context.synchronize()?;
        call_status(|error| {
            // SAFETY: The source slice remains live for the synchronous call.
            unsafe {
                leone_metal_buffer_write(
                    self.raw,
                    bytes.as_ptr().cast(),
                    bytes.len(),
                    error.as_mut_ptr(),
                    error.len(),
                )
            }
        })
    }

    pub(crate) fn read(&self, bytes: &mut [u8]) -> Result<(), String> {
        self.read_raw(bytes.as_mut_ptr().cast(), bytes.len())
    }

    pub(crate) fn read_u32(&self, values: &mut [u32]) -> Result<(), String> {
        let bytes = checked_byte_count(values.len(), size_of::<u32>())?;
        self.read_raw(values.as_mut_ptr().cast(), bytes)?;
        for value in values {
            *value = u32::from_le(*value);
        }
        Ok(())
    }

    pub(crate) fn read_f16(&self, values: &mut [u16]) -> Result<(), String> {
        let bytes = checked_byte_count(values.len(), size_of::<u16>())?;
        self.read_raw(values.as_mut_ptr().cast(), bytes)?;
        for value in values {
            *value = u16::from_le(*value);
        }
        Ok(())
    }

    pub(crate) fn read_f32(&self, values: &mut [f32]) -> Result<(), String> {
        let bytes = checked_byte_count(values.len(), size_of::<f32>())?;
        self.read_raw(values.as_mut_ptr().cast(), bytes)?;
        for value in values {
            *value = f32::from_bits(u32::from_le(value.to_bits()));
        }
        Ok(())
    }

    fn read_raw(&self, destination: *mut c_void, bytes: usize) -> Result<(), String> {
        self.context.synchronize()?;
        call_status(|error| {
            // SAFETY: The pointer comes from a live mutable slice, and `bytes`
            // is that slice's exact byte length. Typed callers use scalar types
            // for which every bit pattern is valid.
            unsafe {
                leone_metal_buffer_read(
                    self.raw,
                    destination,
                    bytes,
                    error.as_mut_ptr(),
                    error.len(),
                )
            }
        })
    }
}

fn checked_byte_count(elements: usize, element_size: usize) -> Result<usize, String> {
    elements
        .checked_mul(element_size)
        .ok_or_else(|| String::from("Metal typed read byte count overflow"))
}

pub(crate) fn host_memory_snapshot() -> Result<HostMemoryInfo, String> {
    let mut info = HostMemoryInfo::default();
    call_status(|error| {
        // SAFETY: The output structure and error buffer remain live for the call.
        unsafe { leone_metal_host_memory_snapshot(&mut info, error.as_mut_ptr(), error.len()) }
    })?;
    Ok(info)
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // MetalBuffer drops this handle before releasing its allocation lease.
        if let Err(error) = self.context.synchronize() {
            self.context.inner.deferred_error.replace(Some(error));
        }
        if !self.raw.is_null() {
            // SAFETY: This is the unique owner of the buffer pointer.
            unsafe { leone_metal_buffer_free(self.raw) };
        }
    }
}

fn call_status(call: impl FnOnce(&mut [c_char; ERROR_BYTES]) -> i32) -> Result<(), String> {
    let mut error = [0 as c_char; ERROR_BYTES];
    let code = call(&mut error);
    if code == 0 {
        Ok(())
    } else {
        Err(error_string(&error))
    }
}

fn error_string(error: &[c_char; ERROR_BYTES]) -> String {
    let bytes = error
        .iter()
        .map(|value| *value as u8)
        .take_while(|value| *value != 0)
        .collect::<Vec<_>>();
    String::from_utf8_lossy(&bytes).into_owned()
}

#[cfg(all(test, target_os = "macos"))]
mod submission_tests {
    use super::*;
    use std::ffi::CString;

    fn context() -> Context {
        let shader = CString::new(crate::SHADER_SOURCE).unwrap();
        Context::new(&shader).unwrap().0
    }

    fn increment_args() -> DispatchArgs {
        DispatchArgs {
            op: 20,
            threads: 1,
            ..DispatchArgs::default()
        }
    }

    fn increment_buffers(counter: &Buffer) -> [Option<&Buffer>; 12] {
        let mut buffers = [None; 12];
        buffers[8] = Some(counter);
        buffers
    }

    #[test]
    fn drop_errors_survive_metadata_and_reject_dispatch_until_sync() {
        let context = context();
        let counter = context.alloc(4).unwrap();
        let guard = context.alloc(4).unwrap();
        counter.write(&0_u32.to_le_bytes()).unwrap();
        let args = increment_args();
        let buffers = increment_buffers(&counter);
        context.dispatch(&args, &buffers).unwrap();
        context
            .inner
            .deferred_error
            .replace(Some("injected completion failure".to_owned()));
        assert!(context.dispatch(&args, &buffers).is_err());
        assert!(context.alloc(4).is_err());
        context.device_info().unwrap();
        drop(guard);
        assert_eq!(
            context.synchronize().unwrap_err(),
            "injected completion failure"
        );
        let mut value = [0_u32];
        counter.read_u32(&mut value).unwrap();
        assert_eq!(value, [1]);
        context.dispatch(&args, &buffers).unwrap();
        drop(context);
        counter.read_u32(&mut value).unwrap();
        assert_eq!(value, [2]);
    }

    #[test]
    fn invalid_sequence_does_not_encode_its_valid_prefix() {
        let context = context();
        let counter = context.alloc(4).unwrap();
        counter.write(&0_u32.to_le_bytes()).unwrap();
        let valid = increment_args();
        let invalid = DispatchArgs { op: 0, ..valid };
        let buffers = increment_buffers(&counter);
        context.dispatch(&valid, &buffers).unwrap();
        assert!(context
            .dispatch_sequence(&[valid, invalid], &[buffers, buffers])
            .is_err());
        let mut value = [0_u32];
        counter.read_u32(&mut value).unwrap();
        assert_eq!(value, [1]);
    }
}
