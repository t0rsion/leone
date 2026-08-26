//! Provides read-only file access without a memory map.
//!
//! Metadata is read once through a buffered clone. Tensor data uses positional
//! reads, so callers can read independent ranges without a shared file cursor.
//! Each requested range is copied. A memory map requires that the file not
//! be truncated while the map is live.

use crate::{Error, Result};
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;

#[derive(Debug)]
pub(crate) struct ReadOnlyFile {
    file: File,
    len: u64,
}

impl ReadOnlyFile {
    pub(crate) fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let len = file.metadata()?.len();
        Ok(Self { file, len })
    }

    pub(crate) fn try_clone(&self) -> Result<File> {
        Ok(self.file.try_clone()?)
    }

    pub(crate) const fn len(&self) -> u64 {
        self.len
    }

    pub(crate) fn read(&self, offset: u64, len: u64) -> Result<Vec<u8>> {
        let end = offset
            .checked_add(len)
            .ok_or(Error::IntegerOverflow("tensor range"))?;
        if end > self.len {
            return Err(Error::TensorOutOfBounds {
                tensor: "requested range".to_owned(),
            });
        }
        let len = usize::try_from(len).map_err(|_| Error::IntegerOverflow("tensor length"))?;
        let mut bytes = vec![0; len];
        self.file.read_exact_at(&mut bytes, offset)?;
        Ok(bytes)
    }
}
