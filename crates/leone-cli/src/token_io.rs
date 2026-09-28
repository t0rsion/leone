use std::fs;
use std::io;
use std::path::Path;

pub(crate) fn read_tokens(path: &Path) -> Result<Vec<u32>, io::Error> {
    let bytes = fs::read(path)?;
    if bytes.len() % 4 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "token file size must be a multiple of four bytes",
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect())
}
