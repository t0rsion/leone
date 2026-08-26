use std::fmt::{self, Display, Formatter};

/// An error produced while reading, validating, or writing a receipt.
#[derive(Debug)]
pub enum Error {
    /// A file operation failed.
    Io(std::io::Error),
    /// JSON encoding or decoding failed.
    Json(serde_json::Error),
    /// A receipt field failed validation.
    Validation(String),
}

impl Error {
    pub(crate) fn validation(message: impl Into<String>) -> Self {
        Self::Validation(message.into())
    }
}

impl Display for Error {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "file operation failed: {error}"),
            Self::Json(error) => write!(formatter, "receipt JSON is invalid: {error}"),
            Self::Validation(message) => write!(formatter, "receipt validation failed: {message}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Json(error) => Some(error),
            Self::Validation(_) => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for Error {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

/// A result returned by receipt operations.
pub type Result<T> = std::result::Result<T, Error>;
