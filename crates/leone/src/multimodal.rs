//! Defines validated image inputs without coupling the runtime to a vision backend.

use thiserror::Error;

/// Requested image preprocessing detail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageDetail {
    Auto,
    Low,
    High,
}

/// A validated image source accepted by the server contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageSource {
    Http(String),
    DataUrl(String),
}

/// One validated image input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageInput {
    source: ImageSource,
    detail: ImageDetail,
}

/// An image source or detail that is not accepted.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ImageInputError {
    #[error("image URL must use http, https, or a base64 image data URL")]
    Scheme,
    #[error("image data URL must have an image media type and base64 payload")]
    DataUrl,
    #[error("image detail must be auto, low, or high")]
    Detail,
}

impl ImageInput {
    /// Validates one OpenAI-compatible `image_url` part.
    pub fn new(url: impl Into<String>, detail: Option<&str>) -> Result<Self, ImageInputError> {
        let url = url.into();
        let source = if url.starts_with("https://") || url.starts_with("http://") {
            ImageSource::Http(url)
        } else if url.starts_with("data:image/") {
            let (_, payload) = url.split_once(";base64,").ok_or(ImageInputError::DataUrl)?;
            if payload.is_empty() {
                return Err(ImageInputError::DataUrl);
            }
            ImageSource::DataUrl(url)
        } else {
            return Err(ImageInputError::Scheme);
        };
        let detail = match detail.unwrap_or("auto") {
            "auto" => ImageDetail::Auto,
            "low" => ImageDetail::Low,
            "high" => ImageDetail::High,
            _ => return Err(ImageInputError::Detail),
        };
        Ok(Self { source, detail })
    }

    pub fn source(&self) -> &ImageSource {
        &self.source
    }

    pub const fn detail(&self) -> ImageDetail {
        self.detail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_remote_and_inline_images() {
        assert!(matches!(
            ImageInput::new("https://example.test/image.png", Some("high"))
                .expect("remote image")
                .source(),
            ImageSource::Http(_)
        ));
        assert!(matches!(
            ImageInput::new("data:image/png;base64,iVBORw0KGgo=", None)
                .expect("inline image")
                .source(),
            ImageSource::DataUrl(_)
        ));
    }

    #[test]
    fn rejects_local_paths_and_untyped_data() {
        assert_eq!(
            ImageInput::new("file:///tmp/image.png", None),
            Err(ImageInputError::Scheme)
        );
        assert_eq!(
            ImageInput::new("data:text/plain;base64,SGVsbG8=", None),
            Err(ImageInputError::Scheme)
        );
    }
}
