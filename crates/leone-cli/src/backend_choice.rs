#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackendChoice {
    Cuda,
    Cpu,
    Metal,
}

pub(crate) fn validate_compiled(backend: BackendChoice) -> Result<(), std::io::Error> {
    match backend {
        BackendChoice::Cpu => Ok(()),
        BackendChoice::Cuda => {
            #[cfg(feature = "cuda")]
            return Ok(());
            #[cfg(not(feature = "cuda"))]
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "the cuda backend is unavailable in this build; use --backend cpu",
            ));
        }
        BackendChoice::Metal => {
            #[cfg(feature = "metal")]
            return Ok(());
            #[cfg(not(feature = "metal"))]
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "the metal backend is unavailable in this build; use --backend cpu or install the native Metal package",
            ));
        }
    }
}

pub(crate) fn default_backend() -> BackendChoice {
    #[cfg(feature = "metal")]
    {
        BackendChoice::Metal
    }
    #[cfg(all(not(feature = "metal"), feature = "cuda"))]
    {
        BackendChoice::Cuda
    }
    #[cfg(all(not(feature = "metal"), not(feature = "cuda")))]
    {
        BackendChoice::Cpu
    }
}
