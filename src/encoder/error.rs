use std::fmt::{Display, Formatter, Result as FmtResult};

/// Error type used by encoding entry points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncoderError {
    /// A caller-provided dimension or buffer length is invalid.
    InvalidParam(&'static str),
    /// Internal encoder state would produce an invalid VP8L bitstream.
    Bitstream(&'static str),
}

impl Display for EncoderError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::InvalidParam(msg) => write!(f, "invalid parameter: {msg}"),
            Self::Bitstream(msg) => write!(f, "bitstream error: {msg}"),
        }
    }
}

impl std::error::Error for EncoderError {}
