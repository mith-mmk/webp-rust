//! Lower-level WebP parsing and decoding APIs.

pub mod alpha;
pub mod animation;
pub mod header;
pub mod lossless;
pub mod lossy;
pub mod quant;
pub mod tree;
pub mod vp8;
pub mod vp8i;

use std::fmt::{Display, Formatter, Result as FmtResult};

pub use alpha::{apply_alpha_plane, decode_alpha_plane, AlphaHeader};
pub use animation::{decode_animation_webp, DecodedAnimation, DecodedAnimationFrame};
pub use header::{
    get_features, parse_animation_webp, parse_still_webp, AnimationHeader, ChunkHeader,
    ParsedAnimationFrame, ParsedAnimationWebp, ParsedWebp, Vp8xHeader, WebpFeatures,
};
pub use lossless::{decode_lossless_vp8l_to_rgba, decode_lossless_webp_to_rgba};
pub use lossy::{
    decode_lossy_vp8_to_rgba, decode_lossy_vp8_to_yuv, decode_lossy_webp_to_rgba,
    decode_lossy_webp_to_yuv, DecodedImage, DecodedYuvImage,
};
pub use vp8::{
    parse_lossy_headers, parse_macroblock_data, parse_macroblock_headers, LosslessInfo,
    LossyHeader, MacroBlockData, MacroBlockDataFrame, MacroBlockHeaders,
};
pub use vp8i::WebpFormat;

/// Error type used by decoding and parsing entry points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecoderError {
    /// A caller-provided buffer size or dimension is invalid.
    InvalidParam(&'static str),
    /// The input ended before a required structure was fully available.
    NotEnoughData(&'static str),
    /// The bitstream violates the WebP container or codec format.
    Bitstream(&'static str),
    /// The input uses a feature that is intentionally not implemented.
    Unsupported(&'static str),
}

impl Display for DecoderError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::InvalidParam(msg) => write!(f, "invalid parameter: {msg}"),
            Self::NotEnoughData(msg) => write!(f, "not enough data: {msg}"),
            Self::Bitstream(msg) => write!(f, "bitstream error: {msg}"),
            Self::Unsupported(msg) => write!(f, "unsupported feature: {msg}"),
        }
    }
}

impl std::error::Error for DecoderError {}
