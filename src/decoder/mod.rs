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

pub use alpha::AlphaHeader;
pub use animation::{
    decode_animation_webp, decode_animation_webp_to_bmp_frames, DecodedAnimation,
    DecodedAnimationFrame,
};
pub use header::{
    get_features, parse_animation_webp, parse_still_webp, AnimationHeader, ChunkHeader,
    ParsedAnimationFrame, ParsedAnimationWebp, ParsedWebp, Vp8xHeader, WebpFeatures,
};
pub use lossless::{
    decode_lossless_vp8l_to_bmp, decode_lossless_vp8l_to_rgba, decode_lossless_webp_to_bmp,
    decode_lossless_webp_to_rgba,
};
pub use lossy::{
    decode_lossy_vp8_to_bmp, decode_lossy_vp8_to_rgba, decode_lossy_vp8_to_yuv,
    decode_lossy_webp_to_bmp, decode_lossy_webp_to_rgba, decode_lossy_webp_to_yuv, DecodedImage,
    DecodedYuvImage,
};
pub use vp8::{
    parse_lossy_headers, parse_macroblock_data, parse_macroblock_headers, LosslessInfo,
    LossyHeader, MacroBlockData, MacroBlockDataFrame, MacroBlockHeaders,
};
pub use vp8i::WebpFormat;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecoderError {
    InvalidParam(&'static str),
    NotEnoughData(&'static str),
    Bitstream(&'static str),
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
