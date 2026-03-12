//! Pure Rust WebP encoder helpers.
//!
//! Current scope is still-image WebP encode.
//! The lossless path targets `VP8L` with transforms, adaptive Huffman
//! coding, simple backward references, and an optional color cache.
//! The lossy path targets opaque still images and emits a minimal
//! intra-only `VP8` bitstream.

mod bit_writer;
mod error;
mod huffman;
mod lossless;
mod lossy;
mod vp8_bool_writer;

pub use error::EncoderError;
pub use lossless::{
    encode_lossless_image_to_webp, encode_lossless_image_to_webp_with_options,
    encode_lossless_rgba_to_vp8l, encode_lossless_rgba_to_vp8l_with_options,
    encode_lossless_rgba_to_webp, encode_lossless_rgba_to_webp_with_options,
    LosslessEncodingOptions,
};
pub use lossy::{
    encode_lossy_image_to_webp, encode_lossy_image_to_webp_with_options, encode_lossy_rgba_to_vp8,
    encode_lossy_rgba_to_vp8_with_options, encode_lossy_rgba_to_webp,
    encode_lossy_rgba_to_webp_with_options, LossyEncodingOptions,
};
