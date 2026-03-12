//! Pure Rust WebP encoder helpers.
//!
//! Current scope is intentionally small: still-image lossless WebP (`VP8L`)
//! using direct literal coding without transforms, color cache, or backward
//! references.

mod bit_writer;
mod error;
mod huffman;
mod lossless;

pub use error::EncoderError;
pub use lossless::{
    encode_lossless_image_to_webp, encode_lossless_rgba_to_vp8l, encode_lossless_rgba_to_webp,
};
