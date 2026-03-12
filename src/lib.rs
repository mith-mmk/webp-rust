//! Pure Rust WebP decode and still-image encode helpers.
//!
//! The top-level API is intentionally small:
//! - [`decode`] decodes a still WebP image into [`ImageBuffer`]
//! - [`encode`] encodes an [`ImageBuffer`] as lossy or lossless WebP
//! - [`encode_lossy`] encodes an [`ImageBuffer`] as lossy WebP
//! - [`encode_lossless`] encodes an [`ImageBuffer`] as lossless WebP
//!
//! Lower-level codec and container entry points remain available under
//! [`decoder`] and [`encoder`].

use std::path::Path;

pub mod decoder;
pub mod encoder;
mod image;
#[doc(hidden)]
pub mod legacy;

pub use decoder::DecoderError;
pub use encoder::{EncoderError, LosslessEncodingOptions, LossyEncodingOptions};
pub use image::ImageBuffer;

/// Top-level still-image WebP compression mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebpEncoding {
    /// Encode as lossless `VP8L`.
    Lossless,
    /// Encode as lossy `VP8`.
    Lossy,
}

/// Decodes a still WebP image from memory into an RGBA buffer.
///
/// Animated WebP is rejected by this helper. Use
/// [`decoder::decode_animation_webp`] for animated input.
pub fn decode(data: &[u8]) -> Result<ImageBuffer, DecoderError> {
    let features = decoder::get_features(data)?;
    if features.has_animation {
        return Err(DecoderError::Unsupported(
            "animated WebP requires animation decoder API",
        ));
    }

    let image = match features.format {
        decoder::WebpFormat::Lossy => decoder::decode_lossy_webp_to_rgba(data)?,
        decoder::WebpFormat::Lossless => decoder::decode_lossless_webp_to_rgba(data)?,
        decoder::WebpFormat::Undefined => {
            return Err(DecoderError::Unsupported("unsupported WebP format"))
        }
    };

    Ok(ImageBuffer {
        width: image.width,
        height: image.height,
        rgba: image.rgba,
    })
}

fn to_lossless_options(optimize: usize) -> Result<LosslessEncodingOptions, EncoderError> {
    let optimization_level = u8::try_from(optimize)
        .map_err(|_| EncoderError::InvalidParam("lossless optimization level must be in 0..=9"))?;
    Ok(LosslessEncodingOptions { optimization_level })
}

fn to_lossy_options(optimize: usize, quality: usize) -> Result<LossyEncodingOptions, EncoderError> {
    let optimization_level = u8::try_from(optimize)
        .map_err(|_| EncoderError::InvalidParam("lossy optimization level must be in 0..=9"))?;
    let quality = u8::try_from(quality)
        .map_err(|_| EncoderError::InvalidParam("quality must be in 0..=100"))?;
    Ok(LossyEncodingOptions {
        quality,
        optimization_level,
    })
}

/// Encodes an image as a still WebP container.
///
/// `optimize` is interpreted as `0..=9` for [`WebpEncoding::Lossless`] and
/// `0..=9` for [`WebpEncoding::Lossy`]. `quality` is used only for lossy
/// encoding and must be in `0..=100`.
///
/// If `exif` is present it is embedded as a raw `EXIF` chunk.
pub fn encode(
    image: &ImageBuffer,
    optimize: usize,
    quality: usize,
    compression: WebpEncoding,
    exif: Option<&[u8]>,
) -> Result<Vec<u8>, EncoderError> {
    match compression {
        WebpEncoding::Lossless => encode_lossless(image, optimize, exif),
        WebpEncoding::Lossy => encode_lossy(image, optimize, quality, exif),
    }
}

/// Encodes an image as a still lossy WebP container.
///
/// `optimize` must be in `0..=9`. `quality` must be in `0..=100`.
///
/// If `exif` is present it is embedded as a raw `EXIF` chunk.
pub fn encode_lossy(
    image: &ImageBuffer,
    optimize: usize,
    quality: usize,
    exif: Option<&[u8]>,
) -> Result<Vec<u8>, EncoderError> {
    let options = to_lossy_options(optimize, quality)?;
    encoder::encode_lossy_image_to_webp_with_options_and_exif(image, &options, exif)
}

/// Encodes an image as a still lossless WebP container.
///
/// `optimize` must be in `0..=9`.
///
/// If `exif` is present it is embedded as a raw `EXIF` chunk.
pub fn encode_lossless(
    image: &ImageBuffer,
    optimize: usize,
    exif: Option<&[u8]>,
) -> Result<Vec<u8>, EncoderError> {
    let options = to_lossless_options(optimize)?;
    encoder::encode_lossless_image_to_webp_with_options_and_exif(image, &options, exif)
}

/// Compatibility alias for [`decode`].
pub fn image_from_bytes(data: &[u8]) -> Result<ImageBuffer, DecoderError> {
    decode(data)
}

/// Reads a still WebP image from disk and decodes it to RGBA.
#[cfg(not(target_family = "wasm"))]
pub fn decode_file<P: AsRef<Path>>(filename: P) -> Result<ImageBuffer, Box<dyn std::error::Error>> {
    let data = std::fs::read(filename)?;
    Ok(decode(&data)?)
}

/// Compatibility alias for [`decode_file`].
#[cfg(not(target_family = "wasm"))]
pub fn image_from_file<P: AsRef<Path>>(
    filename: P,
) -> Result<ImageBuffer, Box<dyn std::error::Error>> {
    decode_file(filename)
}
