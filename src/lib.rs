//! Pure Rust WebP decode and still-image encode helpers.
//!
//! The top-level API is intentionally small:
//! - [`decode`] decodes a still WebP image into [`ImageBuffer`]
//! - [`encode`] encodes an [`ImageBuffer`] as lossless WebP
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

/// Encodes an image as a still lossless WebP container.
///
/// This is the default top-level encoder and preserves the RGBA payload.
/// If `exif` is present it is embedded as a raw `EXIF` chunk.
pub fn encode(image: &ImageBuffer, exif: Option<&[u8]>) -> Result<Vec<u8>, EncoderError> {
    encode_lossless(image, exif)
}

/// Encodes an image as a still lossy WebP container with default quality.
///
/// If `exif` is present it is embedded as a raw `EXIF` chunk.
/// For custom quality, use [`encoder::encode_lossy_image_to_webp_with_options_and_exif`].
pub fn encode_lossy(image: &ImageBuffer, exif: Option<&[u8]>) -> Result<Vec<u8>, EncoderError> {
    encoder::encode_lossy_image_to_webp_with_options_and_exif(
        image,
        &LossyEncodingOptions::default(),
        exif,
    )
}

/// Encodes an image as a still lossless WebP container with default options.
///
/// If `exif` is present it is embedded as a raw `EXIF` chunk.
/// For custom effort, use [`encoder::encode_lossless_image_to_webp_with_options_and_exif`].
pub fn encode_lossless(image: &ImageBuffer, exif: Option<&[u8]>) -> Result<Vec<u8>, EncoderError> {
    encoder::encode_lossless_image_to_webp_with_options_and_exif(
        image,
        &LosslessEncodingOptions::default(),
        exif,
    )
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
