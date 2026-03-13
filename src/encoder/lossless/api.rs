//! Public entry points for lossless still-image WebP encoding.

use super::plans::*;
use super::*;

/// Encodes RGBA pixels to a raw lossless `VP8L` frame payload with explicit options.
pub fn encode_lossless_rgba_to_vp8l_with_options(
    width: usize,
    height: usize,
    rgba: &[u8],
    options: &LosslessEncodingOptions,
) -> Result<Vec<u8>, EncoderError> {
    validate_rgba(width, height, rgba)?;
    validate_options(options)?;

    let argb = rgba_to_argb(rgba);
    let subtract_green = apply_subtract_green_transform(&argb);
    let mut best = None;

    for profile in lossless_candidate_profiles(options.optimization_level) {
        let mut profile_best =
            if let Some(candidate) = build_palette_candidate(width, height, &argb)? {
                Some(encode_palette_candidate_to_vp8l(
                    width, height, rgba, &candidate, &profile,
                )?)
            } else {
                None
            };

        let plans = collect_transform_plans(width, height, &argb, &subtract_green, &profile);
        let ranked_plans = shortlist_transform_plans(width, plans, &profile)?;
        for (estimate, plan) in ranked_plans {
            if profile_best
                .as_ref()
                .map(|encoded| should_stop_transform_search(encoded.len(), estimate, &profile))
                .unwrap_or(false)
            {
                break;
            }

            let encoded = encode_transform_plan_to_vp8l(width, height, rgba, &plan, &profile)?;
            if profile_best
                .as_ref()
                .map(|current| encoded.len() < current.len())
                .unwrap_or(true)
            {
                profile_best = Some(encoded);
            }
        }

        if let Some(encoded) = profile_best {
            if best
                .as_ref()
                .map(|current: &Vec<u8>| encoded.len() < current.len())
                .unwrap_or(true)
            {
                best = Some(encoded);
            }
        }
    }

    best.ok_or(EncoderError::Bitstream(
        "lossless encoder produced no candidate",
    ))
}

/// Encodes RGBA pixels to a raw lossless `VP8L` frame payload.
pub fn encode_lossless_rgba_to_vp8l(
    width: usize,
    height: usize,
    rgba: &[u8],
) -> Result<Vec<u8>, EncoderError> {
    encode_lossless_rgba_to_vp8l_with_options(
        width,
        height,
        rgba,
        &LosslessEncodingOptions::default(),
    )
}

/// Encodes RGBA pixels to a still lossless WebP container with explicit options.
pub fn encode_lossless_rgba_to_webp_with_options(
    width: usize,
    height: usize,
    rgba: &[u8],
    options: &LosslessEncodingOptions,
) -> Result<Vec<u8>, EncoderError> {
    encode_lossless_rgba_to_webp_with_options_and_exif(width, height, rgba, options, None)
}

/// Encodes RGBA pixels to a still lossless WebP container with explicit options and EXIF.
pub fn encode_lossless_rgba_to_webp_with_options_and_exif(
    width: usize,
    height: usize,
    rgba: &[u8],
    options: &LosslessEncodingOptions,
    exif: Option<&[u8]>,
) -> Result<Vec<u8>, EncoderError> {
    let vp8l = encode_lossless_rgba_to_vp8l_with_options(width, height, rgba, options)?;
    wrap_still_webp(
        StillImageChunk {
            fourcc: *b"VP8L",
            payload: &vp8l,
            width,
            height,
            has_alpha: rgba_has_alpha(rgba),
        },
        exif,
    )
}

/// Encodes RGBA pixels to a still lossless WebP container.
pub fn encode_lossless_rgba_to_webp(
    width: usize,
    height: usize,
    rgba: &[u8],
) -> Result<Vec<u8>, EncoderError> {
    encode_lossless_rgba_to_webp_with_options(
        width,
        height,
        rgba,
        &LosslessEncodingOptions::default(),
    )
}

/// Encodes an [`ImageBuffer`] to a still lossless WebP container with explicit options.
pub fn encode_lossless_image_to_webp_with_options(
    image: &ImageBuffer,
    options: &LosslessEncodingOptions,
) -> Result<Vec<u8>, EncoderError> {
    encode_lossless_image_to_webp_with_options_and_exif(image, options, None)
}

/// Encodes an [`ImageBuffer`] to a still lossless WebP container with explicit options and EXIF.
pub fn encode_lossless_image_to_webp_with_options_and_exif(
    image: &ImageBuffer,
    options: &LosslessEncodingOptions,
    exif: Option<&[u8]>,
) -> Result<Vec<u8>, EncoderError> {
    encode_lossless_rgba_to_webp_with_options_and_exif(
        image.width,
        image.height,
        &image.rgba,
        options,
        exif,
    )
}

/// Encodes an [`ImageBuffer`] to a still lossless WebP container.
pub fn encode_lossless_image_to_webp(image: &ImageBuffer) -> Result<Vec<u8>, EncoderError> {
    encode_lossless_image_to_webp_with_options(image, &LosslessEncodingOptions::default())
}
