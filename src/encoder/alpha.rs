use super::lossless::encode_lossless_rgba_to_vp8l;
use super::lossy::{AlphaFilter, LossyEncodingConfig};
use super::EncoderError;

fn predictor(filter: AlphaFilter, left: u8, top: u8, top_left: u8) -> u8 {
    match filter {
        AlphaFilter::None => 0,
        AlphaFilter::Fast => left,
        AlphaFilter::Best => (left as i32 + top as i32 - top_left as i32).clamp(0, 255) as u8,
    }
}

fn filtered_alpha(
    width: usize,
    height: usize,
    rgba: &[u8],
    filter: AlphaFilter,
    quality: u8,
) -> Vec<u8> {
    let mut alpha = vec![0u8; width * height];
    let step = ((100 - quality) / 16).max(1);
    for (index, pixel) in rgba.chunks_exact(4).enumerate() {
        alpha[index] = ((pixel[3] as usize / step as usize) * step as usize).min(255) as u8;
    }

    let mut filtered = vec![0u8; alpha.len()];
    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            let left = if x == 0 { 0 } else { alpha[index - 1] };
            let top = if y == 0 { 0 } else { alpha[index - width] };
            let top_left = if x == 0 || y == 0 {
                0
            } else {
                alpha[index - width - 1]
            };
            let pred = match filter {
                AlphaFilter::None => 0,
                AlphaFilter::Fast => {
                    if x == 0 {
                        if y == 0 {
                            0
                        } else {
                            top
                        }
                    } else {
                        left
                    }
                }
                AlphaFilter::Best => {
                    if x == 0 && y > 0 {
                        top
                    } else if y == 0 {
                        left
                    } else {
                        predictor(filter, left, top, top_left)
                    }
                }
            };
            filtered[index] = alpha[index].wrapping_sub(pred);
        }
    }
    filtered
}

pub(crate) fn encode_alpha_payload(
    width: usize,
    height: usize,
    rgba: &[u8],
    config: &LossyEncodingConfig,
) -> Result<Vec<u8>, EncoderError> {
    let filter = config.alpha_filter;
    let filtered = filtered_alpha(width, height, rgba, filter, config.alpha_quality);
    let filter_bits = match filter {
        AlphaFilter::None => 0,
        AlphaFilter::Fast => 1,
        AlphaFilter::Best => 3,
    };
    let compression = if config.alpha_method == 0 { 0 } else { 1 };
    let header = (filter_bits << 2) | compression;
    let mut output = Vec::new();
    output.push(header);
    if compression == 0 {
        output.extend_from_slice(&filtered);
        return Ok(output);
    }

    let mut alpha_rgba = vec![0u8; filtered.len() * 4];
    for (index, &value) in filtered.iter().enumerate() {
        let pixel = &mut alpha_rgba[index * 4..index * 4 + 4];
        pixel.copy_from_slice(&[0, value, 0, 0xff]);
    }
    let encoded = encode_lossless_rgba_to_vp8l(width, height, &alpha_rgba)?;
    let stream = encoded.get(5..).ok_or(EncoderError::Bitstream(
        "lossless alpha stream is too small",
    ))?;
    output.extend_from_slice(stream);
    Ok(output)
}
