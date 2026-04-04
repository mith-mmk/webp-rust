//! Alpha-plane parsing and reconstruction helpers.

use crate::decoder::lossless::decode_lossless_stream_to_argb;
use crate::decoder::DecoderError;

const ALPHA_HEADER_LEN: usize = 1;
const ALPHA_NO_COMPRESSION: u8 = 0;
const ALPHA_LOSSLESS_COMPRESSION: u8 = 1;
const ALPHA_PREPROCESSED_LEVELS: u8 = 2;
const ALPHA_FILTER_NONE: u8 = 0;
const ALPHA_FILTER_HORIZONTAL: u8 = 1;
const ALPHA_FILTER_VERTICAL: u8 = 2;
const ALPHA_FILTER_GRADIENT: u8 = 3;

/// Parsed one-byte `ALPH` header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AlphaHeader {
    /// Compression method.
    pub compression: u8,
    /// Spatial filter method.
    pub filter: u8,
    /// Alpha preprocessing mode.
    pub preprocessing: u8,
}

/// Parses the one-byte header that prefixes an `ALPH` payload.
pub fn parse_alpha_header(data: &[u8]) -> Result<AlphaHeader, DecoderError> {
    let Some(&header) = data.first() else {
        return Err(DecoderError::NotEnoughData("ALPH header"));
    };

    let reserved = header >> 6;
    if reserved != 0 {
        return Err(DecoderError::Bitstream("ALPH reserved bits must be zero"));
    }

    let alpha = AlphaHeader {
        compression: header & 0x03,
        filter: (header >> 2) & 0x03,
        preprocessing: (header >> 4) & 0x03,
    };

    if alpha.compression > ALPHA_LOSSLESS_COMPRESSION {
        return Err(DecoderError::Bitstream(
            "unsupported ALPH compression method",
        ));
    }
    if alpha.preprocessing > ALPHA_PREPROCESSED_LEVELS {
        return Err(DecoderError::Bitstream(
            "unsupported ALPH preprocessing mode",
        ));
    }

    Ok(alpha)
}

fn gradient_predictor(left: u8, top: u8, top_left: u8) -> u8 {
    (left as i32 + top as i32 - top_left as i32).clamp(0, 255) as u8
}

fn unfilter_row(
    filter: u8,
    prev: Option<&[u8]>,
    deltas: &[u8],
    out: &mut [u8],
) -> Result<(), DecoderError> {
    match filter {
        ALPHA_FILTER_NONE => {
            out.copy_from_slice(deltas);
        }
        ALPHA_FILTER_HORIZONTAL => {
            let mut pred = prev.map_or(0, |line| line[0]);
            for (dst, &delta) in out.iter_mut().zip(deltas.iter()) {
                *dst = pred.wrapping_add(delta);
                pred = *dst;
            }
        }
        ALPHA_FILTER_VERTICAL => {
            if let Some(prev) = prev {
                for ((dst, &delta), &top) in out.iter_mut().zip(deltas.iter()).zip(prev.iter()) {
                    *dst = top.wrapping_add(delta);
                }
            } else {
                unfilter_row(ALPHA_FILTER_HORIZONTAL, None, deltas, out)?;
            }
        }
        ALPHA_FILTER_GRADIENT => {
            if let Some(prev) = prev {
                let mut top_left = prev[0];
                let mut left = prev[0];
                for (x, (dst, &delta)) in out.iter_mut().zip(deltas.iter()).enumerate() {
                    let top = prev[x];
                    left = delta.wrapping_add(gradient_predictor(left, top, top_left));
                    top_left = top;
                    *dst = left;
                }
            } else {
                unfilter_row(ALPHA_FILTER_HORIZONTAL, None, deltas, out)?;
            }
        }
        _ => return Err(DecoderError::Bitstream("invalid ALPH filter")),
    }
    Ok(())
}

fn unfilter_alpha(
    alpha: &[u8],
    filter: u8,
    width: usize,
    height: usize,
) -> Result<Vec<u8>, DecoderError> {
    let expected_len = width
        .checked_mul(height)
        .ok_or(DecoderError::Bitstream("alpha plane size overflow"))?;
    if alpha.len() < expected_len {
        return Err(DecoderError::NotEnoughData("alpha plane payload"));
    }

    let mut decoded = vec![0u8; expected_len];
    for y in 0..height {
        let row_start = y * width;
        let row_end = row_start + width;
        let (head, tail) = decoded.split_at_mut(row_start);
        let prev = if y == 0 {
            None
        } else {
            Some(&head[row_start - width..row_start])
        };
        unfilter_row(filter, prev, &alpha[row_start..row_end], &mut tail[..width])?;
    }
    Ok(decoded)
}

/// Decodes an `ALPH` payload to a single-channel alpha plane.
///
/// The returned buffer contains one alpha byte per pixel in row-major order.
pub fn decode_alpha_plane(
    data: &[u8],
    width: usize,
    height: usize,
) -> Result<Vec<u8>, DecoderError> {
    let header = parse_alpha_header(data)?;
    let payload = data
        .get(ALPHA_HEADER_LEN..)
        .ok_or(DecoderError::NotEnoughData("ALPH payload"))?;
    let pixel_count = width
        .checked_mul(height)
        .ok_or(DecoderError::Bitstream("alpha plane size overflow"))?;

    match header.compression {
        ALPHA_NO_COMPRESSION => {
            if payload.len() < pixel_count {
                return Err(DecoderError::NotEnoughData("ALPH raw payload"));
            }
            unfilter_alpha(&payload[..pixel_count], header.filter, width, height)
        }
        ALPHA_LOSSLESS_COMPRESSION => {
            let argb = decode_lossless_stream_to_argb(payload, width, height)?;
            let mut filtered = vec![0u8; pixel_count];
            for (dst, pixel) in filtered.iter_mut().zip(argb.iter()) {
                *dst = ((pixel >> 8) & 0xff) as u8;
            }
            unfilter_alpha(&filtered, header.filter, width, height)
        }
        _ => Err(DecoderError::Bitstream(
            "unsupported ALPH compression method",
        )),
    }
}

/// Replaces the alpha channel of an RGBA image with a decoded alpha plane.
pub fn apply_alpha_plane(rgba: &mut [u8], alpha: &[u8]) -> Result<(), DecoderError> {
    let expected_len = alpha
        .len()
        .checked_mul(4)
        .ok_or(DecoderError::Bitstream("RGBA buffer size overflow"))?;
    if rgba.len() != expected_len {
        return Err(DecoderError::InvalidParam(
            "RGBA buffer length does not match alpha plane",
        ));
    }

    for (pixel, &value) in rgba.chunks_exact_mut(4).zip(alpha.iter()) {
        pixel[3] = value;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{decode_alpha_plane, ALPHA_FILTER_HORIZONTAL};
    use std::path::PathBuf;

    #[test]
    fn decode_alpha_plane_unfilters_horizontal_rows() {
        let width = 4usize;
        let height = 2usize;
        let plane = [10u8, 20, 25, 40, 5, 7, 9, 11];
        let mut filtered = Vec::with_capacity(1 + plane.len());
        filtered.push(ALPHA_FILTER_HORIZONTAL << 2);

        filtered.push(plane[0]);
        for x in 1..width {
            filtered.push(plane[x].wrapping_sub(plane[x - 1]));
        }
        filtered.push(plane[width].wrapping_sub(plane[0]));
        for x in 1..width {
            filtered.push(plane[width + x].wrapping_sub(plane[width + x - 1]));
        }

        let decoded = decode_alpha_plane(&filtered, width, height).unwrap();

        assert_eq!(decoded, plane);
    }

    #[test]
    fn decode_lossless_alpha_plane_from_viewer_sample() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("test")
            .join("images")
            .join("bundled")
            .join("WML2Viewer_error.webp");
        let data = std::fs::read(path).unwrap();
        let parsed = crate::decoder::header::parse_animation_webp(&data).unwrap();
        let frame = &parsed.frames[0];
        let alpha =
            decode_alpha_plane(frame.alpha_data.unwrap(), frame.width, frame.height).unwrap();
        assert_eq!(alpha.len(), frame.width * frame.height);
    }
}
