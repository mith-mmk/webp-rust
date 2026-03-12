use crate::encoder::bit_writer::BitWriter;
use crate::encoder::huffman::HuffmanCode;
use crate::encoder::EncoderError;
use crate::ImageBuffer;

const MAX_WEBP_DIMENSION: usize = 1 << 14;
const NUM_LITERAL_CODES: usize = 256;
const NUM_LENGTH_CODES: usize = 24;
const NUM_CODE_LENGTH_CODES: usize = 19;
const CODE_LENGTH_CODE_ORDER: [usize; NUM_CODE_LENGTH_CODES] = [
    17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
];

fn validate_rgba(width: usize, height: usize, rgba: &[u8]) -> Result<(), EncoderError> {
    if width == 0 || height == 0 {
        return Err(EncoderError::InvalidParam(
            "image dimensions must be non-zero",
        ));
    }
    if width > MAX_WEBP_DIMENSION || height > MAX_WEBP_DIMENSION {
        return Err(EncoderError::InvalidParam(
            "image dimensions exceed VP8L limits",
        ));
    }

    let expected_len = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or(EncoderError::InvalidParam("image dimensions overflow"))?;
    if rgba.len() != expected_len {
        return Err(EncoderError::InvalidParam(
            "RGBA buffer length does not match dimensions",
        ));
    }

    Ok(())
}

fn has_alpha(rgba: &[u8]) -> bool {
    rgba.chunks_exact(4).any(|pixel| pixel[3] != 0xff)
}

fn write_trimmed_length(bw: &mut BitWriter, trimmed_length: usize) -> Result<(), EncoderError> {
    if trimmed_length < 2 {
        return Err(EncoderError::Bitstream("trimmed Huffman span is too small"));
    }

    let value = trimmed_length - 2;
    let mut nbitpairs = 1usize;
    while nbitpairs <= 8 && value >= (1usize << (nbitpairs * 2)) {
        nbitpairs += 1;
    }
    if nbitpairs > 8 {
        return Err(EncoderError::Bitstream("trimmed Huffman span is too large"));
    }

    bw.put_bits((nbitpairs - 1) as u32, 3)?;
    bw.put_bits(value as u32, nbitpairs * 2)
}

fn write_single_symbol_code_length_tree(
    bw: &mut BitWriter,
    symbol: usize,
) -> Result<(), EncoderError> {
    let num_codes = CODE_LENGTH_CODE_ORDER
        .iter()
        .position(|&value| value == symbol)
        .map(|index| index + 1)
        .ok_or(EncoderError::Bitstream("invalid code-length symbol"))?;
    if num_codes < 4 {
        return Err(EncoderError::Bitstream("code-length tree is too small"));
    }

    bw.put_bits((num_codes - 4) as u32, 4)?;
    for &code in CODE_LENGTH_CODE_ORDER.iter().take(num_codes) {
        let depth = u32::from(code == symbol);
        bw.put_bits(depth, 3)?;
    }
    Ok(())
}

fn write_full_literal_huffman_tree(
    bw: &mut BitWriter,
    total_symbols: usize,
    active_symbols: usize,
) -> Result<(), EncoderError> {
    if active_symbols == 0 || active_symbols > total_symbols || active_symbols > NUM_LITERAL_CODES {
        return Err(EncoderError::InvalidParam(
            "invalid literal Huffman tree shape",
        ));
    }

    bw.put_bits(0, 1)?;
    write_single_symbol_code_length_tree(bw, 8)?;
    if active_symbols < total_symbols {
        bw.put_bits(1, 1)?;
        write_trimmed_length(bw, active_symbols)?;
    } else {
        bw.put_bits(0, 1)?;
    }
    Ok(())
}

fn write_single_symbol_huffman_tree(bw: &mut BitWriter, symbol: usize) -> Result<(), EncoderError> {
    if symbol >= 1 << 8 {
        return Err(EncoderError::InvalidParam(
            "simple Huffman symbol is too large",
        ));
    }

    bw.put_bits(1, 1)?;
    bw.put_bits(0, 1)?;
    if symbol <= 1 {
        bw.put_bits(0, 1)?;
        bw.put_bits(symbol as u32, 1)?;
    } else {
        bw.put_bits(1, 1)?;
        bw.put_bits(symbol as u32, 8)?;
    }
    Ok(())
}

fn write_image_data(
    bw: &mut BitWriter,
    rgba: &[u8],
    byte_codes: &HuffmanCode,
) -> Result<(), EncoderError> {
    for pixel in rgba.chunks_exact(4) {
        let red = pixel[0] as usize;
        let green = pixel[1] as usize;
        let blue = pixel[2] as usize;
        let alpha = pixel[3] as usize;

        byte_codes.write_symbol(bw, green)?;
        byte_codes.write_symbol(bw, red)?;
        byte_codes.write_symbol(bw, blue)?;
        byte_codes.write_symbol(bw, alpha)?;
    }
    Ok(())
}

fn wrap_lossless_webp(vp8l: &[u8]) -> Result<Vec<u8>, EncoderError> {
    let padded_vp8l_size = vp8l.len() + (vp8l.len() & 1);
    let riff_size = 4usize
        .checked_add(8)
        .and_then(|size| size.checked_add(padded_vp8l_size))
        .ok_or(EncoderError::InvalidParam("encoded output is too large"))?;
    let total_size = 8usize
        .checked_add(riff_size)
        .ok_or(EncoderError::InvalidParam("encoded output is too large"))?;

    let mut data = Vec::with_capacity(total_size);
    data.extend_from_slice(b"RIFF");
    data.extend_from_slice(&(riff_size as u32).to_le_bytes());
    data.extend_from_slice(b"WEBP");
    data.extend_from_slice(b"VP8L");
    data.extend_from_slice(&(vp8l.len() as u32).to_le_bytes());
    data.extend_from_slice(vp8l);
    if vp8l.len() & 1 == 1 {
        data.push(0);
    }
    Ok(data)
}

/// Encodes RGBA pixels to a raw lossless `VP8L` frame payload.
pub fn encode_lossless_rgba_to_vp8l(
    width: usize,
    height: usize,
    rgba: &[u8],
) -> Result<Vec<u8>, EncoderError> {
    validate_rgba(width, height, rgba)?;

    let mut bw = BitWriter::default();
    bw.put_bits((width - 1) as u32, 14)?;
    bw.put_bits((height - 1) as u32, 14)?;
    bw.put_bits(has_alpha(rgba) as u32, 1)?;
    bw.put_bits(0, 3)?;

    bw.put_bits(0, 1)?;
    bw.put_bits(0, 1)?;
    bw.put_bits(0, 1)?;

    write_full_literal_huffman_tree(
        &mut bw,
        NUM_LITERAL_CODES + NUM_LENGTH_CODES,
        NUM_LITERAL_CODES,
    )?;
    write_full_literal_huffman_tree(&mut bw, NUM_LITERAL_CODES, NUM_LITERAL_CODES)?;
    write_full_literal_huffman_tree(&mut bw, NUM_LITERAL_CODES, NUM_LITERAL_CODES)?;
    write_full_literal_huffman_tree(&mut bw, NUM_LITERAL_CODES, NUM_LITERAL_CODES)?;
    write_single_symbol_huffman_tree(&mut bw, 0)?;

    let byte_codes = HuffmanCode::from_code_lengths(vec![8; NUM_LITERAL_CODES])?;
    write_image_data(&mut bw, rgba, &byte_codes)?;

    let bitstream = bw.into_bytes();
    let mut vp8l = Vec::with_capacity(1 + bitstream.len());
    vp8l.push(0x2f);
    vp8l.extend_from_slice(&bitstream);
    Ok(vp8l)
}

/// Encodes RGBA pixels to a still lossless WebP container.
pub fn encode_lossless_rgba_to_webp(
    width: usize,
    height: usize,
    rgba: &[u8],
) -> Result<Vec<u8>, EncoderError> {
    let vp8l = encode_lossless_rgba_to_vp8l(width, height, rgba)?;
    wrap_lossless_webp(&vp8l)
}

/// Encodes an [`ImageBuffer`] to a still lossless WebP container.
pub fn encode_lossless_image_to_webp(image: &ImageBuffer) -> Result<Vec<u8>, EncoderError> {
    encode_lossless_rgba_to_webp(image.width, image.height, &image.rgba)
}
