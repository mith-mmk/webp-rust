use std::io::Write;
use std::path::Path;

use crate::decoder::DecoderError;

const FILE_HEADER_SIZE: usize = 14;
const INFO_HEADER_SIZE: usize = 40;
const BMP_HEADER_SIZE: usize = FILE_HEADER_SIZE + INFO_HEADER_SIZE;
const BITS_PER_PIXEL: usize = 24;
const PIXELS_PER_METER: u32 = 3_780;

fn row_stride(width: usize) -> Result<usize, DecoderError> {
    let raw = width
        .checked_mul(3)
        .ok_or(DecoderError::InvalidParam("BMP row size overflow"))?;
    Ok((raw + 3) & !3)
}

pub fn encode_bmp24_from_rgba(
    width: usize,
    height: usize,
    rgba: &[u8],
) -> Result<Vec<u8>, DecoderError> {
    if width == 0 || height == 0 {
        return Err(DecoderError::InvalidParam(
            "BMP dimensions must be non-zero",
        ));
    }

    let expected_len = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or(DecoderError::InvalidParam("RGBA buffer size overflow"))?;
    if rgba.len() != expected_len {
        return Err(DecoderError::InvalidParam(
            "RGBA buffer length does not match dimensions",
        ));
    }

    let stride = row_stride(width)?;
    let pixel_bytes = stride
        .checked_mul(height)
        .ok_or(DecoderError::InvalidParam("BMP pixel storage overflow"))?;
    let file_size = BMP_HEADER_SIZE
        .checked_add(pixel_bytes)
        .ok_or(DecoderError::InvalidParam("BMP file size overflow"))?;

    let mut bmp = vec![0u8; file_size];

    bmp[0..2].copy_from_slice(b"BM");
    bmp[2..6].copy_from_slice(&(file_size as u32).to_le_bytes());
    bmp[10..14].copy_from_slice(&(BMP_HEADER_SIZE as u32).to_le_bytes());

    bmp[14..18].copy_from_slice(&(INFO_HEADER_SIZE as u32).to_le_bytes());
    bmp[18..22].copy_from_slice(&(width as i32).to_le_bytes());
    bmp[22..26].copy_from_slice(&(height as i32).to_le_bytes());
    bmp[26..28].copy_from_slice(&(1u16).to_le_bytes());
    bmp[28..30].copy_from_slice(&(BITS_PER_PIXEL as u16).to_le_bytes());
    bmp[34..38].copy_from_slice(&(pixel_bytes as u32).to_le_bytes());
    bmp[38..42].copy_from_slice(&PIXELS_PER_METER.to_le_bytes());
    bmp[42..46].copy_from_slice(&PIXELS_PER_METER.to_le_bytes());

    let mut dest_offset = BMP_HEADER_SIZE;
    let mut row = vec![0u8; stride];
    for y in (0..height).rev() {
        row.fill(0);
        let src_row = y * width * 4;
        for x in 0..width {
            let src = src_row + x * 4;
            let dst = x * 3;
            row[dst] = rgba[src + 2];
            row[dst + 1] = rgba[src + 1];
            row[dst + 2] = rgba[src];
        }
        bmp[dest_offset..dest_offset + stride].copy_from_slice(&row);
        dest_offset += stride;
    }

    Ok(bmp)
}

pub fn write_bmp24_from_rgba<P: AsRef<Path>>(
    path: P,
    width: usize,
    height: usize,
    rgba: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    let bmp = encode_bmp24_from_rgba(width, height, rgba)?;
    let mut file = std::fs::File::create(path)?;
    file.write_all(&bmp)?;
    file.flush()?;
    Ok(())
}
