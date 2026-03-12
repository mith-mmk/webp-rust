use std::fs;
use std::io::{Error as IoError, ErrorKind};
use std::path::{Path, PathBuf};

use webp_rust::encoder::{
    encode_lossless_image_to_webp_with_options, encode_lossy_image_to_webp_with_options,
};
use webp_rust::{ImageBuffer, LosslessEncodingOptions, LossyEncodingOptions};

type Error = Box<dyn std::error::Error>;

const FILE_HEADER_SIZE: usize = 14;
const MIN_INFO_HEADER_SIZE: usize = 40;

fn invalid_data(message: &'static str) -> Error {
    Box::new(IoError::new(ErrorKind::InvalidData, message))
}

fn invalid_input(message: &'static str) -> Error {
    Box::new(IoError::new(ErrorKind::InvalidInput, message))
}

fn invalid_input_owned(message: String) -> Error {
    Box::new(IoError::new(ErrorKind::InvalidInput, message))
}

fn read_u16_le(data: &[u8], offset: usize) -> Result<u16, Error> {
    let bytes = data
        .get(offset..offset + 2)
        .ok_or_else(|| invalid_data("BMP header is truncated"))?;
    Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
}

fn read_u32_le(data: &[u8], offset: usize) -> Result<u32, Error> {
    let bytes = data
        .get(offset..offset + 4)
        .ok_or_else(|| invalid_data("BMP header is truncated"))?;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn read_i32_le(data: &[u8], offset: usize) -> Result<i32, Error> {
    let bytes = data
        .get(offset..offset + 4)
        .ok_or_else(|| invalid_data("BMP header is truncated"))?;
    Ok(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn row_stride(width: usize, bytes_per_pixel: usize) -> Result<usize, Error> {
    let raw = width
        .checked_mul(bytes_per_pixel)
        .ok_or_else(|| invalid_data("BMP row size overflow"))?;
    Ok((raw + 3) & !3)
}

fn decode_bmp_to_rgba(data: &[u8]) -> Result<ImageBuffer, Error> {
    if data.len() < FILE_HEADER_SIZE + MIN_INFO_HEADER_SIZE {
        return Err(invalid_data("BMP file is too small"));
    }
    if &data[0..2] != b"BM" {
        return Err(invalid_data("expected a BMP file"));
    }

    let pixel_offset = read_u32_le(data, 10)? as usize;
    let dib_header_size = read_u32_le(data, 14)? as usize;
    if dib_header_size < MIN_INFO_HEADER_SIZE {
        return Err(invalid_data("unsupported BMP DIB header"));
    }
    if data.len() < FILE_HEADER_SIZE + dib_header_size {
        return Err(invalid_data("BMP DIB header is truncated"));
    }

    let width_i32 = read_i32_le(data, 18)?;
    let height_i32 = read_i32_le(data, 22)?;
    let planes = read_u16_le(data, 26)?;
    let bits_per_pixel = read_u16_le(data, 28)?;
    let compression = read_u32_le(data, 30)?;

    if planes != 1 {
        return Err(invalid_data("unsupported BMP plane count"));
    }
    if compression != 0 {
        return Err(invalid_data("only uncompressed BMP is supported"));
    }
    if width_i32 <= 0 {
        return Err(invalid_data("BMP width must be positive"));
    }
    if height_i32 == 0 {
        return Err(invalid_data("BMP height must be non-zero"));
    }

    let bytes_per_pixel = match bits_per_pixel {
        24 => 3usize,
        32 => 4usize,
        _ => return Err(invalid_data("only 24bpp and 32bpp BMP are supported")),
    };

    let width = width_i32 as usize;
    let top_down = height_i32 < 0;
    let height = height_i32
        .checked_abs()
        .ok_or_else(|| invalid_data("BMP height is out of range"))? as usize;

    let stride = row_stride(width, bytes_per_pixel)?;
    let pixel_bytes = stride
        .checked_mul(height)
        .ok_or_else(|| invalid_data("BMP pixel storage overflow"))?;
    let pixel_end = pixel_offset
        .checked_add(pixel_bytes)
        .ok_or_else(|| invalid_data("BMP pixel offset overflow"))?;
    if pixel_offset > data.len() || pixel_end > data.len() {
        return Err(invalid_data("BMP pixel data is truncated"));
    }

    let rgba_len = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or_else(|| invalid_data("BMP output size overflow"))?;
    let mut rgba = vec![0u8; rgba_len];

    for y in 0..height {
        let src_y = if top_down { y } else { height - 1 - y };
        let src_row = pixel_offset + src_y * stride;
        let dst_row = y * width * 4;
        for x in 0..width {
            let src = src_row + x * bytes_per_pixel;
            let dst = dst_row + x * 4;
            rgba[dst] = data[src + 2];
            rgba[dst + 1] = data[src + 1];
            rgba[dst + 2] = data[src];
            rgba[dst + 3] = if bytes_per_pixel == 4 {
                data[src + 3]
            } else {
                0xff
            };
        }
    }

    Ok(ImageBuffer {
        width,
        height,
        rgba,
    })
}

fn default_output_path(input: &Path) -> PathBuf {
    let mut output = input.to_path_buf();
    output.set_extension("webp");
    output
}

fn usage() -> &'static str {
    "usage: cargo run --example bmp2webp -- [-z N] [--lossy --quality 0..100 [--lossy-opt-level 0..9]] [--opt-level 0|1|2] <input.bmp> [output.webp]"
}

fn parse_u8_level(value: &str, what: &str) -> Result<u8, Error> {
    value
        .parse::<u8>()
        .map_err(|_| invalid_input_owned(format!("invalid {what}: {value}")))
}

fn parse_optimization_level(value: &str) -> Result<u8, Error> {
    let level = parse_u8_level(value, "optimization level")?;
    if level > 2 {
        return Err(invalid_input("optimization level must be in 0..=2"));
    }
    Ok(level)
}

fn parse_lossy_optimization_level(value: &str) -> Result<u8, Error> {
    let level = parse_u8_level(value, "lossy optimization level")?;
    if level > 9 {
        return Err(invalid_input("lossy optimization level must be in 0..=9"));
    }
    Ok(level)
}

fn parse_quality(value: &str) -> Result<u8, Error> {
    let quality = value
        .parse::<u8>()
        .map_err(|_| invalid_input_owned(format!("invalid quality: {value}")))?;
    if quality > 100 {
        return Err(invalid_input("quality must be in 0..=100"));
    }
    Ok(quality)
}

fn main() -> Result<(), Error> {
    let mut args = std::env::args_os().skip(1);
    let mut input = None;
    let mut output = None;
    let mut options = LosslessEncodingOptions::default();
    let mut lossy = false;
    let mut lossy_options = LossyEncodingOptions::default();
    let mut shared_optimization_level = None;
    let mut lossless_optimization_explicit = false;
    let mut lossy_optimization_explicit = false;

    while let Some(arg) = args.next() {
        match arg.to_string_lossy().as_ref() {
            "--lossy" => lossy = true,
            "--opt" | "--opt-level" | "-O" => {
                let value = args.next().ok_or_else(|| invalid_input(usage()))?;
                options.optimization_level = parse_optimization_level(&value.to_string_lossy())?;
                lossless_optimization_explicit = true;
            }
            "--quality" | "-q" => {
                let value = args.next().ok_or_else(|| invalid_input(usage()))?;
                lossy_options.quality = parse_quality(&value.to_string_lossy())?;
            }
            "-z" => {
                let value = args.next().ok_or_else(|| invalid_input(usage()))?;
                shared_optimization_level = Some(parse_u8_level(
                    &value.to_string_lossy(),
                    "optimization level",
                )?);
            }
            "--lossy-opt-level" => {
                let value = args.next().ok_or_else(|| invalid_input(usage()))?;
                lossy_options.optimization_level =
                    parse_lossy_optimization_level(&value.to_string_lossy())?;
                lossy_optimization_explicit = true;
            }
            _ => {
                if input.is_none() {
                    input = Some(PathBuf::from(arg));
                } else if output.is_none() {
                    output = Some(PathBuf::from(arg));
                } else {
                    return Err(invalid_input(usage()));
                }
            }
        }
    }

    if let Some(level) = shared_optimization_level {
        if lossy {
            if !lossy_optimization_explicit {
                lossy_options.optimization_level = if level <= 9 {
                    level
                } else {
                    return Err(invalid_input("lossy optimization level must be in 0..=9"));
                };
            }
        } else if !lossless_optimization_explicit {
            options.optimization_level = if level <= 2 {
                level
            } else {
                return Err(invalid_input("optimization level must be in 0..=2"));
            };
        }
    }

    let input = input.ok_or_else(|| invalid_input(usage()))?;
    let output = output.unwrap_or_else(|| default_output_path(&input));

    let data = fs::read(&input)?;
    let image = decode_bmp_to_rgba(&data)?;
    let webp = if lossy {
        encode_lossy_image_to_webp_with_options(&image, &lossy_options)?
    } else {
        encode_lossless_image_to_webp_with_options(&image, &options)?
    };

    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    fs::write(&output, webp)?;

    println!("{}", output.display());
    Ok(())
}
