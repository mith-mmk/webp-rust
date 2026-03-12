use std::fs;
use std::path::{Path, PathBuf};

use webp_rust::decode;
use webp_rust::decoder::{decode_animation_webp, get_features};

type Error = Box<dyn std::error::Error>;
type DecoderError = webp_rust::decoder::DecoderError;

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

fn encode_bmp24_from_rgba(
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

fn default_output_path(input: &Path) -> PathBuf {
    let mut output = input.to_path_buf();
    output.set_extension("bmp");
    output
}

fn default_animation_output_prefix(input: &Path) -> PathBuf {
    let mut output = input.to_path_buf();
    output.set_extension("");
    output
}

fn animation_frame_path(prefix: &Path, index: usize) -> PathBuf {
    let parent = prefix.parent().unwrap_or_else(|| Path::new(""));
    let stem = prefix
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or("frame");
    parent.join(format!("{stem}_{index:04}.bmp"))
}

fn main() -> Result<(), Error> {
    let mut args = std::env::args_os().skip(1);
    let input = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("_testdata/sample.webp"));
    let output = args.next().map(PathBuf::from).unwrap_or_else(|| {
        if input == PathBuf::from("_testdata/sample.webp") {
            PathBuf::from("target/sample.bmp")
        } else if input == PathBuf::from("_testdata/sample_animation.webp") {
            PathBuf::from("target/sample_animation")
        } else {
            default_output_path(&input)
        }
    });

    let data = fs::read(&input)?;
    let features = get_features(&data)?;

    if features.has_animation {
        let mut prefix = if output.is_dir() {
            output.join(
                input
                    .file_stem()
                    .and_then(|stem| stem.to_str())
                    .unwrap_or("frame"),
            )
        } else {
            output
        };
        if prefix.extension().is_some() {
            prefix.set_extension("");
        } else if prefix.as_os_str().is_empty() {
            prefix = default_animation_output_prefix(&input);
        }

        let animation = decode_animation_webp(&data)?;
        let mut written_paths = Vec::with_capacity(animation.frames.len());
        for (index, frame) in animation.frames.into_iter().enumerate() {
            let path = animation_frame_path(&prefix, index);
            if let Some(parent) = path.parent() {
                if !parent.as_os_str().is_empty() {
                    fs::create_dir_all(parent)?;
                }
            }
            let bmp = encode_bmp24_from_rgba(animation.width, animation.height, &frame.rgba)?;
            fs::write(&path, bmp)?;
            written_paths.push(path);
        }
        for path in written_paths {
            println!("{}", path.display());
        }
        return Ok(());
    }

    let image = decode(&data)?;
    let bmp = encode_bmp24_from_rgba(image.width, image.height, &image.rgba)?;

    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    fs::write(&output, bmp)?;

    println!("{}", output.display());
    Ok(())
}
