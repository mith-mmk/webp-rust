use std::fs;
use std::io::{Error as IoError, Result as IoResult};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use webp_rust::decode;
use webp_rust::encoder::{encode_lossy_rgba_to_webp_with_config, LossyEncodingConfig};

fn write_bmp(path: &Path, width: usize, height: usize, rgba: &[u8]) -> IoResult<()> {
    let row_stride = (width * 3 + 3) & !3;
    let pixel_size = row_stride * height;
    let mut data = vec![0u8; 54 + pixel_size];
    data[0..2].copy_from_slice(b"BM");
    data[2..6].copy_from_slice(&((54 + pixel_size) as u32).to_le_bytes());
    data[10..14].copy_from_slice(&54u32.to_le_bytes());
    data[14..18].copy_from_slice(&40u32.to_le_bytes());
    data[18..22].copy_from_slice(&(width as i32).to_le_bytes());
    data[22..26].copy_from_slice(&(height as i32).to_le_bytes());
    data[26..28].copy_from_slice(&1u16.to_le_bytes());
    data[28..30].copy_from_slice(&24u16.to_le_bytes());
    data[34..38].copy_from_slice(&(pixel_size as u32).to_le_bytes());
    for y in 0..height {
        let source_y = height - 1 - y;
        let row = 54 + y * row_stride;
        for x in 0..width {
            let source = (source_y * width + x) * 4;
            let target = row + x * 3;
            data[target..target + 3].copy_from_slice(&[
                rgba[source + 2],
                rgba[source + 1],
                rgba[source],
            ]);
        }
    }
    fs::write(path, data)
}

fn sample(name: &str) -> (usize, usize, Vec<u8>) {
    let (width, height) = (320usize, 180usize);
    let mut rgba = vec![0u8; width * height * 4];
    for y in 0..height {
        for x in 0..width {
            let offset = (y * width + x) * 4;
            let value = match name {
                "flat" => 128,
                "gradient" => ((x * 255) / (width - 1)) as u8,
                _ => ((x * 13 + y * 17 + (x ^ y) * 3) & 0xff) as u8,
            };
            rgba[offset] = value;
            rgba[offset + 1] = value.wrapping_add((y / 2) as u8);
            rgba[offset + 2] = value.wrapping_add((x / 3) as u8);
            rgba[offset + 3] = 0xff;
        }
    }
    (width, height, rgba)
}

fn run_cwebp(input: &Path, output: &Path, quality: f32, method: u8) -> Option<(usize, f64)> {
    let started = Instant::now();
    let status = Command::new("cwebp")
        .args([
            "-quiet",
            "-q",
            &quality.to_string(),
            "-m",
            &method.to_string(),
            input.to_string_lossy().as_ref(),
            "-o",
            output.to_string_lossy().as_ref(),
        ])
        .status()
        .ok()?;
    if !status.success() {
        return None;
    }
    let encoded = fs::read(output).ok()?;
    decode(&encoded).ok()?;
    let size = fs::metadata(output).ok()?.len() as usize;
    Some((size, started.elapsed().as_secs_f64() * 1000.0))
}

fn main() -> IoResult<()> {
    let keep = std::env::args().any(|arg| arg == "--keep");
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".test-webp-bench");
    fs::create_dir_all(&root)?;
    let mut cwebp_available = true;

    println!("encoder,sample,method,bytes,time_ms");
    for name in ["flat", "gradient", "texture"] {
        let (width, height, rgba) = sample(name);
        let bmp = root.join(format!("{name}.bmp"));
        write_bmp(&bmp, width, height, &rgba)?;
        for method in [0u8, 4, 6] {
            let config = LossyEncodingConfig {
                quality: 75.0,
                method,
                ..LossyEncodingConfig::default()
            };
            let started = Instant::now();
            let encoded = encode_lossy_rgba_to_webp_with_config(width, height, &rgba, &config)
                .map_err(|error| IoError::other(error.to_string()))?;
            if keep {
                fs::write(root.join(format!("rust-{name}-{method}.webp")), &encoded)?;
            }
            println!(
                "rust,{name},{method},{},{:.3}",
                encoded.len(),
                started.elapsed().as_secs_f64() * 1000.0
            );

            let cwebp_output = root.join(format!("{name}-{method}.webp"));
            match run_cwebp(&bmp, &cwebp_output, config.quality, method) {
                Some((size, elapsed)) => println!("cwebp,{name},{method},{size},{elapsed:.3}"),
                None => cwebp_available = false,
            }
        }
    }

    if !cwebp_available {
        eprintln!("cwebp was not available or failed; Rust measurements were still emitted");
    }
    if !keep {
        let _ = fs::remove_dir_all(root);
    }
    Ok(())
}
