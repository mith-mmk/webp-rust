//! Compare the 0.3.0 encoder with libwebp's cwebp.
//!
//! The comparison uses generated opaque BMP images so both encoders receive
//! the same pixels. It prints one row per image and an average row per option.
//! The temporary files are kept below the OS temp directory only with
//! --keep.

use std::fs;
use std::io::{Error as IoError, ErrorKind, Result as IoResult};
use std::path::Path;
use std::process::Command;
use std::time::Instant;

use webp_rust::decode;
use webp_rust::encoder::{
    encode_lossless_rgba_to_webp_with_config, encode_lossy_rgba_to_webp_with_config,
    LosslessEncodingConfig, LossyEncodingConfig,
};

const WIDTH: usize = 320;
const HEIGHT: usize = 180;

struct Measurement {
    encoded_size: usize,
    psnr: f64,
    time_ms: f64,
}

fn write_bmp(path: &Path, rgba: &[u8]) -> IoResult<()> {
    let row_stride = (WIDTH * 3 + 3) & !3;
    let pixel_size = row_stride * HEIGHT;
    let mut data = vec![0u8; 54 + pixel_size];
    data[0..2].copy_from_slice(b"BM");
    data[2..6].copy_from_slice(&((54 + pixel_size) as u32).to_le_bytes());
    data[10..14].copy_from_slice(&54u32.to_le_bytes());
    data[14..18].copy_from_slice(&40u32.to_le_bytes());
    data[18..22].copy_from_slice(&(WIDTH as i32).to_le_bytes());
    data[22..26].copy_from_slice(&(HEIGHT as i32).to_le_bytes());
    data[26..28].copy_from_slice(&1u16.to_le_bytes());
    data[28..30].copy_from_slice(&24u16.to_le_bytes());
    data[34..38].copy_from_slice(&(pixel_size as u32).to_le_bytes());

    for y in 0..HEIGHT {
        let source_y = HEIGHT - 1 - y;
        let row = 54 + y * row_stride;
        for x in 0..WIDTH {
            let source = (source_y * WIDTH + x) * 4;
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

fn sample(name: &str) -> Vec<u8> {
    let mut rgba = vec![0u8; WIDTH * HEIGHT * 4];
    let mut state = 0x8f3a_21c5_u32;
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let offset = (y * WIDTH + x) * 4;
            let value = match name {
                "flat" => 128,
                "gradient" => ((x * 255) / (WIDTH - 1)) as u8,
                "texture" => ((x * 13 + y * 17 + (x ^ y) * 3) & 0xff) as u8,
                _ => {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    ((state >> 24) & 0xff) as u8
                }
            };
            rgba[offset] = value;
            rgba[offset + 1] = value.wrapping_add((y / 2) as u8);
            rgba[offset + 2] = value.wrapping_add((x / 3) as u8);
            rgba[offset + 3] = 0xff;
        }
    }
    rgba
}

fn psnr(source: &[u8], decoded: &[u8]) -> f64 {
    let mut squared_error = 0_u64;
    let mut samples = 0_u64;
    for (source_pixel, decoded_pixel) in source.chunks_exact(4).zip(decoded.chunks_exact(4)) {
        for channel in 0..3 {
            let difference = source_pixel[channel] as i32 - decoded_pixel[channel] as i32;
            squared_error += (difference * difference) as u64;
            samples += 1;
        }
    }
    if squared_error == 0 {
        return f64::INFINITY;
    }
    let mse = squared_error as f64 / samples as f64;
    10.0 * ((255.0 * 255.0) / mse).log10()
}

fn measure(encoded: Vec<u8>, source: &[u8], started: Instant) -> IoResult<Measurement> {
    let decoded =
        decode(&encoded).map_err(|error| IoError::new(ErrorKind::Other, error.to_string()))?;
    Ok(Measurement {
        encoded_size: encoded.len(),
        psnr: psnr(source, &decoded.rgba),
        time_ms: started.elapsed().as_secs_f64() * 1000.0,
    })
}

fn run_cwebp(
    input: &Path,
    output: &Path,
    mode: &str,
    quality: Option<f32>,
    method: Option<u8>,
    z_level: Option<u8>,
) -> Option<Vec<u8>> {
    let mut command = Command::new("cwebp");
    command.args(["-quiet"]);
    if mode == "lossless" {
        command.args(["-lossless", "-z", &z_level?.to_string()]);
    } else {
        command.args(["-q", &quality?.to_string(), "-m", &method?.to_string()]);
    }
    let status = command
        .args([input.to_string_lossy().as_ref(), "-o"])
        .arg(output)
        .status()
        .ok()?;
    if status.success() {
        fs::read(output).ok()
    } else {
        None
    }
}

fn print_row(
    record: &str,
    encoder: &str,
    mode: &str,
    sample: &str,
    option: &str,
    raw_size: usize,
    measurement: &Measurement,
) {
    let ratio = raw_size as f64 / measurement.encoded_size as f64;
    let saving = (1.0 - measurement.encoded_size as f64 / raw_size as f64) * 100.0;
    println!(
        "{record},{encoder},{mode},{sample},{option},{},{ratio:.3},{saving:.2},{:.2},{:.3}",
        measurement.encoded_size, measurement.psnr, measurement.time_ms
    );
}

fn print_average(
    encoder: &str,
    mode: &str,
    option: &str,
    raw_size: usize,
    measurements: &[Measurement],
) {
    let average_size = measurements
        .iter()
        .map(|measurement| measurement.encoded_size as f64)
        .sum::<f64>()
        / measurements.len() as f64;
    let ratio = raw_size as f64 / average_size;
    let saving = (1.0 - average_size / raw_size as f64) * 100.0;
    let average_psnr = measurements
        .iter()
        .map(|measurement| measurement.psnr)
        .sum::<f64>()
        / measurements.len() as f64;
    let average_time = measurements
        .iter()
        .map(|measurement| measurement.time_ms)
        .sum::<f64>()
        / measurements.len() as f64;
    println!(
        "average,{encoder},{mode},average,{option},{average_size:.1},{ratio:.3},{saving:.2},{average_psnr:.2},{average_time:.3}"
    );
}

fn compare_lossless(
    root: &Path,
    samples: &[(&str, Vec<u8>)],
    raw_size: usize,
    cwebp_available: &mut bool,
) -> IoResult<()> {
    for z_level in [0u8, 2, 4, 6] {
        let option = format!("z={z_level}");
        let mut rust_results = Vec::with_capacity(samples.len());
        let mut libwebp_results = Vec::with_capacity(samples.len());
        for (name, rgba) in samples {
            let input = root.join(format!("{name}.bmp"));
            let started = Instant::now();
            let encoded = encode_lossless_rgba_to_webp_with_config(
                WIDTH,
                HEIGHT,
                rgba,
                &LosslessEncodingConfig { z_level },
            )
            .map_err(|error| IoError::new(ErrorKind::Other, error.to_string()))?;
            let measurement = measure(encoded, rgba, started)?;
            print_row(
                "sample",
                "rust",
                "lossless",
                name,
                &option,
                raw_size,
                &measurement,
            );
            rust_results.push(measurement);

            let output = root.join(format!("libwebp-lossless-{name}-{z_level}.webp"));
            let started = Instant::now();
            let Some(encoded) = run_cwebp(&input, &output, "lossless", None, None, Some(z_level))
            else {
                *cwebp_available = false;
                continue;
            };
            let measurement = measure(encoded, rgba, started)?;
            print_row(
                "sample",
                "libwebp",
                "lossless",
                name,
                &option,
                raw_size,
                &measurement,
            );
            libwebp_results.push(measurement);
        }
        print_average("rust", "lossless", &option, raw_size, &rust_results);
        if libwebp_results.len() == samples.len() {
            print_average("libwebp", "lossless", &option, raw_size, &libwebp_results);
        }
    }
    Ok(())
}

fn compare_lossy(
    root: &Path,
    samples: &[(&str, Vec<u8>)],
    raw_size: usize,
    cwebp_available: &mut bool,
) -> IoResult<()> {
    for quality in [50.0, 75.0, 90.0] {
        for method in [0u8, 4, 6] {
            let option = format!("q={quality:.0};m={method}");
            let mut rust_results = Vec::with_capacity(samples.len());
            let mut libwebp_results = Vec::with_capacity(samples.len());
            for (name, rgba) in samples {
                let input = root.join(format!("{name}.bmp"));
                let config = LossyEncodingConfig {
                    quality,
                    method,
                    ..LossyEncodingConfig::default()
                };
                let started = Instant::now();
                let encoded = encode_lossy_rgba_to_webp_with_config(WIDTH, HEIGHT, rgba, &config)
                    .map_err(|error| IoError::new(ErrorKind::Other, error.to_string()))?;
                let measurement = measure(encoded, rgba, started)?;
                print_row(
                    "sample",
                    "rust",
                    "lossy",
                    name,
                    &option,
                    raw_size,
                    &measurement,
                );
                rust_results.push(measurement);

                let output = root.join(format!("libwebp-lossy-{quality:.0}-{method}-{name}.webp"));
                let started = Instant::now();
                let Some(encoded) =
                    run_cwebp(&input, &output, "lossy", Some(quality), Some(method), None)
                else {
                    *cwebp_available = false;
                    continue;
                };
                let measurement = measure(encoded, rgba, started)?;
                print_row(
                    "sample",
                    "libwebp",
                    "lossy",
                    name,
                    &option,
                    raw_size,
                    &measurement,
                );
                libwebp_results.push(measurement);
            }
            print_average("rust", "lossy", &option, raw_size, &rust_results);
            if libwebp_results.len() == samples.len() {
                print_average("libwebp", "lossy", &option, raw_size, &libwebp_results);
            }
        }
    }
    Ok(())
}

fn main() -> IoResult<()> {
    let keep = std::env::args().any(|arg| arg == "--keep");
    let root = std::env::temp_dir().join(format!(".test-webp-compare-{}", std::process::id()));
    fs::create_dir_all(&root)?;
    let samples = ["flat", "gradient", "texture"]
        .into_iter()
        .map(|name| (name, sample(name)))
        .collect::<Vec<_>>();
    let raw_size = WIDTH * HEIGHT * 4;

    for (name, rgba) in &samples {
        write_bmp(&root.join(format!("{name}.bmp")), rgba)?;
    }

    println!(
        "record,encoder,mode,sample,option,bytes_or_avg,raw_to_webp,saving_percent,rgb_psnr,time_ms"
    );
    let mut cwebp_available = true;
    compare_lossless(&root, &samples, raw_size, &mut cwebp_available)?;
    compare_lossy(&root, &samples, raw_size, &mut cwebp_available)?;

    if !cwebp_available {
        eprintln!("cwebp was not available or failed; Rust measurements were still emitted");
    }
    if keep {
        eprintln!("comparison files kept at {}", root.display());
    } else {
        let _ = fs::remove_dir_all(&root);
    }
    Ok(())
}
