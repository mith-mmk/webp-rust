use std::fs;
use std::hint::black_box;
use std::io::{Error as IoError, ErrorKind, Result as IoResult};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use webp_rust::decoder::{
    decode_animation_webp, decode_lossless_webp_to_rgba, decode_lossy_webp_to_rgba,
    decode_lossy_webp_to_yuv,
};

const DEFAULT_BATCHES: usize = 7;
const DEFAULT_BATCH_MILLIS: u64 = 250;
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Debug)]
struct Options {
    batches: usize,
    batch_duration: Duration,
    output: Option<PathBuf>,
}

#[derive(Debug)]
struct Measurement {
    case_name: &'static str,
    format: &'static str,
    width: usize,
    height: usize,
    pixels_per_decode: usize,
    output_hash: u64,
    median_ns: f64,
    p95_ns: f64,
}

#[derive(Debug, Clone, Copy)]
struct CaseMetadata {
    case_name: &'static str,
    format: &'static str,
    width: usize,
    height: usize,
    pixels_per_decode: usize,
    output_hash: u64,
}

fn invalid_input(message: impl Into<String>) -> IoError {
    IoError::new(ErrorKind::InvalidInput, message.into())
}

fn parse_usize(value: Option<String>, name: &str) -> IoResult<usize> {
    let value = value.ok_or_else(|| invalid_input(format!("missing value for {name}")))?;
    value
        .parse::<usize>()
        .map_err(|_| invalid_input(format!("invalid value for {name}: {value}")))
}

fn parse_options() -> IoResult<Options> {
    let mut options = Options {
        batches: DEFAULT_BATCHES,
        batch_duration: Duration::from_millis(DEFAULT_BATCH_MILLIS),
        output: None,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--batches" => {
                options.batches = parse_usize(args.next(), "--batches")?;
                if options.batches == 0 {
                    return Err(invalid_input("--batches must be greater than zero"));
                }
            }
            "--batch-ms" => {
                let millis = parse_usize(args.next(), "--batch-ms")?;
                if millis == 0 {
                    return Err(invalid_input("--batch-ms must be greater than zero"));
                }
                options.batch_duration = Duration::from_millis(millis as u64);
            }
            "--output" => {
                let path = args
                    .next()
                    .ok_or_else(|| invalid_input("missing value for --output"))?;
                options.output = Some(PathBuf::from(path));
            }
            _ => return Err(invalid_input(format!("unknown argument: {arg}"))),
        }
    }
    Ok(options)
}

fn hash_bytes(mut hash: u64, bytes: &[u8]) -> u64 {
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn hash_usize(hash: u64, value: usize) -> u64 {
    hash_bytes(hash, &value.to_le_bytes())
}

fn calibrate_iterations<F>(batch_duration: Duration, decode: &mut F) -> IoResult<usize>
where
    F: FnMut() -> IoResult<usize>,
{
    let mut iterations = 1usize;
    loop {
        let started = Instant::now();
        for _ in 0..iterations {
            black_box(decode()?);
        }
        let elapsed = started.elapsed();
        if elapsed >= batch_duration {
            return Ok(iterations);
        }
        let elapsed_nanos = elapsed.as_nanos().max(1);
        let target_nanos = batch_duration.as_nanos();
        let scale = target_nanos.div_ceil(elapsed_nanos) as usize;
        iterations = iterations.saturating_mul(scale.clamp(2, 16));
    }
}

fn measure<F>(options: &Options, metadata: CaseMetadata, mut decode: F) -> IoResult<Measurement>
where
    F: FnMut() -> IoResult<usize>,
{
    let iterations = calibrate_iterations(options.batch_duration, &mut decode)?;
    let mut samples = Vec::with_capacity(options.batches);
    for _ in 0..options.batches {
        let started = Instant::now();
        for _ in 0..iterations {
            black_box(decode()?);
        }
        samples.push(started.elapsed().as_nanos() as f64 / iterations as f64);
    }
    samples.sort_by(f64::total_cmp);
    let median_ns = samples[samples.len() / 2];
    let p95_index = ((samples.len() - 1) * 95).div_ceil(100);
    Ok(Measurement {
        case_name: metadata.case_name,
        format: metadata.format,
        width: metadata.width,
        height: metadata.height,
        pixels_per_decode: metadata.pixels_per_decode,
        output_hash: metadata.output_hash,
        median_ns,
        p95_ns: samples[p95_index],
    })
}

fn main() -> IoResult<()> {
    let options = parse_options()?;
    let lossy = include_bytes!("../samples/sample.webp");
    let lossless = include_bytes!("../samples/sample_lossless.webp");
    let animation = include_bytes!("../samples/sample_animation.webp");

    let lossy_rgba = decode_lossy_webp_to_rgba(lossy)
        .map_err(|error| IoError::new(ErrorKind::InvalidData, error))?;
    let lossy_rgba_hash = hash_bytes(FNV_OFFSET, &lossy_rgba.rgba);
    let lossy_rgba_width = lossy_rgba.width;
    let lossy_rgba_height = lossy_rgba.height;

    let lossy_yuv = decode_lossy_webp_to_yuv(lossy)
        .map_err(|error| IoError::new(ErrorKind::InvalidData, error))?;
    let mut lossy_yuv_hash = hash_bytes(FNV_OFFSET, &lossy_yuv.y);
    lossy_yuv_hash = hash_bytes(lossy_yuv_hash, &lossy_yuv.u);
    lossy_yuv_hash = hash_bytes(lossy_yuv_hash, &lossy_yuv.v);
    lossy_yuv_hash = hash_usize(lossy_yuv_hash, lossy_yuv.y_stride);
    lossy_yuv_hash = hash_usize(lossy_yuv_hash, lossy_yuv.uv_stride);

    let lossless_rgba = decode_lossless_webp_to_rgba(lossless)
        .map_err(|error| IoError::new(ErrorKind::InvalidData, error))?;
    let lossless_rgba_hash = hash_bytes(FNV_OFFSET, &lossless_rgba.rgba);
    let lossless_width = lossless_rgba.width;
    let lossless_height = lossless_rgba.height;

    let decoded_animation = decode_animation_webp(animation)
        .map_err(|error| IoError::new(ErrorKind::InvalidData, error))?;
    let mut animation_hash = hash_usize(FNV_OFFSET, decoded_animation.loop_count as usize);
    for frame in &decoded_animation.frames {
        animation_hash = hash_usize(animation_hash, frame.duration);
        animation_hash = hash_bytes(animation_hash, &frame.rgba);
    }
    let animation_width = decoded_animation.width;
    let animation_height = decoded_animation.height;
    let animation_pixels = animation_width * animation_height * decoded_animation.frames.len();

    let measurements = [
        measure(
            &options,
            CaseMetadata {
                case_name: "lossy_rgba",
                format: "VP8",
                width: lossy_rgba_width,
                height: lossy_rgba_height,
                pixels_per_decode: lossy_rgba_width * lossy_rgba_height,
                output_hash: lossy_rgba_hash,
            },
            || {
                let image = decode_lossy_webp_to_rgba(black_box(lossy))
                    .map_err(|error| IoError::new(ErrorKind::InvalidData, error))?;
                Ok(image.rgba.len())
            },
        )?,
        measure(
            &options,
            CaseMetadata {
                case_name: "lossy_yuv",
                format: "VP8",
                width: lossy_yuv.width,
                height: lossy_yuv.height,
                pixels_per_decode: lossy_yuv.width * lossy_yuv.height,
                output_hash: lossy_yuv_hash,
            },
            || {
                let image = decode_lossy_webp_to_yuv(black_box(lossy))
                    .map_err(|error| IoError::new(ErrorKind::InvalidData, error))?;
                Ok(image.y.len() + image.u.len() + image.v.len())
            },
        )?,
        measure(
            &options,
            CaseMetadata {
                case_name: "lossless_rgba",
                format: "VP8L",
                width: lossless_width,
                height: lossless_height,
                pixels_per_decode: lossless_width * lossless_height,
                output_hash: lossless_rgba_hash,
            },
            || {
                let image = decode_lossless_webp_to_rgba(black_box(lossless))
                    .map_err(|error| IoError::new(ErrorKind::InvalidData, error))?;
                Ok(image.rgba.len())
            },
        )?,
        measure(
            &options,
            CaseMetadata {
                case_name: "animation",
                format: "ANIM",
                width: animation_width,
                height: animation_height,
                pixels_per_decode: animation_pixels,
                output_hash: animation_hash,
            },
            || {
                let image = decode_animation_webp(black_box(animation))
                    .map_err(|error| IoError::new(ErrorKind::InvalidData, error))?;
                Ok(image.frames.len())
            },
        )?,
    ];

    let mut csv = String::from(
        "case,format,width,height,pixels_per_decode,output_hash,median_ns,p95_ns,mpixels_per_second\n",
    );
    for measurement in measurements {
        let mpixels_per_second =
            measurement.pixels_per_decode as f64 * 1_000.0 / measurement.median_ns;
        csv.push_str(&format!(
            "{},{},{},{},{},{:016x},{:.0},{:.0},{:.3}\n",
            measurement.case_name,
            measurement.format,
            measurement.width,
            measurement.height,
            measurement.pixels_per_decode,
            measurement.output_hash,
            measurement.median_ns,
            measurement.p95_ns,
            mpixels_per_second,
        ));
    }
    print!("{csv}");
    if let Some(path) = options.output {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, csv)?;
    }
    Ok(())
}
