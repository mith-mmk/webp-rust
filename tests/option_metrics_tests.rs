use webp_rust::decode;
use webp_rust::encoder::{
    encode_lossless_rgba_to_webp_with_config, encode_lossy_rgba_to_webp_with_config, AlphaFilter,
    LosslessEncodingConfig, LossyEncodingConfig, WebpPreset,
};

const WIDTH: usize = 64;
const HEIGHT: usize = 48;

struct TestImage {
    name: &'static str,
    rgba: Vec<u8>,
    opaque: bool,
}

struct LossyCase {
    label: String,
    config: LossyEncodingConfig,
}

#[derive(Debug, Clone, Copy)]
struct CompressionMetrics {
    ratio: f64,
    saving_percent: f64,
}

fn generated_images() -> Vec<TestImage> {
    vec![
        TestImage {
            name: "gradient",
            rgba: gradient_image(),
            opaque: true,
        },
        TestImage {
            name: "palette",
            rgba: palette_image(),
            opaque: true,
        },
        TestImage {
            name: "photo_like",
            rgba: photo_like_image(),
            opaque: true,
        },
        TestImage {
            name: "alpha_graphic",
            rgba: alpha_graphic_image(),
            opaque: false,
        },
    ]
}

fn gradient_image() -> Vec<u8> {
    let mut rgba = vec![0; WIDTH * HEIGHT * 4];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let offset = (y * WIDTH + x) * 4;
            rgba[offset] = (x * 255 / (WIDTH - 1)) as u8;
            rgba[offset + 1] = (y * 255 / (HEIGHT - 1)) as u8;
            rgba[offset + 2] = ((x + y) * 255 / (WIDTH + HEIGHT - 2)) as u8;
            rgba[offset + 3] = 0xff;
        }
    }
    rgba
}

fn palette_image() -> Vec<u8> {
    const COLORS: [[u8; 3]; 8] = [
        [20, 24, 32],
        [235, 240, 245],
        [210, 50, 60],
        [40, 155, 85],
        [45, 105, 220],
        [245, 185, 45],
        [130, 70, 180],
        [30, 185, 200],
    ];

    let mut rgba = vec![0; WIDTH * HEIGHT * 4];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let color = COLORS[((x / 8) + (y / 6) * 3) % COLORS.len()];
            let offset = (y * WIDTH + x) * 4;
            rgba[offset..offset + 3].copy_from_slice(&color);
            rgba[offset + 3] = 0xff;
        }
    }
    rgba
}

fn photo_like_image() -> Vec<u8> {
    let mut rgba = vec![0; WIDTH * HEIGHT * 4];
    let mut state = 0x8f3a_21c5_u32;
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let noise = ((state >> 24) & 0x3f) as usize;
            let offset = (y * WIDTH + x) * 4;
            rgba[offset] = ((x * 3 + y * 2 + noise) & 0xff) as u8;
            rgba[offset + 1] = ((x * 2 + y * 4 + noise / 2 + 35) & 0xff) as u8;
            rgba[offset + 2] = ((x + y * 3 + noise / 3 + 70) & 0xff) as u8;
            rgba[offset + 3] = 0xff;
        }
    }
    rgba
}

fn alpha_graphic_image() -> Vec<u8> {
    let mut rgba = vec![0; WIDTH * HEIGHT * 4];
    let center_x = WIDTH as isize / 2;
    let center_y = HEIGHT as isize / 2;
    let max_distance = (center_x * center_x + center_y * center_y) as usize;
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let dx = x as isize - center_x;
            let dy = y as isize - center_y;
            let distance = (dx * dx + dy * dy) as usize;
            let alpha = 255usize.saturating_sub(distance * 255 / max_distance);
            let offset = (y * WIDTH + x) * 4;
            rgba[offset] = ((x / 8) * 31) as u8;
            rgba[offset + 1] = ((y / 6) * 29) as u8;
            rgba[offset + 2] = (((x + y) / 7) * 23) as u8;
            rgba[offset + 3] = alpha as u8;
        }
    }
    rgba
}

fn compression_metrics(raw_size: usize, encoded_size: usize) -> CompressionMetrics {
    CompressionMetrics {
        ratio: raw_size as f64 / encoded_size as f64,
        saving_percent: (1.0 - encoded_size as f64 / raw_size as f64) * 100.0,
    }
}

fn average(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn psnr_for_channels(source: &[u8], decoded: &[u8], channels: &[usize]) -> f64 {
    let mut squared_error = 0_u64;
    let mut sample_count = 0_u64;
    for (source_pixel, decoded_pixel) in source.chunks_exact(4).zip(decoded.chunks_exact(4)) {
        for &channel in channels {
            let difference = source_pixel[channel] as i32 - decoded_pixel[channel] as i32;
            squared_error += (difference * difference) as u64;
            sample_count += 1;
        }
    }
    if squared_error == 0 {
        return f64::INFINITY;
    }
    let mse = squared_error as f64 / sample_count as f64;
    10.0 * ((255.0 * 255.0) / mse).log10()
}

fn lossy_case(
    label: impl Into<String>,
    configure: impl FnOnce(&mut LossyEncodingConfig),
) -> LossyCase {
    let mut config = LossyEncodingConfig::default();
    configure(&mut config);
    LossyCase {
        label: label.into(),
        config,
    }
}

fn lossy_cases() -> Vec<LossyCase> {
    let mut cases = vec![lossy_case("default", |_| {})];

    for quality in [0.0, 25.0, 50.0, 75.0, 90.0, 100.0] {
        cases.push(lossy_case(format!("quality={quality:.0}"), |config| {
            config.quality = quality;
        }));
    }
    for method in 0..=6 {
        cases.push(lossy_case(format!("method={method}"), |config| {
            config.method = method;
        }));
    }
    for segments in 1..=4 {
        cases.push(lossy_case(format!("segments={segments}"), |config| {
            config.segments = segments;
        }));
    }
    for strength in [0, 50, 100] {
        cases.push(lossy_case(format!("sns_strength={strength}"), |config| {
            config.sns_strength = strength;
        }));
        cases.push(lossy_case(
            format!("filter_strength={strength}"),
            |config| {
                config.filter_strength = strength;
            },
        ));
    }
    for sharpness in [0, 3, 7] {
        cases.push(lossy_case(
            format!("filter_sharpness={sharpness}"),
            |config| {
                config.filter_sharpness = sharpness;
            },
        ));
    }
    for enabled in [false, true] {
        cases.push(lossy_case(format!("strong_filter={enabled}"), |config| {
            config.strong_filter = enabled;
        }));
        cases.push(lossy_case(format!("sharp_yuv={enabled}"), |config| {
            config.sharp_yuv = enabled;
        }));
        cases.push(lossy_case(format!("exact={enabled}"), |config| {
            config.exact = enabled;
        }));
        cases.push(lossy_case(format!("no_alpha={enabled}"), |config| {
            config.no_alpha = enabled;
        }));
    }
    for limit in [0, 50, 100] {
        cases.push(lossy_case(format!("partition_limit={limit}"), |config| {
            config.partition_limit = limit;
        }));
    }
    for quality in [0, 50, 100] {
        cases.push(lossy_case(format!("alpha_quality={quality}"), |config| {
            config.alpha_quality = quality;
        }));
    }
    for method in [0, 1] {
        cases.push(lossy_case(format!("alpha_method={method}"), |config| {
            config.alpha_method = method;
        }));
    }
    for (name, filter) in [
        ("none", AlphaFilter::None),
        ("fast", AlphaFilter::Fast),
        ("best", AlphaFilter::Best),
    ] {
        cases.push(lossy_case(format!("alpha_filter={name}"), |config| {
            config.alpha_filter = filter;
        }));
    }
    for near_lossless in [0, 50, 100] {
        cases.push(lossy_case(
            format!("near_lossless={near_lossless}"),
            |config| {
                config.near_lossless = near_lossless;
            },
        ));
    }
    for (name, preset) in [
        ("default", WebpPreset::Default),
        ("photo", WebpPreset::Photo),
        ("picture", WebpPreset::Picture),
        ("drawing", WebpPreset::Drawing),
        ("icon", WebpPreset::Icon),
        ("text", WebpPreset::Text),
    ] {
        cases.push(LossyCase {
            label: format!("preset={name}"),
            config: LossyEncodingConfig::preset(preset),
        });
    }

    cases
}

#[test]
fn lossless_options_report_average_compression_and_exact_fidelity() {
    let images = generated_images();
    eprintln!(
        "lossless option metrics (average across {} images)",
        images.len()
    );
    eprintln!(
        "{:<12} {:>12} {:>12} {:>12} {:>12}",
        "option", "avg bytes", "raw/webp", "saving %", "exact"
    );

    for z_level in 0..=9 {
        let mut total_raw_size = 0;
        let mut total_encoded_size = 0;
        for image in &images {
            let encoded = encode_lossless_rgba_to_webp_with_config(
                WIDTH,
                HEIGHT,
                &image.rgba,
                &LosslessEncodingConfig { z_level },
            )
            .unwrap_or_else(|error| panic!("z_level={z_level}, {}: {error}", image.name));
            let decoded = decode(&encoded)
                .unwrap_or_else(|error| panic!("z_level={z_level}, {}: {error}", image.name));
            assert_eq!((decoded.width, decoded.height), (WIDTH, HEIGHT));
            assert_eq!(
                decoded.rgba, image.rgba,
                "z_level={z_level}, {}",
                image.name
            );

            total_raw_size += image.rgba.len();
            total_encoded_size += encoded.len();
        }
        let metrics = compression_metrics(total_raw_size, total_encoded_size);
        eprintln!(
            "{:<12} {:>12.1} {:>12.3} {:>12.2} {:>12}",
            format!("z_level={z_level}"),
            total_encoded_size as f64 / images.len() as f64,
            metrics.ratio,
            metrics.saving_percent,
            "yes"
        );
    }
}

#[test]
fn lossy_options_report_average_compression_and_fidelity() {
    let images = generated_images();
    let cases = lossy_cases();
    let mut quality_zero_psnr = None;
    let mut quality_hundred_psnr = None;

    eprintln!(
        "lossy option metrics (average across {} images)",
        images.len()
    );
    eprintln!(
        "{:<28} {:>10} {:>10} {:>10} {:>12} {:>12}",
        "option", "avg bytes", "raw/webp", "saving %", "RGB PSNR", "alpha PSNR"
    );

    for case in cases {
        let mut total_raw_size = 0;
        let mut total_encoded_size = 0;
        let mut color_psnr = Vec::new();
        let mut alpha_psnr = None;

        for image in &images {
            let encoded =
                encode_lossy_rgba_to_webp_with_config(WIDTH, HEIGHT, &image.rgba, &case.config)
                    .unwrap_or_else(|error| panic!("{}, {}: {error}", case.label, image.name));
            let decoded = decode(&encoded)
                .unwrap_or_else(|error| panic!("{}, {}: {error}", case.label, image.name));
            assert_eq!((decoded.width, decoded.height), (WIDTH, HEIGHT));

            total_raw_size += image.rgba.len();
            total_encoded_size += encoded.len();

            if image.opaque {
                color_psnr.push(psnr_for_channels(&image.rgba, &decoded.rgba, &[0, 1, 2]));
            } else {
                alpha_psnr = Some(psnr_for_channels(&image.rgba, &decoded.rgba, &[3]));
                if case.config.no_alpha {
                    assert!(decoded.rgba.chunks_exact(4).all(|pixel| pixel[3] == 0xff));
                } else if case.config.alpha_quality == 100 {
                    assert_eq!(
                        decoded
                            .rgba
                            .chunks_exact(4)
                            .map(|pixel| pixel[3])
                            .collect::<Vec<_>>(),
                        image
                            .rgba
                            .chunks_exact(4)
                            .map(|pixel| pixel[3])
                            .collect::<Vec<_>>(),
                        "{}, alpha fidelity",
                        case.label
                    );
                }
            }
        }

        let average_color_psnr = average(&color_psnr);
        let minimum_psnr = if case.config.quality >= 75.0 {
            14.0
        } else if case.config.quality >= 50.0 {
            12.0
        } else {
            8.0
        };
        assert!(
            average_color_psnr >= minimum_psnr,
            "{}: average RGB PSNR {:.2} dB is below {:.2} dB",
            case.label,
            average_color_psnr,
            minimum_psnr
        );

        if case.label == "quality=0" {
            quality_zero_psnr = Some(average_color_psnr);
        } else if case.label == "quality=100" {
            quality_hundred_psnr = Some(average_color_psnr);
        }

        let metrics = compression_metrics(total_raw_size, total_encoded_size);
        eprintln!(
            "{:<28} {:>10.1} {:>10.3} {:>10.2} {:>12.2} {:>12.2}",
            case.label,
            total_encoded_size as f64 / images.len() as f64,
            metrics.ratio,
            metrics.saving_percent,
            average_color_psnr,
            alpha_psnr.expect("alpha test image must be present")
        );
    }

    assert!(
        quality_hundred_psnr.expect("quality=100 metrics")
            > quality_zero_psnr.expect("quality=0 metrics"),
        "quality=100 must reproduce RGB more faithfully than quality=0"
    );
}
