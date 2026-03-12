use webp_rust::decoder::{
    decode_lossless_vp8l_to_rgba, decode_lossy_vp8_to_rgba, get_features, parse_macroblock_headers,
    WebpFormat,
};
use webp_rust::encoder::{
    encode_lossless_image_to_webp, encode_lossless_rgba_to_vp8l, encode_lossless_rgba_to_webp,
    encode_lossless_rgba_to_webp_with_options, encode_lossy_image_to_webp,
    encode_lossy_rgba_to_vp8, encode_lossy_rgba_to_webp_with_options,
};
use webp_rust::{
    decode, encode, encode_lossless, encode_lossy, ImageBuffer, LosslessEncodingOptions,
    LossyEncodingOptions, WebpEncoding,
};

fn sample_rgba() -> (usize, usize, Vec<u8>) {
    let width = 3;
    let height = 2;
    let rgba = vec![
        0xff, 0x00, 0x00, 0xff, 0x00, 0xff, 0x00, 0x80, 0x00, 0x00, 0xff, 0x40, 0xff, 0xff, 0xff,
        0xff, 0x22, 0x44, 0x66, 0x00, 0x80, 0x20, 0xc0, 0xfe,
    ];
    (width, height, rgba)
}

fn lossy_sample_rgba() -> (usize, usize, Vec<u8>) {
    let width = 19;
    let height = 17;
    let mut rgba = vec![0u8; width * height * 4];
    for y in 0..height {
        for x in 0..width {
            let offset = (y * width + x) * 4;
            rgba[offset] = (x as u8).saturating_mul(12);
            rgba[offset + 1] = (y as u8).saturating_mul(13);
            rgba[offset + 2] = ((x + y) as u8).saturating_mul(7);
            rgba[offset + 3] = 0xff;
        }
    }
    (width, height, rgba)
}

fn average_abs_diff(a: &[u8], b: &[u8]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(lhs, rhs)| (*lhs as i32 - *rhs as i32).unsigned_abs() as f64)
        .sum::<f64>()
        / a.len() as f64
}

#[test]
fn encode_lossless_rgba_to_vp8l_round_trips_pixels() {
    let (width, height, rgba) = sample_rgba();

    let vp8l = encode_lossless_rgba_to_vp8l(width, height, &rgba).unwrap();
    let decoded = decode_lossless_vp8l_to_rgba(&vp8l).unwrap();

    assert_eq!(decoded.width, width);
    assert_eq!(decoded.height, height);
    assert_eq!(decoded.rgba, rgba);
}

#[test]
fn encode_lossless_rgba_to_webp_round_trips_pixels() {
    let (width, height, rgba) = sample_rgba();

    let webp = encode_lossless_rgba_to_webp(width, height, &rgba).unwrap();
    let decoded = decode(&webp).unwrap();

    assert_eq!(decoded.width, width);
    assert_eq!(decoded.height, height);
    assert_eq!(decoded.rgba, rgba);
}

#[test]
fn encode_lossless_rgba_to_webp_round_trips_pixels_at_optimization_level_zero() {
    let (width, height, rgba) = sample_rgba();
    let options = LosslessEncodingOptions {
        optimization_level: 0,
    };

    let webp = encode_lossless_rgba_to_webp_with_options(width, height, &rgba, &options).unwrap();
    let decoded = decode(&webp).unwrap();

    assert_eq!(decoded.width, width);
    assert_eq!(decoded.height, height);
    assert_eq!(decoded.rgba, rgba);
}

#[test]
fn encode_lossless_image_to_webp_sets_lossless_features() {
    let (width, height, rgba) = sample_rgba();
    let image = ImageBuffer {
        width,
        height,
        rgba: rgba.clone(),
    };

    let webp = encode_lossless_image_to_webp(&image).unwrap();
    let features = get_features(&webp).unwrap();

    assert_eq!(features.width, width);
    assert_eq!(features.height, height);
    assert_eq!(features.format, WebpFormat::Lossless);
    assert!(features.has_alpha);
}

#[test]
fn encode_lossless_rgba_to_webp_rejects_mismatched_buffer_length() {
    let error = encode_lossless_rgba_to_webp(2, 2, &[0; 15]).unwrap_err();

    assert_eq!(
        error,
        webp_rust::EncoderError::InvalidParam("RGBA buffer length does not match dimensions")
    );
}

#[test]
fn encode_lossless_rgba_to_webp_rejects_invalid_optimization_level() {
    let error = encode_lossless_rgba_to_webp_with_options(
        1,
        1,
        &[0, 0, 0, 0xff],
        &LosslessEncodingOptions {
            optimization_level: 3,
        },
    )
    .unwrap_err();

    assert_eq!(
        error,
        webp_rust::EncoderError::InvalidParam("lossless optimization level must be in 0..=2")
    );
}

#[test]
fn encode_lossless_rgba_to_webp_compresses_flat_runs() {
    let width = 64;
    let height = 64;
    let mut rgba = vec![0u8; width * height * 4];
    for pixel in rgba.chunks_exact_mut(4) {
        pixel.copy_from_slice(&[0x12, 0x34, 0x56, 0xff]);
    }

    let webp = encode_lossless_rgba_to_webp(width, height, &rgba).unwrap();
    let decoded = decode(&webp).unwrap();

    assert_eq!(decoded.rgba, rgba);
    assert!(
        webp.len() < 200,
        "unexpected flat-image size: {}",
        webp.len()
    );
}

#[test]
fn encode_lossless_higher_optimization_helps_repeated_tiles() {
    let width = 64;
    let height = 64;
    let mut rgba = vec![0u8; width * height * 4];
    for y in 0..height {
        for x in 0..width {
            let sx = (x % 8) as u8;
            let sy = (y % 8) as u8;
            let offset = (y * width + x) * 4;
            rgba[offset] = sx.saturating_mul(29);
            rgba[offset + 1] = sy.saturating_mul(31);
            rgba[offset + 2] = (sx ^ sy).saturating_mul(17);
            rgba[offset + 3] = 0xff;
        }
    }

    let opt0 = encode_lossless_rgba_to_webp_with_options(
        width,
        height,
        &rgba,
        &LosslessEncodingOptions {
            optimization_level: 0,
        },
    )
    .unwrap();
    let opt2 = encode_lossless_rgba_to_webp_with_options(
        width,
        height,
        &rgba,
        &LosslessEncodingOptions {
            optimization_level: 2,
        },
    )
    .unwrap();

    assert_eq!(decode(&opt0).unwrap().rgba, rgba);
    assert_eq!(decode(&opt2).unwrap().rgba, rgba);
    assert!(
        opt2.len() < opt0.len(),
        "expected opt2 to beat opt0 for repeated tiles: opt0={}, opt2={}",
        opt0.len(),
        opt2.len()
    );
}

#[test]
fn encode_lossless_palette_image_round_trips_and_compresses() {
    let width = 32;
    let height = 32;
    let colors = [
        [0x00, 0x00, 0x00, 0xff],
        [0xff, 0x00, 0x00, 0xff],
        [0x00, 0xff, 0x00, 0xff],
        [0x00, 0x00, 0xff, 0xff],
    ];
    let mut rgba = vec![0u8; width * height * 4];
    for y in 0..height {
        for x in 0..width {
            let color = colors[((x / 4) + (y / 4)) % colors.len()];
            let offset = (y * width + x) * 4;
            rgba[offset..offset + 4].copy_from_slice(&color);
        }
    }

    let vp8l = encode_lossless_rgba_to_vp8l(width, height, &rgba).unwrap();
    let decoded = decode_lossless_vp8l_to_rgba(&vp8l).unwrap();

    assert_eq!(decoded.rgba, rgba);
    assert!(vp8l.len() < rgba.len());
}

#[test]
fn encode_lossy_rgba_to_vp8_round_trips_as_lossy_frame() {
    let (width, height, rgba) = lossy_sample_rgba();
    let vp8 = encode_lossy_rgba_to_vp8(width, height, &rgba).unwrap();
    let decoded = decode_lossy_vp8_to_rgba(&vp8).unwrap();
    let diff = average_abs_diff(&decoded.rgba, &rgba);

    assert_eq!(decoded.width, width);
    assert_eq!(decoded.height, height);
    assert!(diff < 26.0, "avg diff: {diff}");
}

#[test]
fn encode_lossy_rgba_to_webp_sets_lossy_features() {
    let (width, height, rgba) = lossy_sample_rgba();
    let options = LossyEncodingOptions {
        quality: 90,
        ..LossyEncodingOptions::default()
    };
    let webp = encode_lossy_rgba_to_webp_with_options(width, height, &rgba, &options).unwrap();
    let decoded = decode(&webp).unwrap();
    let features = get_features(&webp).unwrap();
    let diff = average_abs_diff(&decoded.rgba, &rgba);

    assert_eq!(decoded.width, width);
    assert_eq!(decoded.height, height);
    assert!(diff < 26.0, "avg diff: {diff}");
    assert_eq!(features.format, WebpFormat::Lossy);
    assert!(!features.has_alpha);
}

#[test]
fn encode_lossy_rgba_to_vp8_marks_flat_macroblocks_as_skip() {
    let width = 64;
    let height = 64;
    let mut rgba = vec![0u8; width * height * 4];
    for pixel in rgba.chunks_exact_mut(4) {
        pixel.copy_from_slice(&[0x80, 0x80, 0x80, 0xff]);
    }

    let vp8 = encode_lossy_rgba_to_vp8(width, height, &rgba).unwrap();
    let headers = parse_macroblock_headers(&vp8).unwrap();

    assert!(
        headers.macroblocks.iter().any(|header| header.skip),
        "expected at least one skipped macroblock"
    );
}

#[test]
fn encode_lossy_image_to_webp_accepts_opaque_image_buffer() {
    let (width, height, rgba) = lossy_sample_rgba();
    let image = ImageBuffer {
        width,
        height,
        rgba,
    };
    let webp = encode_lossy_image_to_webp(&image).unwrap();
    let features = get_features(&webp).unwrap();

    assert_eq!(features.width, width);
    assert_eq!(features.height, height);
    assert_eq!(features.format, WebpFormat::Lossy);
}

#[test]
fn encode_lossy_rgba_to_webp_rejects_alpha_input() {
    let rgba = [0u8, 0, 0, 0x7f];
    let error =
        encode_lossy_rgba_to_webp_with_options(1, 1, &rgba, &LossyEncodingOptions::default())
            .unwrap_err();

    assert_eq!(
        error,
        webp_rust::EncoderError::InvalidParam("lossy encoder does not support alpha yet")
    );
}

#[test]
fn encode_lossy_rgba_to_webp_rejects_invalid_quality() {
    let error = encode_lossy_rgba_to_webp_with_options(
        1,
        1,
        &[0, 0, 0, 0xff],
        &LossyEncodingOptions {
            quality: 101,
            ..LossyEncodingOptions::default()
        },
    )
    .unwrap_err();

    assert_eq!(
        error,
        webp_rust::EncoderError::InvalidParam("lossy quality must be in 0..=100")
    );
}

#[test]
fn encode_lossy_rgba_to_webp_rejects_invalid_optimization_level() {
    let error = encode_lossy_rgba_to_webp_with_options(
        1,
        1,
        &[0, 0, 0, 0xff],
        &LossyEncodingOptions {
            optimization_level: 10,
            ..LossyEncodingOptions::default()
        },
    )
    .unwrap_err();

    assert_eq!(
        error,
        webp_rust::EncoderError::InvalidParam("lossy optimization level must be in 0..=9")
    );
}

#[test]
fn top_level_encode_lossless_round_trips() {
    let (width, height, rgba) = sample_rgba();
    let image = ImageBuffer {
        width,
        height,
        rgba: rgba.clone(),
    };

    let webp = encode(&image, 2, 100, WebpEncoding::Lossless, None).unwrap();
    let decoded = decode(&webp).unwrap();
    let features = get_features(&webp).unwrap();

    assert_eq!(decoded.rgba, rgba);
    assert_eq!(features.format, WebpFormat::Lossless);
}

#[test]
fn top_level_encode_lossy_uses_requested_compression() {
    let (width, height, rgba) = lossy_sample_rgba();
    let image = ImageBuffer {
        width,
        height,
        rgba: rgba.clone(),
    };

    let webp = encode(&image, 0, 90, WebpEncoding::Lossy, None).unwrap();
    let decoded = decode(&webp).unwrap();
    let features = get_features(&webp).unwrap();
    let diff = average_abs_diff(&decoded.rgba, &rgba);

    assert_eq!(features.format, WebpFormat::Lossy);
    assert!(diff < 26.0, "avg diff: {diff}");
}

#[test]
fn top_level_encode_variants_embed_exif_chunk() {
    let (width, height, rgba) = sample_rgba();
    let image = ImageBuffer {
        width,
        height,
        rgba,
    };
    let exif = b"Exif\0\0unit-test";

    let lossless = encode_lossless(&image, 2, Some(exif)).unwrap();
    let lossy = encode_lossy(
        &ImageBuffer {
            width: image.width,
            height: image.height,
            rgba: image
                .rgba
                .iter()
                .enumerate()
                .map(|(i, v)| if i % 4 == 3 { 0xff } else { *v })
                .collect(),
        },
        0,
        90,
        Some(exif),
    )
    .unwrap();

    assert!(lossless.windows(4).any(|chunk| chunk == b"EXIF"));
    assert!(lossless.windows(exif.len()).any(|chunk| chunk == exif));
    assert!(lossy.windows(4).any(|chunk| chunk == b"EXIF"));
    assert!(lossy.windows(exif.len()).any(|chunk| chunk == exif));
}
