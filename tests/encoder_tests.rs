use webp_rust::decoder::{decode_lossless_vp8l_to_rgba, get_features, WebpFormat};
use webp_rust::{
    encode_lossless_image_to_webp, encode_lossless_rgba_to_vp8l, encode_lossless_rgba_to_webp,
    image_from_bytes, ImageBuffer,
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
    let decoded = image_from_bytes(&webp).unwrap();

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
