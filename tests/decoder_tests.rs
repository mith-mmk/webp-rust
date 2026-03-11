use webp_rust::decoder::alpha::parse_alpha_header;
use webp_rust::decoder::header::{get_features, parse_animation_webp, parse_still_webp};
use webp_rust::decoder::vp8::{
    parse_lossy_headers, parse_macroblock_data, parse_macroblock_headers,
};
use webp_rust::decoder::WebpFormat;
use webp_rust::decoder::{
    decode_animation_webp, decode_lossless_vp8l_to_rgba, decode_lossless_webp_to_rgba,
    decode_lossy_vp8_to_rgba, decode_lossy_webp_to_bmp, decode_lossy_webp_to_rgba,
};

fn rgba_at(rgba: &[u8], width: usize, x: usize, y: usize) -> [u8; 4] {
    let offset = (y * width + x) * 4;
    rgba[offset..offset + 4].try_into().unwrap()
}

fn assert_rgba_close(actual: [u8; 4], expected: [u8; 4], tolerance: u8) {
    for i in 0..3 {
        let diff = actual[i].abs_diff(expected[i]);
        assert!(
            diff <= tolerance,
            "channel {i} differs too much: actual={}, expected={}, tolerance={tolerance}",
            actual[i],
            expected[i]
        );
    }
    assert_eq!(actual[3], expected[3]);
}

#[test]
fn get_features_parses_lossy_sample() {
    let data = include_bytes!("../_testdata/sample.webp");

    let features = get_features(data).unwrap();

    assert_eq!(features.width, 1920);
    assert_eq!(features.height, 1080);
    assert_eq!(features.format, WebpFormat::Lossy);
    assert!(!features.has_alpha);
    assert!(!features.has_animation);
    assert!(features.vp8x.is_none());
}

#[test]
fn parse_still_webp_exposes_vp8_payload() {
    let data = include_bytes!("../_testdata/sample.webp");

    let parsed = parse_still_webp(data).unwrap();

    assert_eq!(parsed.image_chunk.size, 15_226);
    assert_eq!(parsed.image_data.len(), 15_226);
    assert!(parsed.alpha_chunk.is_none());
    assert!(parsed.alpha_data.is_none());
}

#[test]
fn parse_lossy_headers_reads_sample_partition_headers() {
    let data = include_bytes!("../_testdata/sample.webp");
    let parsed = parse_still_webp(data).unwrap();

    let vp8 = parse_lossy_headers(parsed.image_data).unwrap();

    assert!(vp8.frame.key_frame);
    assert!(vp8.frame.show);
    assert_eq!(vp8.picture.width, 1920);
    assert_eq!(vp8.picture.height, 1080);
    assert_eq!(vp8.macroblock_width, 120);
    assert_eq!(vp8.macroblock_height, 68);
    assert!(!vp8.token_partition_sizes.is_empty());
    assert!(vp8.token_partition_sizes.len() <= 8);
    assert!(vp8.quantization.indices.base_q0 > 0);
}

#[test]
fn parse_macroblock_headers_reads_all_lossy_macroblocks() {
    let data = include_bytes!("../_testdata/sample.webp");
    let parsed = parse_still_webp(data).unwrap();

    let frame = parse_macroblock_headers(parsed.image_data).unwrap();

    assert_eq!(frame.frame.macroblock_width, 120);
    assert_eq!(frame.frame.macroblock_height, 68);
    assert_eq!(frame.macroblocks.len(), 120 * 68);
    assert!(frame.macroblocks.iter().any(|mb| mb.is_i4x4));
    assert!(frame.macroblocks.iter().all(|mb| mb.uv_mode <= 3));
}

#[test]
fn parse_macroblock_data_reads_residual_coefficients() {
    let data = include_bytes!("../_testdata/sample.webp");
    let parsed = parse_still_webp(data).unwrap();

    let frame = parse_macroblock_data(parsed.image_data).unwrap();

    assert_eq!(frame.macroblocks.len(), 120 * 68);
    assert!(frame
        .macroblocks
        .iter()
        .any(|mb| mb.non_zero_y != 0 || mb.non_zero_uv != 0));
}

#[test]
fn decode_lossy_webp_to_rgba_matches_reference_pixels() {
    let data = include_bytes!("../_testdata/sample.webp");

    let image = decode_lossy_webp_to_rgba(data).unwrap();

    assert_eq!(image.width, 1920);
    assert_eq!(image.height, 1080);
    assert_rgba_close(
        rgba_at(&image.rgba, image.width, 0, 0),
        [177, 147, 189, 255],
        0,
    );
    assert_rgba_close(
        rgba_at(&image.rgba, image.width, 960, 540),
        [254, 169, 161, 255],
        0,
    );
    assert_rgba_close(
        rgba_at(&image.rgba, image.width, 0, 1079),
        [253, 190, 2, 255],
        0,
    );
    assert_rgba_close(
        rgba_at(&image.rgba, image.width, 123, 456),
        [243, 179, 167, 255],
        0,
    );
    assert_rgba_close(
        rgba_at(&image.rgba, image.width, 789, 321),
        [222, 167, 181, 255],
        0,
    );
    assert_rgba_close(
        rgba_at(&image.rgba, image.width, 1000, 100),
        [183, 151, 196, 255],
        0,
    );
    assert_rgba_close(
        rgba_at(&image.rgba, image.width, 42, 900),
        [254, 193, 4, 255],
        0,
    );
}

#[test]
fn decode_lossy_vp8_to_rgba_matches_container_decode() {
    let data = include_bytes!("../_testdata/sample.webp");
    let parsed = parse_still_webp(data).unwrap();

    let from_container = decode_lossy_webp_to_rgba(data).unwrap();
    let from_vp8 = decode_lossy_vp8_to_rgba(parsed.image_data).unwrap();

    assert_eq!(from_vp8, from_container);
}

#[test]
fn decode_lossy_webp_to_bmp_matches_reference_file() {
    let data = include_bytes!("../_testdata/sample.webp");
    let bmp = decode_lossy_webp_to_bmp(data).unwrap();
    let expected = include_bytes!("../_testdata/sample-right.bmp");

    assert_eq!(bmp, expected);
}

#[test]
fn get_features_parses_minimal_lossless_webp() {
    let data = [
        b'R', b'I', b'F', b'F', 18, 0, 0, 0, b'W', b'E', b'B', b'P', b'V', b'P', b'8', b'L', 5, 0,
        0, 0, 0x2f, 0x00, 0x00, 0x00, 0x10, 0x00,
    ];

    let features = get_features(&data).unwrap();

    assert_eq!(features.width, 1);
    assert_eq!(features.height, 1);
    assert_eq!(features.format, WebpFormat::Lossless);
    assert!(features.has_alpha);
    assert!(!features.has_animation);
}

#[test]
fn decode_lossless_webp_to_rgba_matches_reference_pixels() {
    let data = include_bytes!("../_testdata/sample_lossless.webp");

    let image = decode_lossless_webp_to_rgba(data).unwrap();

    assert_eq!(image.width, 1152);
    assert_eq!(image.height, 896);
    assert_rgba_close(
        rgba_at(&image.rgba, image.width, 0, 0),
        [23, 65, 103, 255],
        0,
    );
    assert_rgba_close(
        rgba_at(&image.rgba, image.width, 576, 448),
        [197, 156, 160, 255],
        0,
    );
    assert_rgba_close(
        rgba_at(&image.rgba, image.width, 1151, 895),
        [243, 183, 110, 255],
        0,
    );
    assert_rgba_close(
        rgba_at(&image.rgba, image.width, 123, 456),
        [30, 37, 53, 255],
        0,
    );
    assert_rgba_close(
        rgba_at(&image.rgba, image.width, 789, 321),
        [252, 192, 181, 255],
        0,
    );
    assert_rgba_close(
        rgba_at(&image.rgba, image.width, 42, 800),
        [35, 35, 43, 255],
        0,
    );
    assert_rgba_close(
        rgba_at(&image.rgba, image.width, 1000, 100),
        [65, 56, 59, 255],
        0,
    );
}

#[test]
fn decode_lossless_vp8l_to_rgba_matches_container_decode() {
    let data = include_bytes!("../_testdata/sample_lossless.webp");
    let parsed = parse_still_webp(data).unwrap();

    let from_container = decode_lossless_webp_to_rgba(data).unwrap();
    let from_vp8l = decode_lossless_vp8l_to_rgba(parsed.image_data).unwrap();

    assert_eq!(from_vp8l, from_container);
}

#[test]
fn parse_animation_webp_reads_sample_animation_metadata() {
    let data = include_bytes!("../_testdata/sample_animation.webp");

    let parsed = parse_animation_webp(data).unwrap();

    assert_eq!(parsed.features.width, 1200);
    assert_eq!(parsed.features.height, 1200);
    assert!(parsed.features.has_alpha);
    assert!(parsed.features.has_animation);
    assert_eq!(parsed.animation.background_color, 0xffb5_eef8);
    assert_eq!(parsed.animation.loop_count, 0);
    assert_eq!(parsed.frames.len(), 7);
    assert_eq!(parsed.frames[0].width, 1200);
    assert_eq!(parsed.frames[0].height, 1200);
    assert_eq!(parsed.frames[0].x_offset, 0);
    assert_eq!(parsed.frames[0].y_offset, 0);
    assert!(!parsed.frames[0].blend);
    assert!(!parsed.frames[0].dispose_to_background);
    assert_eq!(parsed.frames[1].x_offset, 428);
    assert_eq!(parsed.frames[1].y_offset, 600);
    assert_eq!(parsed.frames[1].width, 313);
    assert_eq!(parsed.frames[1].height, 280);
    assert!(parsed.frames[1].blend);
    assert!(!parsed.frames[1].dispose_to_background);
}

#[test]
fn decode_animation_webp_matches_reference_pixels() {
    let data = include_bytes!("../_testdata/sample_animation.webp");

    let animation = decode_animation_webp(data).unwrap();

    assert_eq!(animation.width, 1200);
    assert_eq!(animation.height, 1200);
    assert_eq!(animation.loop_count, 0);
    assert_eq!(animation.background_color, 0xffb5_eef8);
    assert_eq!(animation.frames.len(), 7);

    assert_rgba_close(
        rgba_at(&animation.frames[0].rgba, animation.width, 556, 601),
        [243, 222, 195, 255],
        0,
    );
    assert_rgba_close(
        rgba_at(&animation.frames[1].rgba, animation.width, 556, 601),
        [201, 195, 169, 255],
        0,
    );
    assert_rgba_close(
        rgba_at(&animation.frames[2].rgba, animation.width, 250, 73),
        [199, 247, 251, 255],
        0,
    );
    assert_rgba_close(
        rgba_at(&animation.frames[3].rgba, animation.width, 250, 73),
        [200, 244, 248, 255],
        0,
    );
    assert_rgba_close(
        rgba_at(&animation.frames[4].rgba, animation.width, 668, 526),
        [3, 0, 0, 255],
        0,
    );
    assert_rgba_close(
        rgba_at(&animation.frames[5].rgba, animation.width, 736, 739),
        [3, 0, 0, 255],
        0,
    );
    assert_rgba_close(
        rgba_at(&animation.frames[6].rgba, animation.width, 568, 651),
        [243, 222, 195, 255],
        0,
    );
}

#[test]
fn get_features_parses_animated_vp8x_header_without_frames() {
    let data = [
        b'R', b'I', b'F', b'F', 22, 0, 0, 0, b'W', b'E', b'B', b'P', b'V', b'P', b'8', b'X', 10, 0,
        0, 0, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x03, 0x00, 0x00,
    ];

    let features = get_features(&data).unwrap();

    assert_eq!(features.width, 3);
    assert_eq!(features.height, 4);
    assert_eq!(features.format, WebpFormat::Undefined);
    assert!(features.has_animation);
}

#[test]
fn parse_alpha_header_decodes_fields() {
    let header = parse_alpha_header(&[0b0001_1001]).unwrap();

    assert_eq!(header.compression, 0b01);
    assert_eq!(header.filter, 0b10);
    assert_eq!(header.preprocessing, 0b01);
}
