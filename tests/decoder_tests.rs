use webp_rust::decoder::alpha::{decode_alpha_plane, parse_alpha_header};
use webp_rust::decoder::header::{get_features, parse_animation_webp, parse_still_webp};
use webp_rust::decoder::vp8::{
    parse_lossy_headers, parse_macroblock_data, parse_macroblock_headers,
};
use webp_rust::decoder::vp8i::{ALPHA_FLAG, ANIMATION_FLAG};
use webp_rust::decoder::WebpFormat;
use webp_rust::decoder::{
    decode_animation_webp, decode_lossless_vp8l_to_rgba, decode_lossless_webp_to_rgba,
    decode_lossy_vp8_to_rgba, decode_lossy_webp_to_rgba,
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

fn le24(value: usize) -> [u8; 3] {
    [
        (value & 0xff) as u8,
        ((value >> 8) & 0xff) as u8,
        ((value >> 16) & 0xff) as u8,
    ]
}

fn make_chunk(fourcc: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut chunk = Vec::with_capacity(8 + payload.len() + (payload.len() & 1));
    chunk.extend_from_slice(fourcc);
    chunk.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    chunk.extend_from_slice(payload);
    if payload.len() & 1 == 1 {
        chunk.push(0);
    }
    chunk
}

fn wrap_riff(chunks: &[Vec<u8>]) -> Vec<u8> {
    let riff_size = 4 + chunks.iter().map(Vec::len).sum::<usize>();
    let mut data = Vec::with_capacity(8 + riff_size);
    data.extend_from_slice(b"RIFF");
    data.extend_from_slice(&(riff_size as u32).to_le_bytes());
    data.extend_from_slice(b"WEBP");
    for chunk in chunks {
        data.extend_from_slice(chunk);
    }
    data
}

fn make_vp8x_payload(flags: u32, width: usize, height: usize) -> Vec<u8> {
    let mut payload = Vec::with_capacity(10);
    payload.extend_from_slice(&flags.to_le_bytes());
    payload.extend_from_slice(&le24(width - 1));
    payload.extend_from_slice(&le24(height - 1));
    payload
}

fn make_alpha_plane(width: usize, height: usize) -> Vec<u8> {
    let mut alpha = vec![0u8; width * height];
    for y in 0..height {
        for x in 0..width {
            alpha[y * width + x] = ((x * 13 + y * 7 + (x ^ y)) & 0xff) as u8;
        }
    }
    alpha
}

fn make_raw_alpha_chunk(alpha: &[u8]) -> Vec<u8> {
    let mut payload = Vec::with_capacity(1 + alpha.len());
    payload.push(0);
    payload.extend_from_slice(alpha);
    payload
}

fn make_lossy_alpha_still_webp(alpha: &[u8]) -> Vec<u8> {
    let sample = include_bytes!("../_testdata/sample.webp");
    let parsed = parse_still_webp(sample).unwrap();
    let vp8x = make_chunk(
        b"VP8X",
        &make_vp8x_payload(ALPHA_FLAG, parsed.features.width, parsed.features.height),
    );
    let alph = make_chunk(b"ALPH", &make_raw_alpha_chunk(alpha));
    let vp8 = make_chunk(b"VP8 ", parsed.image_data);
    wrap_riff(&[vp8x, alph, vp8])
}

fn make_lossy_alpha_animation_webp(alpha: &[u8]) -> Vec<u8> {
    let sample = include_bytes!("../_testdata/sample.webp");
    let parsed = parse_still_webp(sample).unwrap();

    let mut anmf_payload = Vec::new();
    anmf_payload.extend_from_slice(&le24(0));
    anmf_payload.extend_from_slice(&le24(0));
    anmf_payload.extend_from_slice(&le24(parsed.features.width - 1));
    anmf_payload.extend_from_slice(&le24(parsed.features.height - 1));
    anmf_payload.extend_from_slice(&le24(100));
    anmf_payload.push(0x02);
    anmf_payload.extend_from_slice(&make_chunk(b"ALPH", &make_raw_alpha_chunk(alpha)));
    anmf_payload.extend_from_slice(&make_chunk(b"VP8 ", parsed.image_data));

    let vp8x = make_chunk(
        b"VP8X",
        &make_vp8x_payload(
            ALPHA_FLAG | ANIMATION_FLAG,
            parsed.features.width,
            parsed.features.height,
        ),
    );
    let anim = make_chunk(b"ANIM", &[0, 0, 0, 0, 0, 0]);
    let anmf = make_chunk(b"ANMF", &anmf_payload);
    wrap_riff(&[vp8x, anim, anmf])
}

#[test]
fn get_features_parses_lossy_sample() {
    let data = include_bytes!("../_testdata/sample.webp");

    let features = get_features(data).unwrap();

    assert_eq!(features.width, 1152);
    assert_eq!(features.height, 896);
    assert_eq!(features.format, WebpFormat::Lossy);
    assert!(!features.has_alpha);
    assert!(!features.has_animation);
    assert!(features.vp8x.is_none());
}

#[test]
fn parse_still_webp_exposes_vp8_payload() {
    let data = include_bytes!("../_testdata/sample.webp");

    let parsed = parse_still_webp(data).unwrap();

    assert_eq!(parsed.image_chunk.size, 66_702);
    assert_eq!(parsed.image_data.len(), 66_702);
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
    assert_eq!(vp8.picture.width, 1152);
    assert_eq!(vp8.picture.height, 896);
    assert_eq!(vp8.macroblock_width, 72);
    assert_eq!(vp8.macroblock_height, 56);
    assert!(!vp8.token_partition_sizes.is_empty());
    assert!(vp8.token_partition_sizes.len() <= 8);
    assert!(vp8.quantization.indices.base_q0 > 0);
}

#[test]
fn parse_macroblock_headers_reads_all_lossy_macroblocks() {
    let data = include_bytes!("../_testdata/sample.webp");
    let parsed = parse_still_webp(data).unwrap();

    let frame = parse_macroblock_headers(parsed.image_data).unwrap();

    assert_eq!(frame.frame.macroblock_width, 72);
    assert_eq!(frame.frame.macroblock_height, 56);
    assert_eq!(frame.macroblocks.len(), 72 * 56);
    assert!(frame.macroblocks.iter().any(|mb| mb.is_i4x4));
    assert!(frame.macroblocks.iter().all(|mb| mb.uv_mode <= 3));
}

#[test]
fn parse_macroblock_data_reads_residual_coefficients() {
    let data = include_bytes!("../_testdata/sample.webp");
    let parsed = parse_still_webp(data).unwrap();

    let frame = parse_macroblock_data(parsed.image_data).unwrap();

    assert_eq!(frame.macroblocks.len(), 72 * 56);
    assert!(frame
        .macroblocks
        .iter()
        .any(|mb| mb.non_zero_y != 0 || mb.non_zero_uv != 0));
}

#[test]
fn decode_lossy_webp_to_rgba_matches_reference_pixels() {
    let data = include_bytes!("../_testdata/sample.webp");

    let image = decode_lossy_webp_to_rgba(data).unwrap();

    assert_eq!(image.width, 1152);
    assert_eq!(image.height, 896);
    assert_rgba_close(
        rgba_at(&image.rgba, image.width, 0, 0),
        [24, 65, 105, 255],
        0,
    );
    assert_rgba_close(
        rgba_at(&image.rgba, image.width, 576, 448),
        [189, 150, 154, 255],
        0,
    );
    assert_rgba_close(
        rgba_at(&image.rgba, image.width, 0, 895),
        [58, 59, 63, 255],
        0,
    );
    assert_rgba_close(
        rgba_at(&image.rgba, image.width, 123, 456),
        [27, 36, 49, 255],
        0,
    );
    assert_rgba_close(
        rgba_at(&image.rgba, image.width, 789, 321),
        [253, 191, 182, 255],
        0,
    );
    assert_rgba_close(
        rgba_at(&image.rgba, image.width, 1000, 100),
        [65, 55, 56, 255],
        0,
    );
    assert_rgba_close(
        rgba_at(&image.rgba, image.width, 42, 800),
        [34, 37, 45, 255],
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
fn decode_alpha_plane_extracts_green_channel_from_lossless_payload() {
    let data = include_bytes!("../_testdata/sample_lossless.webp");
    let parsed = parse_still_webp(data).unwrap();
    let image = decode_lossless_webp_to_rgba(data).unwrap();

    let mut alpha_data = Vec::with_capacity(1 + parsed.image_data.len());
    alpha_data.push(0x01);
    alpha_data.extend_from_slice(parsed.image_data);

    let alpha = decode_alpha_plane(&alpha_data, image.width, image.height).unwrap();
    let expected: Vec<u8> = image.rgba.chunks_exact(4).map(|pixel| pixel[1]).collect();

    assert_eq!(alpha, expected);
}

#[test]
fn decode_lossy_webp_to_rgba_applies_raw_alpha_chunk() {
    let base = decode_lossy_webp_to_rgba(include_bytes!("../_testdata/sample.webp")).unwrap();
    let alpha = make_alpha_plane(base.width, base.height);
    let webp = make_lossy_alpha_still_webp(&alpha);

    let image = decode_lossy_webp_to_rgba(&webp).unwrap();

    assert_eq!(image.width, base.width);
    assert_eq!(image.height, base.height);
    for &(x, y) in &[
        (0usize, 0usize),
        (123, 456),
        (base.width - 1, base.height - 1),
    ] {
        let expected_alpha = alpha[y * image.width + x];
        let actual = rgba_at(&image.rgba, image.width, x, y);
        let expected = rgba_at(&base.rgba, base.width, x, y);
        assert_eq!(actual[0..3], expected[0..3]);
        assert_eq!(actual[3], expected_alpha);
    }
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
fn decode_animation_webp_handles_lossy_alpha_frames() {
    let base = decode_lossy_webp_to_rgba(include_bytes!("../_testdata/sample.webp")).unwrap();
    let alpha = make_alpha_plane(base.width, base.height);
    let webp = make_lossy_alpha_animation_webp(&alpha);

    let animation = decode_animation_webp(&webp).unwrap();

    assert_eq!(animation.frames.len(), 1);
    for &(x, y) in &[
        (0usize, 0usize),
        (base.width / 2, base.height / 2),
        (base.width - 1, base.height - 1),
    ] {
        let expected_alpha = alpha[y * animation.width + x];
        let actual = rgba_at(&animation.frames[0].rgba, animation.width, x, y);
        let expected = rgba_at(&base.rgba, base.width, x, y);
        assert_eq!(actual[0..3], expected[0..3]);
        assert_eq!(actual[3], expected_alpha);
    }
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
