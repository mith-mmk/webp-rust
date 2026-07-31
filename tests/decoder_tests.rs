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
use webp_rust::encoder::encode_lossless_rgba_to_vp8l;

fn rgba_at(rgba: &[u8], width: usize, x: usize, y: usize) -> [u8; 4] {
    let offset = (y * width + x) * 4;
    rgba[offset..offset + 4].try_into().unwrap()
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
    let sample = include_bytes!("../samples/sample.webp");
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
    let sample = include_bytes!("../samples/sample.webp");
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
    let data = include_bytes!("../samples/sample.webp");

    let features = get_features(data).unwrap();

    assert!(features.width > 0);
    assert!(features.height > 0);
    assert_eq!(features.format, WebpFormat::Lossy);
    assert!(!features.has_alpha);
    assert!(!features.has_animation);
    assert!(features.vp8x.is_none());
}

#[test]
fn parse_still_webp_exposes_vp8_payload() {
    let data = include_bytes!("../samples/sample.webp");

    let parsed = parse_still_webp(data).unwrap();

    assert_eq!(parsed.image_chunk.size, parsed.image_data.len());
    assert!(parsed.image_chunk.size > 0);
    assert!(parsed.alpha_chunk.is_none());
    assert!(parsed.alpha_data.is_none());
}

#[test]
fn parse_lossy_headers_reads_sample_partition_headers() {
    let data = include_bytes!("../samples/sample.webp");
    let parsed = parse_still_webp(data).unwrap();

    let vp8 = parse_lossy_headers(parsed.image_data).unwrap();

    assert!(vp8.frame.key_frame);
    assert!(vp8.frame.show);
    assert_eq!(usize::from(vp8.picture.width), parsed.features.width);
    assert_eq!(usize::from(vp8.picture.height), parsed.features.height);
    assert_eq!(vp8.macroblock_width, parsed.features.width.div_ceil(16));
    assert_eq!(vp8.macroblock_height, parsed.features.height.div_ceil(16));
    assert!(!vp8.token_partition_sizes.is_empty());
    assert!(vp8.token_partition_sizes.len() <= 8);
    assert!(vp8.quantization.indices.base_q0 > 0);
}

#[test]
fn parse_macroblock_headers_reads_all_lossy_macroblocks() {
    let data = include_bytes!("../samples/sample.webp");
    let parsed = parse_still_webp(data).unwrap();

    let frame = parse_macroblock_headers(parsed.image_data).unwrap();

    assert_eq!(
        frame.frame.macroblock_width,
        parsed.features.width.div_ceil(16)
    );
    assert_eq!(
        frame.frame.macroblock_height,
        parsed.features.height.div_ceil(16)
    );
    assert_eq!(
        frame.macroblocks.len(),
        frame.frame.macroblock_width * frame.frame.macroblock_height
    );
    assert!(frame.macroblocks.iter().any(|mb| mb.is_i4x4));
    assert!(frame.macroblocks.iter().all(|mb| mb.uv_mode <= 3));
}

#[test]
fn parse_macroblock_data_reads_residual_coefficients() {
    let data = include_bytes!("../samples/sample.webp");
    let parsed = parse_still_webp(data).unwrap();

    let frame = parse_macroblock_data(parsed.image_data).unwrap();

    assert_eq!(
        frame.macroblocks.len(),
        parsed.features.width.div_ceil(16) * parsed.features.height.div_ceil(16)
    );
    assert!(frame
        .macroblocks
        .iter()
        .any(|mb| mb.non_zero_y != 0 || mb.non_zero_uv != 0));
}

#[test]
fn decode_lossy_webp_to_rgba_matches_reference_pixels() {
    let data = include_bytes!("../samples/sample.webp");

    let image = decode_lossy_webp_to_rgba(data).unwrap();

    let features = get_features(data).unwrap();
    assert_eq!(image.width, features.width);
    assert_eq!(image.height, features.height);
    assert_eq!(image.rgba.len(), image.width * image.height * 4);
    assert!(image.rgba.chunks_exact(4).all(|pixel| pixel[3] == 0xff));
}

#[test]
fn decode_lossy_vp8_to_rgba_matches_container_decode() {
    let data = include_bytes!("../samples/sample.webp");
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
    let data = include_bytes!("../samples/sample_lossless.webp");

    let image = decode_lossless_webp_to_rgba(data).unwrap();

    let features = get_features(data).unwrap();
    assert_eq!(image.width, features.width);
    assert_eq!(image.height, features.height);
    assert_eq!(image.rgba.len(), image.width * image.height * 4);
}

#[test]
fn decode_lossless_vp8l_to_rgba_matches_container_decode() {
    let data = include_bytes!("../samples/sample_lossless.webp");
    let parsed = parse_still_webp(data).unwrap();

    let from_container = decode_lossless_webp_to_rgba(data).unwrap();
    let from_vp8l = decode_lossless_vp8l_to_rgba(parsed.image_data).unwrap();

    assert_eq!(from_vp8l, from_container);
}

#[test]
fn decode_alpha_plane_extracts_green_channel_from_lossless_payload() {
    let width = 19usize;
    let height = 17usize;
    let mut rgba = vec![0u8; width * height * 4];
    for (index, pixel) in rgba.chunks_exact_mut(4).enumerate() {
        pixel.copy_from_slice(&[0, (index * 13 % 256) as u8, 0, 0xff]);
    }
    let payload = encode_lossless_rgba_to_vp8l(width, height, &rgba).unwrap();

    let mut alpha_data = Vec::with_capacity(1 + payload.len());
    alpha_data.push(0x01);
    alpha_data.extend_from_slice(payload.get(5..).unwrap());

    let alpha = decode_alpha_plane(&alpha_data, width, height).unwrap();
    let expected: Vec<u8> = rgba.chunks_exact(4).map(|pixel| pixel[1]).collect();

    assert_eq!(alpha, expected);
}

#[test]
fn decode_lossy_webp_to_rgba_applies_raw_alpha_chunk() {
    let base = decode_lossy_webp_to_rgba(include_bytes!("../samples/sample.webp")).unwrap();
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
    let data = include_bytes!("../samples/sample_animation.webp");

    let parsed = parse_animation_webp(data).unwrap();

    assert!(parsed.features.width > 0);
    assert!(parsed.features.height > 0);
    assert!(parsed.features.has_alpha);
    assert!(parsed.features.has_animation);
    assert_eq!(parsed.animation.loop_count, 0);
    assert!(!parsed.frames.is_empty());
    assert!(parsed.frames[0].width <= parsed.features.width);
    assert!(parsed.frames[0].height <= parsed.features.height);
    assert_eq!(parsed.frames[0].x_offset, 0);
    assert_eq!(parsed.frames[0].y_offset, 0);
    for frame in &parsed.frames {
        assert!(frame.x_offset + frame.width <= parsed.features.width);
        assert!(frame.y_offset + frame.height <= parsed.features.height);
    }
}

#[test]
fn decode_animation_webp_matches_reference_pixels() {
    let data = include_bytes!("../samples/sample_animation.webp");

    let animation = decode_animation_webp(data).unwrap();

    let parsed = parse_animation_webp(data).unwrap();
    assert_eq!(animation.width, parsed.features.width);
    assert_eq!(animation.height, parsed.features.height);
    assert_eq!(animation.loop_count, 0);
    assert_eq!(animation.frames.len(), parsed.frames.len());
    for frame in &animation.frames {
        assert_eq!(frame.rgba.len(), animation.width * animation.height * 4);
    }
}

#[test]
fn decode_animation_webp_handles_lossy_alpha_frames() {
    let base = decode_lossy_webp_to_rgba(include_bytes!("../samples/sample.webp")).unwrap();
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
