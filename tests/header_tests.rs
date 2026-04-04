use bin_rs::reader::BytesReader;
use webp_rust::decoder::header::parse_still_webp;
use webp_rust::legacy::{read_header, read_u24};

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

#[test]
fn read_u24_reads_little_endian_values() {
    let mut reader = BytesReader::from(vec![0x56, 0x34, 0x12]);

    let value = read_u24(&mut reader).unwrap();

    assert_eq!(value, 0x12_34_56);
}

#[test]
fn read_header_rejects_non_riff_data() {
    let mut reader = BytesReader::from(b"NOPE".to_vec());

    let result = read_header(&mut reader);

    assert!(result.is_err());
}

#[test]
fn read_header_parses_lossy_sample() {
    let data = include_bytes!("../samples/sample.webp");
    let mut reader = BytesReader::from(data.to_vec());
    let parsed = parse_still_webp(data).unwrap();

    let header = read_header(&mut reader).unwrap();

    assert!(header.lossy);
    assert_eq!(header.width, parsed.features.width);
    assert_eq!(header.height, parsed.features.height);
    assert_eq!(header.image_chunksize, parsed.image_chunk.size);
    assert_eq!(header.image.len(), parsed.image_data.len());
    assert_eq!(header.canvas_width, 0);
    assert_eq!(header.canvas_height, 0);
    assert!(!header.has_alpha);
    assert!(!header.has_animation);
    assert!(header.alpha.is_none());
    assert!(header.animation.is_none());
}

#[test]
fn read_header_keeps_animation_frame_alpha_payload() {
    let sample = include_bytes!("../samples/sample.webp");
    let parsed = parse_still_webp(sample).unwrap();
    let alpha = vec![0x7f; parsed.features.width * parsed.features.height];

    let mut vp8x_payload = Vec::with_capacity(10);
    vp8x_payload.extend_from_slice(&0x12u32.to_le_bytes());
    vp8x_payload.extend_from_slice(&le24(parsed.features.width - 1));
    vp8x_payload.extend_from_slice(&le24(parsed.features.height - 1));

    let mut alph_payload = Vec::with_capacity(1 + alpha.len());
    alph_payload.push(0);
    alph_payload.extend_from_slice(&alpha);

    let mut anmf_payload = Vec::new();
    anmf_payload.extend_from_slice(&le24(0));
    anmf_payload.extend_from_slice(&le24(0));
    anmf_payload.extend_from_slice(&le24(parsed.features.width - 1));
    anmf_payload.extend_from_slice(&le24(parsed.features.height - 1));
    anmf_payload.extend_from_slice(&le24(50));
    anmf_payload.push(0x02);
    anmf_payload.extend_from_slice(&make_chunk(b"ALPH", &alph_payload));
    anmf_payload.extend_from_slice(&make_chunk(b"VP8 ", parsed.image_data));

    let webp = wrap_riff(&[
        make_chunk(b"VP8X", &vp8x_payload),
        make_chunk(b"ANIM", &[0, 0, 0, 0, 0, 0]),
        make_chunk(b"ANMF", &anmf_payload),
    ]);
    let mut reader = BytesReader::from(webp);

    let header = read_header(&mut reader).unwrap();
    let frame = &header.animation_frame.unwrap()[0];

    assert!(header.has_alpha);
    assert!(header.has_animation);
    assert_eq!(frame.width, parsed.features.width);
    assert_eq!(frame.height, parsed.features.height);
    assert_eq!(frame.frame, parsed.image_data);
    assert_eq!(frame.alpha.as_deref(), Some(alph_payload.as_slice()));
}

#[cfg(not(target_family = "wasm"))]
#[test]
fn image_from_file_decodes_still_webp() {
    let sample = include_bytes!("../samples/sample.webp");
    let decoded = webp_rust::decode(sample).unwrap();
    let path = std::env::temp_dir().join(format!("webp-rust-{}-sample.webp", std::process::id()));
    std::fs::write(&path, sample).unwrap();

    let image = webp_rust::image_from_file(path.to_string_lossy().into_owned()).unwrap();

    let _ = std::fs::remove_file(&path);

    assert_eq!(image.width, decoded.width);
    assert_eq!(image.height, decoded.height);
    assert_eq!(image.rgba, decoded.rgba);
}
