use bin_rs::reader::BytesReader;
use webp_rust::{read_header, read_u24};

#[test]
fn read_u24_reads_little_endian_values() {
    let mut reader = BytesReader::from_vec(vec![0x56, 0x34, 0x12]);

    let value = read_u24(&mut reader).unwrap();

    assert_eq!(value, 0x12_34_56);
}

#[test]
fn read_header_rejects_non_riff_data() {
    let mut reader = BytesReader::from_vec(b"NOPE".to_vec());

    let result = read_header(&mut reader);

    assert!(result.is_err());
}

#[test]
fn read_header_parses_lossy_sample() {
    let data = include_bytes!("../_testdata/sample.webp");
    let mut reader = BytesReader::from_vec(data.to_vec());

    let header = read_header(&mut reader).unwrap();

    assert!(header.lossy);
    assert_eq!(header.width, 1920);
    assert_eq!(header.height, 1080);
    assert_eq!(header.image_chunksize, 15_226);
    assert_eq!(header.image.len(), 15_226);
    assert_eq!(header.canvas_width, 0);
    assert_eq!(header.canvas_height, 0);
    assert!(!header.has_alpha);
    assert!(!header.has_animation);
    assert!(header.alpha.is_none());
    assert!(header.animation.is_none());
}
