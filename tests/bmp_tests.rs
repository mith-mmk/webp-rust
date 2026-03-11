use webp_rust::bmp::encode_bmp24_from_rgba;

#[test]
fn encode_bmp24_writes_bottom_up_bgr_rows() {
    let rgba = [
        0xff, 0x00, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff, // top row: red, green
        0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, // bottom row: blue, white
    ];

    let bmp = encode_bmp24_from_rgba(2, 2, &rgba).unwrap();

    assert_eq!(&bmp[0..2], b"BM");
    assert_eq!(u32::from_le_bytes(bmp[2..6].try_into().unwrap()), 70);
    assert_eq!(u32::from_le_bytes(bmp[10..14].try_into().unwrap()), 54);
    assert_eq!(u32::from_le_bytes(bmp[14..18].try_into().unwrap()), 40);
    assert_eq!(i32::from_le_bytes(bmp[18..22].try_into().unwrap()), 2);
    assert_eq!(i32::from_le_bytes(bmp[22..26].try_into().unwrap()), 2);
    assert_eq!(u16::from_le_bytes(bmp[28..30].try_into().unwrap()), 24);
    assert_eq!(u32::from_le_bytes(bmp[38..42].try_into().unwrap()), 3_780);
    assert_eq!(u32::from_le_bytes(bmp[42..46].try_into().unwrap()), 3_780);

    assert_eq!(
        &bmp[54..70],
        &[
            0xff, 0x00, 0x00, 0xff, 0xff, 0xff, 0x00, 0x00, // bottom row
            0x00, 0x00, 0xff, 0x00, 0xff, 0x00, 0x00, 0x00, // top row
        ]
    );
}

#[test]
fn encode_bmp24_rejects_wrong_buffer_size() {
    let error = encode_bmp24_from_rgba(1, 1, &[0, 1, 2]).unwrap_err();

    assert_eq!(
        error.to_string(),
        "invalid parameter: RGBA buffer length does not match dimensions"
    );
}
