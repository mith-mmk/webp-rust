use super::*;

fn reference_transform(transform: &Transform, input: &[u32]) -> Vec<u32> {
    match transform.kind {
        TransformType::SubtractGreen => input
            .iter()
            .map(|&argb| {
                let green = (argb >> 8) & 0xff;
                let red = (((argb >> 16) & 0xff) + green) & 0xff;
                let blue = ((argb & 0xff) + green) & 0xff;
                (argb & 0xff00_ff00) | (red << 16) | blue
            })
            .collect(),
        TransformType::CrossColor => {
            let tiles_per_row = subsample_size(transform.xsize, transform.bits);
            let mut output = vec![0u32; input.len()];
            for y in 0..transform.ysize {
                for x in 0..transform.xsize {
                    let argb = input[y * transform.xsize + x];
                    let code = transform.data
                        [(y >> transform.bits) * tiles_per_row + (x >> transform.bits)];
                    let green_to_red = code as u8;
                    let green_to_blue = ((code >> 8) & 0xff) as u8;
                    let red_to_blue = ((code >> 16) & 0xff) as u8;
                    let green = ((argb >> 8) & 0xff) as u8;
                    let mut red = ((argb >> 16) & 0xff) as i32;
                    let mut blue = (argb & 0xff) as i32;
                    red = (red + color_transform_delta(green_to_red, green)) & 0xff;
                    blue = (blue + color_transform_delta(green_to_blue, green)) & 0xff;
                    blue = (blue + color_transform_delta(red_to_blue, red as u8)) & 0xff;
                    output[y * transform.xsize + x] =
                        (argb & 0xff00_ff00) | ((red as u32) << 16) | blue as u32;
                }
            }
            output
        }
        TransformType::Predictor => {
            let mut output = vec![0u32; input.len()];
            let tiles_per_row = subsample_size(transform.xsize, transform.bits);
            for y in 0..transform.ysize {
                for x in 0..transform.xsize {
                    let index = y * transform.xsize + x;
                    let pred = if y == 0 {
                        if x == 0 {
                            ARGB_BLACK
                        } else {
                            output[index - 1]
                        }
                    } else if x == 0 {
                        output[index - transform.xsize]
                    } else {
                        let left = output[index - 1];
                        let top = output[index - transform.xsize];
                        let top_left = output[index - transform.xsize - 1];
                        let top_right = if x + 1 < transform.xsize {
                            output[index - transform.xsize + 1]
                        } else {
                            output[index - x]
                        };
                        let mode = ((transform.data
                            [(y >> transform.bits) * tiles_per_row + (x >> transform.bits)]
                            >> 8)
                            & 0x0f) as u8;
                        predictor(mode, left, top, top_left, top_right)
                    };
                    output[index] = add_pixels(input[index], pred);
                }
            }
            output
        }
        TransformType::ColorIndexing => {
            let reduced_width = subsample_size(transform.xsize, transform.bits);
            let bits_per_pixel = 8 >> transform.bits;
            let pixels_per_byte = 1usize << transform.bits;
            let bit_mask = (1u32 << bits_per_pixel) - 1;
            let mut output = vec![0u32; transform.xsize * transform.ysize];
            for y in 0..transform.ysize {
                let mut x = 0usize;
                for &packed in &input[y * reduced_width..(y + 1) * reduced_width] {
                    let mut indices = (packed >> 8) & 0xff;
                    for _ in 0..pixels_per_byte {
                        if x == transform.xsize {
                            break;
                        }
                        output[y * transform.xsize + x] =
                            transform.data[(indices & bit_mask) as usize];
                        indices >>= bits_per_pixel;
                        x += 1;
                    }
                }
            }
            output
        }
    }
}

fn assert_in_place_matches_reference(transform: Transform, mut input: Vec<u32>) {
    let expected = reference_transform(&transform, &input);
    apply_inverse_transform(&transform, &mut input).unwrap();
    assert_eq!(input, expected);
}

#[test]
fn buffered_bit_reader_crosses_byte_boundaries() {
    let mut reader = LosslessBitReader::new(&[0b1010_0101, 0b1100_0011, 0b0110_1001]);
    assert_eq!(reader.read_bits(3).unwrap(), 0b101);
    assert_eq!(reader.read_bits(9).unwrap(), 0b0_0111_0100);
    assert_eq!(reader.read_bits(8).unwrap(), 0b1001_1100);
    assert_eq!(reader.remaining_bits(), 4);
    assert!(reader.read_bits(5).is_err());
}

#[test]
fn huffman_lookup_handles_short_and_fifteen_bit_codes() {
    let empty_error = HuffmanTree::from_code_lengths(&[0, 0, 0, 0, 0]).unwrap_err();
    assert!(matches!(empty_error, DecoderError::Bitstream(_)));
    let single = HuffmanTree::from_code_lengths(&[0, 0, 1, 0, 0]).unwrap();
    assert_eq!(
        single
            .read_symbol(&mut LosslessBitReader::new(&[]))
            .unwrap(),
        2
    );

    let mut lengths = (1u8..=14).collect::<Vec<_>>();
    lengths.extend([15, 15]);
    let tree = HuffmanTree::from_code_lengths(&lengths).unwrap();

    for bits in 1..=15 {
        for &(code, symbol) in &tree.codes_by_len[bits] {
            let encoded = [code as u8, (code >> 8) as u8, 0];
            let mut reader = LosslessBitReader::new(&encoded);
            assert_eq!(tree.read_symbol(&mut reader).unwrap(), symbol);
            let mut reference_reader = LosslessBitReader::new(&encoded);
            assert_eq!(
                tree.read_symbol_slow(&mut reference_reader).unwrap(),
                symbol
            );
        }
    }
}

#[test]
fn huffman_lookup_preserves_validation_and_truncation_errors() {
    assert!(HuffmanTree::from_code_lengths(&[1, 1, 1]).is_err());
    assert!(HuffmanTree::from_code_lengths(&[2, 2]).is_err());

    let tree = HuffmanTree::from_code_lengths(&[1, 2, 2]).unwrap();
    let mut reader = LosslessBitReader::new(&[]);
    assert!(matches!(
        tree.read_symbol(&mut reader),
        Err(DecoderError::NotEnoughData("VP8L bitstream"))
    ));
}

#[test]
fn inverse_transforms_match_allocating_reference() {
    let pixels = (0..35)
        .map(|index| {
            let index = index as u32;
            (0x80u32.wrapping_add(index) << 24)
                | (index.wrapping_mul(37) & 0xff) << 16
                | (index.wrapping_mul(19) & 0xff) << 8
                | (index.wrapping_mul(11) & 0xff)
        })
        .collect::<Vec<_>>();

    assert_in_place_matches_reference(
        Transform {
            kind: TransformType::SubtractGreen,
            bits: 0,
            xsize: 7,
            ysize: 5,
            data: Vec::new(),
        },
        pixels.clone(),
    );
    assert_in_place_matches_reference(
        Transform {
            kind: TransformType::CrossColor,
            bits: 2,
            xsize: 7,
            ysize: 5,
            data: vec![0x0011_22ee, 0x00d3_a419, 0x007f_80c0, 0x0020_f010],
        },
        pixels.clone(),
    );
    assert_in_place_matches_reference(
        Transform {
            kind: TransformType::Predictor,
            bits: 2,
            xsize: 7,
            ysize: 5,
            data: vec![0x0000_0000, 0x0000_0700, 0x0000_0c00, 0x0000_0d00],
        },
        pixels,
    );
}

#[test]
fn color_indexing_in_place_expands_odd_widths_for_all_packings() {
    let palette = (0..256)
        .map(|index| 0xff00_0000 | (index << 16) | (index << 8) | index)
        .collect::<Vec<_>>();
    for bits in 0..=3 {
        let xsize = 11usize;
        let ysize = 3usize;
        let reduced_width = subsample_size(xsize, bits);
        let input = (0..reduced_width * ysize)
            .map(|index| ((index as u32 * 53) & 0xff) << 8)
            .collect::<Vec<_>>();
        assert_in_place_matches_reference(
            Transform {
                kind: TransformType::ColorIndexing,
                bits,
                xsize,
                ysize,
                data: palette.clone(),
            },
            input,
        );
    }
}

#[test]
fn backward_reference_copy_repeats_overlapping_source() {
    let mut repeated_pattern = vec![1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    copy_backward_reference(&mut repeated_pattern, 3, 3, 9);
    assert_eq!(repeated_pattern, [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]);

    let mut repeated_pixel = vec![0xfeed_beef, 0, 0, 0, 0, 0];
    copy_backward_reference(&mut repeated_pixel, 1, 1, 5);
    assert!(repeated_pixel.iter().all(|&pixel| pixel == 0xfeed_beef));
}
