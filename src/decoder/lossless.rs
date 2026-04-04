//! Lossless `VP8L` decode helpers.

use std::collections::HashMap;

use crate::decoder::header::parse_still_webp;
use crate::decoder::lossy::DecodedImage;
use crate::decoder::vp8::get_lossless_info;
use crate::decoder::vp8i::WebpFormat;
use crate::decoder::DecoderError;

const ARGB_BLACK: u32 = 0xff00_0000;
const MAX_ALLOWED_CODE_LENGTH: usize = 15;
const MAX_CACHE_BITS: usize = 11;
const NUM_LITERAL_CODES: usize = 256;
const NUM_LENGTH_CODES: usize = 24;
const NUM_DISTANCE_CODES: usize = 40;
const NUM_CODE_LENGTH_CODES: usize = 19;
const MIN_HUFFMAN_BITS: usize = 2;
const NUM_HUFFMAN_BITS: usize = 3;
const MIN_TRANSFORM_BITS: usize = 2;
const NUM_TRANSFORM_BITS: usize = 3;
const DEFAULT_CODE_LENGTH: u8 = 8;
const CODE_LENGTH_REPEAT_CODE: usize = 16;
const CODE_LENGTH_EXTRA_BITS: [usize; 3] = [2, 3, 7];
const CODE_LENGTH_REPEAT_OFFSETS: [usize; 3] = [3, 3, 11];
const CODE_LENGTH_CODE_ORDER: [usize; NUM_CODE_LENGTH_CODES] = [
    17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
];
const CODE_TO_PLANE: [u8; 120] = [
    0x18, 0x07, 0x17, 0x19, 0x28, 0x06, 0x27, 0x29, 0x16, 0x1a, 0x26, 0x2a, 0x38, 0x05, 0x37, 0x39,
    0x15, 0x1b, 0x36, 0x3a, 0x25, 0x2b, 0x48, 0x04, 0x47, 0x49, 0x14, 0x1c, 0x35, 0x3b, 0x46, 0x4a,
    0x24, 0x2c, 0x58, 0x45, 0x4b, 0x34, 0x3c, 0x03, 0x57, 0x59, 0x13, 0x1d, 0x56, 0x5a, 0x23, 0x2d,
    0x44, 0x4c, 0x55, 0x5b, 0x33, 0x3d, 0x68, 0x02, 0x67, 0x69, 0x12, 0x1e, 0x66, 0x6a, 0x22, 0x2e,
    0x54, 0x5c, 0x43, 0x4d, 0x65, 0x6b, 0x32, 0x3e, 0x78, 0x01, 0x77, 0x79, 0x53, 0x5d, 0x11, 0x1f,
    0x64, 0x6c, 0x42, 0x4e, 0x76, 0x7a, 0x21, 0x2f, 0x75, 0x7b, 0x31, 0x3f, 0x63, 0x6d, 0x52, 0x5e,
    0x00, 0x74, 0x7c, 0x41, 0x4f, 0x10, 0x20, 0x62, 0x6e, 0x30, 0x73, 0x7d, 0x51, 0x5f, 0x40, 0x72,
    0x7e, 0x61, 0x6f, 0x50, 0x71, 0x7f, 0x60, 0x70,
];
const COLOR_CACHE_HASH_MUL: u32 = 0x1e35_a7bd;

#[derive(Debug, Clone)]
struct LosslessBitReader<'a> {
    data: &'a [u8],
    bit_pos: usize,
}

impl<'a> LosslessBitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, bit_pos: 0 }
    }

    fn read_bit(&mut self) -> Result<u32, DecoderError> {
        self.read_bits(1)
    }

    fn read_bits(&mut self, num_bits: usize) -> Result<u32, DecoderError> {
        if num_bits > 24 {
            return Err(DecoderError::InvalidParam("VP8L bit read is too wide"));
        }
        let end = self
            .bit_pos
            .checked_add(num_bits)
            .ok_or(DecoderError::Bitstream("VP8L bit position overflow"))?;
        if end > self.data.len() * 8 {
            return Err(DecoderError::NotEnoughData("VP8L bitstream"));
        }

        let mut value = 0u32;
        for bit_index in 0..num_bits {
            let stream_bit = self.bit_pos + bit_index;
            let byte = self.data[stream_bit >> 3];
            let bit = (byte >> (stream_bit & 7)) & 1;
            value |= (bit as u32) << bit_index;
        }
        self.bit_pos = end;
        Ok(value)
    }
}

#[derive(Debug, Clone)]
struct HuffmanTree {
    single_symbol: Option<u16>,
    by_len: Vec<HashMap<u16, u16>>,
    max_len: usize,
}

impl HuffmanTree {
    fn from_code_lengths(code_lengths: &[u8]) -> Result<Self, DecoderError> {
        let mut counts = [0i32; MAX_ALLOWED_CODE_LENGTH + 1];
        let mut single_symbol = None;
        let mut num_symbols = 0usize;

        for (symbol, &len) in code_lengths.iter().enumerate() {
            let bits = len as usize;
            if bits > MAX_ALLOWED_CODE_LENGTH {
                return Err(DecoderError::Bitstream("invalid VP8L Huffman code length"));
            }
            if bits > 0 {
                counts[bits] += 1;
                single_symbol = Some(symbol as u16);
                num_symbols += 1;
            }
        }

        if num_symbols == 0 {
            return Err(DecoderError::Bitstream("empty VP8L Huffman tree"));
        }
        if num_symbols == 1 {
            return Ok(Self {
                single_symbol,
                by_len: Vec::new(),
                max_len: 0,
            });
        }

        let mut left = 1i32;
        for bits in 1..=MAX_ALLOWED_CODE_LENGTH {
            left = (left << 1) - counts[bits];
            if left < 0 {
                return Err(DecoderError::Bitstream("oversubscribed VP8L Huffman tree"));
            }
        }
        if left != 0 {
            return Err(DecoderError::Bitstream("incomplete VP8L Huffman tree"));
        }

        let mut next_code = [0u32; MAX_ALLOWED_CODE_LENGTH + 1];
        let mut code = 0u32;
        for bits in 1..=MAX_ALLOWED_CODE_LENGTH {
            code = (code + counts[bits - 1] as u32) << 1;
            next_code[bits] = code;
        }

        let mut by_len = (0..=MAX_ALLOWED_CODE_LENGTH)
            .map(|_| HashMap::new())
            .collect::<Vec<_>>();
        let mut max_len = 0usize;

        for (symbol, &len) in code_lengths.iter().enumerate() {
            let bits = len as usize;
            if bits == 0 {
                continue;
            }
            let canonical = next_code[bits];
            next_code[bits] += 1;
            by_len[bits].insert(reverse_bits(canonical, bits), symbol as u16);
            max_len = max_len.max(bits);
        }

        Ok(Self {
            single_symbol: None,
            by_len,
            max_len,
        })
    }

    fn read_symbol(&self, br: &mut LosslessBitReader<'_>) -> Result<u16, DecoderError> {
        if let Some(symbol) = self.single_symbol {
            return Ok(symbol);
        }

        let mut code = 0u16;
        for bits in 1..=self.max_len {
            code |= (br.read_bit()? as u16) << (bits - 1);
            if let Some(&symbol) = self.by_len[bits].get(&code) {
                return Ok(symbol);
            }
        }

        Err(DecoderError::Bitstream("invalid VP8L Huffman symbol"))
    }
}

#[derive(Debug, Clone)]
struct ColorCache {
    colors: Vec<u32>,
    hash_shift: u32,
}

impl ColorCache {
    fn new(hash_bits: usize) -> Result<Self, DecoderError> {
        if !(1..=MAX_CACHE_BITS).contains(&hash_bits) {
            return Err(DecoderError::Bitstream("invalid VP8L color cache size"));
        }
        let size = 1usize << hash_bits;
        Ok(Self {
            colors: vec![0; size],
            hash_shift: (32 - hash_bits) as u32,
        })
    }

    fn insert(&mut self, argb: u32) {
        let key = ((argb.wrapping_mul(COLOR_CACHE_HASH_MUL)) >> self.hash_shift) as usize;
        self.colors[key] = argb;
    }

    fn lookup(&self, key: usize) -> Result<u32, DecoderError> {
        self.colors
            .get(key)
            .copied()
            .ok_or(DecoderError::Bitstream("invalid VP8L color cache lookup"))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TransformType {
    Predictor,
    CrossColor,
    SubtractGreen,
    ColorIndexing,
}

#[derive(Debug, Clone)]
struct Transform {
    kind: TransformType,
    bits: usize,
    xsize: usize,
    ysize: usize,
    data: Vec<u32>,
}

#[derive(Debug, Clone)]
struct HTreeGroup {
    green: HuffmanTree,
    red: HuffmanTree,
    blue: HuffmanTree,
    alpha: HuffmanTree,
    dist: HuffmanTree,
}

#[derive(Debug, Clone)]
struct HuffmanMetadata {
    huffman_subsample_bits: usize,
    huffman_xsize: usize,
    huffman_image: Option<Vec<usize>>,
    groups: Vec<HTreeGroup>,
}

impl HuffmanMetadata {
    fn group_index(&self, x: usize, y: usize) -> usize {
        if let Some(image) = &self.huffman_image {
            image[(y >> self.huffman_subsample_bits) * self.huffman_xsize
                + (x >> self.huffman_subsample_bits)]
        } else {
            0
        }
    }
}

struct LosslessDecoder<'a> {
    br: LosslessBitReader<'a>,
}

impl<'a> LosslessDecoder<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            br: LosslessBitReader::new(data),
        }
    }

    fn decode_image_stream(
        &mut self,
        xsize: usize,
        ysize: usize,
        top_level: bool,
    ) -> Result<Vec<u32>, DecoderError> {
        let mut transforms = Vec::new();
        let mut transform_xsize = xsize;
        let transform_ysize = ysize;

        if top_level {
            let mut transforms_seen = 0u32;
            while self.br.read_bit()? == 1 {
                let transform =
                    self.read_transform(transform_xsize, transform_ysize, &mut transforms_seen)?;
                if matches!(transform.kind, TransformType::ColorIndexing) {
                    transform_xsize = subsample_size(transform_xsize, transform.bits);
                }
                transforms.push(transform);
            }
        }

        let color_cache_bits = if self.br.read_bit()? == 1 {
            let bits = self.br.read_bits(4)? as usize;
            if !(1..=MAX_CACHE_BITS).contains(&bits) {
                return Err(DecoderError::Bitstream("invalid VP8L color cache bits"));
            }
            bits
        } else {
            0
        };

        let metadata = self.read_huffman_codes(
            transform_xsize,
            transform_ysize,
            color_cache_bits,
            top_level,
        )?;
        let mut data = self.decode_image_data(
            transform_xsize,
            transform_ysize,
            color_cache_bits,
            &metadata,
        )?;

        if top_level {
            for transform in transforms.iter().rev() {
                data = apply_inverse_transform(transform, &data)?;
            }
        }

        Ok(data)
    }

    fn read_transform(
        &mut self,
        xsize: usize,
        ysize: usize,
        transforms_seen: &mut u32,
    ) -> Result<Transform, DecoderError> {
        let type_bits = self.br.read_bits(2)? as usize;
        let kind = match type_bits {
            0 => TransformType::Predictor,
            1 => TransformType::CrossColor,
            2 => TransformType::SubtractGreen,
            3 => TransformType::ColorIndexing,
            _ => unreachable!(),
        };

        if (*transforms_seen & (1u32 << type_bits)) != 0 {
            return Err(DecoderError::Bitstream("duplicate VP8L transform"));
        }
        *transforms_seen |= 1u32 << type_bits;

        match kind {
            TransformType::Predictor | TransformType::CrossColor => {
                let bits = MIN_TRANSFORM_BITS + self.br.read_bits(NUM_TRANSFORM_BITS)? as usize;
                let data = self.decode_image_stream(
                    subsample_size(xsize, bits),
                    subsample_size(ysize, bits),
                    false,
                )?;
                Ok(Transform {
                    kind,
                    bits,
                    xsize,
                    ysize,
                    data,
                })
            }
            TransformType::SubtractGreen => Ok(Transform {
                kind,
                bits: 0,
                xsize,
                ysize,
                data: Vec::new(),
            }),
            TransformType::ColorIndexing => {
                let num_colors = self.br.read_bits(8)? as usize + 1;
                let bits = if num_colors > 16 {
                    0
                } else if num_colors > 4 {
                    1
                } else if num_colors > 2 {
                    2
                } else {
                    3
                };
                let palette = self.decode_image_stream(num_colors, 1, false)?;
                let expanded = expand_color_map(&palette, num_colors, bits);
                Ok(Transform {
                    kind,
                    bits,
                    xsize,
                    ysize,
                    data: expanded,
                })
            }
        }
    }

    fn read_huffman_codes(
        &mut self,
        xsize: usize,
        ysize: usize,
        color_cache_bits: usize,
        allow_meta: bool,
    ) -> Result<HuffmanMetadata, DecoderError> {
        let mut huffman_subsample_bits = 0usize;
        let mut huffman_xsize = 0usize;
        let mut huffman_image = None;
        let mapping = if allow_meta && self.br.read_bit()? == 1 {
            huffman_subsample_bits =
                MIN_HUFFMAN_BITS + self.br.read_bits(NUM_HUFFMAN_BITS)? as usize;
            huffman_xsize = subsample_size(xsize, huffman_subsample_bits);
            let huffman_ysize = subsample_size(ysize, huffman_subsample_bits);
            let image = self.decode_image_stream(huffman_xsize, huffman_ysize, false)?;

            let mut max_group = 0usize;
            let raw_groups = image
                .iter()
                .map(|&pixel| ((pixel >> 8) & 0xffff) as usize)
                .inspect(|&group| max_group = max_group.max(group))
                .collect::<Vec<_>>();
            let mut mapping = vec![None; max_group + 1];
            let mut dense_image = Vec::with_capacity(raw_groups.len());
            let mut next_group = 0usize;
            for group in raw_groups {
                let dense = if let Some(index) = mapping[group] {
                    index
                } else {
                    let index = next_group;
                    mapping[group] = Some(index);
                    next_group += 1;
                    index
                };
                dense_image.push(dense);
            }
            huffman_image = Some(dense_image);
            mapping
        } else {
            vec![Some(0)]
        };

        let num_groups = mapping.iter().flatten().count();
        let mut groups = vec![None; num_groups];
        for dense in mapping {
            let group = self.read_htree_group(color_cache_bits)?;
            if let Some(index) = dense {
                groups[index] = Some(group);
            }
        }

        let groups = groups
            .into_iter()
            .map(|group| group.ok_or(DecoderError::Bitstream("missing VP8L Huffman group")))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(HuffmanMetadata {
            huffman_subsample_bits,
            huffman_xsize,
            huffman_image,
            groups,
        })
    }

    fn read_htree_group(&mut self, color_cache_bits: usize) -> Result<HTreeGroup, DecoderError> {
        let green_alphabet_size = NUM_LITERAL_CODES
            + NUM_LENGTH_CODES
            + if color_cache_bits > 0 {
                1usize << color_cache_bits
            } else {
                0
            };

        Ok(HTreeGroup {
            green: self.read_huffman_code(green_alphabet_size)?,
            red: self.read_huffman_code(NUM_LITERAL_CODES)?,
            blue: self.read_huffman_code(NUM_LITERAL_CODES)?,
            alpha: self.read_huffman_code(NUM_LITERAL_CODES)?,
            dist: self.read_huffman_code(NUM_DISTANCE_CODES)?,
        })
    }

    fn read_huffman_code(&mut self, alphabet_size: usize) -> Result<HuffmanTree, DecoderError> {
        let mut code_lengths = vec![0u8; alphabet_size];
        let simple_code = self.br.read_bit()? == 1;

        if simple_code {
            let num_symbols = self.br.read_bit()? as usize + 1;
            let first_symbol_len_code = self.br.read_bit()? as usize;
            let first_bits = if first_symbol_len_code == 0 { 1 } else { 8 };
            let first_symbol = self.br.read_bits(first_bits)? as usize;
            if first_symbol >= alphabet_size {
                return Err(DecoderError::Bitstream(
                    "invalid VP8L simple Huffman symbol",
                ));
            }
            code_lengths[first_symbol] = 1;
            if num_symbols == 2 {
                let second_symbol = self.br.read_bits(8)? as usize;
                if second_symbol >= alphabet_size {
                    return Err(DecoderError::Bitstream(
                        "invalid VP8L simple Huffman symbol",
                    ));
                }
                code_lengths[second_symbol] = 1;
            }
        } else {
            let mut code_length_code_lengths = [0u8; NUM_CODE_LENGTH_CODES];
            let num_codes = self.br.read_bits(4)? as usize + 4;
            if num_codes > NUM_CODE_LENGTH_CODES {
                return Err(DecoderError::Bitstream("too many VP8L code length codes"));
            }
            for i in 0..num_codes {
                code_length_code_lengths[CODE_LENGTH_CODE_ORDER[i]] = self.br.read_bits(3)? as u8;
            }
            let code_length_tree = HuffmanTree::from_code_lengths(&code_length_code_lengths)?;
            self.read_huffman_code_lengths(&code_length_tree, &mut code_lengths)?;
        }

        HuffmanTree::from_code_lengths(&code_lengths)
    }

    fn read_huffman_code_lengths(
        &mut self,
        code_length_tree: &HuffmanTree,
        code_lengths: &mut [u8],
    ) -> Result<(), DecoderError> {
        let num_symbols = code_lengths.len();
        let mut max_symbol = if self.br.read_bit()? == 1 {
            let length_nbits = 2 + 2 * self.br.read_bits(3)? as usize;
            let value = 2 + self.br.read_bits(length_nbits)? as usize;
            if value > num_symbols {
                return Err(DecoderError::Bitstream(
                    "invalid VP8L Huffman code length span",
                ));
            }
            value
        } else {
            num_symbols
        };

        let mut symbol = 0usize;
        let mut prev_code_len = DEFAULT_CODE_LENGTH;
        while symbol < num_symbols {
            if max_symbol == 0 {
                break;
            }
            max_symbol -= 1;

            let code_len = code_length_tree.read_symbol(&mut self.br)? as usize;
            if code_len < CODE_LENGTH_REPEAT_CODE {
                code_lengths[symbol] = code_len as u8;
                if code_len != 0 {
                    prev_code_len = code_len as u8;
                }
                symbol += 1;
                continue;
            }

            let slot = code_len
                .checked_sub(CODE_LENGTH_REPEAT_CODE)
                .ok_or(DecoderError::Bitstream("invalid VP8L repeat code"))?;
            if slot >= CODE_LENGTH_EXTRA_BITS.len() {
                return Err(DecoderError::Bitstream("invalid VP8L repeat code"));
            }
            let repeat = self.br.read_bits(CODE_LENGTH_EXTRA_BITS[slot])? as usize
                + CODE_LENGTH_REPEAT_OFFSETS[slot];
            if symbol + repeat > num_symbols {
                return Err(DecoderError::Bitstream("VP8L repeat overruns code lengths"));
            }
            let value = if code_len == CODE_LENGTH_REPEAT_CODE {
                prev_code_len
            } else {
                0
            };
            for len in &mut code_lengths[symbol..symbol + repeat] {
                *len = value;
            }
            symbol += repeat;
        }

        Ok(())
    }

    fn decode_image_data(
        &mut self,
        width: usize,
        height: usize,
        color_cache_bits: usize,
        metadata: &HuffmanMetadata,
    ) -> Result<Vec<u32>, DecoderError> {
        let mut data = vec![0u32; width * height];
        let mut color_cache = if color_cache_bits > 0 {
            Some(ColorCache::new(color_cache_bits)?)
        } else {
            None
        };
        let mut pos = 0usize;
        let len_code_limit = NUM_LITERAL_CODES + NUM_LENGTH_CODES;
        let color_cache_limit = len_code_limit
            + if color_cache_bits > 0 {
                1usize << color_cache_bits
            } else {
                0
            };

        while pos < data.len() {
            let x = pos % width;
            let y = pos / width;
            let group = &metadata.groups[metadata.group_index(x, y)];
            let code = group.green.read_symbol(&mut self.br)? as usize;

            if code < NUM_LITERAL_CODES {
                let red = group.red.read_symbol(&mut self.br)? as u32;
                let blue = group.blue.read_symbol(&mut self.br)? as u32;
                let alpha = group.alpha.read_symbol(&mut self.br)? as u32;
                let pixel = (alpha << 24) | (red << 16) | ((code as u32) << 8) | blue;
                data[pos] = pixel;
                if let Some(cache) = &mut color_cache {
                    cache.insert(pixel);
                }
                pos += 1;
            } else if code < len_code_limit {
                let length = get_copy_value(code - NUM_LITERAL_CODES, &mut self.br)?;
                let dist_symbol = group.dist.read_symbol(&mut self.br)? as usize;
                let dist_code = get_copy_value(dist_symbol, &mut self.br)?;
                let dist = plane_code_to_distance(width, dist_code);
                if dist > pos || pos + length > data.len() {
                    return Err(DecoderError::Bitstream("invalid VP8L backward reference"));
                }
                for i in 0..length {
                    let pixel = data[pos + i - dist];
                    data[pos + i] = pixel;
                    if let Some(cache) = &mut color_cache {
                        cache.insert(pixel);
                    }
                }
                pos += length;
            } else if code < color_cache_limit {
                let key = code - len_code_limit;
                let cache = color_cache
                    .as_mut()
                    .ok_or(DecoderError::Bitstream("unexpected VP8L color cache code"))?;
                let pixel = cache.lookup(key)?;
                data[pos] = pixel;
                cache.insert(pixel);
                pos += 1;
            } else {
                return Err(DecoderError::Bitstream("invalid VP8L green Huffman symbol"));
            }
        }

        Ok(data)
    }
}

fn reverse_bits(mut code: u32, bits: usize) -> u16 {
    let mut out = 0u32;
    for _ in 0..bits {
        out = (out << 1) | (code & 1);
        code >>= 1;
    }
    out as u16
}

fn subsample_size(size: usize, bits: usize) -> usize {
    (size + (1usize << bits) - 1) >> bits
}

fn get_copy_value(symbol: usize, br: &mut LosslessBitReader<'_>) -> Result<usize, DecoderError> {
    if symbol < 4 {
        Ok(symbol + 1)
    } else {
        let extra_bits = (symbol - 2) >> 1;
        let offset = (2 + (symbol & 1)) << extra_bits;
        Ok(offset + br.read_bits(extra_bits)? as usize + 1)
    }
}

fn plane_code_to_distance(width: usize, plane_code: usize) -> usize {
    if plane_code > CODE_TO_PLANE.len() {
        plane_code - CODE_TO_PLANE.len()
    } else {
        let dist_code = CODE_TO_PLANE[plane_code - 1];
        let y_offset = (dist_code >> 4) as isize;
        let x_offset = 8isize - (dist_code & 0x0f) as isize;
        let dist = y_offset * width as isize + x_offset;
        dist.max(1) as usize
    }
}

fn add_pixels(a: u32, b: u32) -> u32 {
    let alpha = (((a >> 24) as u8).wrapping_add((b >> 24) as u8)) as u32;
    let red = (((a >> 16) as u8).wrapping_add((b >> 16) as u8)) as u32;
    let green = (((a >> 8) as u8).wrapping_add((b >> 8) as u8)) as u32;
    let blue = ((a as u8).wrapping_add(b as u8)) as u32;
    (alpha << 24) | (red << 16) | (green << 8) | blue
}

fn average2(a: u32, b: u32) -> u32 {
    (((a ^ b) & 0xfefe_fefeu32) >> 1) + (a & b)
}

fn clip255(value: i32) -> u32 {
    value.clamp(0, 255) as u32
}

fn clamped_add_subtract_full(left: u32, top: u32, top_left: u32) -> u32 {
    let alpha = clip255((left >> 24) as i32 + (top >> 24) as i32 - (top_left >> 24) as i32);
    let red = clip255(
        ((left >> 16) & 0xff) as i32 + ((top >> 16) & 0xff) as i32
            - ((top_left >> 16) & 0xff) as i32,
    );
    let green = clip255(
        ((left >> 8) & 0xff) as i32 + ((top >> 8) & 0xff) as i32 - ((top_left >> 8) & 0xff) as i32,
    );
    let blue = clip255((left & 0xff) as i32 + (top & 0xff) as i32 - (top_left & 0xff) as i32);
    (alpha << 24) | (red << 16) | (green << 8) | blue
}

fn clamped_add_subtract_half(left: u32, top: u32, top_left: u32) -> u32 {
    let avg = average2(left, top);
    let alpha = clip255((avg >> 24) as i32 + ((avg >> 24) as i32 - (top_left >> 24) as i32) / 2);
    let red = clip255(
        ((avg >> 16) & 0xff) as i32
            + (((avg >> 16) & 0xff) as i32 - ((top_left >> 16) & 0xff) as i32) / 2,
    );
    let green = clip255(
        ((avg >> 8) & 0xff) as i32
            + (((avg >> 8) & 0xff) as i32 - ((top_left >> 8) & 0xff) as i32) / 2,
    );
    let blue = clip255((avg & 0xff) as i32 + ((avg & 0xff) as i32 - (top_left & 0xff) as i32) / 2);
    (alpha << 24) | (red << 16) | (green << 8) | blue
}

fn select_predictor(left: u32, top: u32, top_left: u32) -> u32 {
    let pred_alpha = ((left >> 24) as i32) + ((top >> 24) as i32) - ((top_left >> 24) as i32);
    let pred_red = ((left >> 16) & 0xff) as i32 + ((top >> 16) & 0xff) as i32
        - ((top_left >> 16) & 0xff) as i32;
    let pred_green =
        ((left >> 8) & 0xff) as i32 + ((top >> 8) & 0xff) as i32 - ((top_left >> 8) & 0xff) as i32;
    let pred_blue = (left & 0xff) as i32 + (top & 0xff) as i32 - (top_left & 0xff) as i32;

    let left_distance = (pred_alpha - ((left >> 24) as i32)).abs()
        + (pred_red - (((left >> 16) & 0xff) as i32)).abs()
        + (pred_green - (((left >> 8) & 0xff) as i32)).abs()
        + (pred_blue - ((left & 0xff) as i32)).abs();
    let top_distance = (pred_alpha - ((top >> 24) as i32)).abs()
        + (pred_red - (((top >> 16) & 0xff) as i32)).abs()
        + (pred_green - (((top >> 8) & 0xff) as i32)).abs()
        + (pred_blue - ((top & 0xff) as i32)).abs();

    if left_distance < top_distance {
        left
    } else {
        top
    }
}

fn predictor(mode: u8, left: u32, top: u32, top_left: u32, top_right: u32) -> u32 {
    match mode {
        0 | 14 | 15 => ARGB_BLACK,
        1 => left,
        2 => top,
        3 => top_right,
        4 => top_left,
        5 => average2(average2(left, top_right), top),
        6 => average2(left, top_left),
        7 => average2(left, top),
        8 => average2(top_left, top),
        9 => average2(top, top_right),
        10 => average2(average2(left, top_left), average2(top, top_right)),
        11 => select_predictor(left, top, top_left),
        12 => clamped_add_subtract_full(left, top, top_left),
        13 => clamped_add_subtract_half(left, top, top_left),
        _ => ARGB_BLACK,
    }
}

fn color_transform_delta(transform: u8, color: u8) -> i32 {
    ((transform as i8 as i32) * (color as i8 as i32)) >> 5
}

fn expand_color_map(palette: &[u32], num_colors: usize, bits: usize) -> Vec<u32> {
    let final_num_colors = 1usize << (8 >> bits);
    let mut expanded = vec![0u32; final_num_colors];
    if num_colors == 0 {
        return expanded;
    }

    expanded[0] = palette[0];
    for i in 1..num_colors {
        expanded[i] = add_pixels(palette[i], expanded[i - 1]);
    }
    expanded
}

fn apply_inverse_transform(transform: &Transform, input: &[u32]) -> Result<Vec<u32>, DecoderError> {
    match transform.kind {
        TransformType::SubtractGreen => Ok(input
            .iter()
            .map(|&argb| {
                let green = (argb >> 8) & 0xff;
                let red = (((argb >> 16) & 0xff) + green) & 0xff;
                let blue = ((argb & 0xff) + green) & 0xff;
                (argb & 0xff00_ff00) | (red << 16) | blue
            })
            .collect()),
        TransformType::CrossColor => {
            let expected_len = transform
                .xsize
                .checked_mul(transform.ysize)
                .ok_or(DecoderError::Bitstream("VP8L transform size overflow"))?;
            if input.len() != expected_len {
                return Err(DecoderError::Bitstream("VP8L cross-color size mismatch"));
            }
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
                        (argb & 0xff00_ff00) | ((red as u32) << 16) | (blue as u32);
                }
            }
            Ok(output)
        }
        TransformType::Predictor => {
            let expected_len = transform
                .xsize
                .checked_mul(transform.ysize)
                .ok_or(DecoderError::Bitstream("VP8L transform size overflow"))?;
            if input.len() != expected_len {
                return Err(DecoderError::Bitstream("VP8L predictor size mismatch"));
            }
            let tiles_per_row = subsample_size(transform.xsize, transform.bits);
            let mut output = vec![0u32; input.len()];
            for y in 0..transform.ysize {
                for x in 0..transform.xsize {
                    let residual = input[y * transform.xsize + x];
                    let pred = if y == 0 {
                        if x == 0 {
                            ARGB_BLACK
                        } else {
                            output[y * transform.xsize + x - 1]
                        }
                    } else if x == 0 {
                        output[(y - 1) * transform.xsize]
                    } else {
                        let left = output[y * transform.xsize + x - 1];
                        let top = output[(y - 1) * transform.xsize + x];
                        let top_left = output[(y - 1) * transform.xsize + x - 1];
                        let top_right = if x + 1 < transform.xsize {
                            output[(y - 1) * transform.xsize + x + 1]
                        } else {
                            output[y * transform.xsize]
                        };
                        let mode = ((transform.data
                            [(y >> transform.bits) * tiles_per_row + (x >> transform.bits)]
                            >> 8)
                            & 0x0f) as u8;
                        predictor(mode, left, top, top_left, top_right)
                    };
                    output[y * transform.xsize + x] = add_pixels(residual, pred);
                }
            }
            Ok(output)
        }
        TransformType::ColorIndexing => {
            let reduced_width = subsample_size(transform.xsize, transform.bits);
            let expected_len = reduced_width
                .checked_mul(transform.ysize)
                .ok_or(DecoderError::Bitstream("VP8L transform size overflow"))?;
            if input.len() != expected_len {
                return Err(DecoderError::Bitstream("VP8L color indexing size mismatch"));
            }

            let bits_per_pixel = 8 >> transform.bits;
            let pixels_per_byte = 1usize << transform.bits;
            let bit_mask = (1u32 << bits_per_pixel) - 1;
            let mut output = vec![0u32; transform.xsize * transform.ysize];

            if transform.bits == 0 {
                for (dst, &src) in output.iter_mut().zip(input.iter()) {
                    let index = ((src >> 8) & 0xff) as usize;
                    *dst = transform.data.get(index).copied().unwrap_or(0);
                }
                return Ok(output);
            }

            for y in 0..transform.ysize {
                let src_row = &input[y * reduced_width..(y + 1) * reduced_width];
                let dst_row = &mut output[y * transform.xsize..(y + 1) * transform.xsize];
                let mut x = 0usize;
                for &packed in src_row {
                    let mut indices = (packed >> 8) & 0xff;
                    for _ in 0..pixels_per_byte {
                        if x >= transform.xsize {
                            break;
                        }
                        let index = (indices & bit_mask) as usize;
                        dst_row[x] = transform.data.get(index).copied().unwrap_or(0);
                        indices >>= bits_per_pixel;
                        x += 1;
                    }
                }
            }

            Ok(output)
        }
    }
}

fn argb_to_rgba(argb: &[u32]) -> Vec<u8> {
    let mut rgba = vec![0u8; argb.len() * 4];
    for (index, &pixel) in argb.iter().enumerate() {
        let base = index * 4;
        rgba[base] = ((pixel >> 16) & 0xff) as u8;
        rgba[base + 1] = ((pixel >> 8) & 0xff) as u8;
        rgba[base + 2] = (pixel & 0xff) as u8;
        rgba[base + 3] = (pixel >> 24) as u8;
    }
    rgba
}

pub(crate) fn decode_lossless_vp8l_to_argb(
    data: &[u8],
) -> Result<(usize, usize, Vec<u32>), DecoderError> {
    let info = get_lossless_info(data)?;
    let bitstream = data
        .get(5..)
        .ok_or(DecoderError::NotEnoughData("VP8L frame payload"))?;
    let mut decoder = LosslessDecoder::new(bitstream);
    let argb = decoder.decode_image_stream(info.width, info.height, true)?;
    if argb.len() != info.width * info.height {
        return Err(DecoderError::Bitstream("decoded VP8L image has wrong size"));
    }
    Ok((info.width, info.height, argb))
}

pub(crate) fn decode_lossless_stream_to_argb(
    data: &[u8],
    width: usize,
    height: usize,
) -> Result<Vec<u32>, DecoderError> {
    let mut decoder = LosslessDecoder::new(data);
    let argb = decoder.decode_image_stream(width, height, true)?;
    if argb.len() != width * height {
        return Err(DecoderError::Bitstream("decoded VP8L image has wrong size"));
    }
    Ok(argb)
}

/// Decodes a raw `VP8L` frame payload to RGBA.
pub fn decode_lossless_vp8l_to_rgba(data: &[u8]) -> Result<DecodedImage, DecoderError> {
    let (width, height, argb) = decode_lossless_vp8l_to_argb(data)?;

    Ok(DecodedImage {
        width,
        height,
        rgba: argb_to_rgba(&argb),
    })
}

/// Decodes a still lossless WebP container to RGBA.
pub fn decode_lossless_webp_to_rgba(data: &[u8]) -> Result<DecodedImage, DecoderError> {
    let parsed = parse_still_webp(data)?;
    if parsed.features.format != WebpFormat::Lossless {
        return Err(DecoderError::Unsupported(
            "expected a still lossless WebP image",
        ));
    }
    decode_lossless_vp8l_to_rgba(parsed.image_data)
}
