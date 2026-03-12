use crate::decoder::quant::{AC_TABLE, DC_TABLE};
use crate::decoder::tree::{COEFFS_PROBA0, COEFFS_UPDATE_PROBA};
use crate::encoder::vp8_bool_writer::Vp8BoolWriter;
use crate::encoder::EncoderError;
use crate::ImageBuffer;

const MAX_WEBP_DIMENSION: usize = 1 << 14;
const MAX_PARTITION0_LENGTH: usize = (1 << 19) - 1;
const YUV_FIX: i32 = 16;
const YUV_HALF: i32 = 1 << (YUV_FIX - 1);
const VP8_TRANSFORM_AC3_C1: i32 = 20_091;
const VP8_TRANSFORM_AC3_C2: i32 = 35_468;

const CAT3: [u8; 4] = [173, 148, 140, 0];
const CAT4: [u8; 5] = [176, 155, 140, 135, 0];
const CAT5: [u8; 6] = [180, 157, 141, 134, 130, 0];
const CAT6: [u8; 12] = [254, 254, 243, 230, 196, 177, 153, 140, 133, 130, 129, 0];
const ZIGZAG: [usize; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];
const BANDS: [usize; 17] = [0, 1, 2, 3, 6, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 0];

/// Lossy encoder tuning knobs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LossyEncodingOptions {
    /// Quality from `0` to `100`.
    pub quality: u8,
}

impl Default for LossyEncodingOptions {
    fn default() -> Self {
        Self { quality: 90 }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct NonZeroContext {
    nz: u8,
    nz_dc: u8,
}

#[derive(Debug, Clone, Copy)]
struct QuantMatrices {
    y1: [u16; 2],
    y2: [u16; 2],
    uv: [u16; 2],
}

#[derive(Debug, Clone)]
struct Planes {
    y_stride: usize,
    uv_stride: usize,
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
}

fn validate_rgba(width: usize, height: usize, rgba: &[u8]) -> Result<(), EncoderError> {
    if width == 0 || height == 0 {
        return Err(EncoderError::InvalidParam(
            "image dimensions must be non-zero",
        ));
    }
    if width > MAX_WEBP_DIMENSION || height > MAX_WEBP_DIMENSION {
        return Err(EncoderError::InvalidParam(
            "image dimensions exceed VP8 limits",
        ));
    }
    let expected_len = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or(EncoderError::InvalidParam("image dimensions overflow"))?;
    if rgba.len() != expected_len {
        return Err(EncoderError::InvalidParam(
            "RGBA buffer length does not match dimensions",
        ));
    }
    if rgba.chunks_exact(4).any(|pixel| pixel[3] != 0xff) {
        return Err(EncoderError::InvalidParam(
            "lossy encoder does not support alpha yet",
        ));
    }
    Ok(())
}

fn validate_options(options: &LossyEncodingOptions) -> Result<(), EncoderError> {
    if options.quality > 100 {
        return Err(EncoderError::InvalidParam(
            "lossy quality must be in 0..=100",
        ));
    }
    Ok(())
}

fn base_quantizer_from_quality(quality: u8) -> i32 {
    (((100 - quality as i32) * 127) + 50) / 100
}

fn build_quant_matrices(base_q: i32) -> QuantMatrices {
    let q = base_q.clamp(0, 127) as usize;
    QuantMatrices {
        y1: [DC_TABLE[q] as u16, AC_TABLE[q]],
        y2: [
            (DC_TABLE[q] as u16) * 2,
            ((AC_TABLE[q] as u32 * 101_581) >> 16).max(8) as u16,
        ],
        uv: [DC_TABLE[q.min(117)] as u16, AC_TABLE[q]],
    }
}

fn rgb_to_y(r: u8, g: u8, b: u8) -> u8 {
    let luma = 16_839 * r as i32 + 33_059 * g as i32 + 6_420 * b as i32;
    ((luma + YUV_HALF + (16 << YUV_FIX)) >> YUV_FIX) as u8
}

fn clip_uv(value: i32, rounding: i32) -> u8 {
    let uv = (value + rounding + (128 << (YUV_FIX + 2))) >> (YUV_FIX + 2);
    uv.clamp(0, 255) as u8
}

fn rgb_to_u(r: i32, g: i32, b: i32) -> u8 {
    clip_uv(-9_719 * r - 19_081 * g + 28_800 * b, YUV_HALF << 2)
}

fn rgb_to_v(r: i32, g: i32, b: i32) -> u8 {
    clip_uv(28_800 * r - 24_116 * g - 4_684 * b, YUV_HALF << 2)
}

fn rgba_to_yuv420(
    width: usize,
    height: usize,
    rgba: &[u8],
    mb_width: usize,
    mb_height: usize,
) -> Planes {
    let y_stride = mb_width * 16;
    let uv_stride = mb_width * 8;
    let y_height = mb_height * 16;
    let uv_height = mb_height * 8;
    let mut y = vec![0u8; y_stride * y_height];
    let mut u = vec![0u8; uv_stride * uv_height];
    let mut v = vec![0u8; uv_stride * uv_height];

    for py in 0..y_height {
        let src_y = py.min(height - 1);
        for px in 0..y_stride {
            let src_x = px.min(width - 1);
            let offset = (src_y * width + src_x) * 4;
            y[py * y_stride + px] = rgb_to_y(rgba[offset], rgba[offset + 1], rgba[offset + 2]);
        }
    }

    for py in 0..uv_height {
        for px in 0..uv_stride {
            let mut sum_r = 0i32;
            let mut sum_g = 0i32;
            let mut sum_b = 0i32;
            for dy in 0..2 {
                let src_y = (py * 2 + dy).min(height - 1);
                for dx in 0..2 {
                    let src_x = (px * 2 + dx).min(width - 1);
                    let offset = (src_y * width + src_x) * 4;
                    sum_r += rgba[offset] as i32;
                    sum_g += rgba[offset + 1] as i32;
                    sum_b += rgba[offset + 2] as i32;
                }
            }
            u[py * uv_stride + px] = rgb_to_u(sum_r, sum_g, sum_b);
            v[py * uv_stride + px] = rgb_to_v(sum_r, sum_g, sum_b);
        }
    }

    Planes {
        y_stride,
        uv_stride,
        y,
        u,
        v,
    }
}

fn empty_reconstructed_planes(mb_width: usize, mb_height: usize) -> Planes {
    let y_stride = mb_width * 16;
    let uv_stride = mb_width * 8;
    let y_height = mb_height * 16;
    let uv_height = mb_height * 8;
    Planes {
        y_stride,
        uv_stride,
        y: vec![0; y_stride * y_height],
        u: vec![0; uv_stride * uv_height],
        v: vec![0; uv_stride * uv_height],
    }
}

fn clip_byte(value: i32) -> u8 {
    value.clamp(0, 255) as u8
}

fn mul1(value: i32) -> i32 {
    ((value * VP8_TRANSFORM_AC3_C1) >> 16) + value
}

fn mul2(value: i32) -> i32 {
    (value * VP8_TRANSFORM_AC3_C2) >> 16
}

fn fill_block(
    plane: &mut [u8],
    stride: usize,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    value: u8,
) {
    for row in 0..height {
        let offset = (y + row) * stride + x;
        plane[offset..offset + width].fill(value);
    }
}

fn dc_predict_value(plane: &[u8], stride: usize, x: usize, y: usize, size: usize) -> u8 {
    let has_top = y > 0;
    let has_left = x > 0;
    match (has_top, has_left) {
        (true, true) => {
            let top_row = (y - 1) * stride;
            let sum_top: u32 = (0..size).map(|i| plane[top_row + x + i] as u32).sum();
            let sum_left: u32 = (0..size)
                .map(|i| plane[(y + i) * stride + x - 1] as u32)
                .sum();
            ((sum_top + sum_left + size as u32) >> (size.trailing_zeros() + 1)) as u8
        }
        (true, false) => {
            let top_row = (y - 1) * stride;
            let sum_top: u32 = (0..size).map(|i| plane[top_row + x + i] as u32).sum();
            ((sum_top + (size as u32 >> 1)) >> size.trailing_zeros()) as u8
        }
        (false, true) => {
            let sum_left: u32 = (0..size)
                .map(|i| plane[(y + i) * stride + x - 1] as u32)
                .sum();
            ((sum_left + (size as u32 >> 1)) >> size.trailing_zeros()) as u8
        }
        (false, false) => 128,
    }
}

fn predict_dc_block(plane: &mut [u8], stride: usize, x: usize, y: usize, size: usize) {
    let value = dc_predict_value(plane, stride, x, y, size);
    fill_block(plane, stride, x, y, size, size, value);
}

fn add_transform(plane: &mut [u8], stride: usize, x: usize, y: usize, coeffs: &[i16; 16]) {
    if coeffs.iter().all(|&coeff| coeff == 0) {
        return;
    }

    let mut tmp = [0i32; 16];
    for i in 0..4 {
        let a = coeffs[i] as i32 + coeffs[8 + i] as i32;
        let b = coeffs[i] as i32 - coeffs[8 + i] as i32;
        let c = mul2(coeffs[4 + i] as i32) - mul1(coeffs[12 + i] as i32);
        let d = mul1(coeffs[4 + i] as i32) + mul2(coeffs[12 + i] as i32);
        let base = i * 4;
        tmp[base] = a + d;
        tmp[base + 1] = b + c;
        tmp[base + 2] = b - c;
        tmp[base + 3] = a - d;
    }

    for row in 0..4 {
        let dc = tmp[row] + 4;
        let a = dc + tmp[8 + row];
        let b = dc - tmp[8 + row];
        let c = mul2(tmp[4 + row]) - mul1(tmp[12 + row]);
        let d = mul1(tmp[4 + row]) + mul2(tmp[12 + row]);
        let offset = (y + row) * stride + x;
        plane[offset] = clip_byte(plane[offset] as i32 + ((a + d) >> 3));
        plane[offset + 1] = clip_byte(plane[offset + 1] as i32 + ((b + c) >> 3));
        plane[offset + 2] = clip_byte(plane[offset + 2] as i32 + ((b - c) >> 3));
        plane[offset + 3] = clip_byte(plane[offset + 3] as i32 + ((a - d) >> 3));
    }
}

fn forward_transform(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    x: usize,
    y: usize,
) -> [i16; 16] {
    let mut tmp = [0i32; 16];
    for row in 0..4 {
        let src_offset = (y + row) * src_stride + x;
        let pred_offset = (y + row) * pred_stride + x;
        let d0 = src[src_offset] as i32 - pred[pred_offset] as i32;
        let d1 = src[src_offset + 1] as i32 - pred[pred_offset + 1] as i32;
        let d2 = src[src_offset + 2] as i32 - pred[pred_offset + 2] as i32;
        let d3 = src[src_offset + 3] as i32 - pred[pred_offset + 3] as i32;
        let a0 = d0 + d3;
        let a1 = d1 + d2;
        let a2 = d1 - d2;
        let a3 = d0 - d3;
        tmp[row * 4] = (a0 + a1) * 8;
        tmp[row * 4 + 1] = (a2 * 2_217 + a3 * 5_352 + 1_812) >> 9;
        tmp[row * 4 + 2] = (a0 - a1) * 8;
        tmp[row * 4 + 3] = (a3 * 2_217 - a2 * 5_352 + 937) >> 9;
    }

    let mut out = [0i16; 16];
    for i in 0..4 {
        let a0 = tmp[i] + tmp[12 + i];
        let a1 = tmp[4 + i] + tmp[8 + i];
        let a2 = tmp[4 + i] - tmp[8 + i];
        let a3 = tmp[i] - tmp[12 + i];
        out[i] = ((a0 + a1 + 7) >> 4) as i16;
        out[4 + i] = (((a2 * 2_217 + a3 * 5_352 + 12_000) >> 16) + (a3 != 0) as i32) as i16;
        out[8 + i] = ((a0 - a1 + 7) >> 4) as i16;
        out[12 + i] = ((a3 * 2_217 - a2 * 5_352 + 51_000) >> 16) as i16;
    }
    out
}

fn forward_wht(input: &[i16; 16]) -> [i16; 16] {
    let mut tmp = [0i32; 16];
    for row in 0..4 {
        let base = row * 4;
        let a0 = input[base] as i32 + input[base + 2] as i32;
        let a1 = input[base + 1] as i32 + input[base + 3] as i32;
        let a2 = input[base + 1] as i32 - input[base + 3] as i32;
        let a3 = input[base] as i32 - input[base + 2] as i32;
        tmp[base] = a0 + a1;
        tmp[base + 1] = a3 + a2;
        tmp[base + 2] = a3 - a2;
        tmp[base + 3] = a0 - a1;
    }

    let mut out = [0i16; 16];
    for i in 0..4 {
        let a0 = tmp[i] + tmp[8 + i];
        let a1 = tmp[4 + i] + tmp[12 + i];
        let a2 = tmp[4 + i] - tmp[12 + i];
        let a3 = tmp[i] - tmp[8 + i];
        let b0 = a0 + a1;
        let b1 = a3 + a2;
        let b2 = a3 - a2;
        let b3 = a0 - a1;
        out[i] = (b0 >> 1) as i16;
        out[4 + i] = (b1 >> 1) as i16;
        out[8 + i] = (b2 >> 1) as i16;
        out[12 + i] = (b3 >> 1) as i16;
    }
    out
}

fn inverse_wht(input: &[i16; 16]) -> [i16; 16] {
    let mut tmp = [0i32; 16];
    for i in 0..4 {
        let a0 = input[i] as i32 + input[12 + i] as i32;
        let a1 = input[4 + i] as i32 + input[8 + i] as i32;
        let a2 = input[4 + i] as i32 - input[8 + i] as i32;
        let a3 = input[i] as i32 - input[12 + i] as i32;
        tmp[i] = a0 + a1;
        tmp[8 + i] = a0 - a1;
        tmp[4 + i] = a3 + a2;
        tmp[12 + i] = a3 - a2;
    }

    let mut out = [0i16; 16];
    for row in 0..4 {
        let base = row * 4;
        let dc = tmp[base] + 3;
        let a0 = dc + tmp[base + 3];
        let a1 = tmp[base + 1] + tmp[base + 2];
        let a2 = tmp[base + 1] - tmp[base + 2];
        let a3 = dc - tmp[base + 3];
        out[base] = ((a0 + a1) >> 3) as i16;
        out[base + 1] = ((a3 + a2) >> 3) as i16;
        out[base + 2] = ((a0 - a1) >> 3) as i16;
        out[base + 3] = ((a3 - a2) >> 3) as i16;
    }
    out
}

fn quantize_coefficient(coeff: i16, quant: u16) -> (i16, i16) {
    if quant == 0 {
        return (0, 0);
    }
    let sign = if coeff < 0 { -1 } else { 1 };
    let abs = coeff.unsigned_abs() as i32;
    let quant = quant as i32;
    let level = ((abs + (quant >> 1)) / quant).min(2_047);
    let level = sign * level;
    (level as i16, (level * quant) as i16)
}

fn quantize_block(
    coeffs: &[i16; 16],
    dc_quant: u16,
    ac_quant: u16,
    first: usize,
) -> ([i16; 16], [i16; 16]) {
    let mut levels = [0i16; 16];
    let mut dequantized = [0i16; 16];
    for (index, coeff) in coeffs.iter().copied().enumerate().skip(first) {
        let quant = if index == 0 { dc_quant } else { ac_quant };
        let (level, dequant) = quantize_coefficient(coeff, quant);
        levels[index] = level;
        dequantized[index] = dequant;
    }
    (levels, dequantized)
}

fn coeff_probs(coeff_type: usize, coeff_index: usize, ctx: usize) -> &'static [u8; 11] {
    &COEFFS_PROBA0[coeff_type][BANDS[coeff_index]][ctx]
}

fn last_non_zero(levels: &[i16; 16], first: usize) -> isize {
    for scan in (first..16).rev() {
        if levels[ZIGZAG[scan]] != 0 {
            return scan as isize;
        }
    }
    first as isize - 1
}

fn write_large_value(writer: &mut Vp8BoolWriter, value: u32, probs: &[u8; 11]) {
    if !writer.put_bit(value > 4, probs[3]) {
        if writer.put_bit(value != 2, probs[4]) {
            writer.put_bit(value == 4, probs[5]);
        }
        return;
    }

    if !writer.put_bit(value > 10, probs[6]) {
        if !writer.put_bit(value > 6, probs[7]) {
            writer.put_bit(value == 6, 159);
        } else {
            writer.put_bit(value >= 9, 165);
            writer.put_bit((value & 1) == 0, 145);
        }
        return;
    }

    let (residue, mask, table): (u32, u32, &[u8]) = if value < 19 {
        writer.put_bit(false, probs[8]);
        writer.put_bit(false, probs[9]);
        (value - 11, 1 << 2, &CAT3)
    } else if value < 35 {
        writer.put_bit(false, probs[8]);
        writer.put_bit(true, probs[9]);
        (value - 19, 1 << 3, &CAT4)
    } else if value < 67 {
        writer.put_bit(true, probs[8]);
        writer.put_bit(false, probs[10]);
        (value - 35, 1 << 4, &CAT5)
    } else {
        writer.put_bit(true, probs[8]);
        writer.put_bit(true, probs[10]);
        (value - 67, 1 << 10, &CAT6)
    };

    let mut mask = mask;
    for &prob in table {
        if prob == 0 {
            break;
        }
        writer.put_bit((residue & mask) != 0, prob);
        mask >>= 1;
    }
}

fn encode_coefficients(
    writer: &mut Vp8BoolWriter,
    coeff_type: usize,
    ctx: usize,
    first: usize,
    levels: &[i16; 16],
) -> bool {
    let last = last_non_zero(levels, first);
    let mut scan = first;
    let mut probs = coeff_probs(coeff_type, scan, ctx);
    if !writer.put_bit(last >= scan as isize, probs[0]) {
        return false;
    }

    while scan < 16 {
        let coeff = levels[ZIGZAG[scan]];
        writer.put_bit(coeff != 0, probs[1]);
        scan += 1;
        if coeff == 0 {
            if scan == 16 {
                return false;
            }
            probs = coeff_probs(coeff_type, scan, 0);
            continue;
        }

        let value = coeff.unsigned_abs() as u32;
        let next_ctx = if !writer.put_bit(value > 1, probs[2]) {
            1
        } else {
            write_large_value(writer, value, probs);
            2
        };
        writer.put_bit(coeff < 0, 128);

        if scan == 16 {
            return true;
        }
        probs = coeff_probs(coeff_type, scan, next_ctx);
        if !writer.put_bit(last >= scan as isize, probs[0]) {
            return true;
        }
    }
    true
}

fn encode_partition0(mb_width: usize, mb_height: usize, base_quant: u8) -> Vec<u8> {
    let mut writer = Vp8BoolWriter::new(mb_width * mb_height);
    writer.put_bit_uniform(false);
    writer.put_bit_uniform(false);

    writer.put_bit_uniform(false);

    writer.put_bit_uniform(false);
    writer.put_bits(0, 6);
    writer.put_bits(0, 3);
    writer.put_bit_uniform(false);

    writer.put_bits(0, 2);
    writer.put_bits(base_quant as u32, 7);
    for _ in 0..5 {
        writer.put_signed_bits(0, 4);
    }
    writer.put_bit_uniform(false);

    for tables in COEFFS_UPDATE_PROBA.iter() {
        for bands in tables.iter() {
            for contexts in bands.iter() {
                for &prob in contexts {
                    writer.put_bit(false, prob);
                }
            }
        }
    }
    writer.put_bit_uniform(false);

    for _ in 0..(mb_width * mb_height) {
        writer.put_bit(true, 145);
        writer.put_bit(false, 156);
        writer.put_bit(false, 163);
        writer.put_bit(false, 142);
    }

    writer.finish()
}

fn encode_macroblock(
    writer: &mut Vp8BoolWriter,
    source: &Planes,
    reconstructed: &mut Planes,
    mb_x: usize,
    mb_y: usize,
    quant: &QuantMatrices,
    top: &mut NonZeroContext,
    left: &mut NonZeroContext,
) {
    let y_x = mb_x * 16;
    let y_y = mb_y * 16;
    let uv_x = mb_x * 8;
    let uv_y = mb_y * 8;

    predict_dc_block(&mut reconstructed.y, reconstructed.y_stride, y_x, y_y, 16);
    predict_dc_block(&mut reconstructed.u, reconstructed.uv_stride, uv_x, uv_y, 8);
    predict_dc_block(&mut reconstructed.v, reconstructed.uv_stride, uv_x, uv_y, 8);

    let mut y_levels = [[0i16; 16]; 16];
    let mut y_coeffs = [[0i16; 16]; 16];
    let mut y_dc = [0i16; 16];
    for sub_y in 0..4 {
        for sub_x in 0..4 {
            let block = sub_y * 4 + sub_x;
            let coeffs = forward_transform(
                &source.y,
                source.y_stride,
                &reconstructed.y,
                reconstructed.y_stride,
                y_x + sub_x * 4,
                y_y + sub_y * 4,
            );
            y_dc[block] = coeffs[0];
            let mut ac_only = coeffs;
            ac_only[0] = 0;
            let (levels, coeffs) = quantize_block(&ac_only, quant.y1[0], quant.y1[1], 1);
            y_levels[block] = levels;
            y_coeffs[block] = coeffs;
        }
    }

    let y2_input = forward_wht(&y_dc);
    let (y2_levels, y2_coeffs) = quantize_block(&y2_input, quant.y2[0], quant.y2[1], 0);
    let y2_dc = inverse_wht(&y2_coeffs);
    for block in 0..16 {
        y_coeffs[block][0] = y2_dc[block];
    }

    let has_y2 = encode_coefficients(writer, 1, (top.nz_dc + left.nz_dc) as usize, 0, &y2_levels);
    top.nz_dc = has_y2 as u8;
    left.nz_dc = has_y2 as u8;

    let mut tnz = top.nz & 0x0f;
    let mut lnz = left.nz & 0x0f;
    for sub_y in 0..4 {
        let mut l = lnz & 1;
        for sub_x in 0..4 {
            let block = sub_y * 4 + sub_x;
            let ctx = (l + (tnz & 1)) as usize;
            let has_ac = encode_coefficients(writer, 0, ctx, 1, &y_levels[block]);
            l = has_ac as u8;
            tnz = (tnz >> 1) | (l << 7);
        }
        tnz >>= 4;
        lnz = (lnz >> 1) | (l << 7);
    }
    let mut out_t_nz = tnz;
    let mut out_l_nz = lnz >> 4;

    let mut u_levels = [[0i16; 16]; 4];
    let mut u_coeffs = [[0i16; 16]; 4];
    for sub_y in 0..2 {
        for sub_x in 0..2 {
            let block = sub_y * 2 + sub_x;
            let coeffs = forward_transform(
                &source.u,
                source.uv_stride,
                &reconstructed.u,
                reconstructed.uv_stride,
                uv_x + sub_x * 4,
                uv_y + sub_y * 4,
            );
            let (levels, coeffs) = quantize_block(&coeffs, quant.uv[0], quant.uv[1], 0);
            u_levels[block] = levels;
            u_coeffs[block] = coeffs;
        }
    }

    let mut v_levels = [[0i16; 16]; 4];
    let mut v_coeffs = [[0i16; 16]; 4];
    for sub_y in 0..2 {
        for sub_x in 0..2 {
            let block = sub_y * 2 + sub_x;
            let coeffs = forward_transform(
                &source.v,
                source.uv_stride,
                &reconstructed.v,
                reconstructed.uv_stride,
                uv_x + sub_x * 4,
                uv_y + sub_y * 4,
            );
            let (levels, coeffs) = quantize_block(&coeffs, quant.uv[0], quant.uv[1], 0);
            v_levels[block] = levels;
            v_coeffs[block] = coeffs;
        }
    }

    let mut tnz_u = top.nz >> 4;
    let mut lnz_u = left.nz >> 4;
    for sub_y in 0..2 {
        let mut l = lnz_u & 1;
        for sub_x in 0..2 {
            let block = sub_y * 2 + sub_x;
            let ctx = (l + (tnz_u & 1)) as usize;
            let has_coeffs = encode_coefficients(writer, 2, ctx, 0, &u_levels[block]) as u8;
            l = has_coeffs;
            tnz_u = (tnz_u >> 1) | (has_coeffs << 3);
        }
        tnz_u >>= 2;
        lnz_u = (lnz_u >> 1) | (l << 5);
    }
    out_t_nz |= tnz_u << 4;
    out_l_nz |= lnz_u & 0xf0;

    let mut tnz_v = top.nz >> 6;
    let mut lnz_v = left.nz >> 6;
    for sub_y in 0..2 {
        let mut l = lnz_v & 1;
        for sub_x in 0..2 {
            let block = sub_y * 2 + sub_x;
            let ctx = (l + (tnz_v & 1)) as usize;
            let has_coeffs = encode_coefficients(writer, 2, ctx, 0, &v_levels[block]) as u8;
            l = has_coeffs;
            tnz_v = (tnz_v >> 1) | (has_coeffs << 3);
        }
        tnz_v >>= 2;
        lnz_v = (lnz_v >> 1) | (l << 5);
    }
    out_t_nz |= (tnz_v << 4) << 2;
    out_l_nz |= (lnz_v & 0xf0) << 2;

    top.nz = out_t_nz;
    left.nz = out_l_nz;

    for sub_y in 0..4 {
        for sub_x in 0..4 {
            let block = sub_y * 4 + sub_x;
            add_transform(
                &mut reconstructed.y,
                reconstructed.y_stride,
                y_x + sub_x * 4,
                y_y + sub_y * 4,
                &y_coeffs[block],
            );
        }
    }

    for sub_y in 0..2 {
        for sub_x in 0..2 {
            let block = sub_y * 2 + sub_x;
            add_transform(
                &mut reconstructed.u,
                reconstructed.uv_stride,
                uv_x + sub_x * 4,
                uv_y + sub_y * 4,
                &u_coeffs[block],
            );
            add_transform(
                &mut reconstructed.v,
                reconstructed.uv_stride,
                uv_x + sub_x * 4,
                uv_y + sub_y * 4,
                &v_coeffs[block],
            );
        }
    }
}

fn encode_token_partition(
    source: &Planes,
    mb_width: usize,
    mb_height: usize,
    quant: &QuantMatrices,
) -> (Vec<u8>, Planes) {
    let mut writer = Vp8BoolWriter::new(source.y.len() / 4);
    let mut reconstructed = empty_reconstructed_planes(mb_width, mb_height);
    let mut top_contexts = vec![NonZeroContext::default(); mb_width];

    for mb_y in 0..mb_height {
        let mut left_context = NonZeroContext::default();
        for mb_x in 0..mb_width {
            encode_macroblock(
                &mut writer,
                source,
                &mut reconstructed,
                mb_x,
                mb_y,
                quant,
                &mut top_contexts[mb_x],
                &mut left_context,
            );
        }
    }

    (writer.finish(), reconstructed)
}

fn build_vp8_frame(
    width: usize,
    height: usize,
    partition0: &[u8],
    token_partition: &[u8],
) -> Result<Vec<u8>, EncoderError> {
    if partition0.len() > MAX_PARTITION0_LENGTH {
        return Err(EncoderError::Bitstream("VP8 partition 0 overflow"));
    }

    let payload_size = 10usize
        .checked_add(partition0.len())
        .and_then(|size| size.checked_add(token_partition.len()))
        .ok_or(EncoderError::InvalidParam("encoded output is too large"))?;

    let mut data = Vec::with_capacity(payload_size);
    let frame_bits = ((partition0.len() as u32) << 5) | (1 << 4);
    data.extend_from_slice(&[
        (frame_bits & 0xff) as u8,
        ((frame_bits >> 8) & 0xff) as u8,
        ((frame_bits >> 16) & 0xff) as u8,
        0x9d,
        0x01,
        0x2a,
        (width & 0xff) as u8,
        ((width >> 8) & 0xff) as u8,
        (height & 0xff) as u8,
        ((height >> 8) & 0xff) as u8,
    ]);
    data.extend_from_slice(partition0);
    data.extend_from_slice(token_partition);
    Ok(data)
}

fn wrap_lossy_webp(vp8: &[u8]) -> Result<Vec<u8>, EncoderError> {
    let padded_vp8_size = vp8.len() + (vp8.len() & 1);
    let riff_size = 4usize
        .checked_add(8)
        .and_then(|size| size.checked_add(padded_vp8_size))
        .ok_or(EncoderError::InvalidParam("encoded output is too large"))?;
    let total_size = 8usize
        .checked_add(riff_size)
        .ok_or(EncoderError::InvalidParam("encoded output is too large"))?;

    let mut data = Vec::with_capacity(total_size);
    data.extend_from_slice(b"RIFF");
    data.extend_from_slice(&(riff_size as u32).to_le_bytes());
    data.extend_from_slice(b"WEBP");
    data.extend_from_slice(b"VP8 ");
    data.extend_from_slice(&(vp8.len() as u32).to_le_bytes());
    data.extend_from_slice(vp8);
    if vp8.len() & 1 == 1 {
        data.push(0);
    }
    Ok(data)
}

/// Encodes RGBA pixels to a raw lossy `VP8` frame payload with explicit options.
pub fn encode_lossy_rgba_to_vp8_with_options(
    width: usize,
    height: usize,
    rgba: &[u8],
    options: &LossyEncodingOptions,
) -> Result<Vec<u8>, EncoderError> {
    validate_rgba(width, height, rgba)?;
    validate_options(options)?;

    let mb_width = (width + 15) >> 4;
    let mb_height = (height + 15) >> 4;
    let base_quant = base_quantizer_from_quality(options.quality);
    let quant = build_quant_matrices(base_quant);
    let source = rgba_to_yuv420(width, height, rgba, mb_width, mb_height);
    let partition0 = encode_partition0(mb_width, mb_height, base_quant as u8);
    let (token_partition, _) = encode_token_partition(&source, mb_width, mb_height, &quant);
    build_vp8_frame(width, height, &partition0, &token_partition)
}

/// Encodes RGBA pixels to a raw lossy `VP8` frame payload.
pub fn encode_lossy_rgba_to_vp8(
    width: usize,
    height: usize,
    rgba: &[u8],
) -> Result<Vec<u8>, EncoderError> {
    encode_lossy_rgba_to_vp8_with_options(width, height, rgba, &LossyEncodingOptions::default())
}

/// Encodes RGBA pixels to a still lossy WebP container with explicit options.
pub fn encode_lossy_rgba_to_webp_with_options(
    width: usize,
    height: usize,
    rgba: &[u8],
    options: &LossyEncodingOptions,
) -> Result<Vec<u8>, EncoderError> {
    let vp8 = encode_lossy_rgba_to_vp8_with_options(width, height, rgba, options)?;
    wrap_lossy_webp(&vp8)
}

/// Encodes RGBA pixels to a still lossy WebP container.
pub fn encode_lossy_rgba_to_webp(
    width: usize,
    height: usize,
    rgba: &[u8],
) -> Result<Vec<u8>, EncoderError> {
    encode_lossy_rgba_to_webp_with_options(width, height, rgba, &LossyEncodingOptions::default())
}

/// Encodes an [`ImageBuffer`] to a still lossy WebP container with explicit options.
pub fn encode_lossy_image_to_webp_with_options(
    image: &ImageBuffer,
    options: &LossyEncodingOptions,
) -> Result<Vec<u8>, EncoderError> {
    encode_lossy_rgba_to_webp_with_options(image.width, image.height, &image.rgba, options)
}

/// Encodes an [`ImageBuffer`] to a still lossy WebP container.
pub fn encode_lossy_image_to_webp(image: &ImageBuffer) -> Result<Vec<u8>, EncoderError> {
    encode_lossy_image_to_webp_with_options(image, &LossyEncodingOptions::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::decode_lossy_vp8_to_yuv;

    fn sample_rgba() -> (usize, usize, Vec<u8>) {
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

    #[test]
    fn internal_reconstruction_matches_decoder_output() {
        let (width, height, rgba) = sample_rgba();
        let mb_width = (width + 15) >> 4;
        let mb_height = (height + 15) >> 4;
        let options = LossyEncodingOptions::default();
        let base_quant = base_quantizer_from_quality(options.quality);
        let quant = build_quant_matrices(base_quant);
        let source = rgba_to_yuv420(width, height, &rgba, mb_width, mb_height);
        let partition0 = encode_partition0(mb_width, mb_height, base_quant as u8);
        let (token_partition, reconstructed) =
            encode_token_partition(&source, mb_width, mb_height, &quant);
        let vp8 = build_vp8_frame(width, height, &partition0, &token_partition).unwrap();
        let decoded = decode_lossy_vp8_to_yuv(&vp8).unwrap();
        assert_eq!(decoded.y, reconstructed.y);
        assert_eq!(decoded.u, reconstructed.u);
        assert_eq!(decoded.v, reconstructed.v);
    }
}
