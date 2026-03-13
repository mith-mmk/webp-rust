//! Prediction, transform, and mode-evaluation helpers for lossy encoding.

use super::bitstream::*;
use super::*;

/// Clamps byte.
pub(super) fn clip_byte(value: i32) -> u8 {
    value.clamp(0, 255) as u8
}

/// Internal helper for top left sample.
pub(super) fn top_left_sample(plane: &[u8], stride: usize, x: usize, y: usize) -> u8 {
    if y == 0 {
        127
    } else if x == 0 {
        129
    } else {
        plane[(y - 1) * stride + (x - 1)]
    }
}

/// Internal helper for top samples.
pub(super) fn top_samples<const N: usize>(
    plane: &[u8],
    stride: usize,
    plane_width: usize,
    x: usize,
    y: usize,
) -> [u8; N] {
    let mut out = [0u8; N];
    if y == 0 {
        out.fill(127);
        return out;
    }
    let row = (y - 1) * stride;
    for (i, sample) in out.iter_mut().enumerate() {
        let src_x = (x + i).min(plane_width - 1);
        *sample = plane[row + src_x];
    }
    out
}

/// Internal helper for top samples luma4.
pub(super) fn top_samples_luma4(
    plane: &[u8],
    stride: usize,
    plane_width: usize,
    x: usize,
    y: usize,
) -> [u8; 8] {
    let mut out = [0u8; 8];
    if y == 0 {
        out.fill(127);
        return out;
    }

    let row = (y - 1) * stride;
    for (i, sample) in out.iter_mut().enumerate().take(4) {
        let src_x = (x + i).min(plane_width - 1);
        *sample = plane[row + src_x];
    }

    let local_x = x & 15;
    let local_y = y & 15;
    if local_x == 12 && local_y != 0 {
        let macroblock_y = y - local_y;
        if macroblock_y == 0 {
            out[4..].fill(127);
        } else {
            let top_row = (macroblock_y - 1) * stride;
            for (i, sample) in out.iter_mut().enumerate().skip(4) {
                let src_x = (x + i).min(plane_width - 1);
                *sample = plane[top_row + src_x];
            }
        }
    } else {
        for (i, sample) in out.iter_mut().enumerate().skip(4) {
            let src_x = (x + i).min(plane_width - 1);
            *sample = plane[row + src_x];
        }
    }
    out
}

/// Internal helper for left samples.
pub(super) fn left_samples<const N: usize>(
    plane: &[u8],
    stride: usize,
    x: usize,
    y: usize,
) -> [u8; N] {
    let mut out = [0u8; N];
    if x == 0 {
        out.fill(129);
        return out;
    }
    let src_x = x - 1;
    for (i, sample) in out.iter_mut().enumerate() {
        *sample = plane[(y + i) * stride + src_x];
    }
    out
}

/// Averages two samples with integer rounding.
pub(super) fn avg2(a: u8, b: u8) -> u8 {
    ((a as u16 + b as u16 + 1) >> 1) as u8
}

/// Averages three samples with integer rounding.
pub(super) fn avg3(a: u8, b: u8, c: u8) -> u8 {
    ((a as u16 + 2 * b as u16 + c as u16 + 2) >> 2) as u8
}

/// Internal helper for fill prediction block.
pub(super) fn fill_prediction_block<const N: usize>(
    plane: &[u8],
    stride: usize,
    plane_width: usize,
    x: usize,
    y: usize,
    mode: u8,
    out: &mut [u8],
    out_stride: usize,
) {
    match mode {
        DC_PRED => {
            let value = dc_predict_value(plane, stride, x, y, N);
            for row in 0..N {
                let offset = row * out_stride;
                out[offset..offset + N].fill(value);
            }
        }
        V_PRED => {
            let top = top_samples::<N>(plane, stride, plane_width, x, y);
            for row in 0..N {
                let offset = row * out_stride;
                out[offset..offset + N].copy_from_slice(&top);
            }
        }
        H_PRED => {
            let left = left_samples::<N>(plane, stride, x, y);
            for (row, value) in left.into_iter().enumerate() {
                let offset = row * out_stride;
                out[offset..offset + N].fill(value);
            }
        }
        TM_PRED => {
            let top = top_samples::<N>(plane, stride, plane_width, x, y);
            let left = left_samples::<N>(plane, stride, x, y);
            let top_left = top_left_sample(plane, stride, x, y) as i32;
            for row in 0..N {
                let left_value = left[row] as i32;
                let offset = row * out_stride;
                for col in 0..N {
                    out[offset + col] = clip_byte(left_value + top[col] as i32 - top_left);
                }
            }
        }
        _ => unreachable!("unsupported macroblock prediction mode"),
    }
}

/// Internal helper for fill luma4 prediction block.
pub(super) fn fill_luma4_prediction_block(
    plane: &[u8],
    stride: usize,
    plane_width: usize,
    x: usize,
    y: usize,
    mode: u8,
    out: &mut [u8],
    out_stride: usize,
) {
    let x0 = top_left_sample(plane, stride, x, y);
    let top = top_samples_luma4(plane, stride, plane_width, x, y);
    let left = left_samples::<4>(plane, stride, x, y);

    let a = top[0];
    let b = top[1];
    let c = top[2];
    let d = top[3];
    let e = top[4];
    let f = top[5];
    let g = top[6];
    let h = top[7];
    let i = left[0];
    let j = left[1];
    let k = left[2];
    let l = left[3];

    let mut block = [0u8; 16];
    match mode {
        B_DC_PRED => {
            let sum_top: u32 = [a, b, c, d].into_iter().map(u32::from).sum();
            let sum_left: u32 = [i, j, k, l].into_iter().map(u32::from).sum();
            let dc = ((sum_top + sum_left + 4) >> 3) as u8;
            block.fill(dc);
        }
        B_TM_PRED => {
            let top_left = x0 as i32;
            for row in 0..4 {
                let left_value = left[row] as i32;
                for col in 0..4 {
                    block[row * 4 + col] = clip_byte(left_value + top[col] as i32 - top_left);
                }
            }
        }
        B_VE_PRED => {
            let vals = [avg3(x0, a, b), avg3(a, b, c), avg3(b, c, d), avg3(c, d, e)];
            for row in 0..4 {
                block[row * 4..row * 4 + 4].copy_from_slice(&vals);
            }
        }
        B_HE_PRED => {
            let vals = [avg3(x0, i, j), avg3(i, j, k), avg3(j, k, l), avg3(k, l, l)];
            for (row, value) in vals.into_iter().enumerate() {
                block[row * 4..row * 4 + 4].fill(value);
            }
        }
        B_RD_PRED => {
            block[12] = avg3(j, k, l);
            block[13] = avg3(i, j, k);
            block[8] = block[13];
            block[14] = avg3(x0, i, j);
            block[9] = block[14];
            block[4] = block[14];
            block[15] = avg3(a, x0, i);
            block[10] = block[15];
            block[5] = block[15];
            block[0] = block[15];
            block[11] = avg3(b, a, x0);
            block[6] = block[11];
            block[1] = block[11];
            block[7] = avg3(c, b, a);
            block[2] = block[7];
            block[3] = avg3(d, c, b);
        }
        B_LD_PRED => {
            block[0] = avg3(a, b, c);
            block[1] = avg3(b, c, d);
            block[4] = block[1];
            block[2] = avg3(c, d, e);
            block[5] = block[2];
            block[8] = block[2];
            block[3] = avg3(d, e, f);
            block[6] = block[3];
            block[9] = block[3];
            block[12] = block[3];
            block[7] = avg3(e, f, g);
            block[10] = block[7];
            block[13] = block[7];
            block[11] = avg3(f, g, h);
            block[14] = block[11];
            block[15] = avg3(g, h, h);
        }
        B_VR_PRED => {
            block[0] = avg2(x0, a);
            block[9] = block[0];
            block[1] = avg2(a, b);
            block[10] = block[1];
            block[2] = avg2(b, c);
            block[11] = block[2];
            block[3] = avg2(c, d);
            block[12] = avg3(k, j, i);
            block[8] = avg3(j, i, x0);
            block[4] = avg3(i, x0, a);
            block[13] = block[4];
            block[5] = avg3(x0, a, b);
            block[14] = block[5];
            block[6] = avg3(a, b, c);
            block[15] = block[6];
            block[7] = avg3(b, c, d);
        }
        B_VL_PRED => {
            block[0] = avg2(a, b);
            block[1] = avg2(b, c);
            block[2] = avg2(c, d);
            block[3] = avg2(d, e);
            block[4] = avg3(a, b, c);
            block[5] = avg3(b, c, d);
            block[6] = avg3(c, d, e);
            block[7] = avg3(d, e, f);
            block[8] = block[1];
            block[9] = block[2];
            block[10] = block[3];
            block[11] = avg3(e, f, g);
            block[12] = block[5];
            block[13] = block[6];
            block[14] = block[7];
            block[15] = avg3(f, g, h);
        }
        B_HD_PRED => {
            block[0] = avg2(i, x0);
            block[1] = avg3(i, x0, a);
            block[2] = avg3(x0, a, b);
            block[3] = avg3(a, b, c);
            block[4] = avg2(j, i);
            block[5] = avg3(j, i, x0);
            block[6] = block[0];
            block[7] = block[1];
            block[8] = avg2(k, j);
            block[9] = avg3(k, j, i);
            block[10] = block[4];
            block[11] = block[5];
            block[12] = avg2(l, k);
            block[13] = avg3(l, k, j);
            block[14] = block[8];
            block[15] = block[9];
        }
        B_HU_PRED => {
            block[0] = avg2(i, j);
            block[2] = avg2(j, k);
            block[4] = block[2];
            block[6] = avg2(k, l);
            block[8] = block[6];
            block[1] = avg3(i, j, k);
            block[3] = avg3(j, k, l);
            block[5] = block[3];
            block[7] = avg3(k, l, l);
            block[9] = block[7];
            block[10] = l;
            block[11] = l;
            block[12] = l;
            block[13] = l;
            block[14] = l;
            block[15] = l;
        }
        _ => unreachable!("unsupported 4x4 prediction mode"),
    }

    for row in 0..4 {
        let src = row * 4;
        let dst = row * out_stride;
        out[dst..dst + 4].copy_from_slice(&block[src..src + 4]);
    }
}

/// Predicts block.
pub(super) fn predict_block<const N: usize>(
    plane: &mut [u8],
    stride: usize,
    plane_width: usize,
    x: usize,
    y: usize,
    mode: u8,
) {
    let mut block = vec![0u8; N * N];
    fill_prediction_block::<N>(plane, stride, plane_width, x, y, mode, &mut block, N);
    for row in 0..N {
        let src = row * N;
        let dst = (y + row) * stride + x;
        plane[dst..dst + N].copy_from_slice(&block[src..src + N]);
    }
}

/// Predicts luma4 block.
pub(super) fn predict_luma4_block(
    plane: &mut [u8],
    stride: usize,
    plane_width: usize,
    x: usize,
    y: usize,
    mode: u8,
) {
    let mut block = [0u8; 16];
    fill_luma4_prediction_block(plane, stride, plane_width, x, y, mode, &mut block, 4);
    for row in 0..4 {
        let src = row * 4;
        let dst = (y + row) * stride + x;
        plane[dst..dst + 4].copy_from_slice(&block[src..src + 4]);
    }
}

/// Copies block4.
pub(super) fn copy_block4(plane: &[u8], stride: usize, x: usize, y: usize) -> [u8; 16] {
    let mut block = [0u8; 16];
    for row in 0..4 {
        let src = (y + row) * stride + x;
        block[row * 4..row * 4 + 4].copy_from_slice(&plane[src..src + 4]);
    }
    block
}

/// Restores block4.
pub(super) fn restore_block4(
    plane: &mut [u8],
    stride: usize,
    x: usize,
    y: usize,
    block: &[u8; 16],
) {
    for row in 0..4 {
        let dst = (y + row) * stride + x;
        plane[dst..dst + 4].copy_from_slice(&block[row * 4..row * 4 + 4]);
    }
}

/// Copies block4 from buffer.
pub(super) fn copy_block4_from_buffer(
    buffer: &[u8],
    stride: usize,
    x: usize,
    y: usize,
) -> [u8; 16] {
    let mut block = [0u8; 16];
    for row in 0..4 {
        let src = (y + row) * stride + x;
        block[row * 4..row * 4 + 4].copy_from_slice(&buffer[src..src + 4]);
    }
    block
}

/// Copies block16.
pub(super) fn copy_block16(plane: &[u8], stride: usize, x: usize, y: usize) -> [u8; 256] {
    let mut block = [0u8; 256];
    for row in 0..16 {
        let src = (y + row) * stride + x;
        block[row * 16..row * 16 + 16].copy_from_slice(&plane[src..src + 16]);
    }
    block
}

/// Restores block16.
pub(super) fn restore_block16(
    plane: &mut [u8],
    stride: usize,
    x: usize,
    y: usize,
    block: &[u8; 256],
) {
    for row in 0..16 {
        let dst = (y + row) * stride + x;
        plane[dst..dst + 16].copy_from_slice(&block[row * 16..row * 16 + 16]);
    }
}

/// Applies the first scaled multiply used by the VP8 transform.
pub(super) fn mul1(value: i32) -> i32 {
    ((value * VP8_TRANSFORM_AC3_C1) >> 16) + value
}

/// Applies the second scaled multiply used by the VP8 transform.
pub(super) fn mul2(value: i32) -> i32 {
    (value * VP8_TRANSFORM_AC3_C2) >> 16
}

/// Internal helper for dc predict value.
pub(super) fn dc_predict_value(plane: &[u8], stride: usize, x: usize, y: usize, size: usize) -> u8 {
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

/// Internal helper for add transform.
pub(super) fn add_transform(
    plane: &mut [u8],
    stride: usize,
    x: usize,
    y: usize,
    coeffs: &[i16; 16],
) {
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

/// Internal helper for forward transform at.
pub(super) fn forward_transform_at(
    src: &[u8],
    src_stride: usize,
    src_x: usize,
    src_y: usize,
    pred: &[u8],
    pred_stride: usize,
    pred_x: usize,
    pred_y: usize,
) -> [i16; 16] {
    let mut tmp = [0i32; 16];
    for row in 0..4 {
        let src_offset = (src_y + row) * src_stride + src_x;
        let pred_offset = (pred_y + row) * pred_stride + pred_x;
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

/// Internal helper for forward transform.
pub(super) fn forward_transform(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    x: usize,
    y: usize,
) -> [i16; 16] {
    forward_transform_at(src, src_stride, x, y, pred, pred_stride, x, y)
}

/// Internal helper for forward wht.
pub(super) fn forward_wht(input: &[i16; 16]) -> [i16; 16] {
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

/// Internal helper for inverse wht.
pub(super) fn inverse_wht(input: &[i16; 16]) -> [i16; 16] {
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

/// Quantizes coefficient.
pub(super) fn quantize_coefficient(coeff: i16, quant: u16) -> (i16, i16) {
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

/// Quantizes block.
pub(super) fn quantize_block(
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

/// Dequantizes levels.
pub(super) fn dequantize_levels(levels: &[i16; 16], dc_quant: u16, ac_quant: u16) -> [i16; 16] {
    let mut dequantized = [0i16; 16];
    for (index, level) in levels.iter().copied().enumerate() {
        let quant = if index == 0 { dc_quant } else { ac_quant } as i32;
        dequantized[index] = (i32::from(level) * quant) as i16;
    }
    dequantized
}

/// Internal helper for reconstruct from prediction.
pub(super) fn reconstruct_from_prediction(prediction: &[u8; 16], coeffs: &[i16; 16]) -> [u8; 16] {
    let mut block = *prediction;
    add_transform(&mut block, 4, 0, 0, coeffs);
    block
}

/// Internal helper for block sse 4x4.
pub(super) fn block_sse_4x4(
    source: &[u8],
    stride: usize,
    x: usize,
    y: usize,
    candidate: &[u8; 16],
) -> u64 {
    let mut sse = 0u64;
    for row in 0..4 {
        let src_offset = (y + row) * stride + x;
        let cand_offset = row * 4;
        for col in 0..4 {
            let diff = source[src_offset + col] as i32 - candidate[cand_offset + col] as i32;
            sse += (diff * diff) as u64;
        }
    }
    sse
}

/// Internal helper for reconstruct luma16 from prediction.
pub(super) fn reconstruct_luma16_from_prediction(
    prediction: &[u8; 256],
    ac_coeffs: &[[i16; 16]; 16],
    y2_coeffs: &[i16; 16],
) -> ([u8; 256], [i16; 16]) {
    let mut candidate = *prediction;
    let y2_dc = inverse_wht(y2_coeffs);
    for block in 0..16 {
        let mut coeffs = ac_coeffs[block];
        coeffs[0] = y2_dc[block];
        let sub_x = (block & 3) * 4;
        let sub_y = (block >> 2) * 4;
        add_transform(&mut candidate, 16, sub_x, sub_y, &coeffs);
    }
    (candidate, y2_dc)
}

/// Internal helper for refine levels greedy.
pub(super) fn refine_levels_greedy(
    source: &[u8],
    source_stride: usize,
    x: usize,
    y: usize,
    prediction: &[u8; 16],
    probabilities: &CoeffProbTables,
    coeff_type: usize,
    ctx: usize,
    first: usize,
    dc_quant: u16,
    ac_quant: u16,
    lambda: u32,
    levels: &mut [i16; 16],
) -> [i16; 16] {
    let mut coeffs = dequantize_levels(levels, dc_quant, ac_quant);
    let mut candidate = reconstruct_from_prediction(prediction, &coeffs);
    let mut best_score = rd_score(
        block_sse_4x4(source, source_stride, x, y, &candidate),
        coefficients_rate(probabilities, coeff_type, ctx, first, levels),
        lambda,
    );

    for scan in (first..16).rev() {
        let index = ZIGZAG[scan];
        while levels[index] != 0 {
            let current = levels[index];
            let next = if current > 0 {
                current - 1
            } else {
                current + 1
            };
            let mut trial_levels = *levels;
            trial_levels[index] = next;
            let trial_coeffs = dequantize_levels(&trial_levels, dc_quant, ac_quant);
            let trial_candidate = reconstruct_from_prediction(prediction, &trial_coeffs);
            let trial_score = rd_score(
                block_sse_4x4(source, source_stride, x, y, &trial_candidate),
                coefficients_rate(probabilities, coeff_type, ctx, first, &trial_levels),
                lambda,
            );
            if trial_score <= best_score {
                *levels = trial_levels;
                coeffs = trial_coeffs;
                candidate = trial_candidate;
                best_score = trial_score;
            } else {
                break;
            }
        }
    }

    let _ = candidate;
    coeffs
}

/// Internal helper for refine y2 levels greedy.
pub(super) fn refine_y2_levels_greedy(
    source: &[u8],
    source_stride: usize,
    x: usize,
    y: usize,
    prediction: &[u8; 256],
    ac_coeffs: &[[i16; 16]; 16],
    probabilities: &CoeffProbTables,
    ctx: usize,
    dc_quant: u16,
    ac_quant: u16,
    lambda: u32,
    levels: &mut [i16; 16],
) -> [i16; 16] {
    let mut coeffs = dequantize_levels(levels, dc_quant, ac_quant);
    let (mut candidate, _) = reconstruct_luma16_from_prediction(prediction, ac_coeffs, &coeffs);
    let mut best_score = rd_score(
        block_sse(source, source_stride, x, y, &candidate, 16, 16, 16),
        coefficients_rate(probabilities, 1, ctx, 0, levels),
        lambda,
    );

    for scan in (0..16).rev() {
        let index = ZIGZAG[scan];
        while levels[index] != 0 {
            let current = levels[index];
            let next = if current > 0 {
                current - 1
            } else {
                current + 1
            };
            let mut trial_levels = *levels;
            trial_levels[index] = next;
            let trial_coeffs = dequantize_levels(&trial_levels, dc_quant, ac_quant);
            let (trial_candidate, _) =
                reconstruct_luma16_from_prediction(prediction, ac_coeffs, &trial_coeffs);
            let trial_score = rd_score(
                block_sse(source, source_stride, x, y, &trial_candidate, 16, 16, 16),
                coefficients_rate(probabilities, 1, ctx, 0, &trial_levels),
                lambda,
            );
            if trial_score <= best_score {
                *levels = trial_levels;
                coeffs = trial_coeffs;
                candidate = trial_candidate;
                best_score = trial_score;
            } else {
                break;
            }
        }
    }

    let _ = candidate;
    coeffs
}

/// Optionally refine levels.
pub(super) fn maybe_refine_levels(
    enabled: bool,
    source: &[u8],
    source_stride: usize,
    x: usize,
    y: usize,
    prediction: &[u8; 16],
    probabilities: &CoeffProbTables,
    coeff_type: usize,
    ctx: usize,
    first: usize,
    dc_quant: u16,
    ac_quant: u16,
    lambda: u32,
    levels: &mut [i16; 16],
) -> [i16; 16] {
    if enabled {
        refine_levels_greedy(
            source,
            source_stride,
            x,
            y,
            prediction,
            probabilities,
            coeff_type,
            ctx,
            first,
            dc_quant,
            ac_quant,
            lambda,
            levels,
        )
    } else {
        dequantize_levels(levels, dc_quant, ac_quant)
    }
}

/// Optionally refine y2 levels.
pub(super) fn maybe_refine_y2_levels(
    profile: &LossySearchProfile,
    source: &[u8],
    source_stride: usize,
    x: usize,
    y: usize,
    prediction: &[u8; 256],
    ac_coeffs: &[[i16; 16]; 16],
    probabilities: &CoeffProbTables,
    ctx: usize,
    dc_quant: u16,
    ac_quant: u16,
    lambda: u32,
    levels: &mut [i16; 16],
) -> [i16; 16] {
    if profile.refine_y2 {
        refine_y2_levels_greedy(
            source,
            source_stride,
            x,
            y,
            prediction,
            ac_coeffs,
            probabilities,
            ctx,
            dc_quant,
            ac_quant,
            lambda,
            levels,
        )
    } else {
        dequantize_levels(levels, dc_quant, ac_quant)
    }
}

/// Computes a rate-distortion score for the current candidate.
pub(super) fn rd_score(distortion: u64, rate: u32, lambda: u32) -> u64 {
    distortion * 256 + u64::from(rate) * u64::from(lambda.max(1))
}

/// Internal helper for i16 mode rate.
pub(super) fn i16_mode_rate(mode: u8) -> u32 {
    let mut rate = bit_cost(true, 145);
    match mode {
        DC_PRED => {
            rate += bit_cost(false, 156);
            rate += bit_cost(false, 163);
        }
        V_PRED => {
            rate += bit_cost(false, 156);
            rate += bit_cost(true, 163);
        }
        H_PRED => {
            rate += bit_cost(true, 156);
            rate += bit_cost(false, 128);
        }
        TM_PRED => {
            rate += bit_cost(true, 156);
            rate += bit_cost(true, 128);
        }
        _ => unreachable!("unsupported luma mode"),
    }
    rate
}

/// Internal helper for uv mode rate.
pub(super) fn uv_mode_rate(mode: u8) -> u32 {
    match mode {
        DC_PRED => bit_cost(false, 142),
        V_PRED => bit_cost(true, 142) + bit_cost(false, 114),
        H_PRED => bit_cost(true, 142) + bit_cost(true, 114) + bit_cost(false, 183),
        TM_PRED => bit_cost(true, 142) + bit_cost(true, 114) + bit_cost(true, 183),
        _ => unreachable!("unsupported chroma mode"),
    }
}

/// Internal helper for block sse.
pub(super) fn block_sse(
    source: &[u8],
    source_stride: usize,
    x: usize,
    y: usize,
    reconstructed: &[u8],
    reconstructed_stride: usize,
    width: usize,
    height: usize,
) -> u64 {
    let mut sse = 0u64;
    for row in 0..height {
        let src_offset = (y + row) * source_stride + x;
        let recon_offset = row * reconstructed_stride;
        for col in 0..width {
            let diff = source[src_offset + col] as i32 - reconstructed[recon_offset + col] as i32;
            sse += (diff * diff) as u64;
        }
    }
    sse
}

/// Internal helper for plane sse region.
pub(super) fn plane_sse_region(
    source: &[u8],
    source_stride: usize,
    decoded: &[u8],
    decoded_stride: usize,
    width: usize,
    height: usize,
) -> u64 {
    let mut sse = 0u64;
    for row in 0..height {
        let src_offset = row * source_stride;
        let dec_offset = row * decoded_stride;
        for col in 0..width {
            let diff = source[src_offset + col] as i32 - decoded[dec_offset + col] as i32;
            sse += (diff * diff) as u64;
        }
    }
    sse
}

/// Internal helper for yuv sse.
pub(super) fn yuv_sse(
    source: &Planes,
    width: usize,
    height: usize,
    vp8: &[u8],
) -> Result<u64, EncoderError> {
    let decoded = decode_lossy_vp8_to_yuv(vp8)
        .map_err(|_| EncoderError::Bitstream("internal filter evaluation decode failed"))?;
    let uv_width = width.div_ceil(2);
    let uv_height = height.div_ceil(2);
    Ok(plane_sse_region(
        &source.y,
        source.y_stride,
        &decoded.y,
        decoded.y_stride,
        width,
        height,
    ) + plane_sse_region(
        &source.u,
        source.uv_stride,
        &decoded.u,
        decoded.uv_stride,
        uv_width,
        uv_height,
    ) + plane_sse_region(
        &source.v,
        source.uv_stride,
        &decoded.v,
        decoded.uv_stride,
        uv_width,
        uv_height,
    ))
}

/// Internal helper for evaluate luma mode.
pub(super) fn evaluate_luma_mode(
    source: &Planes,
    reconstructed: &Planes,
    mb_x: usize,
    mb_y: usize,
    profile: &LossySearchProfile,
    quant: &QuantMatrices,
    rd: &RdMultipliers,
    probabilities: &CoeffProbTables,
    top: &NonZeroContext,
    left: &NonZeroContext,
    mode: u8,
) -> u64 {
    let x = mb_x * 16;
    let y = mb_y * 16;
    let mut prediction = [0u8; 16 * 16];
    fill_prediction_block::<16>(
        &reconstructed.y,
        reconstructed.y_stride,
        reconstructed.y_stride,
        x,
        y,
        mode,
        &mut prediction,
        16,
    );
    let mut candidate = prediction;
    let mut y_dc = [0i16; 16];
    let mut y_coeffs = [[0i16; 16]; 16];
    let mut y_levels = [[0i16; 16]; 16];
    let mut rate = 0u32;
    let mut refine_tnz = top.nz & 0x0f;
    let mut refine_lnz = left.nz & 0x0f;

    for sub_y in 0..4 {
        let mut l = refine_lnz & 1;
        for sub_x in 0..4 {
            let block = sub_y * 4 + sub_x;
            let coeffs = forward_transform_at(
                &source.y,
                source.y_stride,
                x + sub_x * 4,
                y + sub_y * 4,
                &prediction,
                16,
                sub_x * 4,
                sub_y * 4,
            );
            y_dc[block] = coeffs[0];
            let mut ac_only = coeffs;
            ac_only[0] = 0;
            let (mut levels, _) = quantize_block(&ac_only, quant.y1[0], quant.y1[1], 1);
            let prediction_block = copy_block4_from_buffer(&prediction, 16, sub_x * 4, sub_y * 4);
            let ctx = (l + (refine_tnz & 1)) as usize;
            let coeffs = maybe_refine_levels(
                profile.refine_i16,
                &source.y,
                source.y_stride,
                x + sub_x * 4,
                y + sub_y * 4,
                &prediction_block,
                probabilities,
                0,
                ctx,
                1,
                quant.y1[0],
                quant.y1[1],
                rd.i16,
                &mut levels,
            );
            y_levels[block] = levels;
            y_coeffs[block] = coeffs;
            let has_ac = block_has_non_zero(&y_levels[block], 1) as u8;
            l = has_ac;
            refine_tnz = (refine_tnz >> 1) | (has_ac << 7);
        }
        refine_tnz >>= 4;
        refine_lnz = (refine_lnz >> 1) | (l << 7);
    }

    let y2_input = forward_wht(&y_dc);
    let mut prediction16 = [0u8; 256];
    prediction16.copy_from_slice(&prediction);
    let (mut y2_levels, _) = quantize_block(&y2_input, quant.y2[0], quant.y2[1], 0);
    let y2_coeffs = maybe_refine_y2_levels(
        profile,
        &source.y,
        source.y_stride,
        x,
        y,
        &prediction16,
        &y_coeffs,
        probabilities,
        (top.nz_dc + left.nz_dc) as usize,
        quant.y2[0],
        quant.y2[1],
        rd.i16,
        &mut y2_levels,
    );
    rate += coefficients_rate(
        probabilities,
        1,
        (top.nz_dc + left.nz_dc) as usize,
        0,
        &y2_levels,
    );
    let y2_dc = inverse_wht(&y2_coeffs);
    for block in 0..16 {
        y_coeffs[block][0] = y2_dc[block];
    }

    let mut tnz = top.nz & 0x0f;
    let mut lnz = left.nz & 0x0f;
    for sub_y in 0..4 {
        let mut l = lnz & 1;
        for sub_x in 0..4 {
            let block = sub_y * 4 + sub_x;
            let ctx = (l + (tnz & 1)) as usize;
            rate += coefficients_rate(probabilities, 0, ctx, 1, &y_levels[block]);
            let has_ac = block_has_non_zero(&y_levels[block], 1) as u8;
            l = has_ac;
            tnz = (tnz >> 1) | (has_ac << 7);
        }
        tnz >>= 4;
        lnz = (lnz >> 1) | (l << 7);
    }

    for sub_y in 0..4 {
        for sub_x in 0..4 {
            let block = sub_y * 4 + sub_x;
            add_transform(&mut candidate, 16, sub_x * 4, sub_y * 4, &y_coeffs[block]);
        }
    }

    let distortion = block_sse(&source.y, source.y_stride, x, y, &candidate, 16, 16, 16);
    rd_score(distortion, rate, rd.i16) + u64::from(i16_mode_rate(mode)) * u64::from(rd.mode.max(1))
}

/// Internal helper for evaluate luma4 mode.
pub(super) fn evaluate_luma4_mode(
    source: &Planes,
    reconstructed: &mut Planes,
    mb_x: usize,
    mb_y: usize,
    profile: &LossySearchProfile,
    quant: &QuantMatrices,
    rd: &RdMultipliers,
    probabilities: &CoeffProbTables,
    top_context: &NonZeroContext,
    left_context: &NonZeroContext,
    top_modes: &[u8],
    left_modes: &[u8; 4],
) -> (u64, [u8; 16]) {
    const MODES: [u8; NUM_BMODES] = [
        B_DC_PRED, B_TM_PRED, B_VE_PRED, B_HE_PRED, B_RD_PRED, B_VR_PRED, B_LD_PRED, B_VL_PRED,
        B_HD_PRED, B_HU_PRED,
    ];

    let x = mb_x * 16;
    let y = mb_y * 16;
    let backup = copy_block16(&reconstructed.y, reconstructed.y_stride, x, y);
    let mut total_score = 0u64;
    let mut sub_modes = [B_DC_PRED; 16];
    let mut local_top = [B_DC_PRED; 4];
    local_top.copy_from_slice(top_modes);
    let mut local_left = *left_modes;
    let mut tnz = top_context.nz & 0x0f;
    let mut lnz = left_context.nz & 0x0f;

    for sub_y in 0..4 {
        let mut left_mode = local_left[sub_y];
        let mut l = lnz & 1;
        for sub_x in 0..4 {
            let block = sub_y * 4 + sub_x;
            let block_x = x + sub_x * 4;
            let block_y = y + sub_y * 4;
            let top_mode = local_top[sub_x];
            let original = copy_block4(&reconstructed.y, reconstructed.y_stride, block_x, block_y);
            let ctx = (l + (tnz & 1)) as usize;

            let mut best_mode = B_DC_PRED;
            let mut best_coeffs = [0i16; 16];
            let mut best_score = u64::MAX;
            let mut best_non_zero = 0u8;
            for mode in MODES {
                restore_block4(
                    &mut reconstructed.y,
                    reconstructed.y_stride,
                    block_x,
                    block_y,
                    &original,
                );
                predict_luma4_block(
                    &mut reconstructed.y,
                    reconstructed.y_stride,
                    reconstructed.y_stride,
                    block_x,
                    block_y,
                    mode,
                );
                let coeffs = forward_transform(
                    &source.y,
                    source.y_stride,
                    &reconstructed.y,
                    reconstructed.y_stride,
                    block_x,
                    block_y,
                );
                let prediction_block =
                    copy_block4(&reconstructed.y, reconstructed.y_stride, block_x, block_y);
                let (mut levels, _) = quantize_block(&coeffs, quant.y1[0], quant.y1[1], 0);
                let dequantized = maybe_refine_levels(
                    profile.refine_i4_search,
                    &source.y,
                    source.y_stride,
                    block_x,
                    block_y,
                    &prediction_block,
                    probabilities,
                    3,
                    ctx,
                    0,
                    quant.y1[0],
                    quant.y1[1],
                    rd.i4,
                    &mut levels,
                );
                add_transform(
                    &mut reconstructed.y,
                    reconstructed.y_stride,
                    block_x,
                    block_y,
                    &dequantized,
                );
                let distortion = block_sse(
                    &source.y,
                    source.y_stride,
                    block_x,
                    block_y,
                    &reconstructed.y[(block_y * reconstructed.y_stride + block_x)..],
                    reconstructed.y_stride,
                    4,
                    4,
                );
                let coeff_rate = coefficients_rate(probabilities, 3, ctx, 0, &levels);
                let score = rd_score(distortion, coeff_rate, rd.i4)
                    + u64::from(intra4_mode_rate(top_mode, left_mode, mode))
                        * u64::from(rd.mode.max(1));
                if score < best_score {
                    best_mode = mode;
                    best_coeffs = dequantized;
                    best_score = score;
                    best_non_zero = block_has_non_zero(&levels, 0) as u8;
                }
            }

            restore_block4(
                &mut reconstructed.y,
                reconstructed.y_stride,
                block_x,
                block_y,
                &original,
            );
            predict_luma4_block(
                &mut reconstructed.y,
                reconstructed.y_stride,
                reconstructed.y_stride,
                block_x,
                block_y,
                best_mode,
            );
            add_transform(
                &mut reconstructed.y,
                reconstructed.y_stride,
                block_x,
                block_y,
                &best_coeffs,
            );

            sub_modes[block] = best_mode;
            total_score += best_score;
            local_top[sub_x] = best_mode;
            left_mode = best_mode;
            l = best_non_zero;
            tnz = (tnz >> 1) | (best_non_zero << 7);
        }
        tnz >>= 4;
        lnz = (lnz >> 1) | (l << 7);
        local_left[sub_y] = left_mode;
    }

    restore_block16(&mut reconstructed.y, reconstructed.y_stride, x, y, &backup);
    (
        total_score + u64::from(bit_cost(false, 145)) * u64::from(rd.mode.max(1)),
        sub_modes,
    )
}

/// Internal helper for evaluate chroma mode.
pub(super) fn evaluate_chroma_mode(
    source: &Planes,
    reconstructed: &Planes,
    mb_x: usize,
    mb_y: usize,
    profile: &LossySearchProfile,
    quant: &QuantMatrices,
    rd: &RdMultipliers,
    probabilities: &CoeffProbTables,
    top: &NonZeroContext,
    left: &NonZeroContext,
    mode: u8,
) -> u64 {
    let x = mb_x * 8;
    let y = mb_y * 8;
    let mut prediction_u = [0u8; 8 * 8];
    let mut prediction_v = [0u8; 8 * 8];
    fill_prediction_block::<8>(
        &reconstructed.u,
        reconstructed.uv_stride,
        reconstructed.uv_stride,
        x,
        y,
        mode,
        &mut prediction_u,
        8,
    );
    fill_prediction_block::<8>(
        &reconstructed.v,
        reconstructed.uv_stride,
        reconstructed.uv_stride,
        x,
        y,
        mode,
        &mut prediction_v,
        8,
    );
    let mut candidate_u = prediction_u;
    let mut candidate_v = prediction_v;
    let mut rate = 0u32;
    let mut tnz_u = top.nz >> 4;
    let mut lnz_u = left.nz >> 4;

    for sub_y in 0..2 {
        let mut l = lnz_u & 1;
        for sub_x in 0..2 {
            let coeffs_u = forward_transform_at(
                &source.u,
                source.uv_stride,
                x + sub_x * 4,
                y + sub_y * 4,
                &prediction_u,
                8,
                sub_x * 4,
                sub_y * 4,
            );
            let prediction_block_u =
                copy_block4_from_buffer(&prediction_u, 8, sub_x * 4, sub_y * 4);
            let (mut levels_u, _) = quantize_block(&coeffs_u, quant.uv[0], quant.uv[1], 0);
            let ctx = (l + (tnz_u & 1)) as usize;
            let coeffs_u = maybe_refine_levels(
                profile.refine_chroma,
                &source.u,
                source.uv_stride,
                x + sub_x * 4,
                y + sub_y * 4,
                &prediction_block_u,
                probabilities,
                2,
                ctx,
                0,
                quant.uv[0],
                quant.uv[1],
                rd.uv,
                &mut levels_u,
            );
            let has_coeffs = block_has_non_zero(&levels_u, 0) as u8;
            rate += coefficients_rate(probabilities, 2, ctx, 0, &levels_u);
            l = has_coeffs;
            tnz_u = (tnz_u >> 1) | (has_coeffs << 3);
            add_transform(&mut candidate_u, 8, sub_x * 4, sub_y * 4, &coeffs_u);
        }
        tnz_u >>= 2;
        lnz_u = (lnz_u >> 1) | (l << 5);
    }

    let mut tnz_v = top.nz >> 6;
    let mut lnz_v = left.nz >> 6;
    for sub_y in 0..2 {
        let mut l = lnz_v & 1;
        for sub_x in 0..2 {
            let coeffs_v = forward_transform_at(
                &source.v,
                source.uv_stride,
                x + sub_x * 4,
                y + sub_y * 4,
                &prediction_v,
                8,
                sub_x * 4,
                sub_y * 4,
            );
            let prediction_block_v =
                copy_block4_from_buffer(&prediction_v, 8, sub_x * 4, sub_y * 4);
            let (mut levels_v, _) = quantize_block(&coeffs_v, quant.uv[0], quant.uv[1], 0);
            let ctx = (l + (tnz_v & 1)) as usize;
            let coeffs_v = maybe_refine_levels(
                profile.refine_chroma,
                &source.v,
                source.uv_stride,
                x + sub_x * 4,
                y + sub_y * 4,
                &prediction_block_v,
                probabilities,
                2,
                ctx,
                0,
                quant.uv[0],
                quant.uv[1],
                rd.uv,
                &mut levels_v,
            );
            let has_coeffs = block_has_non_zero(&levels_v, 0) as u8;
            rate += coefficients_rate(probabilities, 2, ctx, 0, &levels_v);
            l = has_coeffs;
            tnz_v = (tnz_v >> 1) | (has_coeffs << 3);
            add_transform(&mut candidate_v, 8, sub_x * 4, sub_y * 4, &coeffs_v);
        }
        tnz_v >>= 2;
        lnz_v = (lnz_v >> 1) | (l << 5);
    }

    let distortion_u = block_sse(&source.u, source.uv_stride, x, y, &candidate_u, 8, 8, 8);
    let distortion_v = block_sse(&source.v, source.uv_stride, x, y, &candidate_v, 8, 8, 8);
    rd_score(distortion_u + distortion_v, rate, rd.uv)
        + u64::from(uv_mode_rate(mode)) * u64::from(rd.mode.max(1))
}

/// Internal helper for fast luma predictor score.
pub(super) fn fast_luma_predictor_score(
    source: &Planes,
    reconstructed: &Planes,
    mb_x: usize,
    mb_y: usize,
    mode: u8,
) -> u64 {
    let x = mb_x * 16;
    let y = mb_y * 16;
    let mut prediction = [0u8; 16 * 16];
    fill_prediction_block::<16>(
        &reconstructed.y,
        reconstructed.y_stride,
        reconstructed.y_stride,
        x,
        y,
        mode,
        &mut prediction,
        16,
    );
    block_sse(&source.y, source.y_stride, x, y, &prediction, 16, 16, 16)
}

/// Internal helper for fast chroma predictor score.
pub(super) fn fast_chroma_predictor_score(
    source: &Planes,
    reconstructed: &Planes,
    mb_x: usize,
    mb_y: usize,
    mode: u8,
) -> u64 {
    let x = mb_x * 8;
    let y = mb_y * 8;
    let mut prediction_u = [0u8; 8 * 8];
    let mut prediction_v = [0u8; 8 * 8];
    fill_prediction_block::<8>(
        &reconstructed.u,
        reconstructed.uv_stride,
        reconstructed.uv_stride,
        x,
        y,
        mode,
        &mut prediction_u,
        8,
    );
    fill_prediction_block::<8>(
        &reconstructed.v,
        reconstructed.uv_stride,
        reconstructed.uv_stride,
        x,
        y,
        mode,
        &mut prediction_v,
        8,
    );
    block_sse(&source.u, source.uv_stride, x, y, &prediction_u, 8, 8, 8)
        + block_sse(&source.v, source.uv_stride, x, y, &prediction_v, 8, 8, 8)
}

/// Chooses macroblock mode.
pub(super) fn choose_macroblock_mode(
    source: &Planes,
    reconstructed: &mut Planes,
    mb_x: usize,
    mb_y: usize,
    profile: &LossySearchProfile,
    quant: &QuantMatrices,
    rd: &RdMultipliers,
    probabilities: &CoeffProbTables,
    top_context: &NonZeroContext,
    left_context: &NonZeroContext,
    top_modes: &[u8],
    left_modes: &[u8; 4],
) -> MacroblockMode {
    const MODES: [u8; 4] = [DC_PRED, V_PRED, H_PRED, TM_PRED];

    if profile.fast_mode_search {
        let mut best_luma = DC_PRED;
        let mut best_luma_score = u64::MAX;
        for mode in MODES {
            let score = fast_luma_predictor_score(source, reconstructed, mb_x, mb_y, mode);
            if score < best_luma_score {
                best_luma = mode;
                best_luma_score = score;
            }
        }

        let mut best_chroma = DC_PRED;
        let mut best_chroma_score = u64::MAX;
        for mode in MODES {
            let score = fast_chroma_predictor_score(source, reconstructed, mb_x, mb_y, mode);
            if score < best_chroma_score {
                best_chroma = mode;
                best_chroma_score = score;
            }
        }

        return MacroblockMode {
            luma: best_luma,
            sub_luma: [B_DC_PRED; 16],
            chroma: best_chroma,
            segment: 0,
            skip: false,
        };
    }

    let mut best_luma = DC_PRED;
    let mut best_luma_score = u64::MAX;
    for mode in MODES {
        let score = evaluate_luma_mode(
            source,
            reconstructed,
            mb_x,
            mb_y,
            profile,
            quant,
            rd,
            probabilities,
            top_context,
            left_context,
            mode,
        );
        if score < best_luma_score {
            best_luma = mode;
            best_luma_score = score;
        }
    }

    let (best_luma, sub_luma) = if profile.allow_i4x4 {
        let (i4_score, sub_luma) = evaluate_luma4_mode(
            source,
            reconstructed,
            mb_x,
            mb_y,
            profile,
            quant,
            rd,
            probabilities,
            top_context,
            left_context,
            top_modes,
            left_modes,
        );
        if i4_score < best_luma_score {
            (B_PRED, sub_luma)
        } else {
            (best_luma, [B_DC_PRED; 16])
        }
    } else {
        (best_luma, [B_DC_PRED; 16])
    };

    let mut best_chroma = DC_PRED;
    let mut best_chroma_score = u64::MAX;
    for mode in MODES {
        let score = evaluate_chroma_mode(
            source,
            reconstructed,
            mb_x,
            mb_y,
            profile,
            quant,
            rd,
            probabilities,
            top_context,
            left_context,
            mode,
        );
        if score < best_chroma_score {
            best_chroma = mode;
            best_chroma_score = score;
        }
    }

    MacroblockMode {
        luma: best_luma,
        sub_luma,
        chroma: best_chroma,
        segment: 0,
        skip: false,
    }
}
