use crate::decoder::quant::{AC_TABLE, DC_TABLE};
use crate::decoder::tree::{COEFFS_PROBA0, COEFFS_UPDATE_PROBA};
use crate::decoder::vp8i::{DC_PRED, H_PRED, TM_PRED, V_PRED};
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MacroblockMode {
    luma: u8,
    chroma: u8,
    skip: bool,
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

fn top_left_sample(plane: &[u8], stride: usize, x: usize, y: usize) -> u8 {
    if y == 0 {
        127
    } else if x == 0 {
        129
    } else {
        plane[(y - 1) * stride + (x - 1)]
    }
}

fn top_samples<const N: usize>(
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

fn left_samples<const N: usize>(plane: &[u8], stride: usize, x: usize, y: usize) -> [u8; N] {
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

fn fill_prediction_block<const N: usize>(
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

fn predict_block<const N: usize>(
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

fn mul1(value: i32) -> i32 {
    ((value * VP8_TRANSFORM_AC3_C1) >> 16) + value
}

fn mul2(value: i32) -> i32 {
    (value * VP8_TRANSFORM_AC3_C2) >> 16
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

fn forward_transform_at(
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

fn forward_transform(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    x: usize,
    y: usize,
) -> [i16; 16] {
    forward_transform_at(src, src_stride, x, y, pred, pred_stride, x, y)
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

fn non_zero_count(levels: &[i16; 16], first: usize) -> u64 {
    levels
        .iter()
        .enumerate()
        .skip(first)
        .filter(|(_, coeff)| **coeff != 0)
        .count() as u64
}

fn block_sse(
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

fn mode_score(distortion: u64, non_zero_count: u64, ac_quant: u16) -> u64 {
    distortion + non_zero_count * u64::from(ac_quant.max(8)) * 8
}

fn evaluate_luma_mode(
    source: &Planes,
    reconstructed: &Planes,
    mb_x: usize,
    mb_y: usize,
    quant: &QuantMatrices,
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
    let mut non_zero = 0u64;

    for sub_y in 0..4 {
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
            let (levels, coeffs) = quantize_block(&ac_only, quant.y1[0], quant.y1[1], 1);
            non_zero += non_zero_count(&levels, 1);
            y_coeffs[block] = coeffs;
        }
    }

    let y2_input = forward_wht(&y_dc);
    let (y2_levels, y2_coeffs) = quantize_block(&y2_input, quant.y2[0], quant.y2[1], 0);
    non_zero += non_zero_count(&y2_levels, 0);
    let y2_dc = inverse_wht(&y2_coeffs);
    for block in 0..16 {
        y_coeffs[block][0] = y2_dc[block];
    }

    for sub_y in 0..4 {
        for sub_x in 0..4 {
            let block = sub_y * 4 + sub_x;
            add_transform(&mut candidate, 16, sub_x * 4, sub_y * 4, &y_coeffs[block]);
        }
    }

    let distortion = block_sse(&source.y, source.y_stride, x, y, &candidate, 16, 16, 16);
    mode_score(distortion, non_zero, quant.y1[1])
}

fn evaluate_chroma_mode(
    source: &Planes,
    reconstructed: &Planes,
    mb_x: usize,
    mb_y: usize,
    quant: &QuantMatrices,
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
    let mut non_zero = 0u64;

    for sub_y in 0..2 {
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
            let (levels_u, coeffs_u) = quantize_block(&coeffs_u, quant.uv[0], quant.uv[1], 0);
            non_zero += non_zero_count(&levels_u, 0);
            add_transform(&mut candidate_u, 8, sub_x * 4, sub_y * 4, &coeffs_u);

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
            let (levels_v, coeffs_v) = quantize_block(&coeffs_v, quant.uv[0], quant.uv[1], 0);
            non_zero += non_zero_count(&levels_v, 0);
            add_transform(&mut candidate_v, 8, sub_x * 4, sub_y * 4, &coeffs_v);
        }
    }

    let distortion_u = block_sse(&source.u, source.uv_stride, x, y, &candidate_u, 8, 8, 8);
    let distortion_v = block_sse(&source.v, source.uv_stride, x, y, &candidate_v, 8, 8, 8);
    mode_score(distortion_u + distortion_v, non_zero, quant.uv[1])
}

fn choose_macroblock_mode(
    source: &Planes,
    reconstructed: &Planes,
    mb_x: usize,
    mb_y: usize,
    quant: &QuantMatrices,
) -> MacroblockMode {
    const MODES: [u8; 4] = [DC_PRED, V_PRED, H_PRED, TM_PRED];

    let mut best_luma = DC_PRED;
    let mut best_luma_score = u64::MAX;
    for mode in MODES {
        let score = evaluate_luma_mode(source, reconstructed, mb_x, mb_y, quant, mode);
        if score < best_luma_score {
            best_luma = mode;
            best_luma_score = score;
        }
    }

    let mut best_chroma = DC_PRED;
    let mut best_chroma_score = u64::MAX;
    for mode in MODES {
        let score = evaluate_chroma_mode(source, reconstructed, mb_x, mb_y, quant, mode);
        if score < best_chroma_score {
            best_chroma = mode;
            best_chroma_score = score;
        }
    }

    MacroblockMode {
        luma: best_luma,
        chroma: best_chroma,
        skip: false,
    }
}

fn block_has_non_zero(levels: &[i16; 16], first: usize) -> bool {
    levels.iter().skip(first).any(|&level| level != 0)
}

fn compute_skip_probability(modes: &[MacroblockMode]) -> Option<u8> {
    let total = modes.len();
    let skip_count = modes.iter().filter(|mode| mode.skip).count();
    if total == 0 || skip_count == 0 {
        return None;
    }
    let non_skip = total - skip_count;
    let prob_zero = ((non_skip * 255) + total / 2) / total;
    Some(prob_zero.clamp(1, 254) as u8)
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

fn encode_partition0(
    mb_width: usize,
    mb_height: usize,
    base_quant: u8,
    modes: &[MacroblockMode],
) -> Vec<u8> {
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
    let skip_probability = compute_skip_probability(modes);
    if let Some(prob) = skip_probability {
        writer.put_bit_uniform(true);
        writer.put_bits(prob as u32, 8);
    } else {
        writer.put_bit_uniform(false);
    }

    for mode in modes {
        if let Some(prob) = skip_probability {
            writer.put_bit(mode.skip, prob);
        }
        writer.put_bit(true, 145);
        match mode.luma {
            DC_PRED => {
                writer.put_bit(false, 156);
                writer.put_bit(false, 163);
            }
            V_PRED => {
                writer.put_bit(false, 156);
                writer.put_bit(true, 163);
            }
            H_PRED => {
                writer.put_bit(true, 156);
                writer.put_bit(false, 128);
            }
            TM_PRED => {
                writer.put_bit(true, 156);
                writer.put_bit(true, 128);
            }
            _ => unreachable!("unsupported luma mode"),
        }
        match mode.chroma {
            DC_PRED => {
                writer.put_bit(false, 142);
            }
            V_PRED => {
                writer.put_bit(true, 142);
                writer.put_bit(false, 114);
            }
            H_PRED => {
                writer.put_bit(true, 142);
                writer.put_bit(true, 114);
                writer.put_bit(false, 183);
            }
            TM_PRED => {
                writer.put_bit(true, 142);
                writer.put_bit(true, 114);
                writer.put_bit(true, 183);
            }
            _ => unreachable!("unsupported chroma mode"),
        }
    }

    writer.finish()
}

fn encode_macroblock(
    writer: &mut Vp8BoolWriter,
    source: &Planes,
    reconstructed: &mut Planes,
    mb_x: usize,
    mb_y: usize,
    mode: MacroblockMode,
    quant: &QuantMatrices,
    top: &mut NonZeroContext,
    left: &mut NonZeroContext,
) -> bool {
    let y_x = mb_x * 16;
    let y_y = mb_y * 16;
    let uv_x = mb_x * 8;
    let uv_y = mb_y * 8;

    predict_block::<16>(
        &mut reconstructed.y,
        reconstructed.y_stride,
        reconstructed.y_stride,
        y_x,
        y_y,
        mode.luma,
    );
    predict_block::<8>(
        &mut reconstructed.u,
        reconstructed.uv_stride,
        reconstructed.uv_stride,
        uv_x,
        uv_y,
        mode.chroma,
    );
    predict_block::<8>(
        &mut reconstructed.v,
        reconstructed.uv_stride,
        reconstructed.uv_stride,
        uv_x,
        uv_y,
        mode.chroma,
    );

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

    let skip = !block_has_non_zero(&y2_levels, 0)
        && y_levels.iter().all(|levels| !block_has_non_zero(levels, 1))
        && u_levels.iter().all(|levels| !block_has_non_zero(levels, 0))
        && v_levels.iter().all(|levels| !block_has_non_zero(levels, 0));
    if skip {
        top.nz = 0;
        left.nz = 0;
        top.nz_dc = 0;
        left.nz_dc = 0;
        return true;
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

    false
}

fn encode_token_partition(
    source: &Planes,
    mb_width: usize,
    mb_height: usize,
    quant: &QuantMatrices,
) -> (Vec<u8>, Planes, Vec<MacroblockMode>) {
    let mut writer = Vp8BoolWriter::new(source.y.len() / 4);
    let mut reconstructed = empty_reconstructed_planes(mb_width, mb_height);
    let mut top_contexts = vec![NonZeroContext::default(); mb_width];
    let mut modes = Vec::with_capacity(mb_width * mb_height);

    for mb_y in 0..mb_height {
        let mut left_context = NonZeroContext::default();
        for mb_x in 0..mb_width {
            let mut mode = choose_macroblock_mode(source, &reconstructed, mb_x, mb_y, quant);
            mode.skip = encode_macroblock(
                &mut writer,
                source,
                &mut reconstructed,
                mb_x,
                mb_y,
                mode,
                quant,
                &mut top_contexts[mb_x],
                &mut left_context,
            );
            modes.push(mode);
        }
    }

    (writer.finish(), reconstructed, modes)
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
    let (token_partition, _, modes) = encode_token_partition(&source, mb_width, mb_height, &quant);
    let partition0 = encode_partition0(mb_width, mb_height, base_quant as u8, &modes);
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
        let (token_partition, reconstructed, modes) =
            encode_token_partition(&source, mb_width, mb_height, &quant);
        let partition0 = encode_partition0(mb_width, mb_height, base_quant as u8, &modes);
        let vp8 = build_vp8_frame(width, height, &partition0, &token_partition).unwrap();
        let decoded = decode_lossy_vp8_to_yuv(&vp8).unwrap();
        assert_eq!(decoded.y, reconstructed.y);
        assert_eq!(decoded.u, reconstructed.u);
        assert_eq!(decoded.v, reconstructed.v);
    }

    #[test]
    fn mode_search_prefers_vertical_prediction_for_repeated_top_rows() {
        let mb_width = 1;
        let mb_height = 2;
        let mut source = empty_reconstructed_planes(mb_width, mb_height);
        let mut reconstructed = empty_reconstructed_planes(mb_width, mb_height);

        for row in 0..16 {
            for col in 0..16 {
                let value = (col as u8).saturating_mul(9);
                reconstructed.y[row * reconstructed.y_stride + col] = value;
                source.y[(16 + row) * source.y_stride + col] = value;
            }
        }

        for row in 0..8 {
            for col in 0..8 {
                let u = (32 + col * 7) as u8;
                let v = (96 + col * 5) as u8;
                reconstructed.u[row * reconstructed.uv_stride + col] = u;
                reconstructed.v[row * reconstructed.uv_stride + col] = v;
                source.u[(8 + row) * source.uv_stride + col] = u;
                source.v[(8 + row) * source.uv_stride + col] = v;
            }
        }

        let quant = build_quant_matrices(base_quantizer_from_quality(90));
        let mode = choose_macroblock_mode(&source, &reconstructed, 0, 1, &quant);
        assert_eq!(mode.luma, V_PRED);
        assert_eq!(mode.chroma, V_PRED);
    }
}
