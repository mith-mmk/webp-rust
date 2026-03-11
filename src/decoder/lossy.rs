use crate::bmp::encode_bmp24_from_rgba;
use crate::decoder::header::parse_still_webp;
use crate::decoder::vp8::{parse_macroblock_data, FilterType, MacroBlockData, MacroBlockDataFrame};
use crate::decoder::vp8i::{
    WebpFormat, B_DC_PRED, B_HD_PRED, B_HE_PRED, B_HU_PRED, B_LD_PRED, B_RD_PRED, B_TM_PRED,
    B_VE_PRED, B_VL_PRED, B_VR_PRED, DC_PRED, H_PRED, TM_PRED, V_PRED,
};
use crate::decoder::DecoderError;

const VP8_TRANSFORM_AC3_C1: i32 = 20_091;
const VP8_TRANSFORM_AC3_C2: i32 = 35_468;

const RGB_Y_COEFF: i32 = 19_077;
const RGB_V_TO_R_COEFF: i32 = 26_149;
const RGB_U_TO_G_COEFF: i32 = 6_419;
const RGB_V_TO_G_COEFF: i32 = 13_320;
const RGB_U_TO_B_COEFF: i32 = 33_050;
const RGB_R_BIAS: i32 = 14_234;
const RGB_G_BIAS: i32 = 8_708;
const RGB_B_BIAS: i32 = 17_685;
const YUV_FIX2: i32 = 6;
const YUV_MASK2: i32 = (256 << YUV_FIX2) - 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedImage {
    pub width: usize,
    pub height: usize,
    pub rgba: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedYuvImage {
    pub width: usize,
    pub height: usize,
    pub y_stride: usize,
    pub uv_stride: usize,
    pub y: Vec<u8>,
    pub u: Vec<u8>,
    pub v: Vec<u8>,
}

struct Planes {
    width: usize,
    height: usize,
    y_stride: usize,
    uv_stride: usize,
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
}

impl Planes {
    fn new(frame: &MacroBlockDataFrame) -> Self {
        let y_stride = frame.frame.macroblock_width * 16;
        let uv_stride = frame.frame.macroblock_width * 8;
        let height = frame.frame.macroblock_height * 16;
        let uv_height = frame.frame.macroblock_height * 8;
        Self {
            width: frame.frame.picture.width as usize,
            height: frame.frame.picture.height as usize,
            y_stride,
            uv_stride,
            y: vec![0; y_stride * height],
            u: vec![0; uv_stride * uv_height],
            v: vec![0; uv_stride * uv_height],
        }
    }

    fn y_width(&self) -> usize {
        self.y_stride
    }

    fn uv_width(&self) -> usize {
        self.uv_stride
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FilterInfo {
    f_limit: u8,
    f_ilevel: u8,
    f_inner: bool,
    hev_thresh: u8,
}

fn abs_diff(a: u8, b: u8) -> i32 {
    (a as i32 - b as i32).abs()
}

fn clip_signed(value: i32) -> i32 {
    value.clamp(-128, 127)
}

fn clip_filter_value(value: i32) -> i32 {
    value.clamp(-16, 15)
}

fn do_filter2(plane: &mut [u8], pos: usize, step: usize) {
    let p1 = plane[pos - 2 * step] as i32;
    let p0 = plane[pos - step] as i32;
    let q0 = plane[pos] as i32;
    let q1 = plane[pos + step] as i32;
    let a = 3 * (q0 - p0) + clip_signed(p1 - q1);
    let a1 = clip_filter_value((a + 4) >> 3);
    let a2 = clip_filter_value((a + 3) >> 3);
    plane[pos - step] = clip_byte(p0 + a2);
    plane[pos] = clip_byte(q0 - a1);
}

fn do_filter4(plane: &mut [u8], pos: usize, step: usize) {
    let p1 = plane[pos - 2 * step] as i32;
    let p0 = plane[pos - step] as i32;
    let q0 = plane[pos] as i32;
    let q1 = plane[pos + step] as i32;
    let a = 3 * (q0 - p0);
    let a1 = clip_filter_value((a + 4) >> 3);
    let a2 = clip_filter_value((a + 3) >> 3);
    let a3 = (a1 + 1) >> 1;
    plane[pos - 2 * step] = clip_byte(p1 + a3);
    plane[pos - step] = clip_byte(p0 + a2);
    plane[pos] = clip_byte(q0 - a1);
    plane[pos + step] = clip_byte(q1 - a3);
}

fn do_filter6(plane: &mut [u8], pos: usize, step: usize) {
    let p2 = plane[pos - 3 * step] as i32;
    let p1 = plane[pos - 2 * step] as i32;
    let p0 = plane[pos - step] as i32;
    let q0 = plane[pos] as i32;
    let q1 = plane[pos + step] as i32;
    let q2 = plane[pos + 2 * step] as i32;
    let a = clip_signed(3 * (q0 - p0) + clip_signed(p1 - q1));
    let a1 = (27 * a + 63) >> 7;
    let a2 = (18 * a + 63) >> 7;
    let a3 = (9 * a + 63) >> 7;
    plane[pos - 3 * step] = clip_byte(p2 + a3);
    plane[pos - 2 * step] = clip_byte(p1 + a2);
    plane[pos - step] = clip_byte(p0 + a1);
    plane[pos] = clip_byte(q0 - a1);
    plane[pos + step] = clip_byte(q1 - a2);
    plane[pos + 2 * step] = clip_byte(q2 - a3);
}

fn hev(plane: &[u8], pos: usize, step: usize, thresh: i32) -> bool {
    let p1 = plane[pos - 2 * step];
    let p0 = plane[pos - step];
    let q0 = plane[pos];
    let q1 = plane[pos + step];
    abs_diff(p1, p0) > thresh || abs_diff(q1, q0) > thresh
}

fn needs_filter(plane: &[u8], pos: usize, step: usize, thresh: i32) -> bool {
    let p1 = plane[pos - 2 * step];
    let p0 = plane[pos - step];
    let q0 = plane[pos];
    let q1 = plane[pos + step];
    4 * abs_diff(p0, q0) + abs_diff(p1, q1) <= thresh
}

fn needs_filter2(plane: &[u8], pos: usize, step: usize, thresh: i32, inner_thresh: i32) -> bool {
    let p3 = plane[pos - 4 * step];
    let p2 = plane[pos - 3 * step];
    let p1 = plane[pos - 2 * step];
    let p0 = plane[pos - step];
    let q0 = plane[pos];
    let q1 = plane[pos + step];
    let q2 = plane[pos + 2 * step];
    let q3 = plane[pos + 3 * step];
    if 4 * abs_diff(p0, q0) + abs_diff(p1, q1) > thresh {
        return false;
    }
    abs_diff(p3, p2) <= inner_thresh
        && abs_diff(p2, p1) <= inner_thresh
        && abs_diff(p1, p0) <= inner_thresh
        && abs_diff(q3, q2) <= inner_thresh
        && abs_diff(q2, q1) <= inner_thresh
        && abs_diff(q1, q0) <= inner_thresh
}

fn simple_v_filter16(plane: &mut [u8], pos: usize, stride: usize, thresh: i32) {
    let thresh2 = 2 * thresh + 1;
    for i in 0..16 {
        let edge = pos + i;
        if needs_filter(plane, edge, stride, thresh2) {
            do_filter2(plane, edge, stride);
        }
    }
}

fn simple_h_filter16(plane: &mut [u8], pos: usize, stride: usize, thresh: i32) {
    let thresh2 = 2 * thresh + 1;
    for i in 0..16 {
        let edge = pos + i * stride;
        if needs_filter(plane, edge, 1, thresh2) {
            do_filter2(plane, edge, 1);
        }
    }
}

fn simple_v_filter16i(plane: &mut [u8], mut pos: usize, stride: usize, thresh: i32) {
    for _ in (1..=3).rev() {
        pos += 4 * stride;
        simple_v_filter16(plane, pos, stride, thresh);
    }
}

fn simple_h_filter16i(plane: &mut [u8], mut pos: usize, stride: usize, thresh: i32) {
    for _ in (1..=3).rev() {
        pos += 4;
        simple_h_filter16(plane, pos, stride, thresh);
    }
}

fn filter_loop26(
    plane: &mut [u8],
    mut pos: usize,
    hstride: usize,
    vstride: usize,
    size: usize,
    thresh: i32,
    inner_thresh: i32,
    hev_thresh: i32,
) {
    let thresh2 = 2 * thresh + 1;
    for _ in 0..size {
        if needs_filter2(plane, pos, hstride, thresh2, inner_thresh) {
            if hev(plane, pos, hstride, hev_thresh) {
                do_filter2(plane, pos, hstride);
            } else {
                do_filter6(plane, pos, hstride);
            }
        }
        pos += vstride;
    }
}

fn filter_loop24(
    plane: &mut [u8],
    mut pos: usize,
    hstride: usize,
    vstride: usize,
    size: usize,
    thresh: i32,
    inner_thresh: i32,
    hev_thresh: i32,
) {
    let thresh2 = 2 * thresh + 1;
    for _ in 0..size {
        if needs_filter2(plane, pos, hstride, thresh2, inner_thresh) {
            if hev(plane, pos, hstride, hev_thresh) {
                do_filter2(plane, pos, hstride);
            } else {
                do_filter4(plane, pos, hstride);
            }
        }
        pos += vstride;
    }
}

fn v_filter16(
    plane: &mut [u8],
    pos: usize,
    stride: usize,
    thresh: i32,
    inner_thresh: i32,
    hev_thresh: i32,
) {
    filter_loop26(plane, pos, stride, 1, 16, thresh, inner_thresh, hev_thresh);
}

fn h_filter16(
    plane: &mut [u8],
    pos: usize,
    stride: usize,
    thresh: i32,
    inner_thresh: i32,
    hev_thresh: i32,
) {
    filter_loop26(plane, pos, 1, stride, 16, thresh, inner_thresh, hev_thresh);
}

fn v_filter16i(
    plane: &mut [u8],
    mut pos: usize,
    stride: usize,
    thresh: i32,
    inner_thresh: i32,
    hev_thresh: i32,
) {
    for _ in (1..=3).rev() {
        pos += 4 * stride;
        filter_loop24(plane, pos, stride, 1, 16, thresh, inner_thresh, hev_thresh);
    }
}

fn h_filter16i(
    plane: &mut [u8],
    mut pos: usize,
    stride: usize,
    thresh: i32,
    inner_thresh: i32,
    hev_thresh: i32,
) {
    for _ in (1..=3).rev() {
        pos += 4;
        filter_loop24(plane, pos, 1, stride, 16, thresh, inner_thresh, hev_thresh);
    }
}

fn v_filter8(
    plane_u: &mut [u8],
    plane_v: &mut [u8],
    pos: usize,
    stride: usize,
    thresh: i32,
    inner_thresh: i32,
    hev_thresh: i32,
) {
    filter_loop26(plane_u, pos, stride, 1, 8, thresh, inner_thresh, hev_thresh);
    filter_loop26(plane_v, pos, stride, 1, 8, thresh, inner_thresh, hev_thresh);
}

fn h_filter8(
    plane_u: &mut [u8],
    plane_v: &mut [u8],
    pos: usize,
    stride: usize,
    thresh: i32,
    inner_thresh: i32,
    hev_thresh: i32,
) {
    filter_loop26(plane_u, pos, 1, stride, 8, thresh, inner_thresh, hev_thresh);
    filter_loop26(plane_v, pos, 1, stride, 8, thresh, inner_thresh, hev_thresh);
}

fn v_filter8i(
    plane_u: &mut [u8],
    plane_v: &mut [u8],
    pos: usize,
    stride: usize,
    thresh: i32,
    inner_thresh: i32,
    hev_thresh: i32,
) {
    filter_loop24(
        plane_u,
        pos + 4 * stride,
        stride,
        1,
        8,
        thresh,
        inner_thresh,
        hev_thresh,
    );
    filter_loop24(
        plane_v,
        pos + 4 * stride,
        stride,
        1,
        8,
        thresh,
        inner_thresh,
        hev_thresh,
    );
}

fn h_filter8i(
    plane_u: &mut [u8],
    plane_v: &mut [u8],
    pos: usize,
    stride: usize,
    thresh: i32,
    inner_thresh: i32,
    hev_thresh: i32,
) {
    filter_loop24(
        plane_u,
        pos + 4,
        1,
        stride,
        8,
        thresh,
        inner_thresh,
        hev_thresh,
    );
    filter_loop24(
        plane_v,
        pos + 4,
        1,
        stride,
        8,
        thresh,
        inner_thresh,
        hev_thresh,
    );
}

fn macroblock_filter_info(
    frame: &MacroBlockDataFrame,
    macroblock: &MacroBlockData,
) -> Option<FilterInfo> {
    let filter = &frame.frame.filter;
    if filter.filter_type == FilterType::Off {
        return None;
    }

    let segment = &frame.frame.segment;
    let mut base_level = if segment.use_segment {
        let level = segment.filter_strength[macroblock.header.segment as usize] as i32;
        if segment.absolute_delta {
            level
        } else {
            level + filter.level as i32
        }
    } else {
        filter.level as i32
    };

    if filter.use_lf_delta {
        base_level += filter.ref_lf_delta[0] as i32;
        if macroblock.header.is_i4x4 {
            base_level += filter.mode_lf_delta[0] as i32;
        }
    }

    let level = base_level.clamp(0, 63);
    if level == 0 {
        return None;
    }

    let mut ilevel = level;
    if filter.sharpness > 0 {
        if filter.sharpness > 4 {
            ilevel >>= 2;
        } else {
            ilevel >>= 1;
        }
        ilevel = ilevel.min(9 - filter.sharpness as i32);
    }
    if ilevel < 1 {
        ilevel = 1;
    }

    Some(FilterInfo {
        f_limit: (2 * level + ilevel) as u8,
        f_ilevel: ilevel as u8,
        f_inner: macroblock.header.is_i4x4 || (macroblock.non_zero_y | macroblock.non_zero_uv) != 0,
        hev_thresh: if level >= 40 {
            2
        } else if level >= 15 {
            1
        } else {
            0
        },
    })
}

fn filter_macroblock(
    frame: &MacroBlockDataFrame,
    planes: &mut Planes,
    mb_x: usize,
    mb_y: usize,
    macroblock: &MacroBlockData,
) {
    let Some(info) = macroblock_filter_info(frame, macroblock) else {
        return;
    };

    let y_pos = mb_y * 16 * planes.y_stride + mb_x * 16;
    let uv_pos = mb_y * 8 * planes.uv_stride + mb_x * 8;
    let limit = info.f_limit as i32;
    let inner = info.f_ilevel as i32;
    let hev = info.hev_thresh as i32;

    match frame.frame.filter.filter_type {
        FilterType::Off => {}
        FilterType::Simple => {
            if mb_x > 0 {
                simple_h_filter16(&mut planes.y, y_pos, planes.y_stride, limit + 4);
            }
            if info.f_inner {
                simple_h_filter16i(&mut planes.y, y_pos, planes.y_stride, limit);
            }
            if mb_y > 0 {
                simple_v_filter16(&mut planes.y, y_pos, planes.y_stride, limit + 4);
            }
            if info.f_inner {
                simple_v_filter16i(&mut planes.y, y_pos, planes.y_stride, limit);
            }
        }
        FilterType::Complex => {
            if mb_x > 0 {
                h_filter16(&mut planes.y, y_pos, planes.y_stride, limit + 4, inner, hev);
                h_filter8(
                    &mut planes.u,
                    &mut planes.v,
                    uv_pos,
                    planes.uv_stride,
                    limit + 4,
                    inner,
                    hev,
                );
            }
            if info.f_inner {
                h_filter16i(&mut planes.y, y_pos, planes.y_stride, limit, inner, hev);
                h_filter8i(
                    &mut planes.u,
                    &mut planes.v,
                    uv_pos,
                    planes.uv_stride,
                    limit,
                    inner,
                    hev,
                );
            }
            if mb_y > 0 {
                v_filter16(&mut planes.y, y_pos, planes.y_stride, limit + 4, inner, hev);
                v_filter8(
                    &mut planes.u,
                    &mut planes.v,
                    uv_pos,
                    planes.uv_stride,
                    limit + 4,
                    inner,
                    hev,
                );
            }
            if info.f_inner {
                v_filter16i(&mut planes.y, y_pos, planes.y_stride, limit, inner, hev);
                v_filter8i(
                    &mut planes.u,
                    &mut planes.v,
                    uv_pos,
                    planes.uv_stride,
                    limit,
                    inner,
                    hev,
                );
            }
        }
    }
}

fn apply_loop_filter(frame: &MacroBlockDataFrame, planes: &mut Planes) {
    if frame.frame.filter.filter_type == FilterType::Off {
        return;
    }

    for mb_y in 0..frame.frame.macroblock_height {
        for mb_x in 0..frame.frame.macroblock_width {
            let macroblock = &frame.macroblocks[mb_y * frame.frame.macroblock_width + mb_x];
            filter_macroblock(frame, planes, mb_x, mb_y, macroblock);
        }
    }
}

fn mul1(value: i32) -> i32 {
    ((value * VP8_TRANSFORM_AC3_C1) >> 16) + value
}

fn mul2(value: i32) -> i32 {
    (value * VP8_TRANSFORM_AC3_C2) >> 16
}

fn clip_byte(value: i32) -> u8 {
    value.clamp(0, 255) as u8
}

fn avg2(a: u8, b: u8) -> u8 {
    ((a as u16 + b as u16 + 1) >> 1) as u8
}

fn avg3(a: u8, b: u8, c: u8) -> u8 {
    ((a as u16 + 2 * b as u16 + c as u16 + 2) >> 2) as u8
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

fn top_samples_luma4(
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

fn predict_true_motion(
    plane: &mut [u8],
    stride: usize,
    plane_width: usize,
    x: usize,
    y: usize,
    size: usize,
) {
    let top = if size == 4 {
        top_samples::<4>(plane, stride, plane_width, x, y).to_vec()
    } else if size == 8 {
        top_samples::<8>(plane, stride, plane_width, x, y).to_vec()
    } else {
        top_samples::<16>(plane, stride, plane_width, x, y).to_vec()
    };
    let left = if size == 4 {
        left_samples::<4>(plane, stride, x, y).to_vec()
    } else if size == 8 {
        left_samples::<8>(plane, stride, x, y).to_vec()
    } else {
        left_samples::<16>(plane, stride, x, y).to_vec()
    };
    let top_left = top_left_sample(plane, stride, x, y) as i32;
    for row in 0..size {
        let left_value = left[row] as i32;
        let offset = (y + row) * stride + x;
        for col in 0..size {
            plane[offset + col] = clip_byte(left_value + top[col] as i32 - top_left);
        }
    }
}

fn predict_luma16(
    plane: &mut [u8],
    stride: usize,
    plane_width: usize,
    x: usize,
    y: usize,
    mode: u8,
) -> Result<(), DecoderError> {
    match mode {
        DC_PRED => {
            let has_top = y > 0;
            let has_left = x > 0;
            let value = match (has_top, has_left) {
                (true, true) => {
                    let top = top_samples::<16>(plane, stride, plane_width, x, y);
                    let left = left_samples::<16>(plane, stride, x, y);
                    let sum_top: u32 = top.into_iter().map(u32::from).sum();
                    let sum_left: u32 = left.into_iter().map(u32::from).sum();
                    ((sum_top + sum_left + 16) >> 5) as u8
                }
                (true, false) => {
                    let top = top_samples::<16>(plane, stride, plane_width, x, y);
                    let sum_top: u32 = top.into_iter().map(u32::from).sum();
                    ((sum_top + 8) >> 4) as u8
                }
                (false, true) => {
                    let left = left_samples::<16>(plane, stride, x, y);
                    let sum_left: u32 = left.into_iter().map(u32::from).sum();
                    ((sum_left + 8) >> 4) as u8
                }
                (false, false) => 128,
            };
            fill_block(plane, stride, x, y, 16, 16, value);
        }
        TM_PRED => predict_true_motion(plane, stride, plane_width, x, y, 16),
        V_PRED => {
            let top = top_samples::<16>(plane, stride, plane_width, x, y);
            for row in 0..16 {
                let offset = (y + row) * stride + x;
                plane[offset..offset + 16].copy_from_slice(&top);
            }
        }
        H_PRED => {
            let left = left_samples::<16>(plane, stride, x, y);
            for (row, value) in left.into_iter().enumerate() {
                let offset = (y + row) * stride + x;
                plane[offset..offset + 16].fill(value);
            }
        }
        _ => return Err(DecoderError::Bitstream("invalid luma prediction mode")),
    }
    Ok(())
}

fn predict_chroma8(
    plane: &mut [u8],
    stride: usize,
    plane_width: usize,
    x: usize,
    y: usize,
    mode: u8,
) -> Result<(), DecoderError> {
    match mode {
        DC_PRED => {
            let has_top = y > 0;
            let has_left = x > 0;
            let value = match (has_top, has_left) {
                (true, true) => {
                    let top = top_samples::<8>(plane, stride, plane_width, x, y);
                    let left = left_samples::<8>(plane, stride, x, y);
                    let sum_top: u32 = top.into_iter().map(u32::from).sum();
                    let sum_left: u32 = left.into_iter().map(u32::from).sum();
                    ((sum_top + sum_left + 8) >> 4) as u8
                }
                (true, false) => {
                    let top = top_samples::<8>(plane, stride, plane_width, x, y);
                    let sum_top: u32 = top.into_iter().map(u32::from).sum();
                    ((sum_top + 4) >> 3) as u8
                }
                (false, true) => {
                    let left = left_samples::<8>(plane, stride, x, y);
                    let sum_left: u32 = left.into_iter().map(u32::from).sum();
                    ((sum_left + 4) >> 3) as u8
                }
                (false, false) => 128,
            };
            fill_block(plane, stride, x, y, 8, 8, value);
        }
        TM_PRED => predict_true_motion(plane, stride, plane_width, x, y, 8),
        V_PRED => {
            let top = top_samples::<8>(plane, stride, plane_width, x, y);
            for row in 0..8 {
                let offset = (y + row) * stride + x;
                plane[offset..offset + 8].copy_from_slice(&top);
            }
        }
        H_PRED => {
            let left = left_samples::<8>(plane, stride, x, y);
            for (row, value) in left.into_iter().enumerate() {
                let offset = (y + row) * stride + x;
                plane[offset..offset + 8].fill(value);
            }
        }
        _ => return Err(DecoderError::Bitstream("invalid chroma prediction mode")),
    }
    Ok(())
}

fn predict_luma4(
    plane: &mut [u8],
    stride: usize,
    plane_width: usize,
    x: usize,
    y: usize,
    mode: u8,
) -> Result<(), DecoderError> {
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
            block[11] = l;
            block[10] = l;
            block[12] = l;
            block[13] = l;
            block[14] = l;
            block[15] = l;
        }
        _ => return Err(DecoderError::Bitstream("invalid 4x4 prediction mode")),
    }

    for row in 0..4 {
        let offset = (y + row) * stride + x;
        plane[offset..offset + 4].copy_from_slice(&block[row * 4..row * 4 + 4]);
    }
    Ok(())
}

fn add_transform(plane: &mut [u8], stride: usize, x: usize, y: usize, coeffs: &[i16]) {
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

fn reconstruct_macroblock(
    planes: &mut Planes,
    mb_x: usize,
    mb_y: usize,
    macroblock: &MacroBlockData,
) -> Result<(), DecoderError> {
    let y_x = mb_x * 16;
    let y_y = mb_y * 16;
    let y_width = planes.y_width();
    let uv_width = planes.uv_width();

    if macroblock.header.is_i4x4 {
        for sub_y in 0..4 {
            for sub_x in 0..4 {
                let block_index = sub_y * 4 + sub_x;
                let dst_x = y_x + sub_x * 4;
                let dst_y = y_y + sub_y * 4;
                predict_luma4(
                    &mut planes.y,
                    planes.y_stride,
                    y_width,
                    dst_x,
                    dst_y,
                    macroblock.header.sub_modes[block_index],
                )?;
                let coeff_offset = block_index * 16;
                add_transform(
                    &mut planes.y,
                    planes.y_stride,
                    dst_x,
                    dst_y,
                    &macroblock.coeffs[coeff_offset..coeff_offset + 16],
                );
            }
        }
    } else {
        predict_luma16(
            &mut planes.y,
            planes.y_stride,
            y_width,
            y_x,
            y_y,
            macroblock.header.luma_mode,
        )?;
        for sub_y in 0..4 {
            for sub_x in 0..4 {
                let block_index = sub_y * 4 + sub_x;
                let coeff_offset = block_index * 16;
                add_transform(
                    &mut planes.y,
                    planes.y_stride,
                    y_x + sub_x * 4,
                    y_y + sub_y * 4,
                    &macroblock.coeffs[coeff_offset..coeff_offset + 16],
                );
            }
        }
    }

    let uv_x = mb_x * 8;
    let uv_y = mb_y * 8;
    predict_chroma8(
        &mut planes.u,
        planes.uv_stride,
        uv_width,
        uv_x,
        uv_y,
        macroblock.header.uv_mode,
    )?;
    predict_chroma8(
        &mut planes.v,
        planes.uv_stride,
        uv_width,
        uv_x,
        uv_y,
        macroblock.header.uv_mode,
    )?;
    for sub_y in 0..2 {
        for sub_x in 0..2 {
            let block_index = sub_y * 2 + sub_x;
            let dst_x = uv_x + sub_x * 4;
            let dst_y = uv_y + sub_y * 4;
            let u_offset = 16 * 16 + block_index * 16;
            let v_offset = 20 * 16 + block_index * 16;
            add_transform(
                &mut planes.u,
                planes.uv_stride,
                dst_x,
                dst_y,
                &macroblock.coeffs[u_offset..u_offset + 16],
            );
            add_transform(
                &mut planes.v,
                planes.uv_stride,
                dst_x,
                dst_y,
                &macroblock.coeffs[v_offset..v_offset + 16],
            );
        }
    }

    Ok(())
}

fn reconstruct_planes(frame: &MacroBlockDataFrame) -> Result<Planes, DecoderError> {
    let expected = frame.frame.macroblock_width * frame.frame.macroblock_height;
    if frame.macroblocks.len() != expected {
        return Err(DecoderError::Bitstream("macroblock count mismatch"));
    }

    let mut planes = Planes::new(frame);
    for mb_y in 0..frame.frame.macroblock_height {
        for mb_x in 0..frame.frame.macroblock_width {
            let macroblock = &frame.macroblocks[mb_y * frame.frame.macroblock_width + mb_x];
            reconstruct_macroblock(&mut planes, mb_x, mb_y, macroblock)?;
        }
    }
    apply_loop_filter(frame, &mut planes);
    Ok(planes)
}

fn mult_hi(value: i32, coeff: i32) -> i32 {
    (value * coeff) >> 8
}

fn clip_rgb(value: i32) -> u8 {
    if (value & !YUV_MASK2) == 0 {
        (value >> YUV_FIX2) as u8
    } else if value < 0 {
        0
    } else {
        255
    }
}

fn write_rgba(yy: u8, u: i32, v: i32, dst: &mut [u8], offset: usize) {
    let yy = yy as i32;
    dst[offset] = clip_rgb(mult_hi(yy, RGB_Y_COEFF) + mult_hi(v, RGB_V_TO_R_COEFF) - RGB_R_BIAS);
    dst[offset + 1] = clip_rgb(
        mult_hi(yy, RGB_Y_COEFF) - mult_hi(u, RGB_U_TO_G_COEFF) - mult_hi(v, RGB_V_TO_G_COEFF)
            + RGB_G_BIAS,
    );
    dst[offset + 2] =
        clip_rgb(mult_hi(yy, RGB_Y_COEFF) + mult_hi(u, RGB_U_TO_B_COEFF) - RGB_B_BIAS);
    dst[offset + 3] = 255;
}

fn upsample_rgba_line_pair(
    top_y: &[u8],
    bottom_y: Option<&[u8]>,
    top_u: &[u8],
    top_v: &[u8],
    cur_u: &[u8],
    cur_v: &[u8],
    rgba: &mut [u8],
    top_offset: usize,
    bottom_offset: Option<usize>,
    len: usize,
) {
    let last_pixel_pair = (len - 1) >> 1;
    let mut tl_u = top_u[0] as i32;
    let mut tl_v = top_v[0] as i32;
    let mut l_u = cur_u[0] as i32;
    let mut l_v = cur_v[0] as i32;

    let uv0_u = (3 * tl_u + l_u + 2) >> 2;
    let uv0_v = (3 * tl_v + l_v + 2) >> 2;
    write_rgba(top_y[0], uv0_u, uv0_v, rgba, top_offset);
    if let (Some(row), Some(offset)) = (bottom_y, bottom_offset) {
        let uv0_u = (3 * l_u + tl_u + 2) >> 2;
        let uv0_v = (3 * l_v + tl_v + 2) >> 2;
        write_rgba(row[0], uv0_u, uv0_v, rgba, offset);
    }

    for x in 1..=last_pixel_pair {
        let t_u = top_u[x] as i32;
        let t_v = top_v[x] as i32;
        let u = cur_u[x] as i32;
        let v = cur_v[x] as i32;

        let avg_u = tl_u + t_u + l_u + u + 8;
        let avg_v = tl_v + t_v + l_v + v + 8;
        let diag_12_u = (avg_u + 2 * (t_u + l_u)) >> 3;
        let diag_12_v = (avg_v + 2 * (t_v + l_v)) >> 3;
        let diag_03_u = (avg_u + 2 * (tl_u + u)) >> 3;
        let diag_03_v = (avg_v + 2 * (tl_v + v)) >> 3;

        let top_left = (2 * x - 1) * 4;
        let top_right = 2 * x * 4;
        write_rgba(
            top_y[2 * x - 1],
            (diag_12_u + tl_u) >> 1,
            (diag_12_v + tl_v) >> 1,
            rgba,
            top_offset + top_left,
        );
        write_rgba(
            top_y[2 * x],
            (diag_03_u + t_u) >> 1,
            (diag_03_v + t_v) >> 1,
            rgba,
            top_offset + top_right,
        );

        if let (Some(row), Some(offset)) = (bottom_y, bottom_offset) {
            write_rgba(
                row[2 * x - 1],
                (diag_03_u + l_u) >> 1,
                (diag_03_v + l_v) >> 1,
                rgba,
                offset + top_left,
            );
            write_rgba(
                row[2 * x],
                (diag_12_u + u) >> 1,
                (diag_12_v + v) >> 1,
                rgba,
                offset + top_right,
            );
        }

        tl_u = t_u;
        tl_v = t_v;
        l_u = u;
        l_v = v;
    }

    if len & 1 == 0 {
        let last = (len - 1) * 4;
        let uv0_u = (3 * tl_u + l_u + 2) >> 2;
        let uv0_v = (3 * tl_v + l_v + 2) >> 2;
        write_rgba(top_y[len - 1], uv0_u, uv0_v, rgba, top_offset + last);
        if let (Some(row), Some(offset)) = (bottom_y, bottom_offset) {
            let uv0_u = (3 * l_u + tl_u + 2) >> 2;
            let uv0_v = (3 * l_v + tl_v + 2) >> 2;
            write_rgba(row[len - 1], uv0_u, uv0_v, rgba, offset + last);
        }
    }
}

fn yuv_to_rgba_fancy(planes: &Planes) -> Vec<u8> {
    let mut rgba = vec![0u8; planes.width * planes.height * 4];
    if planes.width == 0 || planes.height == 0 {
        return rgba;
    }

    let uv_width = planes.width.div_ceil(2);
    let uv_height = planes.height.div_ceil(2);

    let top_y = &planes.y[..planes.width];
    let top_u = &planes.u[..uv_width];
    let top_v = &planes.v[..uv_width];
    upsample_rgba_line_pair(
        top_y,
        None,
        top_u,
        top_v,
        top_u,
        top_v,
        &mut rgba,
        0,
        None,
        planes.width,
    );

    for uv_row in 1..uv_height {
        let top_row = 2 * uv_row - 1;
        let bottom_row = top_row + 1;
        if bottom_row >= planes.height {
            break;
        }
        let top_y = &planes.y[top_row * planes.y_stride..top_row * planes.y_stride + planes.width];
        let bottom_y =
            &planes.y[bottom_row * planes.y_stride..bottom_row * planes.y_stride + planes.width];
        let prev_u =
            &planes.u[(uv_row - 1) * planes.uv_stride..(uv_row - 1) * planes.uv_stride + uv_width];
        let prev_v =
            &planes.v[(uv_row - 1) * planes.uv_stride..(uv_row - 1) * planes.uv_stride + uv_width];
        let cur_u = &planes.u[uv_row * planes.uv_stride..uv_row * planes.uv_stride + uv_width];
        let cur_v = &planes.v[uv_row * planes.uv_stride..uv_row * planes.uv_stride + uv_width];
        upsample_rgba_line_pair(
            top_y,
            Some(bottom_y),
            prev_u,
            prev_v,
            cur_u,
            cur_v,
            &mut rgba,
            top_row * planes.width * 4,
            Some(bottom_row * planes.width * 4),
            planes.width,
        );
    }

    if planes.height > 1 && planes.height & 1 == 0 {
        let last_row = planes.height - 1;
        let uv_row = uv_height - 1;
        let y_row =
            &planes.y[last_row * planes.y_stride..last_row * planes.y_stride + planes.width];
        let u_row = &planes.u[uv_row * planes.uv_stride..uv_row * planes.uv_stride + uv_width];
        let v_row = &planes.v[uv_row * planes.uv_stride..uv_row * planes.uv_stride + uv_width];
        upsample_rgba_line_pair(
            y_row,
            None,
            u_row,
            v_row,
            u_row,
            v_row,
            &mut rgba,
            last_row * planes.width * 4,
            None,
            planes.width,
        );
    }

    rgba
}

fn into_decoded_yuv(planes: Planes) -> DecodedYuvImage {
    DecodedYuvImage {
        width: planes.width,
        height: planes.height,
        y_stride: planes.y_stride,
        uv_stride: planes.uv_stride,
        y: planes.y,
        u: planes.u,
        v: planes.v,
    }
}

pub fn decode_lossy_vp8_to_yuv(data: &[u8]) -> Result<DecodedYuvImage, DecoderError> {
    let frame = parse_macroblock_data(data)?;
    let planes = reconstruct_planes(&frame)?;
    Ok(into_decoded_yuv(planes))
}

pub fn decode_lossy_vp8_to_rgba(data: &[u8]) -> Result<DecodedImage, DecoderError> {
    let yuv = decode_lossy_vp8_to_yuv(data)?;
    Ok(DecodedImage {
        width: yuv.width,
        height: yuv.height,
        rgba: yuv_to_rgba_fancy(&Planes {
            width: yuv.width,
            height: yuv.height,
            y_stride: yuv.y_stride,
            uv_stride: yuv.uv_stride,
            y: yuv.y,
            u: yuv.u,
            v: yuv.v,
        }),
    })
}

pub fn decode_lossy_vp8_to_bmp(data: &[u8]) -> Result<Vec<u8>, DecoderError> {
    let image = decode_lossy_vp8_to_rgba(data)?;
    encode_bmp24_from_rgba(image.width, image.height, &image.rgba)
}

pub fn decode_lossy_webp_to_rgba(data: &[u8]) -> Result<DecodedImage, DecoderError> {
    let parsed = parse_still_webp(data)?;
    if parsed.features.format != WebpFormat::Lossy {
        return Err(DecoderError::Unsupported(
            "only still lossy WebP is supported",
        ));
    }
    if parsed.alpha_data.is_some() {
        return Err(DecoderError::Unsupported("lossy alpha is not implemented"));
    }
    decode_lossy_vp8_to_rgba(parsed.image_data)
}

pub fn decode_lossy_webp_to_yuv(data: &[u8]) -> Result<DecodedYuvImage, DecoderError> {
    let parsed = parse_still_webp(data)?;
    if parsed.features.format != WebpFormat::Lossy {
        return Err(DecoderError::Unsupported(
            "only still lossy WebP is supported",
        ));
    }
    if parsed.alpha_data.is_some() {
        return Err(DecoderError::Unsupported("lossy alpha is not implemented"));
    }
    decode_lossy_vp8_to_yuv(parsed.image_data)
}

pub fn decode_lossy_webp_to_bmp(data: &[u8]) -> Result<Vec<u8>, DecoderError> {
    let image = decode_lossy_webp_to_rgba(data)?;
    encode_bmp24_from_rgba(image.width, image.height, &image.rgba)
}

#[cfg(test)]
mod tests {
    use super::top_samples_luma4;

    #[test]
    fn top_samples_luma4_uses_macroblock_top_right_for_copy_down() {
        let stride = 32usize;
        let width = 32usize;
        let mut plane = vec![0u8; stride * 32];

        plane[19 * stride + 12..19 * stride + 16].copy_from_slice(&[10, 11, 12, 13]);
        plane[15 * stride + 16..15 * stride + 20].copy_from_slice(&[20, 21, 22, 23]);

        let top = top_samples_luma4(&plane, stride, width, 12, 20);

        assert_eq!(top, [10, 11, 12, 13, 20, 21, 22, 23]);
    }
}
