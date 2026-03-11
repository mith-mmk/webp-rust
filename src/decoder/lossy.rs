use crate::bmp::encode_bmp24_from_rgba;
use crate::decoder::header::parse_still_webp;
use crate::decoder::vp8::{parse_macroblock_data, MacroBlockData, MacroBlockDataFrame};
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
    let top = top_samples::<8>(plane, stride, plane_width, x, y);
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
            block[8] = block[0];
            block[1] = avg2(b, c);
            block[9] = block[1];
            block[2] = avg2(c, d);
            block[10] = block[2];
            block[3] = avg2(d, e);
            block[4] = avg3(a, b, c);
            block[12] = block[4];
            block[5] = avg3(b, c, d);
            block[13] = block[5];
            block[6] = avg3(c, d, e);
            block[14] = block[6];
            block[7] = avg3(d, e, f);
            block[11] = avg3(e, f, g);
            block[15] = avg3(f, g, h);
        }
        B_HD_PRED => {
            block[0] = avg2(i, x0);
            block[9] = block[0];
            block[4] = avg2(j, i);
            block[10] = block[4];
            block[8] = avg2(k, j);
            block[14] = block[8];
            block[12] = avg2(l, k);
            block[3] = avg3(a, b, c);
            block[2] = avg3(x0, a, b);
            block[1] = avg3(i, x0, a);
            block[7] = block[1];
            block[5] = avg3(j, i, x0);
            block[11] = block[5];
            block[6] = avg3(k, j, i);
            block[15] = block[6];
            block[13] = avg3(l, k, j);
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

fn yuv_to_rgba(planes: &Planes) -> Vec<u8> {
    let mut rgba = vec![0u8; planes.width * planes.height * 4];
    for y in 0..planes.height {
        let y_row = y * planes.y_stride;
        let uv_row = (y / 2) * planes.uv_stride;
        let out_row = y * planes.width * 4;
        for x in 0..planes.width {
            let yy = planes.y[y_row + x] as i32;
            let u = planes.u[uv_row + x / 2] as i32;
            let v = planes.v[uv_row + x / 2] as i32;
            let dst = out_row + x * 4;
            rgba[dst] =
                clip_rgb(mult_hi(yy, RGB_Y_COEFF) + mult_hi(v, RGB_V_TO_R_COEFF) - RGB_R_BIAS);
            rgba[dst + 1] = clip_rgb(
                mult_hi(yy, RGB_Y_COEFF)
                    - mult_hi(u, RGB_U_TO_G_COEFF)
                    - mult_hi(v, RGB_V_TO_G_COEFF)
                    + RGB_G_BIAS,
            );
            rgba[dst + 2] =
                clip_rgb(mult_hi(yy, RGB_Y_COEFF) + mult_hi(u, RGB_U_TO_B_COEFF) - RGB_B_BIAS);
            rgba[dst + 3] = 255;
        }
    }
    rgba
}

pub fn decode_lossy_vp8_to_rgba(data: &[u8]) -> Result<DecodedImage, DecoderError> {
    let frame = parse_macroblock_data(data)?;
    let planes = reconstruct_planes(&frame)?;
    Ok(DecodedImage {
        width: planes.width,
        height: planes.height,
        rgba: yuv_to_rgba(&planes),
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

pub fn decode_lossy_webp_to_bmp(data: &[u8]) -> Result<Vec<u8>, DecoderError> {
    let image = decode_lossy_webp_to_rgba(data)?;
    encode_bmp24_from_rgba(image.width, image.height, &image.rgba)
}
