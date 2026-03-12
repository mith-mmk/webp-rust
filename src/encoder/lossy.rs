use crate::decoder::decode_lossy_vp8_to_yuv;
use crate::decoder::quant::{AC_TABLE, DC_TABLE};
use crate::decoder::tree::{BMODES_PROBA, COEFFS_PROBA0, COEFFS_UPDATE_PROBA, Y_MODES_INTRA4};
use crate::decoder::vp8i::{
    B_DC_PRED, B_HD_PRED, B_HE_PRED, B_HU_PRED, B_LD_PRED, B_PRED, B_RD_PRED, B_TM_PRED, B_VE_PRED,
    B_VL_PRED, B_VR_PRED, DC_PRED, H_PRED, MB_FEATURE_TREE_PROBS, NUM_BANDS, NUM_BMODES, NUM_CTX,
    NUM_MB_SEGMENTS, NUM_PROBAS, NUM_TYPES, TM_PRED, V_PRED,
};
use crate::encoder::container::{wrap_still_webp, StillImageChunk};
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

type CoeffProbTables = [[[[u8; NUM_PROBAS]; NUM_CTX]; NUM_BANDS]; NUM_TYPES];
type CoeffStats = [[[[u32; NUM_PROBAS]; NUM_CTX]; NUM_BANDS]; NUM_TYPES];

const DEFAULT_LOSSY_OPTIMIZATION_LEVEL: u8 = 4;
const MAX_LOSSY_OPTIMIZATION_LEVEL: u8 = 9;

/// Lossy encoder tuning knobs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LossyEncodingOptions {
    /// Quality from `0` to `100`.
    pub quality: u8,
    /// Search effort from `0` to `9`.
    ///
    /// The default `4` keeps encode time moderate. `9` enables the heaviest
    /// search profile currently implemented.
    pub optimization_level: u8,
}

impl Default for LossyEncodingOptions {
    fn default() -> Self {
        Self {
            quality: 90,
            optimization_level: DEFAULT_LOSSY_OPTIMIZATION_LEVEL,
        }
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
    sub_luma: [u8; 16],
    chroma: u8,
    segment: u8,
    skip: bool,
}

#[derive(Debug, Clone, Copy)]
struct QuantMatrices {
    y1: [u16; 2],
    y2: [u16; 2],
    uv: [u16; 2],
}

#[derive(Debug, Clone, Copy)]
struct RdMultipliers {
    i16: u32,
    i4: u32,
    uv: u32,
    mode: u32,
}

#[derive(Debug, Clone)]
struct Planes {
    y_stride: usize,
    uv_stride: usize,
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
}

#[derive(Debug, Clone)]
struct SegmentConfig {
    use_segment: bool,
    update_map: bool,
    quantizer: [u8; NUM_MB_SEGMENTS],
    filter_strength: [i8; NUM_MB_SEGMENTS],
    probs: [u8; MB_FEATURE_TREE_PROBS],
    segments: Vec<u8>,
}

#[derive(Debug, Clone, Copy)]
struct FilterConfig {
    simple: bool,
    level: u8,
    sharpness: u8,
}

#[derive(Debug, Clone)]
struct EncodedLossyCandidate {
    base_quant: u8,
    segment: SegmentConfig,
    probabilities: CoeffProbTables,
    modes: Vec<MacroblockMode>,
    token_partition: Vec<u8>,
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
    if options.optimization_level > MAX_LOSSY_OPTIMIZATION_LEVEL {
        return Err(EncoderError::InvalidParam(
            "lossy optimization level must be in 0..=9",
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

fn build_rd_multipliers(quant: &QuantMatrices) -> RdMultipliers {
    let q_i4 = u32::from(quant.y1[1].max(8));
    let q_i16 = u32::from(quant.y2[1].max(8));
    let q_uv = u32::from(quant.uv[1].max(8));
    RdMultipliers {
        i16: ((3 * q_i16 * q_i16).max(128)) >> 0,
        i4: ((3 * q_i4 * q_i4).max(128)) >> 7,
        uv: ((3 * q_uv * q_uv).max(128)) >> 6,
        mode: (q_i4 * q_i4).max(128) >> 7,
    }
}

fn clipped_quantizer(value: i32) -> u8 {
    value.clamp(0, 127) as u8
}

fn filter_candidates(base_quant: i32) -> Vec<FilterConfig> {
    let mut levels = vec![
        0u8,
        clipped_quantizer((base_quant + 1) / 2).min(63),
        clipped_quantizer(base_quant).min(63),
        clipped_quantizer((base_quant * 3 + 1) / 2).min(63),
        clipped_quantizer(base_quant * 2).min(63),
    ];
    levels.sort_unstable();
    levels.dedup();
    levels
        .into_iter()
        .map(|level| FilterConfig {
            simple: false,
            level,
            sharpness: 0,
        })
        .collect()
}

fn heuristic_filter(base_quant: i32) -> FilterConfig {
    let level = if base_quant <= 10 {
        0
    } else {
        clipped_quantizer((base_quant * 3 + 2) / 4).min(63)
    };
    FilterConfig {
        simple: false,
        level,
        sharpness: 0,
    }
}

fn use_exhaustive_segment_search(optimization_level: u8) -> bool {
    optimization_level >= 9
}

fn use_exhaustive_filter_search(optimization_level: u8, mb_count: usize) -> bool {
    if optimization_level >= 9 {
        return true;
    }
    if optimization_level >= 6 {
        return mb_count < 2_048;
    }
    mb_count < 1_024
}

fn segment_with_uniform_filter(segment: &SegmentConfig, level: u8) -> SegmentConfig {
    let mut filtered = segment.clone();
    if filtered.use_segment {
        filtered.filter_strength[..].fill(level as i8);
    }
    filtered
}

fn get_proba(a: usize, b: usize) -> u8 {
    let total = a + b;
    if total == 0 {
        255
    } else {
        ((255 * a + total / 2) / total) as u8
    }
}

fn build_segment_quantizers(segment: &SegmentConfig) -> [QuantMatrices; NUM_MB_SEGMENTS] {
    std::array::from_fn(|index| build_quant_matrices(segment.quantizer[index] as i32))
}

fn disabled_segment_config(mb_count: usize, base_quant: u8) -> SegmentConfig {
    SegmentConfig {
        use_segment: false,
        update_map: false,
        quantizer: [base_quant; NUM_MB_SEGMENTS],
        filter_strength: [0; NUM_MB_SEGMENTS],
        probs: [255; MB_FEATURE_TREE_PROBS],
        segments: vec![0; mb_count],
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

fn macroblock_activity(source: &Planes, mb_x: usize, mb_y: usize) -> u32 {
    let x0 = mb_x * 16;
    let y0 = mb_y * 16;
    let mut activity = 0u32;

    for row in 0..16 {
        let row_offset = (y0 + row) * source.y_stride + x0;
        let pixels = &source.y[row_offset..row_offset + 16];
        for col in 1..16 {
            activity += pixels[col].abs_diff(pixels[col - 1]) as u32;
        }
        if row > 0 {
            let prev_offset = (y0 + row - 1) * source.y_stride + x0;
            let prev = &source.y[prev_offset..prev_offset + 16];
            for col in 0..16 {
                activity += pixels[col].abs_diff(prev[col]) as u32;
            }
        }
    }

    activity
}

fn build_segment_probs(counts: &[usize; NUM_MB_SEGMENTS]) -> [u8; MB_FEATURE_TREE_PROBS] {
    [
        get_proba(counts[0] + counts[1], counts[2] + counts[3]),
        get_proba(counts[0], counts[1]),
        get_proba(counts[2], counts[3]),
    ]
}

fn build_segment_config(
    activities: &[u32],
    sorted_activities: &[u32],
    flat_percent: usize,
    flat_delta: i32,
    detail_delta: i32,
    base_quant: i32,
) -> Option<SegmentConfig> {
    if activities.len() < 8 {
        return None;
    }
    let flat_count = (activities.len() * flat_percent / 100).clamp(1, activities.len() - 1);
    let threshold = sorted_activities[flat_count - 1];

    let mut segments = vec![0u8; activities.len()];
    let mut counts = [0usize; NUM_MB_SEGMENTS];
    for (index, &activity) in activities.iter().enumerate() {
        let segment = if activity <= threshold { 0 } else { 1 };
        segments[index] = segment;
        counts[segment as usize] += 1;
    }
    if counts[0] == 0 || counts[1] == 0 {
        return None;
    }

    let quant0 = clipped_quantizer(base_quant + flat_delta);
    let quant1 = clipped_quantizer(base_quant + detail_delta);
    if quant0 == quant1 {
        return None;
    }

    let probs = build_segment_probs(&counts);
    let update_map = probs.iter().any(|&prob| prob != 255);
    if !update_map {
        return None;
    }

    let mut quantizer = [quant0; NUM_MB_SEGMENTS];
    quantizer[1] = quant1;
    Some(SegmentConfig {
        use_segment: true,
        update_map,
        quantizer,
        filter_strength: [0; NUM_MB_SEGMENTS],
        probs,
        segments,
    })
}

fn build_multi_segment_config(
    activities: &[u32],
    sorted_activities: &[u32],
    percentiles: &[usize],
    deltas: &[i32],
    base_quant: i32,
) -> Option<SegmentConfig> {
    let segment_count = deltas.len();
    if !(2..=NUM_MB_SEGMENTS).contains(&segment_count) || percentiles.len() + 1 != segment_count {
        return None;
    }

    let mut thresholds = Vec::with_capacity(percentiles.len());
    for &percentile in percentiles {
        let split = (activities.len() * percentile / 100).clamp(1, activities.len() - 1);
        thresholds.push(sorted_activities[split - 1]);
    }
    thresholds.sort_unstable();

    let mut segments = vec![0u8; activities.len()];
    let mut counts = [0usize; NUM_MB_SEGMENTS];
    for (index, &activity) in activities.iter().enumerate() {
        let segment = thresholds.partition_point(|&threshold| activity > threshold);
        segments[index] = segment as u8;
        counts[segment] += 1;
    }

    if counts[..segment_count].iter().any(|&count| count == 0) {
        return None;
    }

    let mut quantizer = [clipped_quantizer(base_quant); NUM_MB_SEGMENTS];
    let mut distinct = false;
    for (index, &delta) in deltas.iter().enumerate() {
        quantizer[index] = clipped_quantizer(base_quant + delta);
        if index > 0 && quantizer[index] != quantizer[index - 1] {
            distinct = true;
        }
    }
    if !distinct {
        return None;
    }

    let probs = build_segment_probs(&counts);
    let update_map = probs.iter().any(|&prob| prob != 255);
    if !update_map {
        return None;
    }

    Some(SegmentConfig {
        use_segment: true,
        update_map,
        quantizer,
        filter_strength: [0; NUM_MB_SEGMENTS],
        probs,
        segments,
    })
}

fn build_segment_candidates(
    source: &Planes,
    mb_width: usize,
    mb_height: usize,
    base_quant: i32,
    optimization_level: u8,
) -> Vec<SegmentConfig> {
    let mb_count = mb_width * mb_height;
    let mut candidates = vec![disabled_segment_config(
        mb_count,
        clipped_quantizer(base_quant),
    )];
    if mb_count < 8 || optimization_level == 0 {
        return candidates;
    }

    let mut activities = Vec::with_capacity(mb_count);
    for mb_y in 0..mb_height {
        for mb_x in 0..mb_width {
            activities.push(macroblock_activity(source, mb_x, mb_y));
        }
    }
    let mut sorted = activities.clone();
    sorted.sort_unstable();

    if !use_exhaustive_segment_search(optimization_level) && mb_count >= 1_024 {
        if let Some(config) = build_segment_config(&activities, &sorted, 65, 12, -2, base_quant) {
            return vec![config];
        }
        return candidates;
    }

    let two_segment_presets: &[(usize, i32, i32)] = if optimization_level <= 2 {
        &[(65usize, 12i32, -2i32)]
    } else if mb_count >= 2_048 && !use_exhaustive_segment_search(optimization_level) {
        &[(65usize, 12i32, -2i32), (55, 10, 0)]
    } else {
        &[(55usize, 10i32, 0i32), (65, 12, -2), (45, 8, 0)]
    };
    for &(flat_percent, flat_delta, detail_delta) in two_segment_presets {
        if let Some(config) = build_segment_config(
            &activities,
            &sorted,
            flat_percent,
            flat_delta,
            detail_delta,
            base_quant,
        ) {
            candidates.push(config);
        }
    }

    if optimization_level >= 4
        && (use_exhaustive_segment_search(optimization_level) || mb_count < 2_048)
    {
        for (percentiles, deltas) in [
            (&[35usize, 72usize][..], &[12i32, 4i32, -4i32][..]),
            (
                &[25usize, 50usize, 78usize][..],
                &[16i32, 8i32, 1i32, -7i32][..],
            ),
            (
                &[30usize, 58usize, 84usize][..],
                &[18i32, 10i32, 2i32, -8i32][..],
            ),
        ] {
            if let Some(config) =
                build_multi_segment_config(&activities, &sorted, percentiles, deltas, base_quant)
            {
                candidates.push(config);
            }
        }
    }

    candidates
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

fn avg2(a: u8, b: u8) -> u8 {
    ((a as u16 + b as u16 + 1) >> 1) as u8
}

fn avg3(a: u8, b: u8, c: u8) -> u8 {
    ((a as u16 + 2 * b as u16 + c as u16 + 2) >> 2) as u8
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

fn fill_luma4_prediction_block(
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

fn predict_luma4_block(
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

fn copy_block4(plane: &[u8], stride: usize, x: usize, y: usize) -> [u8; 16] {
    let mut block = [0u8; 16];
    for row in 0..4 {
        let src = (y + row) * stride + x;
        block[row * 4..row * 4 + 4].copy_from_slice(&plane[src..src + 4]);
    }
    block
}

fn restore_block4(plane: &mut [u8], stride: usize, x: usize, y: usize, block: &[u8; 16]) {
    for row in 0..4 {
        let dst = (y + row) * stride + x;
        plane[dst..dst + 4].copy_from_slice(&block[row * 4..row * 4 + 4]);
    }
}

fn copy_block4_from_buffer(buffer: &[u8], stride: usize, x: usize, y: usize) -> [u8; 16] {
    let mut block = [0u8; 16];
    for row in 0..4 {
        let src = (y + row) * stride + x;
        block[row * 4..row * 4 + 4].copy_from_slice(&buffer[src..src + 4]);
    }
    block
}

fn copy_block16(plane: &[u8], stride: usize, x: usize, y: usize) -> [u8; 256] {
    let mut block = [0u8; 256];
    for row in 0..16 {
        let src = (y + row) * stride + x;
        block[row * 16..row * 16 + 16].copy_from_slice(&plane[src..src + 16]);
    }
    block
}

fn restore_block16(plane: &mut [u8], stride: usize, x: usize, y: usize, block: &[u8; 256]) {
    for row in 0..16 {
        let dst = (y + row) * stride + x;
        plane[dst..dst + 16].copy_from_slice(&block[row * 16..row * 16 + 16]);
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

fn dequantize_levels(levels: &[i16; 16], dc_quant: u16, ac_quant: u16) -> [i16; 16] {
    let mut dequantized = [0i16; 16];
    for (index, level) in levels.iter().copied().enumerate() {
        let quant = if index == 0 { dc_quant } else { ac_quant } as i32;
        dequantized[index] = (i32::from(level) * quant) as i16;
    }
    dequantized
}

fn reconstruct_from_prediction(prediction: &[u8; 16], coeffs: &[i16; 16]) -> [u8; 16] {
    let mut block = *prediction;
    add_transform(&mut block, 4, 0, 0, coeffs);
    block
}

fn block_sse_4x4(source: &[u8], stride: usize, x: usize, y: usize, candidate: &[u8; 16]) -> u64 {
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

fn reconstruct_luma16_from_prediction(
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

fn refine_levels_greedy(
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

fn refine_y2_levels_greedy(
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

fn rd_score(distortion: u64, rate: u32, lambda: u32) -> u64 {
    distortion * 256 + u64::from(rate) * u64::from(lambda.max(1))
}

fn i16_mode_rate(mode: u8) -> u32 {
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

fn uv_mode_rate(mode: u8) -> u32 {
    match mode {
        DC_PRED => bit_cost(false, 142),
        V_PRED => bit_cost(true, 142) + bit_cost(false, 114),
        H_PRED => bit_cost(true, 142) + bit_cost(true, 114) + bit_cost(false, 183),
        TM_PRED => bit_cost(true, 142) + bit_cost(true, 114) + bit_cost(true, 183),
        _ => unreachable!("unsupported chroma mode"),
    }
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

fn plane_sse_region(
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

fn yuv_sse(source: &Planes, width: usize, height: usize, vp8: &[u8]) -> Result<u64, EncoderError> {
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

fn evaluate_luma_mode(
    source: &Planes,
    reconstructed: &Planes,
    mb_x: usize,
    mb_y: usize,
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
            let coeffs = refine_levels_greedy(
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
    let y2_coeffs = refine_y2_levels_greedy(
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

fn evaluate_luma4_mode(
    source: &Planes,
    reconstructed: &mut Planes,
    mb_x: usize,
    mb_y: usize,
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
                let dequantized = refine_levels_greedy(
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

fn evaluate_chroma_mode(
    source: &Planes,
    reconstructed: &Planes,
    mb_x: usize,
    mb_y: usize,
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
            let coeffs_u = refine_levels_greedy(
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
            let coeffs_v = refine_levels_greedy(
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

fn choose_macroblock_mode(
    source: &Planes,
    reconstructed: &mut Planes,
    mb_x: usize,
    mb_y: usize,
    quant: &QuantMatrices,
    rd: &RdMultipliers,
    probabilities: &CoeffProbTables,
    top_context: &NonZeroContext,
    left_context: &NonZeroContext,
    top_modes: &[u8],
    left_modes: &[u8; 4],
) -> MacroblockMode {
    const MODES: [u8; 4] = [DC_PRED, V_PRED, H_PRED, TM_PRED];

    let mut best_luma = DC_PRED;
    let mut best_luma_score = u64::MAX;
    for mode in MODES {
        let score = evaluate_luma_mode(
            source,
            reconstructed,
            mb_x,
            mb_y,
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

    let (i4_score, sub_luma) = evaluate_luma4_mode(
        source,
        reconstructed,
        mb_x,
        mb_y,
        quant,
        rd,
        probabilities,
        top_context,
        left_context,
        top_modes,
        left_modes,
    );
    let (best_luma, sub_luma) = if i4_score < best_luma_score {
        (B_PRED, sub_luma)
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

fn block_has_non_zero(levels: &[i16; 16], first: usize) -> bool {
    levels.iter().skip(first).any(|&level| level != 0)
}

fn compute_skip_probability(modes: &[MacroblockMode]) -> Option<u8> {
    const SKIP_PROBA_THRESHOLD: u8 = 250;
    let total = modes.len();
    let skip_count = modes.iter().filter(|mode| mode.skip).count();
    if total == 0 || skip_count == 0 {
        return None;
    }
    let non_skip = total - skip_count;
    let prob_zero = ((non_skip * 255) + total / 2) / total;
    let probability = prob_zero.clamp(1, 254) as u8;
    (probability < SKIP_PROBA_THRESHOLD).then_some(probability)
}

fn intra4_tree_contains(node: i8, mode: u8) -> bool {
    if node <= 0 {
        return (-node) as u8 == mode;
    }
    let node = node as usize;
    intra4_tree_contains(Y_MODES_INTRA4[2 * node], mode)
        || intra4_tree_contains(Y_MODES_INTRA4[2 * node + 1], mode)
}

fn walk_intra4_mode_bits<F: FnMut(bool, u8)>(top_mode: u8, left_mode: u8, mode: u8, emit: &mut F) {
    fn walk<F: FnMut(bool, u8)>(node: usize, mode: u8, probs: &[u8; NUM_BMODES - 1], emit: &mut F) {
        let left = Y_MODES_INTRA4[2 * node];
        let right = Y_MODES_INTRA4[2 * node + 1];
        if intra4_tree_contains(left, mode) {
            emit(false, probs[node]);
            if left > 0 {
                walk(left as usize, mode, probs, emit);
            }
        } else {
            emit(true, probs[node]);
            if right > 0 {
                walk(right as usize, mode, probs, emit);
            }
        }
    }

    let probs = &BMODES_PROBA[top_mode as usize][left_mode as usize];
    walk(0, mode, probs, emit);
}

fn encode_intra4_mode(writer: &mut Vp8BoolWriter, top_mode: u8, left_mode: u8, mode: u8) {
    walk_intra4_mode_bits(top_mode, left_mode, mode, &mut |bit, prob| {
        writer.put_bit(bit, prob);
    });
}

fn intra4_mode_rate(top_mode: u8, left_mode: u8, mode: u8) -> u32 {
    let mut rate = 0u32;
    walk_intra4_mode_bits(top_mode, left_mode, mode, &mut |bit, prob| {
        rate += bit_cost(bit, prob);
    });
    rate
}

fn update_mode_cache(mode: &MacroblockMode, top: &mut [u8], left: &mut [u8; 4]) {
    if mode.luma == B_PRED {
        for sub_y in 0..4 {
            let mut ymode = left[sub_y];
            for sub_x in 0..4 {
                ymode = mode.sub_luma[sub_y * 4 + sub_x];
                top[sub_x] = ymode;
            }
            left[sub_y] = ymode;
        }
    } else {
        top.fill(mode.luma);
        left.fill(mode.luma);
    }
}

fn coeff_probs<'a>(
    probabilities: &'a CoeffProbTables,
    coeff_type: usize,
    coeff_index: usize,
    ctx: usize,
) -> &'a [u8; 11] {
    &probabilities[coeff_type][BANDS[coeff_index]][ctx]
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

fn large_value_rate(value: u32, probs: &[u8; 11]) -> u32 {
    let mut rate = 0;
    if value <= 4 {
        rate += bit_cost(false, probs[3]);
        let not_two = value != 2;
        rate += bit_cost(not_two, probs[4]);
        if not_two {
            rate += bit_cost(value == 4, probs[5]);
        }
        return rate;
    }

    rate += bit_cost(true, probs[3]);
    if value <= 10 {
        rate += bit_cost(false, probs[6]);
        let gt6 = value > 6;
        rate += bit_cost(gt6, probs[7]);
        if !gt6 {
            rate += bit_cost(value == 6, 159);
        } else {
            rate += bit_cost(value >= 9, 165);
            rate += bit_cost((value & 1) == 0, 145);
        }
        return rate;
    }

    rate += bit_cost(true, probs[6]);
    if value < 19 {
        rate += bit_cost(false, probs[8]);
        rate += bit_cost(false, probs[9]);
        let residue = value - 11;
        let mut mask = 1 << 2;
        for &prob in CAT3.iter().take_while(|&&prob| prob != 0) {
            rate += bit_cost((residue & mask) != 0, prob);
            mask >>= 1;
        }
    } else if value < 35 {
        rate += bit_cost(false, probs[8]);
        rate += bit_cost(true, probs[9]);
        let residue = value - 19;
        let mut mask = 1 << 3;
        for &prob in CAT4.iter().take_while(|&&prob| prob != 0) {
            rate += bit_cost((residue & mask) != 0, prob);
            mask >>= 1;
        }
    } else if value < 67 {
        rate += bit_cost(true, probs[8]);
        rate += bit_cost(false, probs[10]);
        let residue = value - 35;
        let mut mask = 1 << 4;
        for &prob in CAT5.iter().take_while(|&&prob| prob != 0) {
            rate += bit_cost((residue & mask) != 0, prob);
            mask >>= 1;
        }
    } else {
        rate += bit_cost(true, probs[8]);
        rate += bit_cost(true, probs[10]);
        let residue = value - 67;
        let mut mask = 1 << 10;
        for &prob in CAT6.iter().take_while(|&&prob| prob != 0) {
            rate += bit_cost((residue & mask) != 0, prob);
            mask >>= 1;
        }
    }
    rate
}

fn coefficients_rate(
    probabilities: &CoeffProbTables,
    coeff_type: usize,
    ctx: usize,
    first: usize,
    levels: &[i16; 16],
) -> u32 {
    let last = last_non_zero(levels, first);
    let mut scan = first;
    let mut probs = coeff_probs(probabilities, coeff_type, scan, ctx);
    let mut rate = bit_cost(last >= scan as isize, probs[0]);
    if last < scan as isize {
        return rate;
    }

    while scan < 16 {
        let coeff = levels[ZIGZAG[scan]];
        rate += bit_cost(coeff != 0, probs[1]);
        scan += 1;
        if coeff == 0 {
            if scan == 16 {
                return rate;
            }
            probs = coeff_probs(probabilities, coeff_type, scan, 0);
            continue;
        }

        let value = coeff.unsigned_abs() as u32;
        let gt1 = value > 1;
        rate += bit_cost(gt1, probs[2]);
        let next_ctx = if gt1 {
            rate += large_value_rate(value, probs);
            2
        } else {
            1
        };
        rate += bit_cost(coeff < 0, 128);

        if scan == 16 {
            return rate;
        }
        probs = coeff_probs(probabilities, coeff_type, scan, next_ctx);
        rate += bit_cost(last >= scan as isize, probs[0]);
        if last < scan as isize {
            return rate;
        }
    }
    rate
}

fn encode_coefficients(
    writer: &mut Vp8BoolWriter,
    probabilities: &CoeffProbTables,
    coeff_type: usize,
    ctx: usize,
    first: usize,
    levels: &[i16; 16],
) -> bool {
    let last = last_non_zero(levels, first);
    let mut scan = first;
    let mut probs = coeff_probs(probabilities, coeff_type, scan, ctx);
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
            probs = coeff_probs(probabilities, coeff_type, scan, 0);
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
        probs = coeff_probs(probabilities, coeff_type, scan, next_ctx);
        if !writer.put_bit(last >= scan as isize, probs[0]) {
            return true;
        }
    }
    true
}

fn record_stat(bit: bool, stat: &mut u32) {
    if *stat >= 0xfffe0000 {
        *stat = ((*stat + 1) >> 1) & 0x7fff7fff;
    }
    *stat += 0x00010000 + bit as u32;
}

fn record_large_value(stats: &mut [u32; NUM_PROBAS], value: u32) {
    let gt4 = value > 4;
    record_stat(gt4, &mut stats[3]);
    if !gt4 {
        let ne2 = value != 2;
        record_stat(ne2, &mut stats[4]);
        if ne2 {
            record_stat(value == 4, &mut stats[5]);
        }
        return;
    }

    let gt10 = value > 10;
    record_stat(gt10, &mut stats[6]);
    if !gt10 {
        record_stat(value > 6, &mut stats[7]);
        return;
    }

    if value < 19 {
        record_stat(false, &mut stats[8]);
        record_stat(false, &mut stats[9]);
    } else if value < 35 {
        record_stat(false, &mut stats[8]);
        record_stat(true, &mut stats[9]);
    } else if value < 67 {
        record_stat(true, &mut stats[8]);
        record_stat(false, &mut stats[10]);
    } else {
        record_stat(true, &mut stats[8]);
        record_stat(true, &mut stats[10]);
    }
}

fn record_coefficients_stats(
    stats: &mut CoeffStats,
    coeff_type: usize,
    ctx: usize,
    first: usize,
    levels: &[i16; 16],
) -> bool {
    let last = last_non_zero(levels, first);
    let mut scan = first;
    let mut current_ctx = ctx;
    record_stat(
        last >= scan as isize,
        &mut stats[coeff_type][BANDS[scan]][current_ctx][0],
    );
    if last < scan as isize {
        return false;
    }

    while scan < 16 {
        let coeff = levels[ZIGZAG[scan]];
        let band = BANDS[scan];
        record_stat(coeff != 0, &mut stats[coeff_type][band][current_ctx][1]);
        scan += 1;
        if coeff == 0 {
            if scan == 16 {
                return false;
            }
            current_ctx = 0;
            continue;
        }

        let value = coeff.unsigned_abs() as u32;
        let gt1 = value > 1;
        record_stat(gt1, &mut stats[coeff_type][band][current_ctx][2]);
        if gt1 {
            record_large_value(&mut stats[coeff_type][band][current_ctx], value);
        }

        if scan == 16 {
            return true;
        }
        current_ctx = if gt1 { 2 } else { 1 };
        record_stat(
            last >= scan as isize,
            &mut stats[coeff_type][BANDS[scan]][current_ctx][0],
        );
        if last < scan as isize {
            return true;
        }
    }
    true
}

fn bit_cost(bit: bool, prob: u8) -> u32 {
    let p = if bit {
        255u16.saturating_sub(prob as u16)
    } else {
        prob as u16
    };
    let p = (p.max(1) as f64) / 256.0;
    ((-p.log2()) * 256.0 + 0.5) as u32
}

fn calc_token_probability(nb: u32, total: u32) -> u8 {
    if nb == 0 {
        255
    } else {
        (255 - nb * 255 / total) as u8
    }
}

fn branch_cost(nb: u32, total: u32, prob: u8) -> u32 {
    nb * bit_cost(true, prob) + (total - nb) * bit_cost(false, prob)
}

fn finalize_token_probabilities(stats: &CoeffStats) -> CoeffProbTables {
    let mut probabilities = COEFFS_PROBA0;
    for t in 0..NUM_TYPES {
        for b in 0..NUM_BANDS {
            for c in 0..NUM_CTX {
                for p in 0..NUM_PROBAS {
                    let stat = stats[t][b][c][p];
                    let nb = stat & 0xffff;
                    let total = stat >> 16;
                    let update_prob = COEFFS_UPDATE_PROBA[t][b][c][p];
                    let old_prob = COEFFS_PROBA0[t][b][c][p];
                    let new_prob = calc_token_probability(nb, total);
                    let old_cost = branch_cost(nb, total, old_prob) + bit_cost(false, update_prob);
                    let new_cost =
                        branch_cost(nb, total, new_prob) + bit_cost(true, update_prob) + 8 * 256;
                    probabilities[t][b][c][p] = if old_cost > new_cost {
                        new_prob
                    } else {
                        old_prob
                    };
                }
            }
        }
    }
    probabilities
}

fn encode_partition0(
    mb_width: usize,
    mb_height: usize,
    base_quant: u8,
    segment: &SegmentConfig,
    filter: &FilterConfig,
    probabilities: &CoeffProbTables,
    modes: &[MacroblockMode],
) -> Vec<u8> {
    let mut writer = Vp8BoolWriter::new(mb_width * mb_height);
    writer.put_bit_uniform(false);
    writer.put_bit_uniform(false);

    writer.put_bit_uniform(segment.use_segment);
    if segment.use_segment {
        writer.put_bit_uniform(segment.update_map);
        writer.put_bit_uniform(true);
        writer.put_bit_uniform(true);
        for &quant in &segment.quantizer {
            writer.put_signed_bits(quant as i32, 7);
        }
        for &strength in &segment.filter_strength {
            writer.put_signed_bits(strength as i32, 6);
        }
        if segment.update_map {
            for &prob in &segment.probs {
                if writer.put_bit_uniform(prob != 255) {
                    writer.put_bits(prob as u32, 8);
                }
            }
        }
    }

    writer.put_bit_uniform(filter.simple);
    writer.put_bits(filter.level as u32, 6);
    writer.put_bits(filter.sharpness as u32, 3);
    writer.put_bit_uniform(false);

    writer.put_bits(0, 2);
    writer.put_bits(base_quant as u32, 7);
    for _ in 0..5 {
        writer.put_signed_bits(0, 4);
    }
    writer.put_bit_uniform(false);

    for t in 0..NUM_TYPES {
        for b in 0..NUM_BANDS {
            for c in 0..NUM_CTX {
                for p in 0..NUM_PROBAS {
                    let update = probabilities[t][b][c][p] != COEFFS_PROBA0[t][b][c][p];
                    writer.put_bit(update, COEFFS_UPDATE_PROBA[t][b][c][p]);
                    if update {
                        writer.put_bits(probabilities[t][b][c][p] as u32, 8);
                    }
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

    let mut top_modes = vec![B_DC_PRED; mb_width * 4];
    let mut left_modes = [B_DC_PRED; 4];
    for (index, mode) in modes.iter().enumerate() {
        if index % mb_width == 0 {
            left_modes = [B_DC_PRED; 4];
        }
        if segment.update_map {
            if writer.put_bit(mode.segment >= 2, segment.probs[0]) {
                writer.put_bit(mode.segment == 3, segment.probs[2]);
            } else {
                writer.put_bit(mode.segment == 1, segment.probs[1]);
            }
        }
        if let Some(prob) = skip_probability {
            writer.put_bit(mode.skip, prob);
        }
        let mb_x = index % mb_width;
        let top = &mut top_modes[mb_x * 4..mb_x * 4 + 4];
        if mode.luma == B_PRED {
            writer.put_bit(false, 145);
            for sub_y in 0..4 {
                let mut ymode = left_modes[sub_y];
                for sub_x in 0..4 {
                    let sub_mode = mode.sub_luma[sub_y * 4 + sub_x];
                    encode_intra4_mode(&mut writer, top[sub_x], ymode, sub_mode);
                    top[sub_x] = sub_mode;
                    ymode = sub_mode;
                }
                left_modes[sub_y] = ymode;
            }
        } else {
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
            top.fill(mode.luma);
            left_modes.fill(mode.luma);
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
    probabilities: &CoeffProbTables,
    source: &Planes,
    reconstructed: &mut Planes,
    mb_x: usize,
    mb_y: usize,
    mode: MacroblockMode,
    quant: &QuantMatrices,
    top: &mut NonZeroContext,
    left: &mut NonZeroContext,
    stats: Option<&mut CoeffStats>,
) -> bool {
    let y_x = mb_x * 16;
    let y_y = mb_y * 16;
    let uv_x = mb_x * 8;
    let uv_y = mb_y * 8;
    let is_i4x4 = mode.luma == B_PRED;
    let mut stats = stats;
    let rd = build_rd_multipliers(quant);

    if !is_i4x4 {
        predict_block::<16>(
            &mut reconstructed.y,
            reconstructed.y_stride,
            reconstructed.y_stride,
            y_x,
            y_y,
            mode.luma,
        );
    }
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
    let mut y2_levels = [0i16; 16];

    if is_i4x4 {
        for sub_y in 0..4 {
            for sub_x in 0..4 {
                let block = sub_y * 4 + sub_x;
                let block_x = y_x + sub_x * 4;
                let block_y = y_y + sub_y * 4;
                predict_luma4_block(
                    &mut reconstructed.y,
                    reconstructed.y_stride,
                    reconstructed.y_stride,
                    block_x,
                    block_y,
                    mode.sub_luma[block],
                );
                let prediction_block =
                    copy_block4(&reconstructed.y, reconstructed.y_stride, block_x, block_y);
                let coeffs = forward_transform(
                    &source.y,
                    source.y_stride,
                    &reconstructed.y,
                    reconstructed.y_stride,
                    block_x,
                    block_y,
                );
                let ctx = ((left.nz >> sub_y) & 1) as usize + ((top.nz >> sub_x) & 1) as usize;
                let (mut levels, _) = quantize_block(&coeffs, quant.y1[0], quant.y1[1], 0);
                let coeffs = refine_levels_greedy(
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
                y_levels[block] = levels;
                y_coeffs[block] = coeffs;
                add_transform(
                    &mut reconstructed.y,
                    reconstructed.y_stride,
                    block_x,
                    block_y,
                    &y_coeffs[block],
                );
            }
        }
    } else {
        let mut y_dc = [0i16; 16];
        let mut refine_tnz = top.nz & 0x0f;
        let mut refine_lnz = left.nz & 0x0f;
        for sub_y in 0..4 {
            let mut l = refine_lnz & 1;
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
                let prediction_block = copy_block4(
                    &reconstructed.y,
                    reconstructed.y_stride,
                    y_x + sub_x * 4,
                    y_y + sub_y * 4,
                );
                let (mut levels, _) = quantize_block(&ac_only, quant.y1[0], quant.y1[1], 1);
                let ctx = (l + (refine_tnz & 1)) as usize;
                let coeffs = refine_levels_greedy(
                    &source.y,
                    source.y_stride,
                    y_x + sub_x * 4,
                    y_y + sub_y * 4,
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
        let prediction_block = copy_block16(&reconstructed.y, reconstructed.y_stride, y_x, y_y);
        let (mut levels, _) = quantize_block(&y2_input, quant.y2[0], quant.y2[1], 0);
        let y2_coeffs = refine_y2_levels_greedy(
            &source.y,
            source.y_stride,
            y_x,
            y_y,
            &prediction_block,
            &y_coeffs,
            probabilities,
            (top.nz_dc + left.nz_dc) as usize,
            quant.y2[0],
            quant.y2[1],
            rd.i16,
            &mut levels,
        );
        y2_levels = levels;
        let y2_dc = inverse_wht(&y2_coeffs);
        for block in 0..16 {
            y_coeffs[block][0] = y2_dc[block];
        }
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
            let prediction_block = copy_block4(
                &reconstructed.u,
                reconstructed.uv_stride,
                uv_x + sub_x * 4,
                uv_y + sub_y * 4,
            );
            let (mut levels, _) = quantize_block(&coeffs, quant.uv[0], quant.uv[1], 0);
            let coeffs = refine_levels_greedy(
                &source.u,
                source.uv_stride,
                uv_x + sub_x * 4,
                uv_y + sub_y * 4,
                &prediction_block,
                probabilities,
                2,
                0,
                0,
                quant.uv[0],
                quant.uv[1],
                rd.uv,
                &mut levels,
            );
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
            let prediction_block = copy_block4(
                &reconstructed.v,
                reconstructed.uv_stride,
                uv_x + sub_x * 4,
                uv_y + sub_y * 4,
            );
            let (mut levels, _) = quantize_block(&coeffs, quant.uv[0], quant.uv[1], 0);
            let coeffs = refine_levels_greedy(
                &source.v,
                source.uv_stride,
                uv_x + sub_x * 4,
                uv_y + sub_y * 4,
                &prediction_block,
                probabilities,
                2,
                0,
                0,
                quant.uv[0],
                quant.uv[1],
                rd.uv,
                &mut levels,
            );
            v_levels[block] = levels;
            v_coeffs[block] = coeffs;
        }
    }

    let skip = (!is_i4x4
        && !block_has_non_zero(&y2_levels, 0)
        && y_levels.iter().all(|levels| !block_has_non_zero(levels, 1))
        || is_i4x4 && y_levels.iter().all(|levels| !block_has_non_zero(levels, 0)))
        && u_levels.iter().all(|levels| !block_has_non_zero(levels, 0))
        && v_levels.iter().all(|levels| !block_has_non_zero(levels, 0));
    if skip {
        top.nz = 0;
        left.nz = 0;
        if !is_i4x4 {
            top.nz_dc = 0;
            left.nz_dc = 0;
        }
        return true;
    }

    let (coeff_type, first) = if is_i4x4 {
        (3, 0)
    } else {
        let ctx = (top.nz_dc + left.nz_dc) as usize;
        let has_y2 = if let Some(stats) = stats.as_deref_mut() {
            let recorded = record_coefficients_stats(stats, 1, ctx, 0, &y2_levels);
            let encoded = encode_coefficients(writer, probabilities, 1, ctx, 0, &y2_levels);
            debug_assert_eq!(recorded, encoded);
            encoded
        } else {
            encode_coefficients(writer, probabilities, 1, ctx, 0, &y2_levels)
        };
        top.nz_dc = has_y2 as u8;
        left.nz_dc = has_y2 as u8;
        (0, 1)
    };

    let mut tnz = top.nz & 0x0f;
    let mut lnz = left.nz & 0x0f;
    for sub_y in 0..4 {
        let mut l = lnz & 1;
        for sub_x in 0..4 {
            let block = sub_y * 4 + sub_x;
            let ctx = (l + (tnz & 1)) as usize;
            let has_ac = if let Some(stats) = stats.as_deref_mut() {
                let recorded =
                    record_coefficients_stats(stats, coeff_type, ctx, first, &y_levels[block]);
                let encoded = encode_coefficients(
                    writer,
                    probabilities,
                    coeff_type,
                    ctx,
                    first,
                    &y_levels[block],
                );
                debug_assert_eq!(recorded, encoded);
                encoded
            } else {
                encode_coefficients(
                    writer,
                    probabilities,
                    coeff_type,
                    ctx,
                    first,
                    &y_levels[block],
                )
            };
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
            let has_coeffs = if let Some(stats) = stats.as_deref_mut() {
                let recorded = record_coefficients_stats(stats, 2, ctx, 0, &u_levels[block]);
                let encoded =
                    encode_coefficients(writer, probabilities, 2, ctx, 0, &u_levels[block]);
                debug_assert_eq!(recorded, encoded);
                encoded
            } else {
                encode_coefficients(writer, probabilities, 2, ctx, 0, &u_levels[block])
            } as u8;
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
            let has_coeffs = if let Some(stats) = stats.as_deref_mut() {
                let recorded = record_coefficients_stats(stats, 2, ctx, 0, &v_levels[block]);
                let encoded =
                    encode_coefficients(writer, probabilities, 2, ctx, 0, &v_levels[block]);
                debug_assert_eq!(recorded, encoded);
                encoded
            } else {
                encode_coefficients(writer, probabilities, 2, ctx, 0, &v_levels[block])
            } as u8;
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

    if !is_i4x4 {
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
    segment: &SegmentConfig,
    segment_quants: &[QuantMatrices; NUM_MB_SEGMENTS],
    probabilities: &CoeffProbTables,
    stats: Option<&mut CoeffStats>,
) -> (Vec<u8>, Planes, Vec<MacroblockMode>) {
    let mut writer = Vp8BoolWriter::new(source.y.len() / 4);
    let mut reconstructed = empty_reconstructed_planes(mb_width, mb_height);
    let mut top_contexts = vec![NonZeroContext::default(); mb_width];
    let mut top_modes = vec![B_DC_PRED; mb_width * 4];
    let mut modes = Vec::with_capacity(mb_width * mb_height);
    let mut stats = stats;
    let segment_rd: [RdMultipliers; NUM_MB_SEGMENTS] =
        std::array::from_fn(|index| build_rd_multipliers(&segment_quants[index]));

    for mb_y in 0..mb_height {
        let mut left_context = NonZeroContext::default();
        let mut left_modes = [B_DC_PRED; 4];
        for mb_x in 0..mb_width {
            let index = mb_y * mb_width + mb_x;
            let segment_id = segment.segments[index] as usize;
            let quant = &segment_quants[segment_id];
            let rd = &segment_rd[segment_id];
            let top = &mut top_modes[mb_x * 4..mb_x * 4 + 4];
            let mut mode = choose_macroblock_mode(
                source,
                &mut reconstructed,
                mb_x,
                mb_y,
                quant,
                &rd,
                probabilities,
                &top_contexts[mb_x],
                &left_context,
                top,
                &left_modes,
            );
            mode.segment = segment_id as u8;
            update_mode_cache(&mode, top, &mut left_modes);
            mode.skip = encode_macroblock(
                &mut writer,
                probabilities,
                source,
                &mut reconstructed,
                mb_x,
                mb_y,
                mode,
                quant,
                &mut top_contexts[mb_x],
                &mut left_context,
                stats.as_deref_mut(),
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

fn build_candidate_vp8_frame(
    width: usize,
    height: usize,
    mb_width: usize,
    mb_height: usize,
    candidate: &EncodedLossyCandidate,
    filter: &FilterConfig,
) -> Result<Vec<u8>, EncoderError> {
    let segment = segment_with_uniform_filter(&candidate.segment, filter.level);
    let partition0 = encode_partition0(
        mb_width,
        mb_height,
        candidate.base_quant,
        &segment,
        filter,
        &candidate.probabilities,
        &candidate.modes,
    );
    build_vp8_frame(width, height, &partition0, &candidate.token_partition)
}

fn encode_lossy_candidate(
    source: &Planes,
    mb_width: usize,
    mb_height: usize,
    segment: &SegmentConfig,
) -> Result<EncodedLossyCandidate, EncoderError> {
    let segment_quants = build_segment_quantizers(segment);
    let mut stats = [[[[0u32; NUM_PROBAS]; NUM_CTX]; NUM_BANDS]; NUM_TYPES];
    let (initial_partition, _, initial_modes) = encode_token_partition(
        source,
        mb_width,
        mb_height,
        segment,
        &segment_quants,
        &COEFFS_PROBA0,
        Some(&mut stats),
    );
    let probabilities = finalize_token_probabilities(&stats);
    let (token_partition, modes) = if probabilities == COEFFS_PROBA0 {
        (initial_partition, initial_modes)
    } else {
        let (token_partition, _, modes) = encode_token_partition(
            source,
            mb_width,
            mb_height,
            segment,
            &segment_quants,
            &probabilities,
            None,
        );
        (token_partition, modes)
    };
    Ok(EncodedLossyCandidate {
        base_quant: segment.quantizer[0],
        segment: segment.clone(),
        probabilities,
        modes,
        token_partition,
    })
}

fn finalize_lossy_candidate(
    width: usize,
    height: usize,
    source: &Planes,
    mb_width: usize,
    mb_height: usize,
    base_quant: i32,
    optimization_level: u8,
    candidate: &EncodedLossyCandidate,
) -> Result<Vec<u8>, EncoderError> {
    let mb_count = mb_width * mb_height;
    if !use_exhaustive_filter_search(optimization_level, mb_count) {
        let filter = heuristic_filter(base_quant);
        return build_candidate_vp8_frame(width, height, mb_width, mb_height, candidate, &filter);
    }

    let filters = filter_candidates(base_quant);
    let mut best = None;
    for filter in &filters {
        let vp8 = build_candidate_vp8_frame(width, height, mb_width, mb_height, candidate, filter)?;
        let distortion = yuv_sse(source, width, height, &vp8)?;
        let replace = match &best {
            Some((best_distortion, best_len, _)) => {
                distortion < *best_distortion
                    || (distortion == *best_distortion && vp8.len() < *best_len)
            }
            None => true,
        };
        if replace {
            best = Some((distortion, vp8.len(), vp8));
        }
    }

    best.map(|(_, _, vp8)| vp8).ok_or(EncoderError::Bitstream(
        "lossy filter search produced no output",
    ))
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
    let source = rgba_to_yuv420(width, height, rgba, mb_width, mb_height);
    let candidates = build_segment_candidates(
        &source,
        mb_width,
        mb_height,
        base_quant,
        options.optimization_level,
    );
    let mut best = None;
    for segment in &candidates {
        let candidate = encode_lossy_candidate(&source, mb_width, mb_height, segment)?;
        let vp8 = finalize_lossy_candidate(
            width,
            height,
            &source,
            mb_width,
            mb_height,
            base_quant,
            options.optimization_level,
            &candidate,
        )?;
        let replace = match &best {
            Some((best_bytes, _)) => vp8.len() < *best_bytes,
            None => true,
        };
        if replace {
            best = Some((vp8.len(), vp8));
        }
    }

    best.map(|(_, vp8)| vp8).ok_or(EncoderError::Bitstream(
        "lossy candidate search produced no output",
    ))
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
    encode_lossy_rgba_to_webp_with_options_and_exif(width, height, rgba, options, None)
}

/// Encodes RGBA pixels to a still lossy WebP container with explicit options and EXIF.
pub fn encode_lossy_rgba_to_webp_with_options_and_exif(
    width: usize,
    height: usize,
    rgba: &[u8],
    options: &LossyEncodingOptions,
    exif: Option<&[u8]>,
) -> Result<Vec<u8>, EncoderError> {
    let vp8 = encode_lossy_rgba_to_vp8_with_options(width, height, rgba, options)?;
    wrap_still_webp(
        StillImageChunk {
            fourcc: *b"VP8 ",
            payload: &vp8,
            width,
            height,
            has_alpha: false,
        },
        exif,
    )
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
    encode_lossy_image_to_webp_with_options_and_exif(image, options, None)
}

/// Encodes an [`ImageBuffer`] to a still lossy WebP container with explicit options and EXIF.
pub fn encode_lossy_image_to_webp_with_options_and_exif(
    image: &ImageBuffer,
    options: &LossyEncodingOptions,
    exif: Option<&[u8]>,
) -> Result<Vec<u8>, EncoderError> {
    encode_lossy_rgba_to_webp_with_options_and_exif(
        image.width,
        image.height,
        &image.rgba,
        options,
        exif,
    )
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
        let source = rgba_to_yuv420(width, height, &rgba, mb_width, mb_height);
        let segment = disabled_segment_config(mb_width * mb_height, clipped_quantizer(base_quant));
        let candidate = encode_lossy_candidate(&source, mb_width, mb_height, &segment).unwrap();
        let partition0 = encode_partition0(
            mb_width,
            mb_height,
            base_quant as u8,
            &segment,
            &FilterConfig {
                simple: false,
                level: 0,
                sharpness: 0,
            },
            &candidate.probabilities,
            &candidate.modes,
        );
        let vp8 = build_vp8_frame(width, height, &partition0, &candidate.token_partition).unwrap();
        let decoded = decode_lossy_vp8_to_yuv(&vp8).unwrap();
        let (_, reconstructed, _) = encode_token_partition(
            &source,
            mb_width,
            mb_height,
            &segment,
            &build_segment_quantizers(&segment),
            &candidate.probabilities,
            None,
        );
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
        let rd = build_rd_multipliers(&quant);
        let top_modes = [B_DC_PRED; 4];
        let left_modes = [B_DC_PRED; 4];
        let top_context = NonZeroContext::default();
        let left_context = NonZeroContext::default();
        let mode = choose_macroblock_mode(
            &source,
            &mut reconstructed,
            0,
            1,
            &quant,
            &rd,
            &COEFFS_PROBA0,
            &top_context,
            &left_context,
            &top_modes,
            &left_modes,
        );
        assert!(matches!(mode.luma, V_PRED | B_PRED));
        assert_eq!(mode.chroma, V_PRED);
    }

    #[test]
    fn segment_candidates_include_segmented_plan_for_mixed_activity() {
        let width = 64;
        let height = 32;
        let mb_width = (width + 15) >> 4;
        let mb_height = (height + 15) >> 4;
        let mut rgba = vec![0u8; width * height * 4];
        for y in 0..height {
            for x in 0..width {
                let offset = (y * width + x) * 4;
                let (r, g, b) = if x < width / 2 {
                    (0x80, 0x80, 0x80)
                } else {
                    (
                        ((x * 17 + y * 3) & 0xff) as u8,
                        ((x * 5 + y * 11) & 0xff) as u8,
                        ((x * 13 + y * 7) & 0xff) as u8,
                    )
                };
                rgba[offset] = r;
                rgba[offset + 1] = g;
                rgba[offset + 2] = b;
                rgba[offset + 3] = 0xff;
            }
        }

        let source = rgba_to_yuv420(width, height, &rgba, mb_width, mb_height);
        let candidates = build_segment_candidates(
            &source,
            mb_width,
            mb_height,
            13,
            DEFAULT_LOSSY_OPTIMIZATION_LEVEL,
        );

        assert!(candidates.iter().any(|candidate| candidate.use_segment));
        assert!(candidates
            .iter()
            .filter(|candidate| candidate.use_segment)
            .any(|candidate| candidate.segments.iter().any(|&segment| segment != 0)));
    }

    #[test]
    fn segment_candidates_can_use_more_than_two_segments() {
        let width = 96;
        let height = 64;
        let mb_width = (width + 15) >> 4;
        let mb_height = (height + 15) >> 4;
        let mut rgba = vec![0u8; width * height * 4];
        for y in 0..height {
            for x in 0..width {
                let offset = (y * width + x) * 4;
                let band = x / 24;
                let value = match band {
                    0 => 96,
                    1 => ((x * 3 + y * 5) & 0xff) as u8,
                    2 => ((x * 9 + y * 13) & 0xff) as u8,
                    _ => ((x * 17 + y * 29) & 0xff) as u8,
                };
                rgba[offset] = value;
                rgba[offset + 1] = value.wrapping_add((band * 17) as u8);
                rgba[offset + 2] = value.wrapping_add((band * 33) as u8);
                rgba[offset + 3] = 0xff;
            }
        }

        let source = rgba_to_yuv420(width, height, &rgba, mb_width, mb_height);
        let candidates = build_segment_candidates(
            &source,
            mb_width,
            mb_height,
            13,
            MAX_LOSSY_OPTIMIZATION_LEVEL,
        );

        assert!(candidates.iter().any(|candidate| {
            candidate.use_segment && candidate.segments.iter().copied().max().unwrap_or(0) >= 2
        }));
    }
}
