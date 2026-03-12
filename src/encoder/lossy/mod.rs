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

const DEFAULT_LOSSY_OPTIMIZATION_LEVEL: u8 = 0;
const MAX_LOSSY_OPTIMIZATION_LEVEL: u8 = 9;

/// Lossy encoder tuning knobs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LossyEncodingOptions {
    /// Quality from `0` to `100`.
    pub quality: u8,
    /// Search effort from `0` to `9`.
    ///
    /// The default `0` favors fast encode speed. `9` enables the heaviest
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

#[derive(Debug, Clone, Copy)]
struct LossySearchProfile {
    fast_mode_search: bool,
    allow_i4x4: bool,
    refine_i16: bool,
    refine_i4_search: bool,
    refine_i4_final: bool,
    refine_chroma: bool,
    refine_y2: bool,
    update_probabilities: bool,
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

fn lossy_search_profile(optimization_level: u8) -> LossySearchProfile {
    match optimization_level {
        0 => LossySearchProfile {
            fast_mode_search: true,
            allow_i4x4: false,
            refine_i16: false,
            refine_i4_search: false,
            refine_i4_final: false,
            refine_chroma: false,
            refine_y2: false,
            update_probabilities: false,
        },
        1 | 2 => LossySearchProfile {
            fast_mode_search: false,
            allow_i4x4: false,
            refine_i16: false,
            refine_i4_search: false,
            refine_i4_final: false,
            refine_chroma: false,
            refine_y2: false,
            update_probabilities: true,
        },
        3 | 4 => LossySearchProfile {
            fast_mode_search: false,
            allow_i4x4: true,
            refine_i16: false,
            refine_i4_search: false,
            refine_i4_final: false,
            refine_chroma: false,
            refine_y2: false,
            update_probabilities: true,
        },
        5 => LossySearchProfile {
            fast_mode_search: false,
            allow_i4x4: true,
            refine_i16: false,
            refine_i4_search: false,
            refine_i4_final: false,
            refine_chroma: true,
            refine_y2: false,
            update_probabilities: true,
        },
        6 => LossySearchProfile {
            fast_mode_search: false,
            allow_i4x4: true,
            refine_i16: true,
            refine_i4_search: false,
            refine_i4_final: true,
            refine_chroma: true,
            refine_y2: false,
            update_probabilities: true,
        },
        7 => LossySearchProfile {
            fast_mode_search: false,
            allow_i4x4: true,
            refine_i16: true,
            refine_i4_search: true,
            refine_i4_final: true,
            refine_chroma: true,
            refine_y2: false,
            update_probabilities: true,
        },
        _ => LossySearchProfile {
            fast_mode_search: false,
            allow_i4x4: true,
            refine_i16: true,
            refine_i4_search: true,
            refine_i4_final: true,
            refine_chroma: true,
            refine_y2: true,
            update_probabilities: true,
        },
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

mod api;
mod bitstream;
mod predict;

pub use api::*;
