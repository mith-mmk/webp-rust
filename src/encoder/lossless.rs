use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::encoder::bit_writer::BitWriter;
use crate::encoder::container::{wrap_still_webp, StillImageChunk};
use crate::encoder::huffman::{compress_huffman_tree, HuffmanCode};
use crate::encoder::EncoderError;
use crate::ImageBuffer;

const MAX_WEBP_DIMENSION: usize = 1 << 14;
const MAX_CACHE_BITS: usize = 11;
const MIN_LENGTH: usize = 4;
const MAX_LENGTH: usize = 4096;
const MIN_TRANSFORM_BITS: usize = 2;
const GLOBAL_CROSS_COLOR_TRANSFORM_BITS: usize = 9;
const GLOBAL_PREDICTOR_TRANSFORM_BITS: usize = 9;
const GLOBAL_PREDICTOR_MODE: u8 = 11;
const CROSS_COLOR_TRANSFORM_BITS: usize = 5;
const PREDICTOR_TRANSFORM_BITS: usize = 5;
const MAX_OPTIMIZATION_LEVEL: u8 = 9;
const DEFAULT_OPTIMIZATION_LEVEL: u8 = 6;
const NUM_PREDICTOR_MODES: u8 = 14;
const NUM_LITERAL_CODES: usize = 256;
const NUM_LENGTH_CODES: usize = 24;
const NUM_DISTANCE_CODES: usize = 40;
const NUM_CODE_LENGTH_CODES: usize = 19;
const NUM_HISTOGRAM_PARTITIONS: usize = 4;
const MIN_HUFFMAN_BITS: usize = 2;
const NUM_HUFFMAN_BITS: usize = 3;
const COLOR_CACHE_HASH_MUL: u32 = 0x1e35_a7bd;
const MATCH_HASH_BITS: usize = 15;
const MATCH_HASH_SIZE: usize = 1 << MATCH_HASH_BITS;
const MATCH_CHAIN_DEPTH_LEVEL1: usize = 4;
const MATCH_CHAIN_DEPTH_LEVEL2: usize = 8;
const MATCH_CHAIN_DEPTH_LEVEL3: usize = 16;
const MATCH_CHAIN_DEPTH_LEVEL4: usize = 32;
const MATCH_CHAIN_DEPTH_LEVEL5: usize = 64;
const MATCH_CHAIN_DEPTH_LEVEL6: usize = 128;
const MATCH_CHAIN_DEPTH_LEVEL7: usize = 192;
const MAX_FALLBACK_DISTANCE: usize = (1 << 20) - 120;
const APPROX_LITERAL_COST_BITS: isize = 32;
const APPROX_CACHE_COST_BITS: isize = 8;
const APPROX_COPY_LENGTH_SYMBOL_BITS: isize = 8;
const APPROX_COPY_DISTANCE_SYMBOL_BITS: isize = 8;
const CODE_LENGTH_CODE_ORDER: [usize; NUM_CODE_LENGTH_CODES] = [
    17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
];
const PLANE_TO_CODE_LUT: [u8; 128] = [
    96, 73, 55, 39, 23, 13, 5, 1, 255, 255, 255, 255, 255, 255, 255, 255, 101, 78, 58, 42, 26, 16,
    8, 2, 0, 3, 9, 17, 27, 43, 59, 79, 102, 86, 62, 46, 32, 20, 10, 6, 4, 7, 11, 21, 33, 47, 63,
    87, 105, 90, 70, 52, 37, 28, 18, 14, 12, 15, 19, 29, 38, 53, 71, 91, 110, 99, 82, 66, 48, 35,
    30, 24, 22, 25, 31, 36, 49, 67, 83, 100, 115, 108, 94, 76, 64, 50, 44, 40, 34, 41, 45, 51, 65,
    77, 95, 109, 118, 113, 103, 92, 80, 68, 60, 56, 54, 57, 61, 69, 81, 93, 104, 114, 119, 116,
    111, 106, 97, 88, 84, 74, 72, 75, 85, 89, 98, 107, 112, 117,
];

#[derive(Debug, Clone, Copy)]
enum Token {
    Literal(u32),
    Cache(usize),
    Copy { distance: usize, length: usize },
}

#[derive(Debug, Clone, Copy)]
struct PrefixCode {
    symbol: usize,
    extra_bits: usize,
    extra_value: usize,
}

#[derive(Debug, Clone, Copy)]
struct CrossColorTransform {
    green_to_red: i8,
    green_to_blue: i8,
    red_to_blue: i8,
}

#[derive(Debug, Clone)]
struct ColorCache {
    colors: Vec<u32>,
    hash_shift: u32,
}

#[derive(Debug, Clone)]
struct TransformPlan {
    use_subtract_green: bool,
    cross_bits: Option<usize>,
    cross_width: usize,
    cross_image: Vec<u32>,
    predictor_bits: Option<usize>,
    predictor_width: usize,
    predictor_image: Vec<u32>,
    predicted: Vec<u32>,
}

#[derive(Debug, Clone)]
struct PaletteCandidate {
    palette: Vec<u32>,
    packed_width: usize,
    packed_indices: Vec<u32>,
}

#[derive(Debug, Clone, Copy)]
struct TokenBuildOptions {
    color_cache_bits: usize,
    match_chain_depth: usize,
    use_window_offsets: bool,
    window_offset_limit: usize,
    lazy_matching: bool,
    use_traceback: bool,
    traceback_max_candidates: usize,
}

#[derive(Debug, Clone, Copy)]
enum TracebackStep {
    Literal,
    Cache { key: usize },
    Copy { distance: usize, length: usize },
}

#[derive(Debug, Clone)]
struct TracebackCostModel {
    literal: Vec<usize>,
    red: Vec<usize>,
    blue: Vec<usize>,
    alpha: Vec<usize>,
    distance: Vec<usize>,
    length_cost_intervals: Vec<(usize, usize, usize)>,
}

type HistogramSet = [Vec<u32>; 5];

#[derive(Debug, Clone)]
struct HuffmanGroupCodes {
    green: HuffmanCode,
    red: HuffmanCode,
    blue: HuffmanCode,
    alpha: HuffmanCode,
    dist: HuffmanCode,
}

#[derive(Debug, Clone)]
struct MetaHuffmanPlan {
    huffman_bits: usize,
    huffman_xsize: usize,
    assignments: Vec<usize>,
    groups: Vec<HuffmanGroupCodes>,
}

#[derive(Debug, Clone)]
struct HistogramCandidate {
    histograms: HistogramSet,
    weight: usize,
}

#[derive(Debug, Clone, Copy)]
struct LosslessSearchProfile {
    transform_search_level: u8,
    match_search_level: u8,
    entropy_search_level: u8,
    use_color_cache: bool,
    shortlist_keep: usize,
    early_stop_ratio_percent: usize,
}

/// Lossless encoder tuning knobs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LosslessEncodingOptions {
    /// Compression effort from `0` to `9`.
    ///
    /// - `0`: fastest path, raw-only search
    /// - `1..=3`: fast presets that should still beat PNG on typical images
    /// - `4..=6`: balanced presets
    /// - `7..=9`: increasingly heavy search, with `9` enabling the slowest trials
    pub optimization_level: u8,
}

impl Default for LosslessEncodingOptions {
    fn default() -> Self {
        Self {
            optimization_level: DEFAULT_OPTIMIZATION_LEVEL,
        }
    }
}

impl ColorCache {
    fn new(hash_bits: usize) -> Result<Self, EncoderError> {
        if !(1..=MAX_CACHE_BITS).contains(&hash_bits) {
            return Err(EncoderError::InvalidParam("invalid VP8L color cache size"));
        }
        let size = 1usize << hash_bits;
        Ok(Self {
            colors: vec![0; size],
            hash_shift: (32 - hash_bits) as u32,
        })
    }

    fn key(&self, argb: u32) -> usize {
        ((argb.wrapping_mul(COLOR_CACHE_HASH_MUL)) >> self.hash_shift) as usize
    }

    fn lookup(&self, argb: u32) -> Option<usize> {
        let key = self.key(argb);
        (self.colors[key] == argb).then_some(key)
    }

    fn insert(&mut self, argb: u32) {
        let key = self.key(argb);
        self.colors[key] = argb;
    }
}

fn validate_rgba(width: usize, height: usize, rgba: &[u8]) -> Result<(), EncoderError> {
    if width == 0 || height == 0 {
        return Err(EncoderError::InvalidParam(
            "image dimensions must be non-zero",
        ));
    }
    if width > MAX_WEBP_DIMENSION || height > MAX_WEBP_DIMENSION {
        return Err(EncoderError::InvalidParam(
            "image dimensions exceed VP8L limits",
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

    Ok(())
}

fn validate_options(options: &LosslessEncodingOptions) -> Result<(), EncoderError> {
    if options.optimization_level > MAX_OPTIMIZATION_LEVEL {
        return Err(EncoderError::InvalidParam(
            "lossless optimization level must be in 0..=9",
        ));
    }
    Ok(())
}

fn lossless_search_profile(optimization_level: u8) -> LosslessSearchProfile {
    match optimization_level {
        0 => LosslessSearchProfile {
            transform_search_level: 0,
            match_search_level: 0,
            entropy_search_level: 0,
            use_color_cache: false,
            shortlist_keep: 1,
            early_stop_ratio_percent: 100,
        },
        1 => LosslessSearchProfile {
            transform_search_level: 1,
            match_search_level: 1,
            entropy_search_level: 0,
            use_color_cache: false,
            shortlist_keep: 2,
            early_stop_ratio_percent: 104,
        },
        2 => LosslessSearchProfile {
            transform_search_level: 2,
            match_search_level: 2,
            entropy_search_level: 1,
            use_color_cache: true,
            shortlist_keep: 2,
            early_stop_ratio_percent: 106,
        },
        3 => LosslessSearchProfile {
            transform_search_level: 3,
            match_search_level: 2,
            entropy_search_level: 1,
            use_color_cache: true,
            shortlist_keep: 3,
            early_stop_ratio_percent: 108,
        },
        4 => LosslessSearchProfile {
            transform_search_level: 4,
            match_search_level: 3,
            entropy_search_level: 2,
            use_color_cache: true,
            shortlist_keep: 3,
            early_stop_ratio_percent: 110,
        },
        5 => LosslessSearchProfile {
            transform_search_level: 5,
            match_search_level: 4,
            entropy_search_level: 2,
            use_color_cache: true,
            shortlist_keep: 4,
            early_stop_ratio_percent: 112,
        },
        6 => LosslessSearchProfile {
            transform_search_level: 6,
            match_search_level: 4,
            entropy_search_level: 3,
            use_color_cache: true,
            shortlist_keep: 4,
            early_stop_ratio_percent: 115,
        },
        7 => LosslessSearchProfile {
            transform_search_level: 7,
            match_search_level: 5,
            entropy_search_level: 4,
            use_color_cache: true,
            shortlist_keep: 5,
            early_stop_ratio_percent: 118,
        },
        8 => LosslessSearchProfile {
            transform_search_level: 7,
            match_search_level: 6,
            entropy_search_level: 5,
            use_color_cache: true,
            shortlist_keep: 6,
            early_stop_ratio_percent: 122,
        },
        _ => LosslessSearchProfile {
            transform_search_level: 7,
            match_search_level: 7,
            entropy_search_level: 6,
            use_color_cache: true,
            shortlist_keep: 8,
            early_stop_ratio_percent: 128,
        },
    }
}

fn lossless_candidate_profiles(optimization_level: u8) -> Vec<LosslessSearchProfile> {
    match optimization_level {
        8 => vec![lossless_search_profile(7)],
        9 => vec![lossless_search_profile(7)],
        _ => vec![lossless_search_profile(optimization_level)],
    }
}

fn rgba_has_alpha(rgba: &[u8]) -> bool {
    rgba.chunks_exact(4).any(|pixel| pixel[3] != 0xff)
}

fn rgba_to_argb(rgba: &[u8]) -> Vec<u32> {
    rgba.chunks_exact(4)
        .map(|pixel| {
            ((pixel[3] as u32) << 24)
                | ((pixel[0] as u32) << 16)
                | ((pixel[1] as u32) << 8)
                | pixel[2] as u32
        })
        .collect()
}

fn apply_subtract_green_transform(argb: &[u32]) -> Vec<u32> {
    argb.iter()
        .map(|&pixel| {
            let alpha = pixel & 0xff00_0000;
            let red = (pixel >> 16) & 0xff;
            let green = (pixel >> 8) & 0xff;
            let blue = pixel & 0xff;
            let red = red.wrapping_sub(green) & 0xff;
            let blue = blue.wrapping_sub(green) & 0xff;
            alpha | (red << 16) | (green << 8) | blue
        })
        .collect()
}

fn color_transform_delta(transform: i8, color: u8) -> i32 {
    ((transform as i32) * (color as i8 as i32)) >> 5
}

fn estimate_transform_coefficient(pairs: &[(i32, i32)]) -> i8 {
    let mut numerator = 0i64;
    let mut denominator = 0i64;
    for &(value, predictor) in pairs {
        numerator += (value as i64) * (predictor as i64);
        denominator += (predictor as i64) * (predictor as i64);
    }
    if denominator == 0 {
        return 0;
    }
    let coefficient = (32 * numerator) / denominator;
    coefficient.clamp(-128, 127) as i8
}

fn estimate_cross_color_transform_region(
    width: usize,
    height: usize,
    argb: &[u32],
    tile_x: usize,
    tile_y: usize,
    bits: usize,
) -> CrossColorTransform {
    let start_x = tile_x << bits;
    let start_y = tile_y << bits;
    let end_x = ((tile_x + 1) << bits).min(width);
    let end_y = ((tile_y + 1) << bits).min(height);
    let capacity = (end_x - start_x) * (end_y - start_y);

    let mut red_pairs = Vec::with_capacity(capacity);
    let mut blue_green_pairs = Vec::with_capacity(capacity);
    for y in start_y..end_y {
        let row = &argb[y * width + start_x..y * width + end_x];
        for &pixel in row {
            let red = (((pixel >> 16) & 0xff) as u8) as i8 as i32;
            let green = (((pixel >> 8) & 0xff) as u8) as i8 as i32;
            let blue = ((pixel & 0xff) as u8) as i8 as i32;
            red_pairs.push((red, green));
            blue_green_pairs.push((blue, green));
        }
    }

    let green_to_red = estimate_transform_coefficient(&red_pairs);
    let green_to_blue = estimate_transform_coefficient(&blue_green_pairs);

    let mut blue_red_pairs = Vec::with_capacity(capacity);
    for y in start_y..end_y {
        let row = &argb[y * width + start_x..y * width + end_x];
        for &pixel in row {
            let red = ((pixel >> 16) & 0xff) as u8;
            let green = ((pixel >> 8) & 0xff) as u8;
            let blue = (pixel & 0xff) as u8;
            let transformed_blue =
                ((blue as i32 - color_transform_delta(green_to_blue, green)) & 0xff) as u8;
            blue_red_pairs.push(((transformed_blue as i8) as i32, (red as i8) as i32));
        }
    }
    let red_to_blue = estimate_transform_coefficient(&blue_red_pairs);

    CrossColorTransform {
        green_to_red,
        green_to_blue,
        red_to_blue,
    }
}

fn estimate_cross_color_transform(argb: &[u32]) -> CrossColorTransform {
    let mut red_pairs = Vec::with_capacity(argb.len());
    let mut blue_green_pairs = Vec::with_capacity(argb.len());
    for &pixel in argb {
        let red = (((pixel >> 16) & 0xff) as u8) as i8 as i32;
        let green = (((pixel >> 8) & 0xff) as u8) as i8 as i32;
        let blue = ((pixel & 0xff) as u8) as i8 as i32;
        red_pairs.push((red, green));
        blue_green_pairs.push((blue, green));
    }

    let green_to_red = estimate_transform_coefficient(&red_pairs);
    let green_to_blue = estimate_transform_coefficient(&blue_green_pairs);

    let mut blue_red_pairs = Vec::with_capacity(argb.len());
    for &pixel in argb {
        let red = ((pixel >> 16) & 0xff) as u8;
        let green = ((pixel >> 8) & 0xff) as u8;
        let blue = (pixel & 0xff) as u8;
        let transformed_blue =
            ((blue as i32 - color_transform_delta(green_to_blue, green)) & 0xff) as u8;
        blue_red_pairs.push(((transformed_blue as i8) as i32, (red as i8) as i32));
    }
    let red_to_blue = estimate_transform_coefficient(&blue_red_pairs);

    CrossColorTransform {
        green_to_red,
        green_to_blue,
        red_to_blue,
    }
}

fn pack_cross_color_transform(transform: CrossColorTransform) -> u32 {
    ((transform.red_to_blue as u8 as u32) << 16)
        | ((transform.green_to_blue as u8 as u32) << 8)
        | (transform.green_to_red as u8 as u32)
}

fn apply_cross_color_transform(
    width: usize,
    height: usize,
    argb: &[u32],
    bits: usize,
    transforms: &[CrossColorTransform],
) -> Vec<u32> {
    let tiles_per_row = subsample_size(width, bits);
    let mut output = Vec::with_capacity(argb.len());
    for y in 0..height {
        for x in 0..width {
            let transform = transforms[(y >> bits) * tiles_per_row + (x >> bits)];
            let pixel = argb[y * width + x];
            let alpha = pixel & 0xff00_0000;
            let red = ((pixel >> 16) & 0xff) as u8;
            let green = ((pixel >> 8) & 0xff) as u8;
            let blue = (pixel & 0xff) as u8;

            let transformed_red =
                ((red as i32 - color_transform_delta(transform.green_to_red, green)) & 0xff) as u32;
            let mut transformed_blue = ((blue as i32
                - color_transform_delta(transform.green_to_blue, green))
                & 0xff) as i32;
            transformed_blue =
                (transformed_blue - color_transform_delta(transform.red_to_blue, red)) & 0xff;

            output.push(
                alpha | (transformed_red << 16) | ((green as u32) << 8) | transformed_blue as u32,
            );
        }
    }
    output
}

fn average2(a: u32, b: u32) -> u32 {
    (((a ^ b) & 0xfefe_fefeu32) >> 1) + (a & b)
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

fn predictor(mode: u8, left: u32, top: u32, top_left: u32, top_right: u32) -> u32 {
    match mode {
        0 => 0xff00_0000,
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
        _ => 0xff00_0000,
    }
}

fn predictor_for_mode(argb: &[u32], width: usize, x: usize, y: usize, mode: u8) -> u32 {
    if y == 0 {
        if x == 0 {
            0xff00_0000
        } else {
            argb[y * width + x - 1]
        }
    } else if x == 0 {
        argb[(y - 1) * width]
    } else {
        let left = argb[y * width + x - 1];
        let top = argb[(y - 1) * width + x];
        let top_left = argb[(y - 1) * width + x - 1];
        let top_right = if x + 1 < width {
            argb[(y - 1) * width + x + 1]
        } else {
            argb[y * width]
        };
        predictor(mode, left, top, top_left, top_right)
    }
}

fn sub_pixels(a: u32, b: u32) -> u32 {
    let alpha = (((a >> 24) as u8).wrapping_sub((b >> 24) as u8)) as u32;
    let red = ((((a >> 16) & 0xff) as u8).wrapping_sub(((b >> 16) & 0xff) as u8)) as u32;
    let green = ((((a >> 8) & 0xff) as u8).wrapping_sub(((b >> 8) & 0xff) as u8)) as u32;
    let blue = (((a & 0xff) as u8).wrapping_sub((b & 0xff) as u8)) as u32;
    (alpha << 24) | (red << 16) | (green << 8) | blue
}

fn wrapped_channel_error(actual: u32, predicted: u32, shift: u32) -> u32 {
    let actual = ((actual >> shift) & 0xff) as i32;
    let predicted = ((predicted >> shift) & 0xff) as i32;
    let delta = (actual - predicted).unsigned_abs();
    delta.min(256 - delta)
}

fn predictor_error(actual: u32, predicted: u32) -> u32 {
    wrapped_channel_error(actual, predicted, 24)
        + wrapped_channel_error(actual, predicted, 16)
        + wrapped_channel_error(actual, predicted, 8)
        + wrapped_channel_error(actual, predicted, 0)
}

fn choose_predictor_mode(
    width: usize,
    height: usize,
    argb: &[u32],
    tile_x: usize,
    tile_y: usize,
    bits: usize,
) -> u8 {
    let start_x = tile_x << bits;
    let start_y = tile_y << bits;
    let end_x = ((tile_x + 1) << bits).min(width);
    let end_y = ((tile_y + 1) << bits).min(height);

    let mut best_mode = 11u8;
    let mut best_cost = u64::MAX;
    for mode in 0..NUM_PREDICTOR_MODES {
        let mut cost = 0u64;
        for y in start_y..end_y {
            for x in start_x..end_x {
                let pred = predictor_for_mode(argb, width, x, y, mode);
                cost += predictor_error(argb[y * width + x], pred) as u64;
            }
        }
        if cost < best_cost {
            best_cost = cost;
            best_mode = mode;
        }
    }
    best_mode
}

fn apply_predictor_transform(
    width: usize,
    height: usize,
    argb: &[u32],
    bits: usize,
    modes: &[u8],
) -> Vec<u32> {
    let tiles_per_row = subsample_size(width, bits);
    let mut residuals = vec![0u32; argb.len()];
    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            let mode = modes[(y >> bits) * tiles_per_row + (x >> bits)];
            let pred = predictor_for_mode(argb, width, x, y, mode);
            residuals[index] = sub_pixels(argb[index], pred);
        }
    }
    residuals
}

fn subsample_size(size: usize, bits: usize) -> usize {
    (size + (1usize << bits) - 1) >> bits
}

fn make_predictor_transform_image(
    width: usize,
    height: usize,
    argb: &[u32],
) -> (usize, usize, Vec<u8>, Vec<u32>) {
    let xsize = subsample_size(width, PREDICTOR_TRANSFORM_BITS);
    let ysize = subsample_size(height, PREDICTOR_TRANSFORM_BITS);
    let mut modes = Vec::with_capacity(xsize * ysize);
    let mut image = Vec::with_capacity(xsize * ysize);
    for tile_y in 0..ysize {
        for tile_x in 0..xsize {
            let mode = choose_predictor_mode(
                width,
                height,
                argb,
                tile_x,
                tile_y,
                PREDICTOR_TRANSFORM_BITS,
            );
            modes.push(mode);
            image.push((mode as u32) << 8);
        }
    }
    (xsize, ysize, modes, image)
}

fn make_uniform_predictor_transform_image(
    width: usize,
    height: usize,
    bits: usize,
    mode: u8,
) -> (usize, usize, Vec<u8>, Vec<u32>) {
    let xsize = subsample_size(width, bits);
    let ysize = subsample_size(height, bits);
    let pixel = (mode as u32) << 8;
    (
        xsize,
        ysize,
        vec![mode; xsize * ysize],
        vec![pixel; xsize * ysize],
    )
}

fn make_cross_color_transform_image(
    width: usize,
    height: usize,
    argb: &[u32],
) -> (usize, usize, Vec<CrossColorTransform>, Vec<u32>) {
    let xsize = subsample_size(width, CROSS_COLOR_TRANSFORM_BITS);
    let ysize = subsample_size(height, CROSS_COLOR_TRANSFORM_BITS);
    let mut transforms = Vec::with_capacity(xsize * ysize);
    let mut image = Vec::with_capacity(xsize * ysize);
    for tile_y in 0..ysize {
        for tile_x in 0..xsize {
            let transform = estimate_cross_color_transform_region(
                width,
                height,
                argb,
                tile_x,
                tile_y,
                CROSS_COLOR_TRANSFORM_BITS,
            );
            transforms.push(transform);
            image.push(pack_cross_color_transform(transform));
        }
    }
    (xsize, ysize, transforms, image)
}

fn make_uniform_cross_color_transform_image(
    width: usize,
    height: usize,
    bits: usize,
    transform: CrossColorTransform,
) -> (usize, usize, Vec<CrossColorTransform>, Vec<u32>) {
    let xsize = subsample_size(width, bits);
    let ysize = subsample_size(height, bits);
    let pixel = pack_cross_color_transform(transform);
    (
        xsize,
        ysize,
        vec![transform; xsize * ysize],
        vec![pixel; xsize * ysize],
    )
}

fn palette_xbits(palette_size: usize) -> usize {
    if palette_size <= 2 {
        3
    } else if palette_size <= 4 {
        2
    } else if palette_size <= 16 {
        1
    } else {
        0
    }
}

fn collect_palette(argb: &[u32]) -> Option<Vec<u32>> {
    let mut unique = HashSet::with_capacity(256);
    for &pixel in argb {
        unique.insert(pixel);
        if unique.len() > 256 {
            return None;
        }
    }
    let mut palette = unique.into_iter().collect::<Vec<_>>();
    palette.sort_unstable();
    Some(palette)
}

fn build_palette_candidate(
    width: usize,
    height: usize,
    argb: &[u32],
) -> Result<Option<PaletteCandidate>, EncoderError> {
    let palette = match collect_palette(argb) {
        Some(palette) if !palette.is_empty() => palette,
        _ => return Ok(None),
    };
    let xbits = palette_xbits(palette.len());
    let packed_width = subsample_size(width, xbits);
    let bits_per_pixel = 8 >> xbits;
    let pixels_per_byte = 1usize << xbits;
    let index_by_color = palette
        .iter()
        .enumerate()
        .map(|(index, &color)| (color, index as u8))
        .collect::<HashMap<_, _>>();
    let mut packed_indices = vec![0u32; packed_width * height];

    for y in 0..height {
        for packed_x in 0..packed_width {
            let mut packed = 0u32;
            for slot in 0..pixels_per_byte {
                let x = packed_x * pixels_per_byte + slot;
                if x >= width {
                    break;
                }
                let index = *index_by_color
                    .get(&argb[y * width + x])
                    .ok_or(EncoderError::Bitstream("palette index lookup failed"))?;
                packed |= (index as u32) << (slot * bits_per_pixel);
            }
            packed_indices[y * packed_width + packed_x] = packed << 8;
        }
    }

    Ok(Some(PaletteCandidate {
        palette,
        packed_width,
        packed_indices,
    }))
}

fn build_global_cross_plan(
    width: usize,
    height: usize,
    input: &[u32],
    use_subtract_green: bool,
) -> TransformPlan {
    let cross_transform = estimate_cross_color_transform(input);
    let (cross_width, _cross_height, cross_transforms, cross_image) =
        make_uniform_cross_color_transform_image(
            width,
            height,
            GLOBAL_CROSS_COLOR_TRANSFORM_BITS,
            cross_transform,
        );
    let cross_colored = apply_cross_color_transform(
        width,
        height,
        input,
        GLOBAL_CROSS_COLOR_TRANSFORM_BITS,
        &cross_transforms,
    );

    TransformPlan {
        use_subtract_green,
        cross_bits: Some(GLOBAL_CROSS_COLOR_TRANSFORM_BITS),
        cross_width,
        cross_image,
        predictor_bits: None,
        predictor_width: 0,
        predictor_image: Vec::new(),
        predicted: cross_colored,
    }
}

fn build_raw_plan(argb: &[u32]) -> TransformPlan {
    TransformPlan {
        use_subtract_green: false,
        cross_bits: None,
        cross_width: 0,
        cross_image: Vec::new(),
        predictor_bits: None,
        predictor_width: 0,
        predictor_image: Vec::new(),
        predicted: argb.to_vec(),
    }
}

fn build_subtract_green_plan(subtract_green: &[u32]) -> TransformPlan {
    TransformPlan {
        use_subtract_green: true,
        cross_bits: None,
        cross_width: 0,
        cross_image: Vec::new(),
        predictor_bits: None,
        predictor_width: 0,
        predictor_image: Vec::new(),
        predicted: subtract_green.to_vec(),
    }
}

fn build_global_predictor_plan(
    width: usize,
    height: usize,
    input: &[u32],
    use_subtract_green: bool,
) -> TransformPlan {
    let (predictor_width, _predictor_height, predictor_modes, predictor_image) =
        make_uniform_predictor_transform_image(
            width,
            height,
            GLOBAL_PREDICTOR_TRANSFORM_BITS,
            GLOBAL_PREDICTOR_MODE,
        );
    let predicted = apply_predictor_transform(
        width,
        height,
        input,
        GLOBAL_PREDICTOR_TRANSFORM_BITS,
        &predictor_modes,
    );

    TransformPlan {
        use_subtract_green,
        cross_bits: None,
        cross_width: 0,
        cross_image: Vec::new(),
        predictor_bits: Some(GLOBAL_PREDICTOR_TRANSFORM_BITS),
        predictor_width,
        predictor_image,
        predicted,
    }
}

fn build_global_transform_plan(
    width: usize,
    height: usize,
    input: &[u32],
    use_subtract_green: bool,
) -> TransformPlan {
    let cross_plan = build_global_cross_plan(width, height, input, use_subtract_green);
    let cross_colored = cross_plan.predicted.clone();
    let (predictor_width, _predictor_height, predictor_modes, predictor_image) =
        make_uniform_predictor_transform_image(
            width,
            height,
            GLOBAL_PREDICTOR_TRANSFORM_BITS,
            GLOBAL_PREDICTOR_MODE,
        );
    let predicted = apply_predictor_transform(
        width,
        height,
        &cross_colored,
        GLOBAL_PREDICTOR_TRANSFORM_BITS,
        &predictor_modes,
    );

    TransformPlan {
        use_subtract_green,
        cross_bits: cross_plan.cross_bits,
        cross_width: cross_plan.cross_width,
        cross_image: cross_plan.cross_image,
        predictor_bits: Some(GLOBAL_PREDICTOR_TRANSFORM_BITS),
        predictor_width,
        predictor_image,
        predicted,
    }
}

fn build_tiled_cross_plan(
    width: usize,
    height: usize,
    input: &[u32],
    use_subtract_green: bool,
) -> TransformPlan {
    let (cross_width, _cross_height, cross_transforms, cross_image) =
        make_cross_color_transform_image(width, height, input);
    let cross_colored = apply_cross_color_transform(
        width,
        height,
        input,
        CROSS_COLOR_TRANSFORM_BITS,
        &cross_transforms,
    );

    TransformPlan {
        use_subtract_green,
        cross_bits: Some(CROSS_COLOR_TRANSFORM_BITS),
        cross_width,
        cross_image,
        predictor_bits: None,
        predictor_width: 0,
        predictor_image: Vec::new(),
        predicted: cross_colored,
    }
}

fn build_tiled_predictor_plan(
    width: usize,
    height: usize,
    input: &[u32],
    use_subtract_green: bool,
) -> TransformPlan {
    let (predictor_width, _predictor_height, predictor_modes, predictor_image) =
        make_predictor_transform_image(width, height, input);
    let predicted = apply_predictor_transform(
        width,
        height,
        input,
        PREDICTOR_TRANSFORM_BITS,
        &predictor_modes,
    );

    TransformPlan {
        use_subtract_green,
        cross_bits: None,
        cross_width: 0,
        cross_image: Vec::new(),
        predictor_bits: Some(PREDICTOR_TRANSFORM_BITS),
        predictor_width,
        predictor_image,
        predicted,
    }
}

fn build_tiled_transform_plan(
    width: usize,
    height: usize,
    input: &[u32],
    use_subtract_green: bool,
) -> TransformPlan {
    let cross_plan = build_tiled_cross_plan(width, height, input, use_subtract_green);
    let cross_colored = cross_plan.predicted.clone();
    let (predictor_width, _predictor_height, predictor_modes, predictor_image) =
        make_predictor_transform_image(width, height, &cross_colored);
    let predicted = apply_predictor_transform(
        width,
        height,
        &cross_colored,
        PREDICTOR_TRANSFORM_BITS,
        &predictor_modes,
    );

    TransformPlan {
        use_subtract_green,
        cross_bits: cross_plan.cross_bits,
        cross_width: cross_plan.cross_width,
        cross_image: cross_plan.cross_image,
        predictor_bits: Some(PREDICTOR_TRANSFORM_BITS),
        predictor_width,
        predictor_image,
        predicted,
    }
}

fn quick_token_build_options(profile: &LosslessSearchProfile) -> TokenBuildOptions {
    token_build_options(profile.match_search_level.min(2), 0)
}

fn estimate_token_stream_cost_bytes(
    width: usize,
    argb: &[u32],
    options: TokenBuildOptions,
) -> Result<usize, EncoderError> {
    let tokens = build_tokens(width, argb, options)?;
    let histograms = build_histograms(&tokens, width, 0)?;
    let group = build_group_codes(&histograms)?;
    let extra_bits = tokens
        .iter()
        .map(|&token| match token {
            Token::Literal(_) | Token::Cache(_) => 0usize,
            Token::Copy { distance, length } => {
                let plane_code = distance_to_plane_code(width, distance);
                prefix_extra_bit_count(length) + prefix_extra_bit_count(plane_code)
            }
        })
        .sum::<usize>();
    let total_bits = histogram_cost(&histograms, &group) + extra_bits + tokens.len();
    Ok(total_bits.div_ceil(8))
}

fn estimate_transform_plan_score(
    width: usize,
    plan: &TransformPlan,
    profile: &LosslessSearchProfile,
) -> Result<usize, EncoderError> {
    let transform_options = TokenBuildOptions {
        color_cache_bits: 0,
        match_chain_depth: 0,
        use_window_offsets: false,
        window_offset_limit: 0,
        lazy_matching: false,
        use_traceback: false,
        traceback_max_candidates: 0,
    };
    let mut score = estimate_token_stream_cost_bytes(
        width,
        &plan.predicted,
        quick_token_build_options(profile),
    )?;
    if plan.use_subtract_green {
        score += 1;
    }
    if !plan.cross_image.is_empty() {
        score += 2 + estimate_token_stream_cost_bytes(
            plan.cross_width,
            &plan.cross_image,
            transform_options,
        )?;
    }
    if !plan.predictor_image.is_empty() {
        score += 2 + estimate_token_stream_cost_bytes(
            plan.predictor_width,
            &plan.predictor_image,
            transform_options,
        )?;
    }
    Ok(score)
}

fn collect_transform_plans(
    width: usize,
    height: usize,
    argb: &[u32],
    subtract_green: &[u32],
    profile: &LosslessSearchProfile,
) -> Vec<TransformPlan> {
    let subtract_is_distinct = subtract_green != argb;
    let mut plans = vec![build_raw_plan(argb)];

    if subtract_is_distinct && profile.transform_search_level >= 1 {
        plans.push(build_subtract_green_plan(subtract_green));
    }
    if profile.transform_search_level >= 2 {
        plans.push(build_global_cross_plan(width, height, argb, false));
        plans.push(build_global_predictor_plan(width, height, argb, false));
    }
    if subtract_is_distinct && profile.transform_search_level >= 3 {
        plans.push(build_global_cross_plan(width, height, subtract_green, true));
        plans.push(build_global_predictor_plan(
            width,
            height,
            subtract_green,
            true,
        ));
    }
    if profile.transform_search_level >= 4 {
        plans.push(build_global_transform_plan(width, height, argb, false));
        if subtract_is_distinct {
            plans.push(build_global_transform_plan(
                width,
                height,
                subtract_green,
                true,
            ));
        }
    }
    if profile.transform_search_level >= 5 {
        plans.push(build_tiled_cross_plan(width, height, argb, false));
        plans.push(build_tiled_predictor_plan(width, height, argb, false));
    }
    if subtract_is_distinct && profile.transform_search_level >= 6 {
        plans.push(build_tiled_cross_plan(width, height, subtract_green, true));
        plans.push(build_tiled_predictor_plan(
            width,
            height,
            subtract_green,
            true,
        ));
    }
    if profile.transform_search_level >= 7 {
        plans.push(build_tiled_transform_plan(width, height, argb, false));
        if subtract_is_distinct {
            plans.push(build_tiled_transform_plan(
                width,
                height,
                subtract_green,
                true,
            ));
        }
    }

    plans
}

fn shortlist_transform_plans(
    width: usize,
    plans: Vec<TransformPlan>,
    profile: &LosslessSearchProfile,
) -> Result<Vec<(usize, TransformPlan)>, EncoderError> {
    let mut ranked = Vec::with_capacity(plans.len());
    for plan in plans {
        ranked.push((estimate_transform_plan_score(width, &plan, profile)?, plan));
    }
    ranked.sort_by_key(|(score, _)| *score);
    ranked.truncate(profile.shortlist_keep.min(ranked.len()));
    Ok(ranked)
}

fn should_stop_transform_search(
    best_len: usize,
    next_estimate: usize,
    profile: &LosslessSearchProfile,
) -> bool {
    profile.early_stop_ratio_percent != usize::MAX
        && next_estimate.saturating_mul(100)
            >= best_len.saturating_mul(profile.early_stop_ratio_percent)
}

fn encode_transform_plan_to_vp8l(
    width: usize,
    height: usize,
    rgba: &[u8],
    plan: &TransformPlan,
    profile: &LosslessSearchProfile,
) -> Result<Vec<u8>, EncoderError> {
    let no_cache_options = token_build_options(profile.match_search_level, 0);
    let mut best = encode_transform_plan_to_vp8l_with_cache(
        width,
        height,
        rgba,
        plan,
        no_cache_options,
        profile.entropy_search_level,
    )?;
    if profile.use_color_cache && plan.predicted.len() >= 64 {
        let base_tokens = build_tokens(width, &plan.predicted, no_cache_options)?;
        let best_cache_bits =
            select_best_color_cache_bits(width, height, &plan.predicted, &base_tokens, profile)?;
        let with_cache = encode_transform_plan_to_vp8l_with_cache(
            width,
            height,
            rgba,
            plan,
            token_build_options(profile.match_search_level, best_cache_bits),
            profile.entropy_search_level,
        )?;
        if best_cache_bits > 0 && with_cache.len() < best.len() {
            best = with_cache;
        }
    }
    Ok(best)
}

fn encode_transform_plan_to_vp8l_with_cache(
    width: usize,
    height: usize,
    rgba: &[u8],
    plan: &TransformPlan,
    token_options: TokenBuildOptions,
    entropy_search_level: u8,
) -> Result<Vec<u8>, EncoderError> {
    let transform_options = TokenBuildOptions {
        color_cache_bits: 0,
        match_chain_depth: 0,
        use_window_offsets: false,
        window_offset_limit: 0,
        lazy_matching: false,
        use_traceback: false,
        traceback_max_candidates: 0,
    };
    let mut bw = BitWriter::default();
    bw.put_bits((width - 1) as u32, 14)?;
    bw.put_bits((height - 1) as u32, 14)?;
    bw.put_bits(rgba_has_alpha(rgba) as u32, 1)?;
    bw.put_bits(0, 3)?;

    if plan.use_subtract_green {
        bw.put_bits(1, 1)?;
        bw.put_bits(2, 2)?;
    }
    if let Some(cross_bits) = plan.cross_bits {
        bw.put_bits(1, 1)?;
        bw.put_bits(1, 2)?;
        bw.put_bits((cross_bits - MIN_TRANSFORM_BITS) as u32, 3)?;
        write_image_stream(
            &mut bw,
            plan.cross_width,
            &plan.cross_image,
            false,
            0,
            transform_options,
        )?;
    }
    if let Some(predictor_bits) = plan.predictor_bits {
        bw.put_bits(1, 1)?;
        bw.put_bits(0, 2)?;
        bw.put_bits((predictor_bits - MIN_TRANSFORM_BITS) as u32, 3)?;
        write_image_stream(
            &mut bw,
            plan.predictor_width,
            &plan.predictor_image,
            false,
            0,
            transform_options,
        )?;
    }
    bw.put_bits(0, 1)?;
    write_image_stream(
        &mut bw,
        width,
        &plan.predicted,
        true,
        entropy_search_level,
        token_options,
    )?;

    let bitstream = bw.into_bytes();
    let mut vp8l = Vec::with_capacity(1 + bitstream.len());
    vp8l.push(0x2f);
    vp8l.extend_from_slice(&bitstream);
    Ok(vp8l)
}

fn encode_palette_candidate_to_vp8l(
    width: usize,
    height: usize,
    rgba: &[u8],
    candidate: &PaletteCandidate,
    profile: &LosslessSearchProfile,
) -> Result<Vec<u8>, EncoderError> {
    let transform_options = TokenBuildOptions {
        color_cache_bits: 0,
        match_chain_depth: 0,
        use_window_offsets: false,
        window_offset_limit: 0,
        lazy_matching: false,
        use_traceback: false,
        traceback_max_candidates: 0,
    };
    let no_cache_options = token_build_options(profile.match_search_level, 0);
    let mut token_options = no_cache_options;
    if profile.use_color_cache && candidate.packed_indices.len() >= 64 {
        let base_tokens = build_tokens(
            candidate.packed_width,
            &candidate.packed_indices,
            no_cache_options,
        )?;
        let best_cache_bits = select_best_color_cache_bits(
            candidate.packed_width,
            height,
            &candidate.packed_indices,
            &base_tokens,
            profile,
        )?;
        token_options = token_build_options(profile.match_search_level, best_cache_bits);
    }

    let mut palette_image = Vec::with_capacity(candidate.palette.len());
    for (index, &color) in candidate.palette.iter().enumerate() {
        if index == 0 {
            palette_image.push(color);
        } else {
            palette_image.push(sub_pixels(color, candidate.palette[index - 1]));
        }
    }

    let mut bw = BitWriter::default();
    bw.put_bits((width - 1) as u32, 14)?;
    bw.put_bits((height - 1) as u32, 14)?;
    bw.put_bits(rgba_has_alpha(rgba) as u32, 1)?;
    bw.put_bits(0, 3)?;

    bw.put_bits(1, 1)?;
    bw.put_bits(3, 2)?;
    bw.put_bits((candidate.palette.len() - 1) as u32, 8)?;
    write_image_stream(
        &mut bw,
        candidate.palette.len(),
        &palette_image,
        false,
        0,
        transform_options,
    )?;

    bw.put_bits(0, 1)?;
    write_image_stream(
        &mut bw,
        candidate.packed_width,
        &candidate.packed_indices,
        true,
        profile.entropy_search_level,
        token_options,
    )?;

    let bitstream = bw.into_bytes();
    let mut vp8l = Vec::with_capacity(1 + bitstream.len());
    vp8l.push(0x2f);
    vp8l.extend_from_slice(&bitstream);
    Ok(vp8l)
}

fn find_match_length(argb: &[u32], first: usize, second: usize, max_len: usize) -> usize {
    let mut len = 0usize;
    while len < max_len && argb[first + len] == argb[second + len] {
        len += 1;
    }
    len
}

fn token_build_options(match_search_level: u8, color_cache_bits: usize) -> TokenBuildOptions {
    let (match_chain_depth, use_window_offsets, window_offset_limit, lazy_matching) =
        match match_search_level {
            0 => (0, false, 0, false),
            1 => (MATCH_CHAIN_DEPTH_LEVEL1, false, 0, false),
            2 => (MATCH_CHAIN_DEPTH_LEVEL2, true, 16, false),
            3 => (MATCH_CHAIN_DEPTH_LEVEL3, true, 32, false),
            4 => (MATCH_CHAIN_DEPTH_LEVEL4, true, 64, true),
            5 => (MATCH_CHAIN_DEPTH_LEVEL5, true, 96, true),
            6 => (MATCH_CHAIN_DEPTH_LEVEL6, true, 128, true),
            _ => (MATCH_CHAIN_DEPTH_LEVEL7, true, 160, true),
        };
    let (use_traceback, traceback_max_candidates) = match match_search_level {
        0..=4 => (false, 0),
        5 => (true, 4),
        6 => (true, 6),
        _ => (true, 8),
    };
    TokenBuildOptions {
        color_cache_bits,
        match_chain_depth,
        use_window_offsets,
        window_offset_limit,
        lazy_matching,
        use_traceback,
        traceback_max_candidates,
    }
}

fn max_color_cache_bits_for_profile(profile: &LosslessSearchProfile) -> usize {
    if !profile.use_color_cache {
        return 0;
    }
    match profile.entropy_search_level {
        0 => 0,
        1 => 7,
        2 => 8,
        3 => 9,
        4 => 10,
        _ => MAX_CACHE_BITS,
    }
}

fn shortlist_color_cache_candidates_for_profile(profile: &LosslessSearchProfile) -> usize {
    match profile.entropy_search_level {
        0 | 1 => 1,
        2 | 3 => 2,
        _ => 3,
    }
}

fn meta_huffman_candidates(
    entropy_search_level: u8,
    width: usize,
    height: usize,
) -> &'static [(usize, usize)] {
    match entropy_search_level {
        0 => &[],
        1 => &[(5usize, 4usize)],
        2 => &[(6usize, 2usize), (5usize, 4usize)],
        3 => &[(6usize, 2usize), (5usize, 4usize), (4usize, 4usize)],
        4 if width * height >= 512 * 512 => &[
            (6usize, 2usize),
            (5usize, 4usize),
            (5usize, 6usize),
            (4usize, 4usize),
        ],
        4 => &[
            (6usize, 2usize),
            (5usize, 4usize),
            (5usize, 6usize),
            (4usize, 4usize),
            (4usize, 6usize),
        ],
        5 if width * height >= 512 * 512 => &[
            (6usize, 2usize),
            (5usize, 4usize),
            (5usize, 6usize),
            (4usize, 4usize),
            (4usize, 6usize),
        ],
        5 => &[
            (6usize, 2usize),
            (5usize, 4usize),
            (5usize, 6usize),
            (4usize, 4usize),
            (4usize, 6usize),
            (4usize, 8usize),
        ],
        _ if width * height >= 512 * 512 => &[
            (6usize, 2usize),
            (5usize, 4usize),
            (5usize, 6usize),
            (4usize, 4usize),
            (4usize, 6usize),
            (4usize, 8usize),
        ],
        _ => &[
            (6usize, 2usize),
            (5usize, 4usize),
            (5usize, 6usize),
            (4usize, 4usize),
            (4usize, 6usize),
            (4usize, 8usize),
            (3usize, 8usize),
        ],
    }
}

fn suggested_max_color_cache_bits(argb: &[u32], max_cache_bits: usize) -> usize {
    if max_cache_bits == 0 {
        return 0;
    }

    let unique_limit = 1usize << max_cache_bits;
    let mut unique = HashSet::with_capacity(unique_limit.min(argb.len()));
    for &pixel in argb {
        unique.insert(pixel);
        if unique.len() > unique_limit {
            return max_cache_bits;
        }
    }

    if unique.len() <= 1 {
        return 0;
    }
    let mut bits = 0usize;
    let mut capacity = 1usize;
    while capacity < unique.len() && bits < max_cache_bits {
        bits += 1;
        capacity <<= 1;
    }
    bits.min(max_cache_bits)
}

fn build_window_offsets(width: usize, max_plane_codes: usize) -> Vec<usize> {
    if max_plane_codes == 0 {
        return Vec::new();
    };
    let radius = if max_plane_codes <= 32 {
        6isize
    } else {
        12isize
    };
    let mut by_plane_code = vec![0usize; max_plane_codes];
    for y in 0..=radius {
        for x in -radius..=radius {
            let offset = y as isize * width as isize + x;
            if offset <= 0 {
                continue;
            }
            let offset = offset as usize;
            let plane_code = distance_to_plane_code(width, offset).saturating_sub(1);
            if plane_code < max_plane_codes && by_plane_code[plane_code] == 0 {
                by_plane_code[plane_code] = offset;
            }
        }
    }
    by_plane_code
        .into_iter()
        .filter(|&offset| offset != 0)
        .collect()
}

fn min_match_length_for_distance(width: usize, distance: usize) -> usize {
    if distance == 1 || distance == width {
        return MIN_LENGTH;
    }
    let plane_code = distance_to_plane_code(width, distance);
    if plane_code <= 32 {
        MIN_LENGTH
    } else if plane_code <= 80 {
        MIN_LENGTH + 1
    } else if plane_code <= 512 {
        MIN_LENGTH + 2
    } else {
        MIN_LENGTH + 3
    }
}

fn prefix_extra_bit_count(value: usize) -> usize {
    if value <= 4 {
        0
    } else {
        let value = value - 1;
        let highest_bit = usize::BITS as usize - 1 - value.leading_zeros() as usize;
        highest_bit - 1
    }
}

fn copy_cost_bits(width: usize, distance: usize, length: usize) -> isize {
    let plane_code = distance_to_plane_code(width, distance);
    APPROX_COPY_LENGTH_SYMBOL_BITS
        + prefix_extra_bit_count(length) as isize
        + APPROX_COPY_DISTANCE_SYMBOL_BITS
        + prefix_extra_bit_count(plane_code) as isize
}

fn match_gain_bits(width: usize, distance: usize, length: usize) -> isize {
    APPROX_LITERAL_COST_BITS * length as isize - copy_cost_bits(width, distance, length)
}

fn consider_match(
    width: usize,
    best_match: &mut Option<(usize, usize)>,
    distance: usize,
    length: usize,
) {
    if length < min_match_length_for_distance(width, distance) {
        return;
    }

    let candidate_score = match_gain_bits(width, distance, length);
    if best_match
        .map(|(best_distance, best_length)| {
            let best_score = match_gain_bits(width, best_distance, best_length);
            candidate_score > best_score
                || (candidate_score == best_score
                    && (length > best_length
                        || (length == best_length && distance < best_distance)))
        })
        .unwrap_or(true)
    {
        *best_match = Some((distance, length));
    }
}

fn preview_update_match_chain(
    argb: &[u32],
    index: usize,
    heads: &mut [usize],
    prev: &mut [usize],
) -> Option<(usize, usize, usize)> {
    if index + MIN_LENGTH > argb.len() {
        return None;
    }
    let hash = hash_match_pixels(argb, index);
    let old_prev = prev[index];
    let old_head = heads[hash];
    update_match_chain(argb, index, heads, prev);
    Some((hash, old_prev, old_head))
}

fn restore_previewed_match_chain(
    index: usize,
    preview: Option<(usize, usize, usize)>,
    heads: &mut [usize],
    prev: &mut [usize],
) {
    if let Some((hash, old_prev, old_head)) = preview {
        prev[index] = old_prev;
        heads[hash] = old_head;
    }
}

fn hash_match_pixels(argb: &[u32], index: usize) -> usize {
    let a = argb[index];
    let b = argb[index + 1].rotate_left(7);
    let c = argb[index + 2].rotate_left(13);
    let d = argb[index + 3].rotate_left(21);
    let hash = a ^ b ^ c ^ d.wrapping_mul(COLOR_CACHE_HASH_MUL);
    ((hash.wrapping_mul(COLOR_CACHE_HASH_MUL)) >> (32 - MATCH_HASH_BITS)) as usize
}

fn update_match_chain(argb: &[u32], index: usize, heads: &mut [usize], prev: &mut [usize]) {
    if index + MIN_LENGTH > argb.len() {
        return;
    }
    let hash = hash_match_pixels(argb, index);
    prev[index] = heads[hash];
    heads[hash] = index;
}

fn find_best_hash_match(
    width: usize,
    argb: &[u32],
    index: usize,
    max_len: usize,
    heads: &[usize],
    prev: &[usize],
    match_chain_depth: usize,
) -> Option<(usize, usize)> {
    if match_chain_depth == 0 || max_len < MIN_LENGTH || index + MIN_LENGTH > argb.len() {
        return None;
    }

    let hash = hash_match_pixels(argb, index);
    let mut candidate = heads[hash];
    let mut best = None;
    let mut remaining = match_chain_depth;

    while candidate != usize::MAX && remaining > 0 {
        remaining -= 1;
        if candidate >= index {
            break;
        }
        let distance = index - candidate;
        if distance <= MAX_FALLBACK_DISTANCE {
            let length = find_match_length(argb, index, candidate, max_len);
            if length >= MIN_LENGTH {
                consider_match(width, &mut best, distance, length);
            }
            if length == max_len {
                break;
            }
        }
        candidate = prev[candidate];
    }

    best
}

fn find_best_window_offset_match(
    width: usize,
    argb: &[u32],
    index: usize,
    max_len: usize,
    window_offsets: &[usize],
) -> Option<(usize, usize)> {
    let mut best_match = None;
    for &distance in window_offsets {
        if distance > index || distance > MAX_FALLBACK_DISTANCE {
            continue;
        }
        let candidate_index = index - distance;
        let length = find_match_length(argb, index, candidate_index, max_len);
        consider_match(width, &mut best_match, distance, length);
    }
    best_match
}

fn single_pixel_cost_bits(cache_hit: bool) -> isize {
    if cache_hit {
        APPROX_CACHE_COST_BITS
    } else {
        APPROX_LITERAL_COST_BITS
    }
}

fn find_best_match(
    width: usize,
    argb: &[u32],
    index: usize,
    options: TokenBuildOptions,
    heads: &[usize],
    prev: &[usize],
    window_offsets: &[usize],
) -> Option<(usize, usize)> {
    let max_len = (argb.len() - index).min(MAX_LENGTH);
    let mut best_match = None;

    if index > 0 {
        let rle_len = find_match_length(argb, index, index - 1, max_len);
        consider_match(width, &mut best_match, 1, rle_len);
    }
    if index >= width {
        let prev_row_len = find_match_length(argb, index, index - width, max_len);
        consider_match(width, &mut best_match, width, prev_row_len);
    }
    if options.use_window_offsets {
        if let Some((distance, length)) =
            find_best_window_offset_match(width, argb, index, max_len, window_offsets)
        {
            consider_match(width, &mut best_match, distance, length);
        }
    }
    if let Some((distance, length)) = find_best_hash_match(
        width,
        argb,
        index,
        max_len,
        heads,
        prev,
        options.match_chain_depth,
    ) {
        consider_match(width, &mut best_match, distance, length);
    }

    best_match
}

fn build_tokens_greedy(
    width: usize,
    argb: &[u32],
    options: TokenBuildOptions,
) -> Result<Vec<Token>, EncoderError> {
    if argb.is_empty() {
        return Ok(Vec::new());
    }

    let mut tokens = Vec::with_capacity(argb.len());
    let mut cache = if options.color_cache_bits > 0 {
        Some(ColorCache::new(options.color_cache_bits)?)
    } else {
        None
    };
    let mut heads = vec![usize::MAX; MATCH_HASH_SIZE];
    let mut prev = vec![usize::MAX; argb.len()];
    let window_offsets = if options.use_window_offsets {
        build_window_offsets(width, options.window_offset_limit)
    } else {
        Vec::new()
    };

    let mut index = 0usize;
    while index < argb.len() {
        let cache_key = cache.as_ref().and_then(|cache| cache.lookup(argb[index]));
        let mut best_match =
            find_best_match(width, argb, index, options, &heads, &prev, &window_offsets);

        if options.lazy_matching {
            if let Some((distance, length)) = best_match {
                if length < 64 && index + 1 < argb.len() {
                    let preview = preview_update_match_chain(argb, index, &mut heads, &mut prev);
                    let next_match = find_best_match(
                        width,
                        argb,
                        index + 1,
                        options,
                        &heads,
                        &prev,
                        &window_offsets,
                    );
                    restore_previewed_match_chain(index, preview, &mut heads, &mut prev);

                    let current_gain = match_gain_bits(width, distance, length);
                    let next_choice = next_match.map(|(next_distance, next_length)| {
                        (
                            next_length,
                            match_gain_bits(width, next_distance, next_length)
                                + APPROX_LITERAL_COST_BITS
                                - single_pixel_cost_bits(cache_key.is_some()),
                        )
                    });
                    if next_choice
                        .map(|(next_length, next_gain)| {
                            index + 1 + next_length >= index + length && next_gain > current_gain
                        })
                        .unwrap_or(false)
                    {
                        best_match = None;
                    } else {
                        best_match = Some((distance, length));
                    }
                }
            }
        }

        if let Some((distance, length)) = best_match {
            tokens.push(Token::Copy { distance, length });
            if let Some(cache) = &mut cache {
                for &pixel in &argb[index..index + length] {
                    cache.insert(pixel);
                }
            }
            for position in index..index + length {
                update_match_chain(argb, position, &mut heads, &mut prev);
            }
            index += length;
        } else if let Some(key) = cache_key {
            tokens.push(Token::Cache(key));
            if let Some(cache) = &mut cache {
                cache.insert(argb[index]);
            }
            update_match_chain(argb, index, &mut heads, &mut prev);
            index += 1;
        } else {
            tokens.push(Token::Literal(argb[index]));
            if let Some(cache) = &mut cache {
                cache.insert(argb[index]);
            }
            update_match_chain(argb, index, &mut heads, &mut prev);
            index += 1;
        }
    }

    Ok(tokens)
}

fn build_traceback_cost_model(
    width: usize,
    tokens: &[Token],
    color_cache_bits: usize,
) -> Result<TracebackCostModel, EncoderError> {
    let histograms = build_histograms(tokens, width, color_cache_bits)?;
    let group = build_group_codes(&histograms)?;
    let mut length_cost_intervals = Vec::new();
    let mut start = 1usize;
    let mut current_cost = {
        let prefix = prefix_encode(1)?;
        group.green.code_lengths()[NUM_LITERAL_CODES + prefix.symbol] as usize + prefix.extra_bits
    };
    for length in 2..=MAX_LENGTH {
        let prefix = prefix_encode(length)?;
        let cost = group.green.code_lengths()[NUM_LITERAL_CODES + prefix.symbol] as usize
            + prefix.extra_bits;
        if cost != current_cost {
            length_cost_intervals.push((start, length, current_cost));
            start = length;
            current_cost = cost;
        }
    }
    length_cost_intervals.push((start, MAX_LENGTH + 1, current_cost));
    Ok(TracebackCostModel {
        literal: group
            .green
            .code_lengths()
            .iter()
            .map(|&bits| bits as usize)
            .collect(),
        red: group
            .red
            .code_lengths()
            .iter()
            .map(|&bits| bits as usize)
            .collect(),
        blue: group
            .blue
            .code_lengths()
            .iter()
            .map(|&bits| bits as usize)
            .collect(),
        alpha: group
            .alpha
            .code_lengths()
            .iter()
            .map(|&bits| bits as usize)
            .collect(),
        distance: group
            .dist
            .code_lengths()
            .iter()
            .map(|&bits| bits as usize)
            .collect(),
        length_cost_intervals,
    })
}

impl TracebackCostModel {
    fn literal_cost(&self, argb: u32) -> usize {
        self.alpha[((argb >> 24) & 0xff) as usize]
            + self.red[((argb >> 16) & 0xff) as usize]
            + self.literal[((argb >> 8) & 0xff) as usize]
            + self.blue[(argb & 0xff) as usize]
    }

    fn distance_cost(&self, width: usize, distance: usize) -> Result<usize, EncoderError> {
        let plane_code = distance_to_plane_code(width, distance);
        let dist_prefix = prefix_encode(plane_code)?;
        Ok(self.distance[dist_prefix.symbol] + dist_prefix.extra_bits)
    }

    fn cache_cost(&self, key: usize) -> usize {
        self.literal[NUM_LITERAL_CODES + NUM_LENGTH_CODES + key]
    }
}

fn push_match_candidate(
    width: usize,
    candidates: &mut Vec<(usize, usize)>,
    distance: usize,
    length: usize,
) {
    if length < min_match_length_for_distance(width, distance) {
        return;
    }
    if let Some(existing) = candidates
        .iter_mut()
        .find(|(existing_distance, _)| *existing_distance == distance)
    {
        existing.1 = existing.1.max(length);
        return;
    }
    candidates.push((distance, length));
}

fn collect_match_candidates(
    width: usize,
    argb: &[u32],
    index: usize,
    options: TokenBuildOptions,
    heads: &[usize],
    prev: &[usize],
    window_offsets: &[usize],
) -> Vec<(usize, usize)> {
    let max_len = (argb.len() - index).min(MAX_LENGTH);
    let mut candidates = Vec::with_capacity(options.traceback_max_candidates.max(4));

    if index > 0 {
        let rle_len = find_match_length(argb, index, index - 1, max_len);
        push_match_candidate(width, &mut candidates, 1, rle_len);
    }
    if index >= width {
        let prev_row_len = find_match_length(argb, index, index - width, max_len);
        push_match_candidate(width, &mut candidates, width, prev_row_len);
    }
    if options.use_window_offsets {
        for &distance in window_offsets {
            if distance > index || distance > MAX_FALLBACK_DISTANCE {
                continue;
            }
            let length = find_match_length(argb, index, index - distance, max_len);
            push_match_candidate(width, &mut candidates, distance, length);
        }
    }
    if options.match_chain_depth > 0 && max_len >= MIN_LENGTH && index + MIN_LENGTH <= argb.len() {
        let hash = hash_match_pixels(argb, index);
        let mut candidate = heads[hash];
        let mut remaining = options.match_chain_depth;
        while candidate != usize::MAX && remaining > 0 {
            remaining -= 1;
            if candidate >= index {
                break;
            }
            let distance = index - candidate;
            if distance <= MAX_FALLBACK_DISTANCE {
                let length = find_match_length(argb, index, candidate, max_len);
                push_match_candidate(width, &mut candidates, distance, length);
                if length == max_len {
                    break;
                }
            }
            candidate = prev[candidate];
        }
    }

    candidates.sort_by(|lhs, rhs| {
        let lhs_score = match_gain_bits(width, lhs.0, lhs.1);
        let rhs_score = match_gain_bits(width, rhs.0, rhs.1);
        rhs_score
            .cmp(&lhs_score)
            .then_with(|| rhs.1.cmp(&lhs.1))
            .then_with(|| lhs.0.cmp(&rhs.0))
    });
    candidates.truncate(options.traceback_max_candidates.max(1));
    candidates
}

fn build_cache_keys(
    argb: &[u32],
    color_cache_bits: usize,
) -> Result<Vec<Option<usize>>, EncoderError> {
    if color_cache_bits == 0 {
        return Ok(vec![None; argb.len()]);
    }

    let mut cache = ColorCache::new(color_cache_bits)?;
    let mut keys = Vec::with_capacity(argb.len());
    for &pixel in argb {
        keys.push(cache.lookup(pixel));
        cache.insert(pixel);
    }
    Ok(keys)
}

fn build_tokens_with_traceback(
    width: usize,
    argb: &[u32],
    options: TokenBuildOptions,
    cost_model: &TracebackCostModel,
) -> Result<Vec<Token>, EncoderError> {
    let mut best_costs = vec![usize::MAX; argb.len() + 1];
    let mut previous = vec![usize::MAX; argb.len() + 1];
    let mut steps = vec![None; argb.len() + 1];
    let mut heads = vec![usize::MAX; MATCH_HASH_SIZE];
    let mut prev = vec![usize::MAX; argb.len()];
    let cache_keys = build_cache_keys(argb, options.color_cache_bits)?;
    let window_offsets = if options.use_window_offsets {
        build_window_offsets(width, options.window_offset_limit)
    } else {
        Vec::new()
    };
    let mut pending: BinaryHeap<Reverse<(usize, usize, usize, usize, usize)>> = BinaryHeap::new();
    let mut active: BinaryHeap<Reverse<(usize, usize, usize, usize)>> = BinaryHeap::new();

    best_costs[0] = 0;
    for index in 0..=argb.len() {
        while let Some(Reverse((start, end_exclusive, cost, source, distance))) =
            pending.peek().copied()
        {
            if start > index {
                break;
            }
            pending.pop();
            if end_exclusive > index {
                active.push(Reverse((cost, end_exclusive, source, distance)));
            }
        }
        while let Some(Reverse((_, end_exclusive, _, _))) = active.peek().copied() {
            if end_exclusive > index {
                break;
            }
            active.pop();
        }
        if let Some(Reverse((cost, _, source, distance))) = active.peek().copied() {
            if cost < best_costs[index] {
                best_costs[index] = cost;
                previous[index] = source;
                steps[index] = Some(TracebackStep::Copy {
                    distance,
                    length: index - source,
                });
            }
        }
        if index == argb.len() {
            break;
        }

        let base_cost = best_costs[index];
        if base_cost == usize::MAX {
            update_match_chain(argb, index, &mut heads, &mut prev);
            continue;
        }

        if let Some(key) = cache_keys[index] {
            let cache_cost = base_cost.saturating_add(cost_model.cache_cost(key));
            if cache_cost < best_costs[index + 1] {
                best_costs[index + 1] = cache_cost;
                previous[index + 1] = index;
                steps[index + 1] = Some(TracebackStep::Cache { key });
            }
        }

        let literal_cost = base_cost.saturating_add(cost_model.literal_cost(argb[index]));
        if literal_cost < best_costs[index + 1] {
            best_costs[index + 1] = literal_cost;
            previous[index + 1] = index;
            steps[index + 1] = Some(TracebackStep::Literal);
        }

        for (distance, length) in
            collect_match_candidates(width, argb, index, options, &heads, &prev, &window_offsets)
        {
            let min_length = min_match_length_for_distance(width, distance);
            let distance_cost =
                base_cost.saturating_add(cost_model.distance_cost(width, distance)?);
            for &(start_length, end_length_exclusive, length_cost) in
                &cost_model.length_cost_intervals
            {
                if start_length > length {
                    break;
                }
                let start = min_length.max(start_length);
                let end_exclusive = (length + 1).min(end_length_exclusive);
                if start < end_exclusive {
                    pending.push(Reverse((
                        index + start,
                        index + end_exclusive,
                        distance_cost.saturating_add(length_cost),
                        index,
                        distance,
                    )));
                }
            }
        }

        update_match_chain(argb, index, &mut heads, &mut prev);
    }

    let mut tokens = Vec::with_capacity(argb.len());
    let mut cursor = argb.len();
    while cursor > 0 {
        match steps[cursor].ok_or(EncoderError::Bitstream("traceback path is incomplete"))? {
            TracebackStep::Literal => {
                tokens.push(Token::Literal(argb[cursor - 1]));
                cursor = previous[cursor];
            }
            TracebackStep::Cache { key } => {
                tokens.push(Token::Cache(key));
                cursor = previous[cursor];
            }
            TracebackStep::Copy { distance, length } => {
                tokens.push(Token::Copy { distance, length });
                let start = cursor.saturating_sub(length);
                if previous[cursor] != start {
                    return Err(EncoderError::Bitstream(
                        "traceback predecessor is inconsistent",
                    ));
                }
                cursor = start;
            }
        }
        if cursor != 0 && steps[cursor].is_none() {
            return Err(EncoderError::Bitstream("traceback predecessor is missing"));
        }
    }
    tokens.reverse();
    Ok(tokens)
}

fn build_tokens(
    width: usize,
    argb: &[u32],
    options: TokenBuildOptions,
) -> Result<Vec<Token>, EncoderError> {
    let greedy = build_tokens_greedy(width, argb, options)?;
    if !options.use_traceback {
        return Ok(greedy);
    }

    let cost_model = build_traceback_cost_model(width, &greedy, options.color_cache_bits)?;
    let traceback = build_tokens_with_traceback(width, argb, options, &cost_model)?;
    let height = argb.len() / width;
    let greedy_cost =
        estimate_image_stream_size(width, height, &greedy, options.color_cache_bits, false, 0)?;
    let traceback_cost = estimate_image_stream_size(
        width,
        height,
        &traceback,
        options.color_cache_bits,
        false,
        0,
    )?;
    if traceback_cost <= greedy_cost {
        Ok(traceback)
    } else {
        Ok(greedy)
    }
}

fn prefix_encode(value: usize) -> Result<PrefixCode, EncoderError> {
    if value == 0 {
        return Err(EncoderError::InvalidParam("prefix value must be non-zero"));
    }

    if value <= 4 {
        return Ok(PrefixCode {
            symbol: value - 1,
            extra_bits: 0,
            extra_value: 0,
        });
    }

    let value = value - 1;
    let highest_bit = usize::BITS as usize - 1 - value.leading_zeros() as usize;
    let second_highest_bit = (value >> (highest_bit - 1)) & 1;
    let extra_bits = highest_bit - 1;
    let extra_value = value & ((1usize << extra_bits) - 1);

    Ok(PrefixCode {
        symbol: 2 * highest_bit + second_highest_bit,
        extra_bits,
        extra_value,
    })
}

fn distance_to_plane_code(width: usize, distance: usize) -> usize {
    let yoffset = distance / width;
    let xoffset = distance - yoffset * width;

    if xoffset <= 8 && yoffset < 8 {
        PLANE_TO_CODE_LUT[yoffset * 16 + 8 - xoffset] as usize + 1
    } else if xoffset > width.saturating_sub(8) && yoffset < 7 {
        PLANE_TO_CODE_LUT[(yoffset + 1) * 16 + 8 + (width - xoffset)] as usize + 1
    } else {
        distance + 120
    }
}

fn write_simple_huffman_tree(bw: &mut BitWriter, symbols: &[usize]) -> Result<(), EncoderError> {
    if symbols.is_empty() || symbols.len() > 2 {
        return Err(EncoderError::InvalidParam(
            "simple Huffman tree expects one or two symbols",
        ));
    }

    for &symbol in symbols {
        if symbol >= (1 << 8) {
            return Err(EncoderError::InvalidParam(
                "simple Huffman symbol is too large",
            ));
        }
    }

    bw.put_bits(1, 1)?;
    bw.put_bits((symbols.len() - 1) as u32, 1)?;

    let first = symbols[0];
    if first <= 1 {
        bw.put_bits(0, 1)?;
        bw.put_bits(first as u32, 1)?;
    } else {
        bw.put_bits(1, 1)?;
        bw.put_bits(first as u32, 8)?;
    }

    if let Some(&second) = symbols.get(1) {
        bw.put_bits(second as u32, 8)?;
    }

    Ok(())
}

fn write_trimmed_length(bw: &mut BitWriter, trimmed_length: usize) -> Result<(), EncoderError> {
    if trimmed_length < 2 {
        return Err(EncoderError::Bitstream("trimmed Huffman span is too small"));
    }
    if trimmed_length == 2 {
        bw.put_bits(0, 5)?;
        return Ok(());
    }

    let nbits = usize::BITS as usize - 1 - (trimmed_length - 2).leading_zeros() as usize;
    let nbitpairs = nbits / 2 + 1;
    if nbitpairs > 8 {
        return Err(EncoderError::Bitstream("trimmed Huffman span is too large"));
    }
    bw.put_bits((nbitpairs - 1) as u32, 3)?;
    bw.put_bits((trimmed_length - 2) as u32, nbitpairs * 2)
}

fn write_huffman_tree(bw: &mut BitWriter, code: &HuffmanCode) -> Result<(), EncoderError> {
    let symbols = code.used_symbols();
    if symbols.is_empty() {
        return Err(EncoderError::Bitstream("empty Huffman tree"));
    }
    if symbols.len() <= 2 && symbols.iter().all(|&symbol| symbol < (1 << 8)) {
        return write_simple_huffman_tree(bw, &symbols);
    }

    bw.put_bits(0, 1)?;
    let tokens = compress_huffman_tree(code.code_lengths());

    let mut token_histogram = vec![0u32; NUM_CODE_LENGTH_CODES];
    for token in &tokens {
        token_histogram[token.code as usize] += 1;
    }
    let token_code = HuffmanCode::from_histogram(&token_histogram, 7)?;

    let code_length_bitdepth = token_code.code_lengths();
    let mut codes_to_store = NUM_CODE_LENGTH_CODES;
    while codes_to_store > 4
        && code_length_bitdepth[CODE_LENGTH_CODE_ORDER[codes_to_store - 1]] == 0
    {
        codes_to_store -= 1;
    }
    bw.put_bits((codes_to_store - 4) as u32, 4)?;
    for &ordered_symbol in CODE_LENGTH_CODE_ORDER.iter().take(codes_to_store) {
        bw.put_bits(code_length_bitdepth[ordered_symbol] as u32, 3)?;
    }

    let mut trailing_zero_bits = 0usize;
    let mut trimmed_length = tokens.len();
    let mut index = tokens.len();
    while index > 0 {
        index -= 1;
        let token = tokens[index];
        if token.code == 0 || token.code == 17 || token.code == 18 {
            trimmed_length -= 1;
            trailing_zero_bits += code_length_bitdepth[token.code as usize] as usize;
            if token.code == 17 {
                trailing_zero_bits += 3;
            } else if token.code == 18 {
                trailing_zero_bits += 7;
            }
        } else {
            break;
        }
    }

    let write_trimmed = trimmed_length > 1 && trailing_zero_bits > 12;
    bw.put_bits(write_trimmed as u32, 1)?;
    let length = if write_trimmed {
        write_trimmed_length(bw, trimmed_length)?;
        trimmed_length
    } else {
        tokens.len()
    };

    for token in tokens.iter().take(length) {
        token_code.write_symbol(bw, token.code as usize)?;
        match token.code {
            16 => bw.put_bits(token.extra_bits as u32, 2)?,
            17 => bw.put_bits(token.extra_bits as u32, 3)?,
            18 => bw.put_bits(token.extra_bits as u32, 7)?,
            _ => {}
        }
    }

    Ok(())
}

fn build_histograms(
    tokens: &[Token],
    width: usize,
    color_cache_bits: usize,
) -> Result<HistogramSet, EncoderError> {
    let mut histograms = new_histograms(color_cache_bits);
    for &token in tokens {
        add_token_to_histograms(&mut histograms, width, token)?;
    }
    normalize_histograms(&mut histograms);
    Ok(histograms)
}

fn new_histograms(color_cache_bits: usize) -> HistogramSet {
    [
        vec![
            0u32;
            NUM_LITERAL_CODES
                + NUM_LENGTH_CODES
                + if color_cache_bits > 0 {
                    1usize << color_cache_bits
                } else {
                    0
                }
        ],
        vec![0u32; NUM_LITERAL_CODES],
        vec![0u32; NUM_LITERAL_CODES],
        vec![0u32; NUM_LITERAL_CODES],
        vec![0u32; NUM_DISTANCE_CODES],
    ]
}

fn add_token_to_histograms(
    histograms: &mut HistogramSet,
    width: usize,
    token: Token,
) -> Result<(), EncoderError> {
    match token {
        Token::Literal(argb) => {
            histograms[0][((argb >> 8) & 0xff) as usize] += 1;
            histograms[1][((argb >> 16) & 0xff) as usize] += 1;
            histograms[2][(argb & 0xff) as usize] += 1;
            histograms[3][((argb >> 24) & 0xff) as usize] += 1;
        }
        Token::Cache(key) => {
            histograms[0][NUM_LITERAL_CODES + NUM_LENGTH_CODES + key] += 1;
        }
        Token::Copy { distance, length } => {
            let length_prefix = prefix_encode(length)?;
            histograms[0][NUM_LITERAL_CODES + length_prefix.symbol] += 1;

            let plane_code = distance_to_plane_code(width, distance);
            let dist_prefix = prefix_encode(plane_code)?;
            histograms[4][dist_prefix.symbol] += 1;
        }
    }
    Ok(())
}

fn normalize_histograms(histograms: &mut HistogramSet) {
    for histogram in histograms.iter_mut().take(4) {
        if histogram.iter().all(|&count| count == 0) {
            histogram[0] = 1;
        }
    }
    if histograms[4].iter().all(|&count| count == 0) {
        histograms[4][0] = 1;
    }
}

fn merge_histograms(dst: &mut HistogramSet, src: &HistogramSet) {
    for (dst_histogram, src_histogram) in dst.iter_mut().zip(src.iter()) {
        for (dst_count, src_count) in dst_histogram.iter_mut().zip(src_histogram.iter()) {
            *dst_count += *src_count;
        }
    }
}

fn build_group_codes(histograms: &HistogramSet) -> Result<HuffmanGroupCodes, EncoderError> {
    Ok(HuffmanGroupCodes {
        green: HuffmanCode::from_histogram(&histograms[0], 15)?,
        red: HuffmanCode::from_histogram(&histograms[1], 15)?,
        blue: HuffmanCode::from_histogram(&histograms[2], 15)?,
        alpha: HuffmanCode::from_histogram(&histograms[3], 15)?,
        dist: HuffmanCode::from_histogram(&histograms[4], 15)?,
    })
}

fn token_len(token: Token) -> usize {
    match token {
        Token::Copy { length, .. } => length,
        Token::Literal(_) | Token::Cache(_) => 1,
    }
}

fn tile_index_for_pos(
    width: usize,
    huffman_bits: usize,
    huffman_xsize: usize,
    pos: usize,
) -> usize {
    let x = pos % width;
    let y = pos / width;
    (y >> huffman_bits) * huffman_xsize + (x >> huffman_bits)
}

fn histogram_cost(histograms: &HistogramSet, codes: &HuffmanGroupCodes) -> usize {
    histograms[0]
        .iter()
        .zip(codes.green.code_lengths())
        .map(|(&count, &bits)| count as usize * bits as usize)
        .sum::<usize>()
        + histograms[1]
            .iter()
            .zip(codes.red.code_lengths())
            .map(|(&count, &bits)| count as usize * bits as usize)
            .sum::<usize>()
        + histograms[2]
            .iter()
            .zip(codes.blue.code_lengths())
            .map(|(&count, &bits)| count as usize * bits as usize)
            .sum::<usize>()
        + histograms[3]
            .iter()
            .zip(codes.alpha.code_lengths())
            .map(|(&count, &bits)| count as usize * bits as usize)
            .sum::<usize>()
        + histograms[4]
            .iter()
            .zip(codes.dist.code_lengths())
            .map(|(&count, &bits)| count as usize * bits as usize)
            .sum::<usize>()
}

fn histogram_entropy_cost(histogram: &[u32]) -> f64 {
    let total = histogram.iter().map(|&count| count as f64).sum::<f64>();
    if total == 0.0 {
        return 0.0;
    }

    histogram
        .iter()
        .filter(|&&count| count != 0)
        .map(|&count| {
            let count = count as f64;
            count * (total / count).log2()
        })
        .sum()
}

fn histogram_signature_costs(histograms: &HistogramSet) -> [f64; 3] {
    [
        histogram_entropy_cost(&histograms[0]),
        histogram_entropy_cost(&histograms[1]),
        histogram_entropy_cost(&histograms[2]),
    ]
}

fn histogram_set_entropy_cost(histograms: &HistogramSet) -> f64 {
    histograms
        .iter()
        .map(|histogram| histogram_entropy_cost(histogram))
        .sum()
}

fn histogram_merge_penalty(lhs: &HistogramSet, rhs: &HistogramSet) -> f64 {
    let mut merged = lhs.clone();
    merge_histograms(&mut merged, rhs);
    histogram_set_entropy_cost(&merged)
        - histogram_set_entropy_cost(lhs)
        - histogram_set_entropy_cost(rhs)
}

fn histogram_partition_index(
    value: f64,
    min_value: f64,
    max_value: f64,
    partitions: usize,
) -> usize {
    if partitions <= 1 || max_value <= min_value {
        return 0;
    }

    let normalized = ((value - min_value) / (max_value - min_value)).clamp(0.0, 1.0);
    let index = (normalized * partitions as f64) as usize;
    index.min(partitions - 1)
}

fn entropy_histogram_candidates(
    non_empty_tiles: &[(usize, usize)],
    tile_histograms: &[HistogramSet],
    target_count: usize,
) -> Vec<HistogramCandidate> {
    if target_count == 0 || non_empty_tiles.is_empty() {
        return Vec::new();
    }

    let signatures = non_empty_tiles
        .iter()
        .map(|&(tile, _)| histogram_signature_costs(&tile_histograms[tile]))
        .collect::<Vec<_>>();

    let mins = [
        signatures
            .iter()
            .map(|costs| costs[0])
            .fold(f64::INFINITY, f64::min),
        signatures
            .iter()
            .map(|costs| costs[1])
            .fold(f64::INFINITY, f64::min),
        signatures
            .iter()
            .map(|costs| costs[2])
            .fold(f64::INFINITY, f64::min),
    ];
    let maxs = [
        signatures
            .iter()
            .map(|costs| costs[0])
            .fold(f64::NEG_INFINITY, f64::max),
        signatures
            .iter()
            .map(|costs| costs[1])
            .fold(f64::NEG_INFINITY, f64::max),
        signatures
            .iter()
            .map(|costs| costs[2])
            .fold(f64::NEG_INFINITY, f64::max),
    ];

    let bin_count = NUM_HISTOGRAM_PARTITIONS * NUM_HISTOGRAM_PARTITIONS * NUM_HISTOGRAM_PARTITIONS;
    let mut bins = vec![None::<HistogramCandidate>; bin_count];

    for (&(tile, weight), signature) in non_empty_tiles.iter().zip(signatures.iter()) {
        let green_bin =
            histogram_partition_index(signature[0], mins[0], maxs[0], NUM_HISTOGRAM_PARTITIONS);
        let red_bin =
            histogram_partition_index(signature[1], mins[1], maxs[1], NUM_HISTOGRAM_PARTITIONS);
        let blue_bin =
            histogram_partition_index(signature[2], mins[2], maxs[2], NUM_HISTOGRAM_PARTITIONS);
        let bin_index = green_bin * NUM_HISTOGRAM_PARTITIONS * NUM_HISTOGRAM_PARTITIONS
            + red_bin * NUM_HISTOGRAM_PARTITIONS
            + blue_bin;

        match &mut bins[bin_index] {
            Some(candidate) => {
                merge_histograms(&mut candidate.histograms, &tile_histograms[tile]);
                candidate.weight += weight;
            }
            slot @ None => {
                let mut histograms = tile_histograms[tile].clone();
                normalize_histograms(&mut histograms);
                *slot = Some(HistogramCandidate { histograms, weight });
            }
        }
    }

    let mut candidates = bins.into_iter().flatten().collect::<Vec<_>>();
    if candidates.is_empty() {
        return Vec::new();
    }

    while candidates.len() > target_count {
        let mut best_pair = None;
        let mut best_penalty = f64::INFINITY;
        for lhs in 0..candidates.len() {
            for rhs in lhs + 1..candidates.len() {
                let penalty = histogram_merge_penalty(
                    &candidates[lhs].histograms,
                    &candidates[rhs].histograms,
                );
                if penalty < best_penalty {
                    best_penalty = penalty;
                    best_pair = Some((lhs, rhs));
                }
            }
        }

        let Some((lhs, rhs)) = best_pair else {
            break;
        };
        let rhs_candidate = candidates.swap_remove(rhs);
        merge_histograms(&mut candidates[lhs].histograms, &rhs_candidate.histograms);
        normalize_histograms(&mut candidates[lhs].histograms);
        candidates[lhs].weight += rhs_candidate.weight;
    }

    candidates
}

fn build_entropy_seed_histograms(
    non_empty_tiles: &[(usize, usize)],
    tile_histograms: &[HistogramSet],
    group_count: usize,
) -> Vec<HistogramSet> {
    let mut candidates =
        entropy_histogram_candidates(non_empty_tiles, tile_histograms, group_count);
    candidates.sort_by(|lhs, rhs| rhs.weight.cmp(&lhs.weight));
    candidates
        .into_iter()
        .take(group_count)
        .map(|candidate| candidate.histograms)
        .collect()
}

fn build_weighted_seed_histograms(
    non_empty_tiles: &[(usize, usize)],
    tile_histograms: &[HistogramSet],
    group_count: usize,
) -> Vec<HistogramSet> {
    non_empty_tiles
        .iter()
        .take(group_count)
        .map(|&(tile, _)| {
            let mut histograms = tile_histograms[tile].clone();
            normalize_histograms(&mut histograms);
            histograms
        })
        .collect()
}

fn assign_tiles_to_groups(
    non_empty_tiles: &[(usize, usize)],
    tile_histograms: &[HistogramSet],
    group_codes: &[HuffmanGroupCodes],
    assignments: &mut [usize],
) {
    for &(tile, _) in non_empty_tiles {
        let mut best_group = 0usize;
        let mut best_cost = usize::MAX;
        for (group_index, codes) in group_codes.iter().enumerate() {
            let cost = histogram_cost(&tile_histograms[tile], codes);
            if cost < best_cost {
                best_cost = cost;
                best_group = group_index;
            }
        }
        assignments[tile] = best_group;
    }
}

fn refine_meta_huffman_plan(
    tile_count: usize,
    color_cache_bits: usize,
    non_empty_tiles: &[(usize, usize)],
    tile_histograms: &[HistogramSet],
    seed_histograms: Vec<HistogramSet>,
) -> Result<Option<MetaHuffmanPlan>, EncoderError> {
    if seed_histograms.len() <= 1 {
        return Ok(None);
    }

    let mut group_codes = seed_histograms
        .iter()
        .map(build_group_codes)
        .collect::<Result<Vec<_>, _>>()?;
    let mut assignments = vec![0usize; tile_count];

    for _ in 0..4 {
        assign_tiles_to_groups(
            non_empty_tiles,
            tile_histograms,
            &group_codes,
            &mut assignments,
        );

        let mut remap = vec![usize::MAX; group_codes.len()];
        let mut merged_histograms = Vec::new();
        for (group_index, _) in group_codes.iter().enumerate() {
            let mut histograms = new_histograms(color_cache_bits);
            let mut used = false;
            for &(tile, _) in non_empty_tiles {
                if assignments[tile] == group_index {
                    merge_histograms(&mut histograms, &tile_histograms[tile]);
                    used = true;
                }
            }
            if used {
                normalize_histograms(&mut histograms);
                remap[group_index] = merged_histograms.len();
                merged_histograms.push(histograms);
            }
        }
        if merged_histograms.len() <= 1 {
            return Ok(None);
        }
        for &(tile, _) in non_empty_tiles {
            assignments[tile] = remap[assignments[tile]];
        }
        group_codes = merged_histograms
            .iter()
            .map(build_group_codes)
            .collect::<Result<Vec<_>, _>>()?;
    }

    Ok(Some(MetaHuffmanPlan {
        huffman_bits: 0,
        huffman_xsize: 0,
        assignments,
        groups: group_codes,
    }))
}

fn meta_huffman_assignment_cost(
    non_empty_tiles: &[(usize, usize)],
    tile_histograms: &[HistogramSet],
    plan: &MetaHuffmanPlan,
) -> usize {
    non_empty_tiles
        .iter()
        .map(|&(tile, _)| {
            histogram_cost(&tile_histograms[tile], &plan.groups[plan.assignments[tile]])
        })
        .sum()
}

fn apply_color_cache_to_tokens(
    argb: &[u32],
    tokens: &[Token],
    color_cache_bits: usize,
) -> Result<Vec<Token>, EncoderError> {
    if color_cache_bits == 0 {
        return Ok(tokens.to_vec());
    }

    let mut cache = ColorCache::new(color_cache_bits)?;
    let mut cached_tokens = Vec::with_capacity(tokens.len());
    let mut pixel_index = 0usize;

    for &token in tokens {
        match token {
            Token::Literal(pixel) => {
                if let Some(key) = cache.lookup(pixel) {
                    cached_tokens.push(Token::Cache(key));
                } else {
                    cached_tokens.push(Token::Literal(pixel));
                    cache.insert(pixel);
                }
                pixel_index += 1;
            }
            Token::Cache(key) => {
                cached_tokens.push(Token::Cache(key));
                pixel_index += 1;
            }
            Token::Copy { distance, length } => {
                cached_tokens.push(Token::Copy { distance, length });
                for &pixel in &argb[pixel_index..pixel_index + length] {
                    cache.insert(pixel);
                }
                pixel_index += length;
            }
        }
    }

    Ok(cached_tokens)
}

fn build_meta_huffman_plan(
    width: usize,
    height: usize,
    tokens: &[Token],
    color_cache_bits: usize,
    huffman_bits: usize,
    max_groups: usize,
) -> Result<Option<MetaHuffmanPlan>, EncoderError> {
    if !(MIN_HUFFMAN_BITS..MIN_HUFFMAN_BITS + (1 << NUM_HUFFMAN_BITS)).contains(&huffman_bits) {
        return Ok(None);
    }

    let huffman_xsize = subsample_size(width, huffman_bits);
    let huffman_ysize = subsample_size(height, huffman_bits);
    let tile_count = huffman_xsize * huffman_ysize;
    if tile_count <= 1 {
        return Ok(None);
    }

    let mut tile_histograms = vec![new_histograms(color_cache_bits); tile_count];
    let mut tile_weights = vec![0usize; tile_count];
    let mut pos = 0usize;
    for &token in tokens {
        let tile = tile_index_for_pos(width, huffman_bits, huffman_xsize, pos);
        add_token_to_histograms(&mut tile_histograms[tile], width, token)?;
        tile_weights[tile] += token_len(token);
        pos += token_len(token);
    }

    let mut non_empty_tiles = tile_weights
        .iter()
        .enumerate()
        .filter_map(|(index, &weight)| (weight != 0).then_some((index, weight)))
        .collect::<Vec<_>>();
    if non_empty_tiles.len() <= 1 {
        return Ok(None);
    }
    non_empty_tiles.sort_by(|lhs, rhs| rhs.1.cmp(&lhs.1));

    let group_count = max_groups.min(non_empty_tiles.len());
    if group_count <= 1 {
        return Ok(None);
    }

    let seed_candidates = vec![
        build_weighted_seed_histograms(&non_empty_tiles, &tile_histograms, group_count),
        build_entropy_seed_histograms(&non_empty_tiles, &tile_histograms, group_count),
    ];

    let mut best_plan = None;
    let mut best_cost = usize::MAX;
    for seed_histograms in seed_candidates {
        if let Some(mut plan) = refine_meta_huffman_plan(
            tile_count,
            color_cache_bits,
            &non_empty_tiles,
            &tile_histograms,
            seed_histograms,
        )? {
            plan.huffman_bits = huffman_bits;
            plan.huffman_xsize = huffman_xsize;
            let cost = meta_huffman_assignment_cost(&non_empty_tiles, &tile_histograms, &plan);
            if cost < best_cost {
                best_cost = cost;
                best_plan = Some(plan);
            }
        }
    }

    Ok(best_plan)
}

fn write_huffman_group(bw: &mut BitWriter, group: &HuffmanGroupCodes) -> Result<(), EncoderError> {
    write_huffman_tree(bw, &group.green)?;
    write_huffman_tree(bw, &group.red)?;
    write_huffman_tree(bw, &group.blue)?;
    write_huffman_tree(bw, &group.alpha)?;
    write_huffman_tree(bw, &group.dist)
}

fn write_tokens_with_meta(
    bw: &mut BitWriter,
    tokens: &[Token],
    width: usize,
    plan: &MetaHuffmanPlan,
) -> Result<(), EncoderError> {
    let mut pos = 0usize;
    for &token in tokens {
        let tile = tile_index_for_pos(width, plan.huffman_bits, plan.huffman_xsize, pos);
        let group = &plan.groups[plan.assignments[tile]];
        match token {
            Token::Literal(argb) => {
                let green = ((argb >> 8) & 0xff) as usize;
                let red = ((argb >> 16) & 0xff) as usize;
                let blue = (argb & 0xff) as usize;
                let alpha = ((argb >> 24) & 0xff) as usize;

                group.green.write_symbol(bw, green)?;
                group.red.write_symbol(bw, red)?;
                group.blue.write_symbol(bw, blue)?;
                group.alpha.write_symbol(bw, alpha)?;
            }
            Token::Cache(key) => {
                group
                    .green
                    .write_symbol(bw, NUM_LITERAL_CODES + NUM_LENGTH_CODES + key)?;
            }
            Token::Copy { distance, length } => {
                let length_prefix = prefix_encode(length)?;
                group
                    .green
                    .write_symbol(bw, NUM_LITERAL_CODES + length_prefix.symbol)?;
                if length_prefix.extra_bits > 0 {
                    bw.put_bits(length_prefix.extra_value as u32, length_prefix.extra_bits)?;
                }

                let plane_code = distance_to_plane_code(width, distance);
                let dist_prefix = prefix_encode(plane_code)?;
                group.dist.write_symbol(bw, dist_prefix.symbol)?;
                if dist_prefix.extra_bits > 0 {
                    bw.put_bits(dist_prefix.extra_value as u32, dist_prefix.extra_bits)?;
                }
            }
        }
        pos += token_len(token);
    }
    Ok(())
}

fn write_tokens(
    bw: &mut BitWriter,
    tokens: &[Token],
    width: usize,
    green_codes: &HuffmanCode,
    red_codes: &HuffmanCode,
    blue_codes: &HuffmanCode,
    alpha_codes: &HuffmanCode,
    dist_codes: &HuffmanCode,
) -> Result<(), EncoderError> {
    for token in tokens {
        match *token {
            Token::Literal(argb) => {
                let green = ((argb >> 8) & 0xff) as usize;
                let red = ((argb >> 16) & 0xff) as usize;
                let blue = (argb & 0xff) as usize;
                let alpha = ((argb >> 24) & 0xff) as usize;

                green_codes.write_symbol(bw, green)?;
                red_codes.write_symbol(bw, red)?;
                blue_codes.write_symbol(bw, blue)?;
                alpha_codes.write_symbol(bw, alpha)?;
            }
            Token::Cache(key) => {
                green_codes.write_symbol(bw, NUM_LITERAL_CODES + NUM_LENGTH_CODES + key)?;
            }
            Token::Copy { distance, length } => {
                let length_prefix = prefix_encode(length)?;
                green_codes.write_symbol(bw, NUM_LITERAL_CODES + length_prefix.symbol)?;
                if length_prefix.extra_bits > 0 {
                    bw.put_bits(length_prefix.extra_value as u32, length_prefix.extra_bits)?;
                }

                let plane_code = distance_to_plane_code(width, distance);
                let dist_prefix = prefix_encode(plane_code)?;
                dist_codes.write_symbol(bw, dist_prefix.symbol)?;
                if dist_prefix.extra_bits > 0 {
                    bw.put_bits(dist_prefix.extra_value as u32, dist_prefix.extra_bits)?;
                }
            }
        }
    }

    Ok(())
}

fn write_single_group_image_stream(
    bw: &mut BitWriter,
    width: usize,
    tokens: &[Token],
    allow_meta_huffman: bool,
    color_cache_bits: usize,
    group: &HuffmanGroupCodes,
) -> Result<(), EncoderError> {
    bw.put_bits((color_cache_bits > 0) as u32, 1)?;
    if color_cache_bits > 0 {
        bw.put_bits(color_cache_bits as u32, 4)?;
    }
    if allow_meta_huffman {
        bw.put_bits(0, 1)?;
    }

    write_huffman_group(bw, group)?;

    write_tokens(
        bw,
        tokens,
        width,
        &group.green,
        &group.red,
        &group.blue,
        &group.alpha,
        &group.dist,
    )
}

fn write_meta_huffman_image_stream(
    bw: &mut BitWriter,
    width: usize,
    tokens: &[Token],
    color_cache_bits: usize,
    plan: &MetaHuffmanPlan,
) -> Result<(), EncoderError> {
    bw.put_bits((color_cache_bits > 0) as u32, 1)?;
    if color_cache_bits > 0 {
        bw.put_bits(color_cache_bits as u32, 4)?;
    }
    bw.put_bits(1, 1)?;
    bw.put_bits(
        (plan.huffman_bits - MIN_HUFFMAN_BITS) as u32,
        NUM_HUFFMAN_BITS,
    )?;

    let huffman_image = plan
        .assignments
        .iter()
        .map(|&group| (((group >> 8) as u32) << 16) | (((group & 0xff) as u32) << 8))
        .collect::<Vec<_>>();
    write_image_stream(
        bw,
        plan.huffman_xsize,
        &huffman_image,
        false,
        0,
        TokenBuildOptions {
            color_cache_bits: 0,
            match_chain_depth: 0,
            use_window_offsets: false,
            window_offset_limit: 0,
            lazy_matching: false,
            use_traceback: false,
            traceback_max_candidates: 0,
        },
    )?;

    for group in &plan.groups {
        write_huffman_group(bw, group)?;
    }
    write_tokens_with_meta(bw, tokens, width, plan)
}

fn write_image_stream_from_tokens(
    bw: &mut BitWriter,
    width: usize,
    height: usize,
    tokens: &[Token],
    emit_meta_huffman_flag: bool,
    entropy_search_level: u8,
    color_cache_bits: usize,
) -> Result<(), EncoderError> {
    let histograms = build_histograms(tokens, width, color_cache_bits)?;
    let group = build_group_codes(&histograms)?;

    let meta_candidates = if emit_meta_huffman_flag {
        meta_huffman_candidates(entropy_search_level, width, height)
    } else {
        &[]
    };
    if !meta_candidates.is_empty() {
        let single_size =
            estimate_single_group_image_stream_size(width, tokens, color_cache_bits, true, &group)?;
        let mut best_meta = None;
        let mut best_meta_size = usize::MAX;
        for &(huffman_bits, group_count) in meta_candidates {
            if let Some(plan) = build_meta_huffman_plan(
                width,
                height,
                tokens,
                color_cache_bits,
                huffman_bits,
                group_count,
            )? {
                let size = estimate_meta_huffman_image_stream_size(
                    width,
                    tokens,
                    color_cache_bits,
                    &plan,
                )?;
                if size < best_meta_size {
                    best_meta_size = size;
                    best_meta = Some(plan);
                }
            }
        }
        if let Some(plan) = best_meta {
            if best_meta_size < single_size {
                return write_meta_huffman_image_stream(bw, width, tokens, color_cache_bits, &plan);
            }
        }
    }

    write_single_group_image_stream(
        bw,
        width,
        tokens,
        emit_meta_huffman_flag,
        color_cache_bits,
        &group,
    )
}

fn write_image_stream(
    bw: &mut BitWriter,
    width: usize,
    argb: &[u32],
    emit_meta_huffman_flag: bool,
    entropy_search_level: u8,
    options: TokenBuildOptions,
) -> Result<(), EncoderError> {
    let tokens = build_tokens(width, argb, options)?;
    write_image_stream_from_tokens(
        bw,
        width,
        argb.len() / width,
        &tokens,
        emit_meta_huffman_flag,
        entropy_search_level,
        options.color_cache_bits,
    )
}

fn estimate_single_group_image_stream_size(
    width: usize,
    tokens: &[Token],
    color_cache_bits: usize,
    allow_meta_huffman: bool,
    group: &HuffmanGroupCodes,
) -> Result<usize, EncoderError> {
    let mut bw = BitWriter::default();
    write_single_group_image_stream(
        &mut bw,
        width,
        tokens,
        allow_meta_huffman,
        color_cache_bits,
        group,
    )?;
    Ok(bw.into_bytes().len())
}

fn estimate_meta_huffman_image_stream_size(
    width: usize,
    tokens: &[Token],
    color_cache_bits: usize,
    plan: &MetaHuffmanPlan,
) -> Result<usize, EncoderError> {
    let mut bw = BitWriter::default();
    write_meta_huffman_image_stream(&mut bw, width, tokens, color_cache_bits, plan)?;
    Ok(bw.into_bytes().len())
}

fn estimate_image_stream_size(
    width: usize,
    height: usize,
    tokens: &[Token],
    color_cache_bits: usize,
    emit_meta_huffman_flag: bool,
    entropy_search_level: u8,
) -> Result<usize, EncoderError> {
    let mut bw = BitWriter::default();
    write_image_stream_from_tokens(
        &mut bw,
        width,
        height,
        tokens,
        emit_meta_huffman_flag,
        entropy_search_level,
        color_cache_bits,
    )?;
    Ok(bw.into_bytes().len())
}

fn estimate_cache_candidate_cost(
    width: usize,
    tokens: &[Token],
    color_cache_bits: usize,
) -> Result<usize, EncoderError> {
    let histograms = build_histograms(tokens, width, color_cache_bits)?;
    let group = build_group_codes(&histograms)?;
    Ok(histogram_cost(&histograms, &group))
}

fn select_best_color_cache_bits(
    width: usize,
    height: usize,
    argb: &[u32],
    base_tokens: &[Token],
    profile: &LosslessSearchProfile,
) -> Result<usize, EncoderError> {
    let max_cache_bits =
        suggested_max_color_cache_bits(argb, max_color_cache_bits_for_profile(profile));
    let shortlist_size = shortlist_color_cache_candidates_for_profile(profile);

    let mut cheap_candidates = Vec::with_capacity(max_cache_bits + 1);
    cheap_candidates.push((
        estimate_cache_candidate_cost(width, base_tokens, 0)?,
        0usize,
    ));
    for cache_bits in 1..=max_cache_bits {
        let tokens = apply_color_cache_to_tokens(argb, base_tokens, cache_bits)?;
        let cost = estimate_cache_candidate_cost(width, &tokens, cache_bits)?;
        cheap_candidates.push((cost, cache_bits));
    }

    cheap_candidates.sort_by_key(|(cost, bits)| (*cost, *bits));
    let mut shortlist = cheap_candidates
        .into_iter()
        .take(shortlist_size.max(1))
        .map(|(_, bits)| bits)
        .collect::<Vec<_>>();
    if !shortlist.contains(&0) {
        shortlist.push(0);
    }

    let mut best_cache_bits = 0usize;
    let mut best_size = usize::MAX;
    for cache_bits in shortlist {
        let cheap_cost = if cache_bits == 0 {
            estimate_cache_candidate_cost(width, base_tokens, 0)?
        } else {
            let tokens = apply_color_cache_to_tokens(argb, base_tokens, cache_bits)?;
            estimate_cache_candidate_cost(width, &tokens, cache_bits)?
        };
        if best_size != usize::MAX
            && should_stop_transform_search(best_size, cheap_cost.div_ceil(8), profile)
        {
            break;
        }
        let size = if cache_bits == 0 {
            estimate_image_stream_size(width, height, base_tokens, 0, false, 0)?
        } else {
            let tokens = build_tokens(
                width,
                argb,
                token_build_options(profile.match_search_level, cache_bits),
            )?;
            estimate_image_stream_size(width, height, &tokens, cache_bits, false, 0)?
        };
        if size < best_size {
            best_size = size;
            best_cache_bits = cache_bits;
        }
    }

    Ok(best_cache_bits)
}

/// Encodes RGBA pixels to a raw lossless `VP8L` frame payload with explicit options.
pub fn encode_lossless_rgba_to_vp8l_with_options(
    width: usize,
    height: usize,
    rgba: &[u8],
    options: &LosslessEncodingOptions,
) -> Result<Vec<u8>, EncoderError> {
    validate_rgba(width, height, rgba)?;
    validate_options(options)?;

    let argb = rgba_to_argb(rgba);
    let subtract_green = apply_subtract_green_transform(&argb);
    let mut best = None;

    for profile in lossless_candidate_profiles(options.optimization_level) {
        let mut profile_best =
            if let Some(candidate) = build_palette_candidate(width, height, &argb)? {
                Some(encode_palette_candidate_to_vp8l(
                    width, height, rgba, &candidate, &profile,
                )?)
            } else {
                None
            };

        let plans = collect_transform_plans(width, height, &argb, &subtract_green, &profile);
        let ranked_plans = shortlist_transform_plans(width, plans, &profile)?;
        for (estimate, plan) in ranked_plans {
            if profile_best
                .as_ref()
                .map(|encoded| should_stop_transform_search(encoded.len(), estimate, &profile))
                .unwrap_or(false)
            {
                break;
            }

            let encoded = encode_transform_plan_to_vp8l(width, height, rgba, &plan, &profile)?;
            if profile_best
                .as_ref()
                .map(|current| encoded.len() < current.len())
                .unwrap_or(true)
            {
                profile_best = Some(encoded);
            }
        }

        if let Some(encoded) = profile_best {
            if best
                .as_ref()
                .map(|current: &Vec<u8>| encoded.len() < current.len())
                .unwrap_or(true)
            {
                best = Some(encoded);
            }
        }
    }

    best.ok_or(EncoderError::Bitstream(
        "lossless encoder produced no candidate",
    ))
}

/// Encodes RGBA pixels to a raw lossless `VP8L` frame payload.
pub fn encode_lossless_rgba_to_vp8l(
    width: usize,
    height: usize,
    rgba: &[u8],
) -> Result<Vec<u8>, EncoderError> {
    encode_lossless_rgba_to_vp8l_with_options(
        width,
        height,
        rgba,
        &LosslessEncodingOptions::default(),
    )
}

/// Encodes RGBA pixels to a still lossless WebP container with explicit options.
pub fn encode_lossless_rgba_to_webp_with_options(
    width: usize,
    height: usize,
    rgba: &[u8],
    options: &LosslessEncodingOptions,
) -> Result<Vec<u8>, EncoderError> {
    encode_lossless_rgba_to_webp_with_options_and_exif(width, height, rgba, options, None)
}

/// Encodes RGBA pixels to a still lossless WebP container with explicit options and EXIF.
pub fn encode_lossless_rgba_to_webp_with_options_and_exif(
    width: usize,
    height: usize,
    rgba: &[u8],
    options: &LosslessEncodingOptions,
    exif: Option<&[u8]>,
) -> Result<Vec<u8>, EncoderError> {
    let vp8l = encode_lossless_rgba_to_vp8l_with_options(width, height, rgba, options)?;
    wrap_still_webp(
        StillImageChunk {
            fourcc: *b"VP8L",
            payload: &vp8l,
            width,
            height,
            has_alpha: rgba_has_alpha(rgba),
        },
        exif,
    )
}

/// Encodes RGBA pixels to a still lossless WebP container.
pub fn encode_lossless_rgba_to_webp(
    width: usize,
    height: usize,
    rgba: &[u8],
) -> Result<Vec<u8>, EncoderError> {
    encode_lossless_rgba_to_webp_with_options(
        width,
        height,
        rgba,
        &LosslessEncodingOptions::default(),
    )
}

/// Encodes an [`ImageBuffer`] to a still lossless WebP container with explicit options.
pub fn encode_lossless_image_to_webp_with_options(
    image: &ImageBuffer,
    options: &LosslessEncodingOptions,
) -> Result<Vec<u8>, EncoderError> {
    encode_lossless_image_to_webp_with_options_and_exif(image, options, None)
}

/// Encodes an [`ImageBuffer`] to a still lossless WebP container with explicit options and EXIF.
pub fn encode_lossless_image_to_webp_with_options_and_exif(
    image: &ImageBuffer,
    options: &LosslessEncodingOptions,
    exif: Option<&[u8]>,
) -> Result<Vec<u8>, EncoderError> {
    encode_lossless_rgba_to_webp_with_options_and_exif(
        image.width,
        image.height,
        &image.rgba,
        options,
        exif,
    )
}

/// Encodes an [`ImageBuffer`] to a still lossless WebP container.
pub fn encode_lossless_image_to_webp(image: &ImageBuffer) -> Result<Vec<u8>, EncoderError> {
    encode_lossless_image_to_webp_with_options(image, &LosslessEncodingOptions::default())
}
