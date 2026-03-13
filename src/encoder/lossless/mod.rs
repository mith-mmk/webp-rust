//! Shared state and search profiles for the lossless `VP8L` encoder.

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
    /// Returns the balanced default lossless settings.
    fn default() -> Self {
        Self {
            optimization_level: DEFAULT_OPTIMIZATION_LEVEL,
        }
    }
}

impl ColorCache {
    /// Allocates a lossless color cache with the requested hash width.
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

    /// Computes the cache slot for a packed ARGB pixel.
    fn key(&self, argb: u32) -> usize {
        ((argb.wrapping_mul(COLOR_CACHE_HASH_MUL)) >> self.hash_shift) as usize
    }

    /// Returns the cache key when the pixel is already present.
    fn lookup(&self, argb: u32) -> Option<usize> {
        let key = self.key(argb);
        (self.colors[key] == argb).then_some(key)
    }

    /// Inserts or replaces one packed ARGB pixel in the cache.
    fn insert(&mut self, argb: u32) {
        let key = self.key(argb);
        self.colors[key] = argb;
    }
}

/// Validates rgba.
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

/// Validates options.
fn validate_options(options: &LosslessEncodingOptions) -> Result<(), EncoderError> {
    if options.optimization_level > MAX_OPTIMIZATION_LEVEL {
        return Err(EncoderError::InvalidParam(
            "lossless optimization level must be in 0..=9",
        ));
    }
    Ok(())
}

/// Builds the lossless search profile for a given optimization level.
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

/// Expands a lossless optimization level into candidate search profiles.
fn lossless_candidate_profiles(optimization_level: u8) -> Vec<LosslessSearchProfile> {
    match optimization_level {
        8 => vec![lossless_search_profile(7)],
        9 => vec![lossless_search_profile(7)],
        _ => vec![lossless_search_profile(optimization_level)],
    }
}

/// Returns whether the RGBA input contains any non-opaque pixels.
fn rgba_has_alpha(rgba: &[u8]) -> bool {
    rgba.chunks_exact(4).any(|pixel| pixel[3] != 0xff)
}

/// Reorders RGBA bytes into packed ARGB pixels.
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

mod api;
mod entropy;
mod plans;
mod tokens;

pub use api::*;
