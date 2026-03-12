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
const MAX_OPTIMIZATION_LEVEL: u8 = 2;
const NUM_PREDICTOR_MODES: u8 = 14;
const NUM_LITERAL_CODES: usize = 256;
const NUM_LENGTH_CODES: usize = 24;
const NUM_DISTANCE_CODES: usize = 40;
const NUM_CODE_LENGTH_CODES: usize = 19;
const MIN_HUFFMAN_BITS: usize = 2;
const NUM_HUFFMAN_BITS: usize = 3;
const COLOR_CACHE_HASH_MUL: u32 = 0x1e35_a7bd;
const MATCH_HASH_BITS: usize = 15;
const MATCH_HASH_SIZE: usize = 1 << MATCH_HASH_BITS;
const MATCH_CHAIN_DEPTH_LEVEL1: usize = 8;
const MATCH_CHAIN_DEPTH_LEVEL2: usize = 32;
const MAX_FALLBACK_DISTANCE: usize = (1 << 20) - 120;
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

#[derive(Debug, Clone, Copy)]
struct TokenBuildOptions {
    color_cache_bits: usize,
    match_chain_depth: usize,
    use_window_offsets: bool,
    lazy_matching: bool,
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

/// Lossless encoder tuning knobs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LosslessEncodingOptions {
    /// Compression effort from `0` to `2`.
    ///
    /// - `0`: global transforms only
    /// - `1`: global transforms + color-cache trial + short backward search
    /// - `2`: global/tiled transform search + color-cache trial + deeper backward search
    pub optimization_level: u8,
}

impl Default for LosslessEncodingOptions {
    fn default() -> Self {
        Self {
            optimization_level: MAX_OPTIMIZATION_LEVEL,
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
            "lossless optimization level must be in 0..=2",
        ));
    }
    Ok(())
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

fn build_global_cross_plan(width: usize, height: usize, subtract_green: &[u32]) -> TransformPlan {
    let cross_transform = estimate_cross_color_transform(subtract_green);
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
        subtract_green,
        GLOBAL_CROSS_COLOR_TRANSFORM_BITS,
        &cross_transforms,
    );

    TransformPlan {
        use_subtract_green: true,
        cross_bits: Some(GLOBAL_CROSS_COLOR_TRANSFORM_BITS),
        cross_width,
        cross_image,
        predictor_bits: None,
        predictor_width: 0,
        predictor_image: Vec::new(),
        predicted: cross_colored,
    }
}

fn build_global_transform_plan(
    width: usize,
    height: usize,
    subtract_green: &[u32],
) -> TransformPlan {
    let cross_plan = build_global_cross_plan(width, height, subtract_green);
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
        use_subtract_green: true,
        cross_bits: cross_plan.cross_bits,
        cross_width: cross_plan.cross_width,
        cross_image: cross_plan.cross_image,
        predictor_bits: Some(GLOBAL_PREDICTOR_TRANSFORM_BITS),
        predictor_width,
        predictor_image,
        predicted,
    }
}

fn build_tiled_cross_plan(width: usize, height: usize, subtract_green: &[u32]) -> TransformPlan {
    let (cross_width, _cross_height, cross_transforms, cross_image) =
        make_cross_color_transform_image(width, height, subtract_green);
    let cross_colored = apply_cross_color_transform(
        width,
        height,
        subtract_green,
        CROSS_COLOR_TRANSFORM_BITS,
        &cross_transforms,
    );

    TransformPlan {
        use_subtract_green: true,
        cross_bits: Some(CROSS_COLOR_TRANSFORM_BITS),
        cross_width,
        cross_image,
        predictor_bits: None,
        predictor_width: 0,
        predictor_image: Vec::new(),
        predicted: cross_colored,
    }
}

fn build_tiled_transform_plan(
    width: usize,
    height: usize,
    subtract_green: &[u32],
) -> TransformPlan {
    let cross_plan = build_tiled_cross_plan(width, height, subtract_green);
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
        use_subtract_green: true,
        cross_bits: cross_plan.cross_bits,
        cross_width: cross_plan.cross_width,
        cross_image: cross_plan.cross_image,
        predictor_bits: Some(PREDICTOR_TRANSFORM_BITS),
        predictor_width,
        predictor_image,
        predicted,
    }
}

fn encode_transform_plan_to_vp8l(
    width: usize,
    height: usize,
    rgba: &[u8],
    plan: &TransformPlan,
    optimization_level: u8,
    use_color_cache: bool,
) -> Result<Vec<u8>, EncoderError> {
    let no_cache_options = token_build_options(optimization_level, 0);
    let mut best =
        encode_transform_plan_to_vp8l_with_cache(width, height, rgba, plan, no_cache_options)?;
    if use_color_cache && plan.predicted.len() >= 64 {
        let base_tokens = build_tokens(width, &plan.predicted, no_cache_options)?;
        let best_cache_bits =
            select_best_color_cache_bits(width, &plan.predicted, &base_tokens, MAX_CACHE_BITS)?;
        let with_cache = encode_transform_plan_to_vp8l_with_cache(
            width,
            height,
            rgba,
            plan,
            token_build_options(optimization_level, best_cache_bits),
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
) -> Result<Vec<u8>, EncoderError> {
    let transform_options = TokenBuildOptions {
        color_cache_bits: 0,
        match_chain_depth: 0,
        use_window_offsets: false,
        lazy_matching: false,
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
            transform_options,
        )?;
    }
    bw.put_bits(0, 1)?;
    write_image_stream(&mut bw, width, &plan.predicted, true, token_options)?;

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

fn token_build_options(optimization_level: u8, color_cache_bits: usize) -> TokenBuildOptions {
    let (match_chain_depth, use_window_offsets, lazy_matching) = match optimization_level {
        0 => (0, false, false),
        1 => (MATCH_CHAIN_DEPTH_LEVEL1, false, false),
        _ => (MATCH_CHAIN_DEPTH_LEVEL2, true, true),
    };
    TokenBuildOptions {
        color_cache_bits,
        match_chain_depth,
        use_window_offsets,
        lazy_matching,
    }
}

fn build_window_offsets(width: usize) -> Vec<usize> {
    const MAX_WINDOW_OFFSETS: usize = 32;
    let mut by_plane_code = [0usize; MAX_WINDOW_OFFSETS];
    for y in 0..=6usize {
        for x in -6isize..=6isize {
            let offset = y as isize * width as isize + x;
            if offset <= 0 {
                continue;
            }
            let offset = offset as usize;
            let plane_code = distance_to_plane_code(width, offset).saturating_sub(1);
            if plane_code < MAX_WINDOW_OFFSETS {
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

fn match_score(width: usize, distance: usize, length: usize) -> isize {
    let plane_code = distance_to_plane_code(width, distance);
    let penalty = if distance == 1 || distance == width {
        0
    } else if plane_code <= 32 {
        1
    } else if plane_code <= 128 {
        2
    } else if plane_code <= 512 {
        3
    } else {
        4
    };
    length as isize - penalty
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

    let candidate_score = match_score(width, distance, length);
    if best_match
        .map(|(best_distance, best_length)| {
            let best_score = match_score(width, best_distance, best_length);
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
    let mut best_length = 0usize;
    let mut remaining = match_chain_depth;

    while candidate != usize::MAX && remaining > 0 {
        remaining -= 1;
        if candidate >= index {
            break;
        }
        let distance = index - candidate;
        if distance <= MAX_FALLBACK_DISTANCE {
            let length = find_match_length(argb, index, candidate, max_len);
            if length >= MIN_LENGTH
                && (length > best_length
                    || (length == best_length
                        && best
                            .map(|(best_distance, _)| distance < best_distance)
                            .unwrap_or(true)))
            {
                best = Some((distance, length));
                best_length = length;
                if length == max_len {
                    break;
                }
            }
        }
        candidate = prev[candidate];
    }

    if let Some((distance, length)) = best {
        if distance == 1 || distance == width {
            return Some((distance, length));
        }
        return Some((distance, length));
    }

    None
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

fn build_tokens(
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
        build_window_offsets(width)
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
                if length < 32 && index + 1 < argb.len() {
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

                    if next_match
                        .map(|(_, next_length)| next_length + 1 > length)
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

    let seed_tiles = non_empty_tiles
        .iter()
        .take(group_count)
        .map(|&(index, _)| index)
        .collect::<Vec<_>>();
    let mut group_codes = seed_tiles
        .iter()
        .map(|&tile| {
            let mut histograms = tile_histograms[tile].clone();
            normalize_histograms(&mut histograms);
            build_group_codes(&histograms)
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut assignments = vec![0usize; tile_count];
    for _ in 0..2 {
        assign_tiles_to_groups(
            &non_empty_tiles,
            &tile_histograms,
            &group_codes,
            &mut assignments,
        );

        let mut remap = vec![usize::MAX; group_codes.len()];
        let mut merged_histograms = Vec::new();
        for (group_index, _) in group_codes.iter().enumerate() {
            let mut histograms = new_histograms(color_cache_bits);
            let mut used = false;
            for &(tile, _) in &non_empty_tiles {
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
        for &(tile, _) in &non_empty_tiles {
            assignments[tile] = remap[assignments[tile]];
        }
        group_codes = merged_histograms
            .iter()
            .map(build_group_codes)
            .collect::<Result<Vec<_>, _>>()?;
    }

    Ok(Some(MetaHuffmanPlan {
        huffman_bits,
        huffman_xsize,
        assignments,
        groups: group_codes,
    }))
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
        TokenBuildOptions {
            color_cache_bits: 0,
            match_chain_depth: 0,
            use_window_offsets: false,
            lazy_matching: false,
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
    allow_meta_huffman: bool,
    color_cache_bits: usize,
) -> Result<(), EncoderError> {
    let histograms = build_histograms(tokens, width, color_cache_bits)?;
    let group = build_group_codes(&histograms)?;

    if allow_meta_huffman {
        let single_size =
            estimate_single_group_image_stream_size(width, tokens, color_cache_bits, true, &group)?;
        let mut best_meta = None;
        let mut best_meta_size = usize::MAX;
        for &(huffman_bits, max_groups) in &[(5usize, 4usize), (4usize, 4usize)] {
            for group_count in 2..=max_groups {
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
        allow_meta_huffman,
        color_cache_bits,
        &group,
    )
}

fn write_image_stream(
    bw: &mut BitWriter,
    width: usize,
    argb: &[u32],
    allow_meta_huffman: bool,
    options: TokenBuildOptions,
) -> Result<(), EncoderError> {
    let base_tokens = build_tokens(
        width,
        argb,
        TokenBuildOptions {
            color_cache_bits: 0,
            ..options
        },
    )?;
    let tokens = apply_color_cache_to_tokens(argb, &base_tokens, options.color_cache_bits)?;
    write_image_stream_from_tokens(
        bw,
        width,
        argb.len() / width,
        &tokens,
        allow_meta_huffman,
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
    tokens: &[Token],
    color_cache_bits: usize,
    allow_meta_huffman: bool,
) -> Result<usize, EncoderError> {
    let mut bw = BitWriter::default();
    write_image_stream_from_tokens(
        &mut bw,
        width,
        1,
        tokens,
        allow_meta_huffman,
        color_cache_bits,
    )?;
    Ok(bw.into_bytes().len())
}

fn select_best_color_cache_bits(
    width: usize,
    argb: &[u32],
    base_tokens: &[Token],
    max_cache_bits: usize,
) -> Result<usize, EncoderError> {
    let mut best_cache_bits = 0usize;
    let mut best_size = estimate_image_stream_size(width, base_tokens, 0, false)?;

    for cache_bits in 1..=max_cache_bits {
        let tokens = apply_color_cache_to_tokens(argb, base_tokens, cache_bits)?;
        let size = estimate_image_stream_size(width, &tokens, cache_bits, false)?;
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

    let subtract_green = apply_subtract_green_transform(&rgba_to_argb(rgba));
    let use_color_cache = options.optimization_level >= 1;
    let global_plan = build_global_transform_plan(width, height, &subtract_green);
    let mut best = encode_transform_plan_to_vp8l(
        width,
        height,
        rgba,
        &global_plan,
        options.optimization_level,
        use_color_cache,
    )?;

    if options.optimization_level >= 2 {
        let tiled_plan = build_tiled_transform_plan(width, height, &subtract_green);
        let tiled = encode_transform_plan_to_vp8l(
            width,
            height,
            rgba,
            &tiled_plan,
            options.optimization_level,
            use_color_cache,
        )?;
        if tiled.len() < best.len() {
            best = tiled;
        }
    }

    Ok(best)
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
