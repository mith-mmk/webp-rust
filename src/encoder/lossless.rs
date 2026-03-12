use crate::encoder::bit_writer::BitWriter;
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
const DEFAULT_COLOR_CACHE_BITS: usize = 10;
const NUM_LITERAL_CODES: usize = 256;
const NUM_LENGTH_CODES: usize = 24;
const NUM_DISTANCE_CODES: usize = 40;
const NUM_CODE_LENGTH_CODES: usize = 19;
const COLOR_CACHE_HASH_MUL: u32 = 0x1e35_a7bd;
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
    cross_bits: usize,
    cross_width: usize,
    cross_image: Vec<u32>,
    predictor_bits: usize,
    predictor_width: usize,
    predictor_image: Vec<u32>,
    predicted: Vec<u32>,
}

/// Lossless encoder tuning knobs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LosslessEncodingOptions {
    /// Compression effort from `0` to `2`.
    ///
    /// - `0`: global transforms only
    /// - `1`: global transforms + color-cache trial
    /// - `2`: global/tiled transform search + color-cache trial
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

fn has_alpha(rgba: &[u8]) -> bool {
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

fn build_global_transform_plan(
    width: usize,
    height: usize,
    subtract_green: &[u32],
) -> TransformPlan {
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
        cross_bits: GLOBAL_CROSS_COLOR_TRANSFORM_BITS,
        cross_width,
        cross_image,
        predictor_bits: GLOBAL_PREDICTOR_TRANSFORM_BITS,
        predictor_width,
        predictor_image,
        predicted,
    }
}

fn build_tiled_transform_plan(
    width: usize,
    height: usize,
    subtract_green: &[u32],
) -> TransformPlan {
    let (cross_width, _cross_height, cross_transforms, cross_image) =
        make_cross_color_transform_image(width, height, subtract_green);
    let cross_colored = apply_cross_color_transform(
        width,
        height,
        subtract_green,
        CROSS_COLOR_TRANSFORM_BITS,
        &cross_transforms,
    );
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
        cross_bits: CROSS_COLOR_TRANSFORM_BITS,
        cross_width,
        cross_image,
        predictor_bits: PREDICTOR_TRANSFORM_BITS,
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
    use_color_cache: bool,
) -> Result<Vec<u8>, EncoderError> {
    let mut best = encode_transform_plan_to_vp8l_with_cache(width, height, rgba, plan, 0)?;
    if use_color_cache && plan.predicted.len() >= 64 {
        let with_cache = encode_transform_plan_to_vp8l_with_cache(
            width,
            height,
            rgba,
            plan,
            DEFAULT_COLOR_CACHE_BITS,
        )?;
        if with_cache.len() < best.len() {
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
    color_cache_bits: usize,
) -> Result<Vec<u8>, EncoderError> {
    let mut bw = BitWriter::default();
    bw.put_bits((width - 1) as u32, 14)?;
    bw.put_bits((height - 1) as u32, 14)?;
    bw.put_bits(has_alpha(rgba) as u32, 1)?;
    bw.put_bits(0, 3)?;

    bw.put_bits(1, 1)?;
    bw.put_bits(2, 2)?;
    bw.put_bits(1, 1)?;
    bw.put_bits(1, 2)?;
    bw.put_bits((plan.cross_bits - MIN_TRANSFORM_BITS) as u32, 3)?;
    write_image_stream(&mut bw, plan.cross_width, &plan.cross_image, false, 0)?;
    bw.put_bits(1, 1)?;
    bw.put_bits(0, 2)?;
    bw.put_bits((plan.predictor_bits - MIN_TRANSFORM_BITS) as u32, 3)?;
    write_image_stream(
        &mut bw,
        plan.predictor_width,
        &plan.predictor_image,
        false,
        0,
    )?;
    bw.put_bits(0, 1)?;
    write_image_stream(&mut bw, width, &plan.predicted, true, color_cache_bits)?;

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

fn build_tokens(
    width: usize,
    argb: &[u32],
    color_cache_bits: usize,
) -> Result<Vec<Token>, EncoderError> {
    if argb.is_empty() {
        return Ok(Vec::new());
    }

    let mut tokens = Vec::with_capacity(argb.len());
    let mut cache = if color_cache_bits > 0 {
        Some(ColorCache::new(color_cache_bits)?)
    } else {
        None
    };

    let mut index = 0usize;
    while index < argb.len() {
        let max_len = (argb.len() - index).min(MAX_LENGTH);
        let cache_key = cache.as_ref().and_then(|cache| cache.lookup(argb[index]));
        let rle_len = if index == 0 {
            0
        } else {
            find_match_length(argb, index, index - 1, max_len)
        };
        let prev_row_len = if index < width {
            0
        } else {
            find_match_length(argb, index, index - width, max_len)
        };

        let mut best_match = None;
        if rle_len >= MIN_LENGTH {
            best_match = Some((1usize, rle_len));
        }
        if prev_row_len >= MIN_LENGTH
            && best_match
                .map(|(_, best_length)| prev_row_len > best_length)
                .unwrap_or(true)
        {
            best_match = Some((width, prev_row_len));
        }

        if let Some((distance, length)) = best_match {
            tokens.push(Token::Copy { distance, length });
            if let Some(cache) = &mut cache {
                for &pixel in &argb[index..index + length] {
                    cache.insert(pixel);
                }
            }
            index += length;
        } else if let Some(key) = cache_key {
            tokens.push(Token::Cache(key));
            if let Some(cache) = &mut cache {
                cache.insert(argb[index]);
            }
            index += 1;
        } else {
            tokens.push(Token::Literal(argb[index]));
            if let Some(cache) = &mut cache {
                cache.insert(argb[index]);
            }
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

fn wrap_lossless_webp(vp8l: &[u8]) -> Result<Vec<u8>, EncoderError> {
    let padded_vp8l_size = vp8l.len() + (vp8l.len() & 1);
    let riff_size = 4usize
        .checked_add(8)
        .and_then(|size| size.checked_add(padded_vp8l_size))
        .ok_or(EncoderError::InvalidParam("encoded output is too large"))?;
    let total_size = 8usize
        .checked_add(riff_size)
        .ok_or(EncoderError::InvalidParam("encoded output is too large"))?;

    let mut data = Vec::with_capacity(total_size);
    data.extend_from_slice(b"RIFF");
    data.extend_from_slice(&(riff_size as u32).to_le_bytes());
    data.extend_from_slice(b"WEBP");
    data.extend_from_slice(b"VP8L");
    data.extend_from_slice(&(vp8l.len() as u32).to_le_bytes());
    data.extend_from_slice(vp8l);
    if vp8l.len() & 1 == 1 {
        data.push(0);
    }
    Ok(data)
}

fn build_histograms(
    tokens: &[Token],
    width: usize,
    color_cache_bits: usize,
) -> Result<[Vec<u32>; 5], EncoderError> {
    let mut green = vec![
        0u32;
        NUM_LITERAL_CODES
            + NUM_LENGTH_CODES
            + if color_cache_bits > 0 {
                1usize << color_cache_bits
            } else {
                0
            }
    ];
    let mut red = vec![0u32; NUM_LITERAL_CODES];
    let mut blue = vec![0u32; NUM_LITERAL_CODES];
    let mut alpha = vec![0u32; NUM_LITERAL_CODES];
    let mut dist = vec![0u32; NUM_DISTANCE_CODES];

    for token in tokens {
        match *token {
            Token::Literal(argb) => {
                green[((argb >> 8) & 0xff) as usize] += 1;
                red[((argb >> 16) & 0xff) as usize] += 1;
                blue[(argb & 0xff) as usize] += 1;
                alpha[((argb >> 24) & 0xff) as usize] += 1;
            }
            Token::Cache(key) => {
                green[NUM_LITERAL_CODES + NUM_LENGTH_CODES + key] += 1;
            }
            Token::Copy { distance, length } => {
                let length_prefix = prefix_encode(length)?;
                green[NUM_LITERAL_CODES + length_prefix.symbol] += 1;

                let plane_code = distance_to_plane_code(width, distance);
                let dist_prefix = prefix_encode(plane_code)?;
                dist[dist_prefix.symbol] += 1;
            }
        }
    }

    if dist.iter().all(|&count| count == 0) {
        dist[0] = 1;
    }

    Ok([green, red, blue, alpha, dist])
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

fn write_image_stream(
    bw: &mut BitWriter,
    width: usize,
    argb: &[u32],
    allow_meta_huffman: bool,
    color_cache_bits: usize,
) -> Result<(), EncoderError> {
    let tokens = build_tokens(width, argb, color_cache_bits)?;
    let [green_hist, red_hist, blue_hist, alpha_hist, dist_hist] =
        build_histograms(&tokens, width, color_cache_bits)?;

    let green_codes = HuffmanCode::from_histogram(&green_hist, 15)?;
    let red_codes = HuffmanCode::from_histogram(&red_hist, 15)?;
    let blue_codes = HuffmanCode::from_histogram(&blue_hist, 15)?;
    let alpha_codes = HuffmanCode::from_histogram(&alpha_hist, 15)?;
    let dist_codes = HuffmanCode::from_histogram(&dist_hist, 15)?;

    bw.put_bits((color_cache_bits > 0) as u32, 1)?;
    if color_cache_bits > 0 {
        bw.put_bits(color_cache_bits as u32, 4)?;
    }
    if allow_meta_huffman {
        bw.put_bits(0, 1)?;
    }

    write_huffman_tree(bw, &green_codes)?;
    write_huffman_tree(bw, &red_codes)?;
    write_huffman_tree(bw, &blue_codes)?;
    write_huffman_tree(bw, &alpha_codes)?;
    write_huffman_tree(bw, &dist_codes)?;

    write_tokens(
        bw,
        &tokens,
        width,
        &green_codes,
        &red_codes,
        &blue_codes,
        &alpha_codes,
        &dist_codes,
    )
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
    let global_plan = build_global_transform_plan(width, height, &subtract_green);
    let use_color_cache = options.optimization_level >= 1;
    let mut best =
        encode_transform_plan_to_vp8l(width, height, rgba, &global_plan, use_color_cache)?;

    if options.optimization_level >= 2 {
        let tiled_plan = build_tiled_transform_plan(width, height, &subtract_green);
        let tiled =
            encode_transform_plan_to_vp8l(width, height, rgba, &tiled_plan, use_color_cache)?;
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
    let vp8l = encode_lossless_rgba_to_vp8l_with_options(width, height, rgba, options)?;
    wrap_lossless_webp(&vp8l)
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
    encode_lossless_rgba_to_webp_with_options(image.width, image.height, &image.rgba, options)
}

/// Encodes an [`ImageBuffer`] to a still lossless WebP container.
pub fn encode_lossless_image_to_webp(image: &ImageBuffer) -> Result<Vec<u8>, EncoderError> {
    encode_lossless_image_to_webp_with_options(image, &LosslessEncodingOptions::default())
}
