use super::entropy::*;
use super::tokens::*;
use super::*;
use crate::encoder::writer::ByteWriter;

pub(super) fn apply_subtract_green_transform(argb: &[u32]) -> Vec<u32> {
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

pub(super) fn color_transform_delta(transform: i8, color: u8) -> i32 {
    ((transform as i32) * (color as i8 as i32)) >> 5
}

pub(super) fn estimate_transform_coefficient(pairs: &[(i32, i32)]) -> i8 {
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

pub(super) fn estimate_cross_color_transform_region(
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

pub(super) fn estimate_cross_color_transform(argb: &[u32]) -> CrossColorTransform {
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

pub(super) fn pack_cross_color_transform(transform: CrossColorTransform) -> u32 {
    ((transform.red_to_blue as u8 as u32) << 16)
        | ((transform.green_to_blue as u8 as u32) << 8)
        | (transform.green_to_red as u8 as u32)
}

pub(super) fn apply_cross_color_transform(
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

pub(super) fn average2(a: u32, b: u32) -> u32 {
    (((a ^ b) & 0xfefe_fefeu32) >> 1) + (a & b)
}

pub(super) fn select_predictor(left: u32, top: u32, top_left: u32) -> u32 {
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

pub(super) fn clip255(value: i32) -> u32 {
    value.clamp(0, 255) as u32
}

pub(super) fn clamped_add_subtract_full(left: u32, top: u32, top_left: u32) -> u32 {
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

pub(super) fn clamped_add_subtract_half(left: u32, top: u32, top_left: u32) -> u32 {
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

pub(super) fn predictor(mode: u8, left: u32, top: u32, top_left: u32, top_right: u32) -> u32 {
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

pub(super) fn predictor_for_mode(argb: &[u32], width: usize, x: usize, y: usize, mode: u8) -> u32 {
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

pub(super) fn sub_pixels(a: u32, b: u32) -> u32 {
    let alpha = (((a >> 24) as u8).wrapping_sub((b >> 24) as u8)) as u32;
    let red = ((((a >> 16) & 0xff) as u8).wrapping_sub(((b >> 16) & 0xff) as u8)) as u32;
    let green = ((((a >> 8) & 0xff) as u8).wrapping_sub(((b >> 8) & 0xff) as u8)) as u32;
    let blue = (((a & 0xff) as u8).wrapping_sub((b & 0xff) as u8)) as u32;
    (alpha << 24) | (red << 16) | (green << 8) | blue
}

pub(super) fn wrapped_channel_error(actual: u32, predicted: u32, shift: u32) -> u32 {
    let actual = ((actual >> shift) & 0xff) as i32;
    let predicted = ((predicted >> shift) & 0xff) as i32;
    let delta = (actual - predicted).unsigned_abs();
    delta.min(256 - delta)
}

pub(super) fn predictor_error(actual: u32, predicted: u32) -> u32 {
    wrapped_channel_error(actual, predicted, 24)
        + wrapped_channel_error(actual, predicted, 16)
        + wrapped_channel_error(actual, predicted, 8)
        + wrapped_channel_error(actual, predicted, 0)
}

pub(super) fn choose_predictor_mode(
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

pub(super) fn apply_predictor_transform(
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

pub(super) fn subsample_size(size: usize, bits: usize) -> usize {
    (size + (1usize << bits) - 1) >> bits
}

pub(super) fn make_predictor_transform_image(
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

pub(super) fn make_uniform_predictor_transform_image(
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

pub(super) fn make_cross_color_transform_image(
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

pub(super) fn make_uniform_cross_color_transform_image(
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

pub(super) fn palette_xbits(palette_size: usize) -> usize {
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

pub(super) fn collect_palette(argb: &[u32]) -> Option<Vec<u32>> {
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

pub(super) fn build_palette_candidate(
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

pub(super) fn build_global_cross_plan(
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

pub(super) fn build_raw_plan(argb: &[u32]) -> TransformPlan {
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

pub(super) fn build_subtract_green_plan(subtract_green: &[u32]) -> TransformPlan {
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

pub(super) fn build_global_predictor_plan(
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

pub(super) fn build_global_transform_plan(
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

pub(super) fn build_tiled_cross_plan(
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

pub(super) fn build_tiled_predictor_plan(
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

pub(super) fn build_tiled_transform_plan(
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

pub(super) fn quick_token_build_options(profile: &LosslessSearchProfile) -> TokenBuildOptions {
    token_build_options(profile.match_search_level.min(2), 0)
}

pub(super) fn estimate_token_stream_cost_bytes(
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

pub(super) fn estimate_transform_plan_score(
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

pub(super) fn collect_transform_plans(
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

pub(super) fn shortlist_transform_plans(
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

pub(super) fn should_stop_transform_search(
    best_len: usize,
    next_estimate: usize,
    profile: &LosslessSearchProfile,
) -> bool {
    profile.early_stop_ratio_percent != usize::MAX
        && next_estimate.saturating_mul(100)
            >= best_len.saturating_mul(profile.early_stop_ratio_percent)
}

pub(super) fn encode_transform_plan_to_vp8l(
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

pub(super) fn encode_transform_plan_to_vp8l_with_cache(
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
    let mut vp8l = ByteWriter::with_capacity(1 + bitstream.len());
    vp8l.write_byte(0x2f);
    vp8l.write_bytes(&bitstream);
    Ok(vp8l.into_bytes())
}

pub(super) fn encode_palette_candidate_to_vp8l(
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
    let mut vp8l = ByteWriter::with_capacity(1 + bitstream.len());
    vp8l.write_byte(0x2f);
    vp8l.write_bytes(&bitstream);
    Ok(vp8l.into_bytes())
}
