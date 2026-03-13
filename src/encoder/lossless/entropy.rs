//! Huffman coding and image-stream emission for lossless encoding.

use super::plans::*;
use super::tokens::*;
use super::*;

/// Internal helper for prefix encode.
pub(super) fn prefix_encode(value: usize) -> Result<PrefixCode, EncoderError> {
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

/// Internal helper for distance to plane code.
pub(super) fn distance_to_plane_code(width: usize, distance: usize) -> usize {
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

/// Writes simple huffman tree.
pub(super) fn write_simple_huffman_tree(
    bw: &mut BitWriter,
    symbols: &[usize],
) -> Result<(), EncoderError> {
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

/// Writes trimmed length.
pub(super) fn write_trimmed_length(
    bw: &mut BitWriter,
    trimmed_length: usize,
) -> Result<(), EncoderError> {
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

/// Writes huffman tree.
pub(super) fn write_huffman_tree(
    bw: &mut BitWriter,
    code: &HuffmanCode,
) -> Result<(), EncoderError> {
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

/// Builds histograms.
pub(super) fn build_histograms(
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

/// Internal helper for new histograms.
pub(super) fn new_histograms(color_cache_bits: usize) -> HistogramSet {
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

/// Internal helper for add token to histograms.
pub(super) fn add_token_to_histograms(
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

/// Internal helper for normalize histograms.
pub(super) fn normalize_histograms(histograms: &mut HistogramSet) {
    for histogram in histograms.iter_mut().take(4) {
        if histogram.iter().all(|&count| count == 0) {
            histogram[0] = 1;
        }
    }
    if histograms[4].iter().all(|&count| count == 0) {
        histograms[4][0] = 1;
    }
}

/// Internal helper for merge histograms.
pub(super) fn merge_histograms(dst: &mut HistogramSet, src: &HistogramSet) {
    for (dst_histogram, src_histogram) in dst.iter_mut().zip(src.iter()) {
        for (dst_count, src_count) in dst_histogram.iter_mut().zip(src_histogram.iter()) {
            *dst_count += *src_count;
        }
    }
}

/// Builds group codes.
pub(super) fn build_group_codes(
    histograms: &HistogramSet,
) -> Result<HuffmanGroupCodes, EncoderError> {
    Ok(HuffmanGroupCodes {
        green: HuffmanCode::from_histogram(&histograms[0], 15)?,
        red: HuffmanCode::from_histogram(&histograms[1], 15)?,
        blue: HuffmanCode::from_histogram(&histograms[2], 15)?,
        alpha: HuffmanCode::from_histogram(&histograms[3], 15)?,
        dist: HuffmanCode::from_histogram(&histograms[4], 15)?,
    })
}

/// Returns the number of pixels consumed by a lossless token.
pub(super) fn token_len(token: Token) -> usize {
    match token {
        Token::Copy { length, .. } => length,
        Token::Literal(_) | Token::Cache(_) => 1,
    }
}

/// Maps a token position to its histogram tile index.
pub(super) fn tile_index_for_pos(
    width: usize,
    huffman_bits: usize,
    huffman_xsize: usize,
    pos: usize,
) -> usize {
    let x = pos % width;
    let y = pos / width;
    (y >> huffman_bits) * huffman_xsize + (x >> huffman_bits)
}

/// Internal helper for histogram cost.
pub(super) fn histogram_cost(histograms: &HistogramSet, codes: &HuffmanGroupCodes) -> usize {
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

/// Internal helper for histogram entropy cost.
pub(super) fn histogram_entropy_cost(histogram: &[u32]) -> f64 {
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

/// Internal helper for histogram signature costs.
pub(super) fn histogram_signature_costs(histograms: &HistogramSet) -> [f64; 3] {
    [
        histogram_entropy_cost(&histograms[0]),
        histogram_entropy_cost(&histograms[1]),
        histogram_entropy_cost(&histograms[2]),
    ]
}

/// Internal helper for histogram set entropy cost.
pub(super) fn histogram_set_entropy_cost(histograms: &HistogramSet) -> f64 {
    histograms
        .iter()
        .map(|histogram| histogram_entropy_cost(histogram))
        .sum()
}

/// Internal helper for histogram merge penalty.
pub(super) fn histogram_merge_penalty(lhs: &HistogramSet, rhs: &HistogramSet) -> f64 {
    let mut merged = lhs.clone();
    merge_histograms(&mut merged, rhs);
    histogram_set_entropy_cost(&merged)
        - histogram_set_entropy_cost(lhs)
        - histogram_set_entropy_cost(rhs)
}

/// Internal helper for histogram partition index.
pub(super) fn histogram_partition_index(
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

/// Internal helper for entropy histogram candidates.
pub(super) fn entropy_histogram_candidates(
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

/// Builds entropy seed histograms.
pub(super) fn build_entropy_seed_histograms(
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

/// Builds weighted seed histograms.
pub(super) fn build_weighted_seed_histograms(
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

/// Internal helper for assign tiles to groups.
pub(super) fn assign_tiles_to_groups(
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

/// Internal helper for refine meta huffman plan.
pub(super) fn refine_meta_huffman_plan(
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

/// Internal helper for meta huffman assignment cost.
pub(super) fn meta_huffman_assignment_cost(
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

/// Applies color cache to tokens.
pub(super) fn apply_color_cache_to_tokens(
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

/// Builds meta huffman plan.
pub(super) fn build_meta_huffman_plan(
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

/// Writes huffman group.
pub(super) fn write_huffman_group(
    bw: &mut BitWriter,
    group: &HuffmanGroupCodes,
) -> Result<(), EncoderError> {
    write_huffman_tree(bw, &group.green)?;
    write_huffman_tree(bw, &group.red)?;
    write_huffman_tree(bw, &group.blue)?;
    write_huffman_tree(bw, &group.alpha)?;
    write_huffman_tree(bw, &group.dist)
}

/// Writes tokens with meta.
pub(super) fn write_tokens_with_meta(
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

/// Writes tokens.
pub(super) fn write_tokens(
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

/// Writes single group image stream.
pub(super) fn write_single_group_image_stream(
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

/// Writes meta huffman image stream.
pub(super) fn write_meta_huffman_image_stream(
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

/// Writes image stream from tokens.
pub(super) fn write_image_stream_from_tokens(
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

/// Writes image stream.
pub(super) fn write_image_stream(
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

/// Estimates single group image stream size.
pub(super) fn estimate_single_group_image_stream_size(
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

/// Estimates meta huffman image stream size.
pub(super) fn estimate_meta_huffman_image_stream_size(
    width: usize,
    tokens: &[Token],
    color_cache_bits: usize,
    plan: &MetaHuffmanPlan,
) -> Result<usize, EncoderError> {
    let mut bw = BitWriter::default();
    write_meta_huffman_image_stream(&mut bw, width, tokens, color_cache_bits, plan)?;
    Ok(bw.into_bytes().len())
}

/// Estimates image stream size.
pub(super) fn estimate_image_stream_size(
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

/// Estimates cache candidate cost.
pub(super) fn estimate_cache_candidate_cost(
    width: usize,
    tokens: &[Token],
    color_cache_bits: usize,
) -> Result<usize, EncoderError> {
    let histograms = build_histograms(tokens, width, color_cache_bits)?;
    let group = build_group_codes(&histograms)?;
    Ok(histogram_cost(&histograms, &group))
}

/// Selects best color cache bits.
pub(super) fn select_best_color_cache_bits(
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
