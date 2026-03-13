//! Backward references, cache selection, and tokenization for lossless encoding.

use super::entropy::*;
use super::*;

/// Finds match length.
pub(super) fn find_match_length(
    argb: &[u32],
    first: usize,
    second: usize,
    max_len: usize,
) -> usize {
    let mut len = 0usize;
    while len < max_len && argb[first + len] == argb[second + len] {
        len += 1;
    }
    len
}

/// Internal helper for token build options.
pub(super) fn token_build_options(
    match_search_level: u8,
    color_cache_bits: usize,
) -> TokenBuildOptions {
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

/// Returns the maximum color cache bits for profile.
pub(super) fn max_color_cache_bits_for_profile(profile: &LosslessSearchProfile) -> usize {
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

/// Builds a shortlist of color cache candidates for profile.
pub(super) fn shortlist_color_cache_candidates_for_profile(
    profile: &LosslessSearchProfile,
) -> usize {
    match profile.entropy_search_level {
        0 | 1 => 1,
        2 | 3 => 2,
        _ => 3,
    }
}

/// Internal helper for meta huffman candidates.
pub(super) fn meta_huffman_candidates(
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

/// Internal helper for suggested max color cache bits.
pub(super) fn suggested_max_color_cache_bits(argb: &[u32], max_cache_bits: usize) -> usize {
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

/// Builds window offsets.
pub(super) fn build_window_offsets(width: usize, max_plane_codes: usize) -> Vec<usize> {
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

/// Returns the minimum match length for distance.
pub(super) fn min_match_length_for_distance(width: usize, distance: usize) -> usize {
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

/// Internal helper for prefix extra bit count.
pub(super) fn prefix_extra_bit_count(value: usize) -> usize {
    if value <= 4 {
        0
    } else {
        let value = value - 1;
        let highest_bit = usize::BITS as usize - 1 - value.leading_zeros() as usize;
        highest_bit - 1
    }
}

/// Copies cost bits.
pub(super) fn copy_cost_bits(width: usize, distance: usize, length: usize) -> isize {
    let plane_code = distance_to_plane_code(width, distance);
    APPROX_COPY_LENGTH_SYMBOL_BITS
        + prefix_extra_bit_count(length) as isize
        + APPROX_COPY_DISTANCE_SYMBOL_BITS
        + prefix_extra_bit_count(plane_code) as isize
}

/// Internal helper for match gain bits.
pub(super) fn match_gain_bits(width: usize, distance: usize, length: usize) -> isize {
    APPROX_LITERAL_COST_BITS * length as isize - copy_cost_bits(width, distance, length)
}

/// Internal helper for consider match.
pub(super) fn consider_match(
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

/// Internal helper for preview update match chain.
pub(super) fn preview_update_match_chain(
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

/// Restores previewed match chain.
pub(super) fn restore_previewed_match_chain(
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

/// Internal helper for hash match pixels.
pub(super) fn hash_match_pixels(argb: &[u32], index: usize) -> usize {
    let a = argb[index];
    let b = argb[index + 1].rotate_left(7);
    let c = argb[index + 2].rotate_left(13);
    let d = argb[index + 3].rotate_left(21);
    let hash = a ^ b ^ c ^ d.wrapping_mul(COLOR_CACHE_HASH_MUL);
    ((hash.wrapping_mul(COLOR_CACHE_HASH_MUL)) >> (32 - MATCH_HASH_BITS)) as usize
}

/// Updates match chain.
pub(super) fn update_match_chain(
    argb: &[u32],
    index: usize,
    heads: &mut [usize],
    prev: &mut [usize],
) {
    if index + MIN_LENGTH > argb.len() {
        return;
    }
    let hash = hash_match_pixels(argb, index);
    prev[index] = heads[hash];
    heads[hash] = index;
}

/// Finds best hash match.
pub(super) fn find_best_hash_match(
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

/// Finds best window offset match.
pub(super) fn find_best_window_offset_match(
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

/// Returns the approximate bit cost of emitting a single pixel.
pub(super) fn single_pixel_cost_bits(cache_hit: bool) -> isize {
    if cache_hit {
        APPROX_CACHE_COST_BITS
    } else {
        APPROX_LITERAL_COST_BITS
    }
}

/// Finds best match.
pub(super) fn find_best_match(
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

/// Builds tokens greedy.
pub(super) fn build_tokens_greedy(
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

/// Builds traceback cost model.
pub(super) fn build_traceback_cost_model(
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

/// Internal helper for push match candidate.
pub(super) fn push_match_candidate(
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

/// Collects match candidates.
pub(super) fn collect_match_candidates(
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

/// Builds cache keys.
pub(super) fn build_cache_keys(
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

/// Builds tokens with traceback.
pub(super) fn build_tokens_with_traceback(
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

/// Builds tokens.
pub(super) fn build_tokens(
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
    if traceback_cost < greedy_cost {
        Ok(traceback)
    } else {
        Ok(greedy)
    }
}
