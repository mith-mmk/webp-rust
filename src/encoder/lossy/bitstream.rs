//! Partition, coefficient, and mode bitstream helpers for lossy encoding.

use super::predict::*;
use super::*;

/// Internal helper for block has non zero.
pub(super) fn block_has_non_zero(levels: &[i16; 16], first: usize) -> bool {
    levels.iter().skip(first).any(|&level| level != 0)
}

/// Computes skip probability.
pub(super) fn compute_skip_probability(modes: &[MacroblockMode]) -> Option<u8> {
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

/// Internal helper for intra4 tree contains.
pub(super) fn intra4_tree_contains(node: i8, mode: u8) -> bool {
    if node <= 0 {
        return (-node) as u8 == mode;
    }
    let node = node as usize;
    intra4_tree_contains(Y_MODES_INTRA4[2 * node], mode)
        || intra4_tree_contains(Y_MODES_INTRA4[2 * node + 1], mode)
}

/// Internal helper for walk intra4 mode bits.
pub(super) fn walk_intra4_mode_bits<F: FnMut(bool, u8)>(
    top_mode: u8,
    left_mode: u8,
    mode: u8,
    emit: &mut F,
) {
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

/// Encodes intra4 mode.
pub(super) fn encode_intra4_mode(
    writer: &mut Vp8BoolWriter,
    top_mode: u8,
    left_mode: u8,
    mode: u8,
) {
    walk_intra4_mode_bits(top_mode, left_mode, mode, &mut |bit, prob| {
        writer.put_bit(bit, prob);
    });
}

/// Internal helper for intra4 mode rate.
pub(super) fn intra4_mode_rate(top_mode: u8, left_mode: u8, mode: u8) -> u32 {
    let mut rate = 0u32;
    walk_intra4_mode_bits(top_mode, left_mode, mode, &mut |bit, prob| {
        rate += bit_cost(bit, prob);
    });
    rate
}

/// Updates mode cache.
pub(super) fn update_mode_cache(mode: &MacroblockMode, top: &mut [u8], left: &mut [u8; 4]) {
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

/// Internal helper for coeff probs.
pub(super) fn coeff_probs<'a>(
    probabilities: &'a CoeffProbTables,
    coeff_type: usize,
    coeff_index: usize,
    ctx: usize,
) -> &'a [u8; 11] {
    &probabilities[coeff_type][BANDS[coeff_index]][ctx]
}

/// Internal helper for last non zero.
pub(super) fn last_non_zero(levels: &[i16; 16], first: usize) -> isize {
    for scan in (first..16).rev() {
        if levels[ZIGZAG[scan]] != 0 {
            return scan as isize;
        }
    }
    first as isize - 1
}

/// Writes large value.
pub(super) fn write_large_value(writer: &mut Vp8BoolWriter, value: u32, probs: &[u8; 11]) {
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

/// Internal helper for large value rate.
pub(super) fn large_value_rate(value: u32, probs: &[u8; 11]) -> u32 {
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

/// Internal helper for coefficients rate.
pub(super) fn coefficients_rate(
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

/// Encodes coefficients.
pub(super) fn encode_coefficients(
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

/// Records stat.
pub(super) fn record_stat(bit: bool, stat: &mut u32) {
    if *stat >= 0xfffe0000 {
        *stat = ((*stat + 1) >> 1) & 0x7fff7fff;
    }
    *stat += 0x00010000 + bit as u32;
}

/// Records large value.
pub(super) fn record_large_value(stats: &mut [u32; NUM_PROBAS], value: u32) {
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

/// Records coefficients stats.
pub(super) fn record_coefficients_stats(
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

/// Returns the bit cost of a boolean decision at the given probability.
pub(super) fn bit_cost(bit: bool, prob: u8) -> u32 {
    let p = if bit {
        255u16.saturating_sub(prob as u16)
    } else {
        prob as u16
    };
    let p = (p.max(1) as f64) / 256.0;
    ((-p.log2()) * 256.0 + 0.5) as u32
}

/// Calculates token probability.
pub(super) fn calc_token_probability(nb: u32, total: u32) -> u8 {
    if nb == 0 {
        255
    } else {
        (255 - nb * 255 / total) as u8
    }
}

/// Returns the modeled cost of one probability branch.
pub(super) fn branch_cost(nb: u32, total: u32, prob: u8) -> u32 {
    nb * bit_cost(true, prob) + (total - nb) * bit_cost(false, prob)
}

/// Finalizes token probabilities.
pub(super) fn finalize_token_probabilities(stats: &CoeffStats) -> CoeffProbTables {
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

/// Encodes partition0.
pub(super) fn encode_partition0(
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

/// Encodes macroblock.
pub(super) fn encode_macroblock(
    writer: &mut Vp8BoolWriter,
    probabilities: &CoeffProbTables,
    source: &Planes,
    reconstructed: &mut Planes,
    mb_x: usize,
    mb_y: usize,
    profile: &LossySearchProfile,
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
                let coeffs = maybe_refine_levels(
                    profile.refine_i4_final,
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
                let coeffs = maybe_refine_levels(
                    profile.refine_i16,
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
        let y2_coeffs = maybe_refine_y2_levels(
            profile,
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
            let coeffs = maybe_refine_levels(
                profile.refine_chroma,
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
            let coeffs = maybe_refine_levels(
                profile.refine_chroma,
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

/// Encodes token partition.
pub(super) fn encode_token_partition(
    source: &Planes,
    mb_width: usize,
    mb_height: usize,
    profile: &LossySearchProfile,
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
                profile,
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
                profile,
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
