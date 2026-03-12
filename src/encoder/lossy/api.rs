use super::bitstream::*;
use super::predict::*;
use super::*;
use crate::encoder::writer::ByteWriter;

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

    let mut data = ByteWriter::with_capacity(payload_size);
    let frame_bits = ((partition0.len() as u32) << 5) | (1 << 4);
    data.write_u24_le(frame_bits);
    data.write_bytes(&[0x9d, 0x01, 0x2a]);
    data.write_u16_le(width as u16);
    data.write_u16_le(height as u16);
    data.write_bytes(partition0);
    data.write_bytes(token_partition);
    Ok(data.into_bytes())
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
    profile: &LossySearchProfile,
    segment: &SegmentConfig,
) -> Result<EncodedLossyCandidate, EncoderError> {
    let segment_quants = build_segment_quantizers(segment);
    let (token_partition, probabilities, modes) = if profile.update_probabilities {
        let mut stats = [[[[0u32; NUM_PROBAS]; NUM_CTX]; NUM_BANDS]; NUM_TYPES];
        let (initial_partition, _, initial_modes) = encode_token_partition(
            source,
            mb_width,
            mb_height,
            profile,
            segment,
            &segment_quants,
            &COEFFS_PROBA0,
            Some(&mut stats),
        );
        let probabilities = finalize_token_probabilities(&stats);
        if probabilities == COEFFS_PROBA0 {
            (initial_partition, probabilities, initial_modes)
        } else {
            let (token_partition, _, modes) = encode_token_partition(
                source,
                mb_width,
                mb_height,
                profile,
                segment,
                &segment_quants,
                &probabilities,
                None,
            );
            (token_partition, probabilities, modes)
        }
    } else {
        let (token_partition, _, modes) = encode_token_partition(
            source,
            mb_width,
            mb_height,
            profile,
            segment,
            &segment_quants,
            &COEFFS_PROBA0,
            None,
        );
        (token_partition, COEFFS_PROBA0, modes)
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
    let profile = lossy_search_profile(options.optimization_level);
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
        let candidate = encode_lossy_candidate(&source, mb_width, mb_height, &profile, segment)?;
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
        let profile = lossy_search_profile(options.optimization_level);
        let source = rgba_to_yuv420(width, height, &rgba, mb_width, mb_height);
        let segment = disabled_segment_config(mb_width * mb_height, clipped_quantizer(base_quant));
        let candidate =
            encode_lossy_candidate(&source, mb_width, mb_height, &profile, &segment).unwrap();
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
            &profile,
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
        let profile = lossy_search_profile(MAX_LOSSY_OPTIMIZATION_LEVEL);
        let top_modes = [B_DC_PRED; 4];
        let left_modes = [B_DC_PRED; 4];
        let top_context = NonZeroContext::default();
        let left_context = NonZeroContext::default();
        let mode = choose_macroblock_mode(
            &source,
            &mut reconstructed,
            0,
            1,
            &profile,
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
            MAX_LOSSY_OPTIMIZATION_LEVEL,
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
