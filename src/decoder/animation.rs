use crate::bmp::encode_bmp24_from_rgba;
use crate::decoder::header::{parse_animation_webp, ParsedAnimationFrame};
use crate::decoder::lossless::decode_lossless_vp8l_to_rgba;
use crate::decoder::lossy::{decode_lossy_vp8_to_rgba, DecodedImage};
use crate::decoder::DecoderError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedAnimationFrame {
    pub duration: usize,
    pub rgba: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedAnimation {
    pub width: usize,
    pub height: usize,
    pub background_color: u32,
    pub loop_count: u16,
    pub frames: Vec<DecodedAnimationFrame>,
}

fn argb_to_rgba(argb: u32) -> [u8; 4] {
    [
        ((argb >> 16) & 0xff) as u8,
        ((argb >> 8) & 0xff) as u8,
        (argb & 0xff) as u8,
        (argb >> 24) as u8,
    ]
}

fn fill_rect(
    canvas: &mut [u8],
    canvas_width: usize,
    x_offset: usize,
    y_offset: usize,
    width: usize,
    height: usize,
    rgba: [u8; 4],
) {
    for y in 0..height {
        let row = ((y_offset + y) * canvas_width + x_offset) * 4;
        for x in 0..width {
            let dst = row + x * 4;
            canvas[dst..dst + 4].copy_from_slice(&rgba);
        }
    }
}

fn blend_channel(src: u8, src_alpha: u32, dst: u8, dst_factor_alpha: u32, scale: u32) -> u8 {
    let blended = (src as u32 * src_alpha + dst as u32 * dst_factor_alpha) * scale;
    (blended >> 24) as u8
}

fn blend_pixel_non_premult(src: [u8; 4], dst: [u8; 4]) -> [u8; 4] {
    let src_alpha = src[3] as u32;
    if src_alpha == 0 {
        return dst;
    }
    if src_alpha == 255 {
        return src;
    }

    let dst_alpha = dst[3] as u32;
    let dst_factor_alpha = (dst_alpha * (256 - src_alpha)) >> 8;
    let blend_alpha = src_alpha + dst_factor_alpha;
    let scale = (1u32 << 24) / blend_alpha;

    [
        blend_channel(src[0], src_alpha, dst[0], dst_factor_alpha, scale),
        blend_channel(src[1], src_alpha, dst[1], dst_factor_alpha, scale),
        blend_channel(src[2], src_alpha, dst[2], dst_factor_alpha, scale),
        blend_alpha as u8,
    ]
}

fn composite_frame(
    canvas: &mut [u8],
    canvas_width: usize,
    frame_rgba: &[u8],
    frame: &ParsedAnimationFrame<'_>,
) {
    for y in 0..frame.height {
        let src_row = y * frame.width * 4;
        let dst_row = ((frame.y_offset + y) * canvas_width + frame.x_offset) * 4;
        for x in 0..frame.width {
            let src = src_row + x * 4;
            let dst = dst_row + x * 4;
            if frame.blend {
                let src_pixel = [
                    frame_rgba[src],
                    frame_rgba[src + 1],
                    frame_rgba[src + 2],
                    frame_rgba[src + 3],
                ];
                let dst_pixel = [
                    canvas[dst],
                    canvas[dst + 1],
                    canvas[dst + 2],
                    canvas[dst + 3],
                ];
                let out = blend_pixel_non_premult(src_pixel, dst_pixel);
                canvas[dst..dst + 4].copy_from_slice(&out);
            } else {
                canvas[dst..dst + 4].copy_from_slice(&frame_rgba[src..src + 4]);
            }
        }
    }
}

fn decode_frame_image(frame: &ParsedAnimationFrame<'_>) -> Result<DecodedImage, DecoderError> {
    let image = match &frame.image_chunk.fourcc {
        b"VP8L" => {
            if frame.alpha_chunk.is_some() {
                return Err(DecoderError::Bitstream(
                    "VP8L animation frame must not carry ALPH chunk",
                ));
            }
            decode_lossless_vp8l_to_rgba(frame.image_data)?
        }
        b"VP8 " => {
            if frame.alpha_chunk.is_some() {
                return Err(DecoderError::Unsupported(
                    "VP8 animation frames with ALPH are not implemented",
                ));
            }
            decode_lossy_vp8_to_rgba(frame.image_data)?
        }
        _ => return Err(DecoderError::Bitstream("unsupported animation frame chunk")),
    };

    if image.width != frame.width || image.height != frame.height {
        return Err(DecoderError::Bitstream(
            "animation frame dimensions do not match bitstream",
        ));
    }
    Ok(image)
}

pub fn decode_animation_webp(data: &[u8]) -> Result<DecodedAnimation, DecoderError> {
    let parsed = parse_animation_webp(data)?;
    let background = argb_to_rgba(parsed.animation.background_color);
    let mut canvas = vec![0u8; parsed.features.width * parsed.features.height * 4];
    fill_rect(
        &mut canvas,
        parsed.features.width,
        0,
        0,
        parsed.features.width,
        parsed.features.height,
        background,
    );

    let mut previous_rect = None;
    let mut frames = Vec::with_capacity(parsed.frames.len());
    for frame in &parsed.frames {
        if let Some((x_offset, y_offset, width, height)) = previous_rect.take() {
            fill_rect(
                &mut canvas,
                parsed.features.width,
                x_offset,
                y_offset,
                width,
                height,
                background,
            );
        }

        let decoded = decode_frame_image(frame)?;
        composite_frame(&mut canvas, parsed.features.width, &decoded.rgba, frame);
        frames.push(DecodedAnimationFrame {
            duration: frame.duration,
            rgba: canvas.clone(),
        });

        if frame.dispose_to_background {
            previous_rect = Some((frame.x_offset, frame.y_offset, frame.width, frame.height));
        }
    }

    Ok(DecodedAnimation {
        width: parsed.features.width,
        height: parsed.features.height,
        background_color: parsed.animation.background_color,
        loop_count: parsed.animation.loop_count,
        frames,
    })
}

pub fn decode_animation_webp_to_bmp_frames(data: &[u8]) -> Result<Vec<Vec<u8>>, DecoderError> {
    let animation = decode_animation_webp(data)?;
    animation
        .frames
        .iter()
        .map(|frame| encode_bmp24_from_rgba(animation.width, animation.height, &frame.rgba))
        .collect()
}
