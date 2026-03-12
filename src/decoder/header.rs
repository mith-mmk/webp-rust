//! WebP RIFF container and chunk parsing helpers.

use crate::decoder::alpha::{parse_alpha_header, AlphaHeader};
use crate::decoder::vp8::{get_info, get_lossless_info};
use crate::decoder::vp8i::{
    WebpFormat, ALPHA_FLAG, ANIMATION_FLAG, CHUNK_HEADER_SIZE, MAX_CHUNK_PAYLOAD, MAX_IMAGE_AREA,
    RIFF_HEADER_SIZE, TAG_SIZE, VP8X_CHUNK_SIZE,
};
use crate::decoder::DecoderError;

/// Common metadata for a RIFF chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkHeader {
    /// FourCC tag.
    pub fourcc: [u8; 4],
    /// Chunk start offset in the source buffer.
    pub offset: usize,
    /// Unpadded payload size.
    pub size: usize,
    /// Payload size including RIFF padding.
    pub padded_size: usize,
    /// Start offset of the chunk payload.
    pub data_offset: usize,
}

/// Parsed `VP8X` extended header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Vp8xHeader {
    /// Raw feature flags from the header.
    pub flags: u32,
    /// Canvas width in pixels.
    pub canvas_width: usize,
    /// Canvas height in pixels.
    pub canvas_height: usize,
}

/// High-level image features derived from the container and bitstream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WebpFeatures {
    /// Image or canvas width in pixels.
    pub width: usize,
    /// Image or canvas height in pixels.
    pub height: usize,
    /// Whether alpha is present.
    pub has_alpha: bool,
    /// Whether the container is animated.
    pub has_animation: bool,
    /// Underlying still-image codec kind.
    pub format: WebpFormat,
    /// Optional extended header.
    pub vp8x: Option<Vp8xHeader>,
}

/// Parsed still-image WebP container.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParsedWebp<'a> {
    /// High-level image features.
    pub features: WebpFeatures,
    /// RIFF size field when the input is RIFF-wrapped.
    pub riff_size: Option<usize>,
    /// Primary image chunk header.
    pub image_chunk: ChunkHeader,
    /// Primary image payload (`VP8 ` or `VP8L`).
    pub image_data: &'a [u8],
    /// Optional `ALPH` chunk header.
    pub alpha_chunk: Option<ChunkHeader>,
    /// Optional `ALPH` payload.
    pub alpha_data: Option<&'a [u8]>,
    /// Optional parsed `ALPH` header byte.
    pub alpha_header: Option<AlphaHeader>,
}

/// Parsed `ANIM` chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnimationHeader {
    /// Canvas background color in little-endian ARGB order.
    pub background_color: u32,
    /// Loop count from the container. `0` means infinite loop.
    pub loop_count: u16,
}

/// Parsed animation frame entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParsedAnimationFrame<'a> {
    /// Enclosing `ANMF` chunk header.
    pub frame_chunk: ChunkHeader,
    /// X offset on the canvas in pixels.
    pub x_offset: usize,
    /// Y offset on the canvas in pixels.
    pub y_offset: usize,
    /// Frame width in pixels.
    pub width: usize,
    /// Frame height in pixels.
    pub height: usize,
    /// Display duration in milliseconds.
    pub duration: usize,
    /// Whether the frame should be alpha-blended.
    pub blend: bool,
    /// Whether the frame should be disposed to background.
    pub dispose_to_background: bool,
    /// Embedded `VP8 ` or `VP8L` image chunk.
    pub image_chunk: ChunkHeader,
    /// Embedded image payload.
    pub image_data: &'a [u8],
    /// Optional embedded `ALPH` chunk header.
    pub alpha_chunk: Option<ChunkHeader>,
    /// Optional embedded `ALPH` payload.
    pub alpha_data: Option<&'a [u8]>,
    /// Optional parsed `ALPH` header byte.
    pub alpha_header: Option<AlphaHeader>,
}

/// Parsed animated WebP container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedAnimationWebp<'a> {
    /// High-level canvas features.
    pub features: WebpFeatures,
    /// RIFF size field.
    pub riff_size: Option<usize>,
    /// Global animation settings.
    pub animation: AnimationHeader,
    /// Parsed animation frames in display order.
    pub frames: Vec<ParsedAnimationFrame<'a>>,
}

fn read_le24(bytes: &[u8]) -> usize {
    bytes[0] as usize | ((bytes[1] as usize) << 8) | ((bytes[2] as usize) << 16)
}

fn read_le32(bytes: &[u8]) -> usize {
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize
}

fn read_le16(bytes: &[u8]) -> u16 {
    u16::from_le_bytes([bytes[0], bytes[1]])
}

fn padded_payload_size(size: usize) -> usize {
    size + (size & 1)
}

fn parse_chunk(
    data: &[u8],
    offset: usize,
    riff_limit: Option<usize>,
) -> Result<ChunkHeader, DecoderError> {
    if data.len() < offset + CHUNK_HEADER_SIZE {
        return Err(DecoderError::NotEnoughData("chunk header"));
    }
    let size = read_le32(&data[offset + TAG_SIZE..offset + CHUNK_HEADER_SIZE]);
    if size > MAX_CHUNK_PAYLOAD {
        return Err(DecoderError::Bitstream("invalid chunk size"));
    }

    let padded_size = padded_payload_size(size);
    let total_size = CHUNK_HEADER_SIZE + padded_size;
    let end = offset + total_size;
    if let Some(limit) = riff_limit {
        if end > limit {
            return Err(DecoderError::Bitstream("chunk exceeds RIFF payload"));
        }
    }
    if data.len() < end {
        return Err(DecoderError::NotEnoughData("chunk payload"));
    }

    Ok(ChunkHeader {
        fourcc: data[offset..offset + TAG_SIZE].try_into().unwrap(),
        offset,
        size,
        padded_size,
        data_offset: offset + CHUNK_HEADER_SIZE,
    })
}

fn parse_riff(data: &[u8]) -> Result<(Option<usize>, usize), DecoderError> {
    if data.len() < RIFF_HEADER_SIZE {
        return Err(DecoderError::NotEnoughData("RIFF header"));
    }
    if &data[..4] != b"RIFF" {
        return Ok((None, 0));
    }
    if &data[8..12] != b"WEBP" {
        return Err(DecoderError::Bitstream("wrong RIFF WEBP signature"));
    }

    let riff_size = read_le32(&data[4..8]);
    if riff_size < TAG_SIZE + CHUNK_HEADER_SIZE {
        return Err(DecoderError::Bitstream("RIFF payload is too small"));
    }
    if riff_size > MAX_CHUNK_PAYLOAD {
        return Err(DecoderError::Bitstream("RIFF payload is too large"));
    }
    if riff_size > data.len() - CHUNK_HEADER_SIZE {
        return Err(DecoderError::NotEnoughData("truncated RIFF payload"));
    }

    Ok((Some(riff_size), RIFF_HEADER_SIZE))
}

fn parse_vp8x(data: &[u8], offset: usize) -> Result<(Option<Vp8xHeader>, usize), DecoderError> {
    if data.len() < offset + CHUNK_HEADER_SIZE {
        return Ok((None, offset));
    }
    if &data[offset..offset + TAG_SIZE] != b"VP8X" {
        return Ok((None, offset));
    }

    let chunk = parse_chunk(data, offset, None)?;
    if chunk.size != VP8X_CHUNK_SIZE {
        return Err(DecoderError::Bitstream("wrong VP8X chunk size"));
    }

    let flags = read_le32(&data[offset + 8..offset + 12]) as u32;
    let canvas_width = read_le24(&data[offset + 12..offset + 15]) + 1;
    let canvas_height = read_le24(&data[offset + 15..offset + 18]) + 1;
    if (canvas_width as u64) * (canvas_height as u64) >= MAX_IMAGE_AREA {
        return Err(DecoderError::Bitstream("canvas is too large"));
    }

    Ok((
        Some(Vp8xHeader {
            flags,
            canvas_width,
            canvas_height,
        }),
        offset + CHUNK_HEADER_SIZE + chunk.padded_size,
    ))
}

/// Returns high-level WebP features without fully decoding the image.
pub fn get_features(data: &[u8]) -> Result<WebpFeatures, DecoderError> {
    let (riff_size, mut offset) = parse_riff(data)?;
    let riff_limit = riff_size.map(|size| size + CHUNK_HEADER_SIZE);

    let (vp8x, next_offset) = parse_vp8x(data, offset)?;
    offset = next_offset;
    if riff_size.is_none() && vp8x.is_some() {
        return Err(DecoderError::Bitstream("VP8X chunk requires RIFF"));
    }

    let mut has_alpha = vp8x
        .map(|chunk| (chunk.flags & ALPHA_FLAG) != 0)
        .unwrap_or(false);
    let has_animation = vp8x
        .map(|chunk| (chunk.flags & ANIMATION_FLAG) != 0)
        .unwrap_or(false);

    if let Some(vp8x) = vp8x {
        if has_animation {
            return Ok(WebpFeatures {
                width: vp8x.canvas_width,
                height: vp8x.canvas_height,
                has_alpha,
                has_animation,
                format: WebpFormat::Undefined,
                vp8x: Some(vp8x),
            });
        }
    }

    if data.len() < offset + TAG_SIZE {
        return Err(DecoderError::NotEnoughData("chunk tag"));
    }

    if (riff_size.is_some() && vp8x.is_some())
        || (riff_size.is_none() && vp8x.is_none() && &data[offset..offset + TAG_SIZE] == b"ALPH")
    {
        loop {
            let chunk = parse_chunk(data, offset, riff_limit)?;
            if &chunk.fourcc == b"VP8 " || &chunk.fourcc == b"VP8L" {
                break;
            }
            if &chunk.fourcc == b"ALPH" {
                has_alpha = true;
            }
            offset += CHUNK_HEADER_SIZE + chunk.padded_size;
        }
    }

    let chunk = parse_chunk(data, offset, riff_limit)?;
    let payload = &data[chunk.data_offset..chunk.data_offset + chunk.size];
    let (format, width, height) = if &chunk.fourcc == b"VP8 " {
        let (width, height) = get_info(payload, chunk.size)?;
        (WebpFormat::Lossy, width, height)
    } else if &chunk.fourcc == b"VP8L" {
        let info = get_lossless_info(payload)?;
        has_alpha |= info.has_alpha;
        (WebpFormat::Lossless, info.width, info.height)
    } else {
        return Err(DecoderError::Bitstream("missing VP8/VP8L image chunk"));
    };

    if let Some(vp8x) = vp8x {
        if vp8x.canvas_width != width || vp8x.canvas_height != height {
            return Err(DecoderError::Bitstream(
                "VP8X canvas does not match image size",
            ));
        }
    }

    Ok(WebpFeatures {
        width,
        height,
        has_alpha,
        has_animation,
        format,
        vp8x,
    })
}

/// Parses a still-image WebP container and returns raw chunk slices.
pub fn parse_still_webp(data: &[u8]) -> Result<ParsedWebp<'_>, DecoderError> {
    let (riff_size, mut offset) = parse_riff(data)?;
    let riff_limit = riff_size.map(|size| size + CHUNK_HEADER_SIZE);

    let (vp8x, next_offset) = parse_vp8x(data, offset)?;
    offset = next_offset;
    if riff_size.is_none() && vp8x.is_some() {
        return Err(DecoderError::Bitstream("VP8X chunk requires RIFF"));
    }
    if vp8x
        .map(|chunk| (chunk.flags & ANIMATION_FLAG) != 0)
        .unwrap_or(false)
    {
        return Err(DecoderError::Unsupported(
            "animated WebP is not implemented",
        ));
    }

    let mut alpha_chunk = None;
    if data.len() < offset + TAG_SIZE {
        return Err(DecoderError::NotEnoughData("chunk tag"));
    }
    if (riff_size.is_some() && vp8x.is_some())
        || (riff_size.is_none() && vp8x.is_none() && &data[offset..offset + TAG_SIZE] == b"ALPH")
    {
        loop {
            let chunk = parse_chunk(data, offset, riff_limit)?;
            if &chunk.fourcc == b"VP8 " || &chunk.fourcc == b"VP8L" {
                break;
            }
            if &chunk.fourcc == b"ALPH" {
                alpha_chunk = Some(chunk);
            }
            offset += CHUNK_HEADER_SIZE + chunk.padded_size;
        }
    }

    let image_chunk = parse_chunk(data, offset, riff_limit)?;
    if &image_chunk.fourcc != b"VP8 " && &image_chunk.fourcc != b"VP8L" {
        return Err(DecoderError::Bitstream("missing VP8/VP8L image chunk"));
    }
    let image_data = &data[image_chunk.data_offset..image_chunk.data_offset + image_chunk.size];
    let mut features = get_features(data)?;
    let alpha_data =
        alpha_chunk.map(|chunk| &data[chunk.data_offset..chunk.data_offset + chunk.size]);
    let alpha_header = alpha_data.map(parse_alpha_header).transpose()?;
    if alpha_chunk.is_some() {
        features.has_alpha = true;
    }

    Ok(ParsedWebp {
        features,
        riff_size,
        image_chunk,
        image_data,
        alpha_chunk,
        alpha_data,
        alpha_header,
    })
}

fn parse_animation_frame<'a>(
    data: &'a [u8],
    features: WebpFeatures,
    chunk: ChunkHeader,
    riff_limit: Option<usize>,
) -> Result<ParsedAnimationFrame<'a>, DecoderError> {
    if chunk.size < 16 {
        return Err(DecoderError::Bitstream("ANMF chunk is too small"));
    }

    let header = &data[chunk.data_offset..chunk.data_offset + 16];
    let x_offset = read_le24(&header[0..3]) * 2;
    let y_offset = read_le24(&header[3..6]) * 2;
    let width = read_le24(&header[6..9]) + 1;
    let height = read_le24(&header[9..12]) + 1;
    let duration = read_le24(&header[12..15]);
    let flags = header[15];
    if flags >> 2 != 0 {
        return Err(DecoderError::Bitstream("ANMF reserved bits must be zero"));
    }
    if x_offset + width > features.width || y_offset + height > features.height {
        return Err(DecoderError::Bitstream(
            "ANMF frame exceeds animation canvas",
        ));
    }

    let mut offset = chunk.data_offset + 16;
    let frame_limit = Some(chunk.data_offset + chunk.size);
    let mut alpha_chunk = None;
    let image_chunk;
    loop {
        let subchunk = parse_chunk(data, offset, frame_limit)?;
        if &subchunk.fourcc == b"VP8 " || &subchunk.fourcc == b"VP8L" {
            image_chunk = subchunk;
            break;
        }
        if &subchunk.fourcc == b"ALPH" {
            alpha_chunk = Some(subchunk);
        }
        offset += CHUNK_HEADER_SIZE + subchunk.padded_size;
        if let Some(limit) = riff_limit {
            if offset > limit {
                return Err(DecoderError::Bitstream(
                    "ANMF frame data exceeds RIFF payload",
                ));
            }
        }
    }

    let image_data = &data[image_chunk.data_offset..image_chunk.data_offset + image_chunk.size];
    let alpha_data = alpha_chunk
        .map(|subchunk| &data[subchunk.data_offset..subchunk.data_offset + subchunk.size]);
    let alpha_header = alpha_data.map(parse_alpha_header).transpose()?;

    Ok(ParsedAnimationFrame {
        frame_chunk: chunk,
        x_offset,
        y_offset,
        width,
        height,
        duration,
        blend: (flags & 0x02) == 0,
        dispose_to_background: (flags & 0x01) != 0,
        image_chunk,
        image_data,
        alpha_chunk,
        alpha_data,
        alpha_header,
    })
}

/// Parses an animated WebP container and returns frame-level chunk slices.
pub fn parse_animation_webp(data: &[u8]) -> Result<ParsedAnimationWebp<'_>, DecoderError> {
    let (riff_size, mut offset) = parse_riff(data)?;
    let riff_limit = riff_size.map(|size| size + CHUNK_HEADER_SIZE);

    let (vp8x, next_offset) = parse_vp8x(data, offset)?;
    offset = next_offset;
    let vp8x = vp8x.ok_or(DecoderError::Bitstream("animated WebP requires VP8X"))?;
    if (vp8x.flags & ANIMATION_FLAG) == 0 {
        return Err(DecoderError::Unsupported("animated WebP flag is not set"));
    }

    let anim_chunk = parse_chunk(data, offset, riff_limit)?;
    if &anim_chunk.fourcc != b"ANIM" {
        return Err(DecoderError::Bitstream("missing ANIM chunk"));
    }
    if anim_chunk.size != 6 {
        return Err(DecoderError::Bitstream("wrong ANIM chunk size"));
    }
    let animation = AnimationHeader {
        background_color: u32::from_le_bytes(
            data[anim_chunk.data_offset..anim_chunk.data_offset + 4]
                .try_into()
                .unwrap(),
        ),
        loop_count: read_le16(&data[anim_chunk.data_offset + 4..anim_chunk.data_offset + 6]),
    };
    offset += CHUNK_HEADER_SIZE + anim_chunk.padded_size;

    let features = WebpFeatures {
        width: vp8x.canvas_width,
        height: vp8x.canvas_height,
        has_alpha: (vp8x.flags & ALPHA_FLAG) != 0,
        has_animation: true,
        format: WebpFormat::Undefined,
        vp8x: Some(vp8x),
    };

    let limit = riff_limit.unwrap_or(data.len());
    let mut frames = Vec::new();
    while offset + CHUNK_HEADER_SIZE <= limit {
        let chunk = parse_chunk(data, offset, riff_limit)?;
        if &chunk.fourcc != b"ANMF" {
            break;
        }
        let frame = parse_animation_frame(data, features, chunk, riff_limit)?;
        frames.push(frame);
        offset += CHUNK_HEADER_SIZE + chunk.padded_size;
    }

    if frames.is_empty() {
        return Err(DecoderError::Bitstream("animated WebP has no ANMF frames"));
    }

    Ok(ParsedAnimationWebp {
        features,
        riff_size,
        animation,
        frames,
    })
}
