use crate::decoder::alpha::{parse_alpha_header, AlphaHeader};
use crate::decoder::vp8::{get_info, get_lossless_info};
use crate::decoder::vp8i::{
    WebpFormat, ALPHA_FLAG, ANIMATION_FLAG, CHUNK_HEADER_SIZE, MAX_CHUNK_PAYLOAD, MAX_IMAGE_AREA,
    RIFF_HEADER_SIZE, TAG_SIZE, VP8X_CHUNK_SIZE,
};
use crate::decoder::DecoderError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkHeader {
    pub fourcc: [u8; 4],
    pub offset: usize,
    pub size: usize,
    pub padded_size: usize,
    pub data_offset: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Vp8xHeader {
    pub flags: u32,
    pub canvas_width: usize,
    pub canvas_height: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WebpFeatures {
    pub width: usize,
    pub height: usize,
    pub has_alpha: bool,
    pub has_animation: bool,
    pub format: WebpFormat,
    pub vp8x: Option<Vp8xHeader>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParsedWebp<'a> {
    pub features: WebpFeatures,
    pub riff_size: Option<usize>,
    pub image_chunk: ChunkHeader,
    pub image_data: &'a [u8],
    pub alpha_chunk: Option<ChunkHeader>,
    pub alpha_data: Option<&'a [u8]>,
    pub alpha_header: Option<AlphaHeader>,
}

fn read_le24(bytes: &[u8]) -> usize {
    bytes[0] as usize | ((bytes[1] as usize) << 8) | ((bytes[2] as usize) << 16)
}

fn read_le32(bytes: &[u8]) -> usize {
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize
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
