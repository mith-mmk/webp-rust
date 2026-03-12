use crate::decoder::vp8i::ALPHA_FLAG;
use crate::encoder::EncoderError;

const EXIF_FLAG: u32 = 0x0000_0008;

pub(crate) struct StillImageChunk<'a> {
    pub fourcc: [u8; 4],
    pub payload: &'a [u8],
    pub width: usize,
    pub height: usize,
    pub has_alpha: bool,
}

fn padded_len(size: usize) -> Result<usize, EncoderError> {
    size.checked_add(size & 1)
        .ok_or(EncoderError::InvalidParam("encoded output is too large"))
}

fn append_chunk(data: &mut Vec<u8>, fourcc: &[u8; 4], payload: &[u8]) -> Result<(), EncoderError> {
    let _ = u32::try_from(payload.len())
        .map_err(|_| EncoderError::InvalidParam("encoded output is too large"))?;
    data.extend_from_slice(fourcc);
    data.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    data.extend_from_slice(payload);
    if payload.len() & 1 == 1 {
        data.push(0);
    }
    Ok(())
}

fn extend_riff(body: Vec<u8>) -> Result<Vec<u8>, EncoderError> {
    let riff_size = u32::try_from(body.len())
        .map_err(|_| EncoderError::InvalidParam("encoded output is too large"))?;
    let mut data = Vec::with_capacity(8 + body.len());
    data.extend_from_slice(b"RIFF");
    data.extend_from_slice(&riff_size.to_le_bytes());
    data.extend_from_slice(&body);
    Ok(data)
}

fn encode_le24(value: usize) -> Result<[u8; 3], EncoderError> {
    let encoded = value.checked_sub(1).ok_or(EncoderError::InvalidParam(
        "image dimensions must be non-zero",
    ))?;
    if encoded > 0x00ff_ffff {
        return Err(EncoderError::InvalidParam(
            "image dimensions exceed VP8X limits",
        ));
    }
    Ok([
        (encoded & 0xff) as u8,
        ((encoded >> 8) & 0xff) as u8,
        ((encoded >> 16) & 0xff) as u8,
    ])
}

pub(crate) fn wrap_still_webp(
    image: StillImageChunk<'_>,
    exif: Option<&[u8]>,
) -> Result<Vec<u8>, EncoderError> {
    let padded_image_size = padded_len(image.payload.len())?;
    if exif.is_none() {
        let body_size = 4usize
            .checked_add(8)
            .and_then(|size| size.checked_add(padded_image_size))
            .ok_or(EncoderError::InvalidParam("encoded output is too large"))?;
        let mut body = Vec::with_capacity(body_size);
        body.extend_from_slice(b"WEBP");
        append_chunk(&mut body, &image.fourcc, image.payload)?;
        return extend_riff(body);
    }

    let exif = exif.unwrap();
    let vp8x_payload_size = 10usize;
    let padded_exif_size = padded_len(exif.len())?;
    let body_size = 4usize
        .checked_add(8 + vp8x_payload_size)
        .and_then(|size| size.checked_add(8 + padded_image_size))
        .and_then(|size| size.checked_add(8 + padded_exif_size))
        .ok_or(EncoderError::InvalidParam("encoded output is too large"))?;
    let mut body = Vec::with_capacity(body_size);
    body.extend_from_slice(b"WEBP");

    let mut flags = EXIF_FLAG;
    if image.has_alpha {
        flags |= ALPHA_FLAG;
    }
    let width = encode_le24(image.width)?;
    let height = encode_le24(image.height)?;
    let mut vp8x_payload = [0u8; 10];
    vp8x_payload[0..4].copy_from_slice(&flags.to_le_bytes());
    vp8x_payload[4..7].copy_from_slice(&width);
    vp8x_payload[7..10].copy_from_slice(&height);

    append_chunk(&mut body, b"VP8X", &vp8x_payload)?;
    append_chunk(&mut body, &image.fourcc, image.payload)?;
    append_chunk(&mut body, b"EXIF", exif)?;
    extend_riff(body)
}
