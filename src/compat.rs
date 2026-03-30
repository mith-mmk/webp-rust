//! Compatibility wrapper for callback-based decode flows used by `wml2`.
//!
//! This module intentionally mirrors the shape of the `wml2` draw-side API so
//! the WebP codec core can stay in `webp-rust` while callers keep a thin
//! adapter around their own image buffer and metadata model.

use crate::decoder::lossless::decode_lossless_webp_to_rgba;
use crate::decoder::lossy::{decode_lossy_vp8_frame_to_rgba, decode_lossy_webp_to_rgba};
use crate::decoder::{
    get_features, parse_animation_webp, DecodedImage, DecoderError, ParsedAnimationFrame,
    WebpFormat,
};
use bin_rs::io::read_u32_le;
use bin_rs::reader::BinaryReader;
use std::collections::HashMap;

type Error = Box<dyn std::error::Error>;

/// Metadata map used by the compatibility wrapper.
pub type Metadata = HashMap<String, DataMap>;

/// Minimal metadata value type used by the compatibility wrapper.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataMap {
    UInt(u64),
    UIntAllay(Vec<u64>),
    Raw(Vec<u8>),
    Ascii(String),
    None,
}

/// RGBA color value used by animation initialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RGBA {
    pub red: u8,
    pub green: u8,
    pub blue: u8,
    pub alpha: u8,
}

/// Callback response command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponseCommand {
    Abort,
    Continue,
}

/// Response returned by compatibility callbacks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CallbackResponse {
    pub response: ResponseCommand,
}

impl CallbackResponse {
    pub fn abort() -> Self {
        Self {
            response: ResponseCommand::Abort,
        }
    }

    pub fn cont() -> Self {
        Self {
            response: ResponseCommand::Continue,
        }
    }
}

/// Receives decoded image data from compatibility decode entry points.
pub trait DrawCallback: Sync + Send {
    fn init(
        &mut self,
        width: usize,
        height: usize,
        option: Option<InitOptions>,
    ) -> Result<Option<CallbackResponse>, Error>;
    fn draw(
        &mut self,
        start_x: usize,
        start_y: usize,
        width: usize,
        height: usize,
        data: &[u8],
        option: Option<DrawOptions>,
    ) -> Result<Option<CallbackResponse>, Error>;
    fn terminate(
        &mut self,
        term: Option<TerminateOptions>,
    ) -> Result<Option<CallbackResponse>, Error>;
    fn next(&mut self, next: Option<NextOptions>) -> Result<Option<CallbackResponse>, Error>;
    fn verbose(
        &mut self,
        verbose: &str,
        option: Option<VerboseOptions>,
    ) -> Result<Option<CallbackResponse>, Error>;
    fn set_metadata(
        &mut self,
        key: &str,
        value: DataMap,
    ) -> Result<Option<CallbackResponse>, Error>;
}

/// Decoder initialization options.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InitOptions {
    pub loop_count: u32,
    pub background: Option<RGBA>,
    pub animation: bool,
}

/// Draw options placeholder kept for shape compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DrawOptions {}

/// Termination options placeholder kept for shape compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TerminateOptions {}

/// Verbose options placeholder kept for shape compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VerboseOptions {}

/// Frame transition commands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NextOption {
    Continue,
    Next,
    Dispose,
    ClearAbort,
    Terminate,
}

/// Disposal mode for an animation frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NextDispose {
    None,
    Override,
    Background,
    Previous,
}

/// Blend mode for an animation frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NextBlend {
    Source,
    Override,
}

/// Destination rectangle for a frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageRect {
    pub start_x: i32,
    pub start_y: i32,
    pub width: usize,
    pub height: usize,
}

/// Per-frame transition options.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NextOptions {
    pub flag: NextOption,
    pub await_time: u64,
    pub image_rect: Option<ImageRect>,
    pub dispose_option: Option<NextDispose>,
    pub blend: Option<NextBlend>,
}

/// Decoder call options.
pub struct DecodeOptions<'a> {
    pub debug_flag: usize,
    pub drawer: &'a mut dyn DrawCallback,
    pub options: Option<Metadata>,
}

impl<'a> DecodeOptions<'a> {
    pub fn new(drawer: &'a mut dyn DrawCallback) -> Self {
        Self {
            debug_flag: 0,
            drawer,
            options: None,
        }
    }
}

fn argb_to_rgba(argb: u32) -> RGBA {
    RGBA {
        red: ((argb >> 16) & 0xff) as u8,
        green: ((argb >> 8) & 0xff) as u8,
        blue: (argb & 0xff) as u8,
        alpha: (argb >> 24) as u8,
    }
}

fn map_error(error: DecoderError) -> Error {
    Box::new(error)
}

fn read_container<B: BinaryReader>(reader: &mut B) -> Result<Vec<u8>, Error> {
    let header = reader.read_bytes_no_move(12)?;
    if header.len() < 12 || &header[0..4] != b"RIFF" || &header[8..12] != b"WEBP" {
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "not a WebP RIFF container",
        )));
    }

    let riff_size = read_u32_le(&header, 4) as usize;
    let total_size = riff_size + 8;
    if total_size < 12 {
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "invalid WebP container length",
        )));
    }

    Ok(reader.read_bytes_as_vec(total_size)?)
}

fn next_options(frame: &ParsedAnimationFrame<'_>) -> NextOptions {
    NextOptions {
        flag: NextOption::Continue,
        await_time: frame.duration as u64,
        image_rect: Some(ImageRect {
            start_x: frame.x_offset as i32,
            start_y: frame.y_offset as i32,
            width: frame.width,
            height: frame.height,
        }),
        dispose_option: Some(if frame.dispose_to_background {
            NextDispose::Background
        } else {
            NextDispose::None
        }),
        blend: Some(if frame.blend {
            NextBlend::Source
        } else {
            NextBlend::Override
        }),
    }
}

fn decode_frame_rgba(frame: &ParsedAnimationFrame<'_>) -> Result<DecodedImage, DecoderError> {
    let image = match &frame.image_chunk.fourcc {
        b"VP8L" => {
            if frame.alpha_chunk.is_some() {
                return Err(DecoderError::Bitstream(
                    "VP8L animation frame must not carry ALPH chunk",
                ));
            }
            crate::decoder::decode_lossless_vp8l_to_rgba(frame.image_data)?
        }
        b"VP8 " => decode_lossy_vp8_frame_to_rgba(frame.image_data, frame.alpha_data)?,
        _ => return Err(DecoderError::Bitstream("unsupported animation frame chunk")),
    };

    if image.width != frame.width || image.height != frame.height {
        return Err(DecoderError::Bitstream(
            "animation frame dimensions do not match bitstream",
        ));
    }
    Ok(image)
}

fn read_le32(bytes: &[u8]) -> usize {
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize
}

fn scan_chunks<'a>(data: &'a [u8]) -> Result<Vec<([u8; 4], &'a [u8])>, DecoderError> {
    if data.len() < 12 {
        return Err(DecoderError::NotEnoughData("RIFF header"));
    }
    if &data[0..4] != b"RIFF" || &data[8..12] != b"WEBP" {
        return Err(DecoderError::Bitstream("wrong RIFF WEBP signature"));
    }

    let riff_size = read_le32(&data[4..8]);
    let limit = riff_size + 8;
    if limit > data.len() {
        return Err(DecoderError::NotEnoughData("truncated RIFF payload"));
    }

    let mut offset = 12;
    let mut chunks = Vec::new();
    while offset + 8 <= limit {
        let size = read_le32(&data[offset + 4..offset + 8]);
        let padded_size = size + (size & 1);
        let chunk_end = offset + 8 + padded_size;
        if chunk_end > limit {
            return Err(DecoderError::NotEnoughData("chunk payload"));
        }

        let fourcc: [u8; 4] = data[offset..offset + 4]
            .try_into()
            .expect("valid fourcc slice");
        let payload = &data[offset + 8..offset + 8 + size];
        chunks.push((fourcc, payload));
        offset = chunk_end;
    }

    Ok(chunks)
}

fn webp_codec_name(format: WebpFormat, animated: bool) -> &'static str {
    if animated {
        "Animated"
    } else {
        match format {
            WebpFormat::Lossy => "Lossy",
            WebpFormat::Lossless => "Lossless",
            WebpFormat::Undefined => "Undefined",
        }
    }
}

fn make_metadata(data: &[u8]) -> Result<Metadata, DecoderError> {
    let features = get_features(data)?;
    let chunks = scan_chunks(data)?;
    let mut map = HashMap::new();

    map.insert("Format".to_string(), DataMap::Ascii("WEBP".to_string()));
    map.insert("width".to_string(), DataMap::UInt(features.width as u64));
    map.insert("height".to_string(), DataMap::UInt(features.height as u64));
    map.insert(
        "WebP codec".to_string(),
        DataMap::Ascii(webp_codec_name(features.format, features.has_animation).to_string()),
    );
    map.insert(
        "WebP has alpha".to_string(),
        DataMap::Ascii(features.has_alpha.to_string()),
    );
    map.insert(
        "WebP animated".to_string(),
        DataMap::Ascii(features.has_animation.to_string()),
    );

    if let Some(vp8x) = features.vp8x {
        map.insert(
            "canvas width".to_string(),
            DataMap::UInt(vp8x.canvas_width as u64),
        );
        map.insert(
            "canvas height".to_string(),
            DataMap::UInt(vp8x.canvas_height as u64),
        );
    }

    if features.has_animation {
        let parsed = parse_animation_webp(data)?;
        map.insert(
            "Animation frames".to_string(),
            DataMap::UInt(parsed.frames.len() as u64),
        );
        map.insert(
            "Animation loop count".to_string(),
            DataMap::UInt(parsed.animation.loop_count as u64),
        );
        map.insert(
            "Animation background color".to_string(),
            DataMap::UInt(parsed.animation.background_color as u64),
        );
        map.insert(
            "Animation frame durations".to_string(),
            DataMap::UIntAllay(
                parsed
                    .frames
                    .iter()
                    .map(|frame| frame.duration as u64)
                    .collect(),
            ),
        );
    }

    for (fourcc, payload) in chunks {
        match &fourcc {
            b"ICCP" => {
                map.insert("ICC Profile".to_string(), DataMap::Raw(payload.to_vec()));
            }
            b"EXIF" => {
                map.insert("EXIF Raw".to_string(), DataMap::Raw(payload.to_vec()));
            }
            b"XMP " => match String::from_utf8(payload.to_vec()) {
                Ok(xmp) => {
                    map.insert("XMP".to_string(), DataMap::Ascii(xmp));
                }
                Err(_) => {
                    map.insert("XMP Raw".to_string(), DataMap::Raw(payload.to_vec()));
                }
            },
            _ => {}
        }
    }

    Ok(map)
}

/// Decodes a WebP image using a callback-based interface compatible with
/// `wml2`'s draw-side flow.
pub fn decode<B: BinaryReader>(
    reader: &mut B,
    option: &mut DecodeOptions<'_>,
) -> Result<(), Error> {
    let data = read_container(reader)?;
    let metadata = make_metadata(&data).map_err(map_error)?;
    let features = get_features(&data).map_err(map_error)?;

    if features.has_animation {
        let parsed = parse_animation_webp(&data).map_err(map_error)?;
        let init = InitOptions {
            loop_count: parsed.animation.loop_count as u32,
            background: Some(argb_to_rgba(parsed.animation.background_color)),
            animation: true,
        };
        option
            .drawer
            .init(parsed.features.width, parsed.features.height, Some(init))?;

        let mut allow_multi_image = false;
        for (index, frame) in parsed.frames.iter().enumerate() {
            let decoded = decode_frame_rgba(frame).map_err(map_error)?;
            if index == 0 {
                option.drawer.draw(
                    frame.x_offset,
                    frame.y_offset,
                    frame.width,
                    frame.height,
                    &decoded.rgba,
                    None,
                )?;

                let result = option.drawer.next(Some(next_options(frame)))?;
                if let Some(response) = result {
                    if response.response == ResponseCommand::Continue {
                        allow_multi_image = true;
                        option
                            .drawer
                            .draw(0, 0, frame.width, frame.height, &decoded.rgba, None)?;
                    }
                }
                continue;
            }

            if !allow_multi_image {
                continue;
            }

            let result = option.drawer.next(Some(next_options(frame)))?;
            if let Some(response) = result {
                if response.response == ResponseCommand::Abort {
                    break;
                }
            }

            option
                .drawer
                .draw(0, 0, frame.width, frame.height, &decoded.rgba, None)?;
        }
    } else {
        let init = InitOptions {
            loop_count: 0,
            background: None,
            animation: false,
        };
        option
            .drawer
            .init(features.width, features.height, Some(init))?;

        let decoded = match features.format {
            WebpFormat::Lossy => decode_lossy_webp_to_rgba(&data).map_err(map_error)?,
            WebpFormat::Lossless => decode_lossless_webp_to_rgba(&data).map_err(map_error)?,
            WebpFormat::Undefined => {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::Unsupported,
                    "unsupported WebP format",
                )));
            }
        };

        option
            .drawer
            .draw(0, 0, decoded.width, decoded.height, &decoded.rgba, None)?;
    }

    for (key, value) in metadata {
        option.drawer.set_metadata(&key, value)?;
    }
    option.drawer.terminate(None)?;

    Ok(())
}
