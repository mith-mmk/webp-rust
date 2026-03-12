//! webp-rust is a pure Rust WebP decoder with a small lossless encoder.
//! This library is based on the official WebP codec in C. It supports lossy
//! and lossless WebP decode, animated WebP decode, and still-image lossless
//! `VP8L` encode from RGBA input.
//!
//! The crate exposes a small top-level API for decoding still images to RGBA
//! and a lower-level [`decoder`] module for container parsing and animation
//! decoding.

type Error = Box<dyn std::error::Error>;
use bin_rs::reader::{BinaryReader, BytesReader};

/// Lower-level WebP parsing and decoding APIs.
pub mod decoder;
/// Lossless WebP encoder APIs.
pub mod encoder;

pub use encoder::{
    encode_lossless_image_to_webp, encode_lossless_rgba_to_vp8l, encode_lossless_rgba_to_webp,
    EncoderError,
};

const MB_FEATURE_TREE_PROBS: usize = 3;
const NUM_MB_SEGMENTS: usize = 4;

pub(crate) struct BitReader {
    pub buffer: Vec<u8>,
    ptr: usize,
    left_bits: usize,
    last_byte: u32,
    warning: bool,
}

impl BitReader {
    pub fn new(data: &[u8]) -> Self {
        let this = Self {
            buffer: data.to_vec(),
            last_byte: 0,
            ptr: 0,
            left_bits: 0,
            warning: false,
        };
        this
    }

    fn look_bits(&mut self, size: usize) -> Result<usize, Error> {
        while self.left_bits <= 24 {
            if self.ptr >= self.buffer.len() {
                if self.left_bits <= 8 && self.left_bits < size {
                    self.warning = true;
                    if size >= 12 {
                        return Ok(0x1); // send EOL
                    } else {
                        return Ok(0x0);
                    }
                }
            }

            self.last_byte = (self.last_byte << 8) | (self.buffer[self.ptr] as u32);
            self.ptr += 1;
            self.left_bits += 8;
        }

        let bits = (self.last_byte >> (self.left_bits - size)) & ((1 << size) - 1);

        Ok(bits as usize)
    }

    fn skip_bits(&mut self, size: usize) {
        if self.left_bits > size {
            self.left_bits -= size;
        } else {
            let r = self.look_bits(size);
            if r.is_ok() {
                self.left_bits -= size;
            } else {
                self.left_bits = 0;
            }
        }
    }

    fn get_bits(&mut self, size: usize) -> Result<usize, Error> {
        let bits = self.look_bits(size);
        self.skip_bits(size);
        bits
    }

    fn get_signed_bits(&mut self, size: usize) -> Result<isize, Error> {
        let bits = self.get_bits(size - 1)? as isize;
        let sign = self.get_bits(1)?;
        if sign == 1 {
            Ok(bits)
        } else {
            Ok(-bits)
        }
    }
}

// Paragraph 14.1
#[allow(dead_code)]
const DC_TABLE: [u8; 128] = [
    4, 5, 6, 7, 8, 9, 10, 10, 11, 12, 13, 14, 15, 16, 17, 17, 18, 19, 20, 20, 21, 21, 22, 22, 23,
    23, 24, 25, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 37, 38, 39, 40, 41, 42, 43, 44,
    45, 46, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
    68, 69, 70, 71, 72, 73, 74, 75, 76, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 91,
    93, 95, 96, 98, 100, 101, 102, 104, 106, 108, 110, 112, 114, 116, 118, 122, 124, 126, 128, 130,
    132, 134, 136, 138, 140, 143, 145, 148, 151, 154, 157,
];

#[allow(dead_code)]
const AC_TABLE: [u16; 128] = [
    4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
    29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
    53, 54, 55, 56, 57, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94,
    96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 119, 122, 125, 128, 131, 134, 137, 140,
    143, 146, 149, 152, 155, 158, 161, 164, 167, 170, 173, 177, 181, 185, 189, 193, 197, 201, 205,
    209, 213, 217, 221, 225, 229, 234, 239, 245, 249, 254, 259, 264, 269, 274, 279, 284,
];

/// Global animation parameters stored in the `ANIM` chunk.
pub struct AnimationControl {
    /// Canvas background color in little-endian ARGB order.
    pub backgroud_color: u32,
    /// Loop count from the container. `0` means infinite loop.
    pub loop_count: u16,
}

/// RGBA pixel buffer for a decoded still image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageBuffer {
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
    /// Packed RGBA8 pixels in row-major order.
    pub rgba: Vec<u8>,
}

impl ImageBuffer {
    /// Returns the image width in pixels.
    pub fn get_width(self) -> usize {
        self.width
    }

    /// Returns the image height in pixels.
    pub fn get_height(self) -> usize {
        self.height
    }

    /// Returns the packed RGBA8 buffer.
    pub fn buffer(self) -> Vec<u8> {
        self.rgba.clone()
    }
}

/// One animation frame entry parsed from an `ANMF` chunk.
pub struct AnimationFrame {
    /// Frame x offset on the animation canvas in pixels.
    pub frame_x: usize,
    /// Frame y offset on the animation canvas in pixels.
    pub frame_y: usize,
    /// Frame width in pixels.
    pub width: usize,
    /// Frame height in pixels.
    pub height: usize,
    /// Frame duration in milliseconds.
    pub duration: usize,
    /// Whether the frame should be alpha-blended onto the canvas.
    pub alpha_blending: bool,
    /// Whether the frame should be disposed to background after display.
    pub disopse: bool,
    /// Raw `VP8 ` or `VP8L` frame payload.
    pub frame: Vec<u8>,
    /// Optional raw `ALPH` payload associated with the frame.
    pub alpha: Option<Vec<u8>>,
}

/// Container-level metadata returned by [`read_header`].
pub struct WebpHeader {
    /// Image width for still images.
    pub width: usize,
    /// Image height for still images.
    pub height: usize,
    /// Canvas width from `VP8X`, when present.
    pub canvas_width: usize,
    /// Canvas height from `VP8X`, when present.
    pub canvas_height: usize,
    /// Encoded size of the primary image chunk.
    pub image_chunksize: usize,
    /// Whether an ICC profile is advertised.
    pub has_icc_profile: bool,
    /// Whether alpha is advertised or present.
    pub has_alpha: bool,
    /// Whether EXIF metadata is advertised.
    pub has_exif: bool,
    /// Whether XMP metadata is advertised.
    pub has_xmp: bool,
    /// Whether animation is advertised.
    pub has_animation: bool,
    /// `true` for `VP8 `, `false` for `VP8L`.
    pub lossy: bool,
    /// Raw primary image payload.
    pub image: Vec<u8>,
    /// Optional ICC profile payload.
    pub icc_profile: Option<Vec<u8>>,
    /// Optional still-image `ALPH` payload.
    pub alpha: Option<Vec<u8>>,
    /// Optional EXIF payload.
    pub exif: Option<Vec<u8>>,
    /// Optional XMP payload.
    pub xmp: Option<Vec<u8>>,
    /// Optional animation control block.
    pub animation: Option<AnimationControl>,
    /// Optional parsed animation frame entries.
    pub animation_frame: Option<Vec<AnimationFrame>>,
}

impl WebpHeader {
    pub fn new() -> Self {
        Self {
            width: 0,
            height: 0,
            canvas_width: 0,
            canvas_height: 0,
            image_chunksize: 0,
            has_icc_profile: false,
            has_alpha: false,
            has_exif: false,
            has_xmp: false,
            has_animation: false,
            lossy: false,
            image: vec![],
            icc_profile: None,
            exif: None,
            alpha: None,
            xmp: None,
            animation: None,
            animation_frame: None,
        }
    }
}

/// Reads a 24-bit little-endian integer from a [`BinaryReader`].
pub fn read_u24<B: BinaryReader>(reader: &mut B) -> Result<u32, Error> {
    let mut b = [0_u8; 3];
    reader.read_bytes(&mut b)?;
    let val = (b[0] as u32) | ((b[1] as u32) << 8) | ((b[2] as u32) << 16);
    Ok(val)
}

/// Decodes a still WebP image from memory into an RGBA buffer.
///
/// Animated WebP is rejected by this helper. Use
/// [`decoder::decode_animation_webp`] for animated input.
pub fn image_from_bytes(data: &[u8]) -> Result<ImageBuffer, decoder::DecoderError> {
    let features = decoder::get_features(data)?;
    if features.has_animation {
        return Err(decoder::DecoderError::Unsupported(
            "animated WebP requires animation decoder API",
        ));
    }

    let image = match features.format {
        decoder::WebpFormat::Lossy => decoder::decode_lossy_webp_to_rgba(data)?,
        decoder::WebpFormat::Lossless => decoder::decode_lossless_webp_to_rgba(data)?,
        decoder::WebpFormat::Undefined => {
            return Err(decoder::DecoderError::Unsupported(
                "unsupported WebP format",
            ))
        }
    };

    Ok(ImageBuffer {
        width: image.width,
        height: image.height,
        rgba: image.rgba,
    })
}

#[cfg(not(target_family = "wasm"))]
/// Reads a still WebP image from disk and decodes it to RGBA.
pub fn image_from_file(filename: String) -> Result<ImageBuffer, Error> {
    let data = std::fs::read(filename)?;
    Ok(image_from_bytes(&data)?)
}

fn parse_animation_frame_payload(data: &[u8]) -> Result<(Vec<u8>, Option<Vec<u8>>), Error> {
    let mut reader = BytesReader::from_vec(data.to_vec());
    let mut frame = None;
    let mut alpha = None;

    while (reader.offset()? as usize) + 8 <= data.len() {
        let chunk_id = reader.read_ascii_string(4)?;
        let size = reader.read_u32_le()? as usize;
        let chunk = reader.read_bytes_as_vec(size)?;
        match chunk_id.as_str() {
            "ALPH" => alpha = Some(chunk),
            "VP8 " | "VP8L" => {
                frame = Some(chunk);
                break;
            }
            _ => {}
        }
        if size & 1 == 1 && (reader.offset()? as usize) < data.len() {
            reader.skip_ptr(1)?;
        }
    }

    frame
        .map(|frame| (frame, alpha))
        .ok_or_else(|| Box::new(std::io::Error::from(std::io::ErrorKind::Other)) as Error)
}

// RIFF Decode
/// Parses the RIFF container and returns raw chunk-oriented metadata.
///
/// This is a legacy low-level parser. For actual pixel decode, prefer
/// [`image_from_bytes`] or the functions in [`decoder`].
pub fn read_header<B: BinaryReader>(reader: &mut B) -> Result<WebpHeader, Error> {
    let riff = reader.read_ascii_string(4)?;
    if riff != "RIFF" {
        //Error
        return Err(Box::new(std::io::Error::from(std::io::ErrorKind::Other)));
    }
    let mut cksize = reader.read_u32_le()? as usize;
    let webp = reader.read_ascii_string(4)?;
    if webp != "WEBP" {
        //Error
        println!("This is not WEBP");
        return Err(Box::new(std::io::Error::from(std::io::ErrorKind::Other)));
    }
    cksize -= 4;
    let mut webp_header = WebpHeader::new();

    loop {
        let vp8: &str = &reader.read_ascii_string(4)?;
        println!("{}", vp8);
        let size = reader.read_u32_le()? as usize;
        let padded_size = size + (size & 1);
        match vp8 {
            "VP8 " => {
                println!("This is Lossy");
                webp_header.lossy = true;
                webp_header.image_chunksize = size;
                // Pragraph 9.1 Uncompressed Data Chunk
                let buf = reader.read_bytes_as_vec(size)?;
                let flags = buf[0] as usize | ((buf[1] as usize) << 8) | ((buf[2] as usize) << 8);
                let key_flame = if flags & 0x0001 == 0 { true } else { false };
                let profile = (flags >> 1) & 0x0007;
                let show = if (flags >> 4) & 0x0001 == 1 {
                    true
                } else {
                    false
                };
                let length = flags >> 5;
                // flags
                println!("key {} profile {} show {}", key_flame, profile, show);
                println!("length {}", length);
                // magic number 0x9d 0x01 0x2a
                println!("{:02x} {:02x} {:02x}", buf[3], buf[4], buf[5]);
                // width
                let w = buf[6] as usize | ((buf[7] as usize) << 8);
                let width = w & 0x3fff;
                let w_scale = w >> 14;
                webp_header.width = width;
                println!("{} {}", width, w_scale);
                // height
                let w = buf[8] as usize | ((buf[9] as usize) << 8);
                let height = w & 0x3fff;
                let h_scale = w >> 14;
                webp_header.height = height;
                println!("{} {}", height, h_scale);
                // Pragraph 9.2 color Space and Pixel Type
                let mut reader = BitReader::new(&buf[10..]);
                let (mut yuv, mut clamp) = (0, 0);
                if key_flame {
                    yuv = reader.get_bits(1)?; // L(1)
                    clamp = reader.get_bits(1)?; // L(1)
                }
                println!("{} {}", yuv, clamp);

                // Pragraph 9.3 Segment-Based Adjustments
                // parse_segment_header

                let mut quant = [0_isize; NUM_MB_SEGMENTS];
                let mut filter = [0_isize; NUM_MB_SEGMENTS];
                let mut seg = [0_usize; MB_FEATURE_TREE_PROBS];

                let segmentation_enabled = reader.get_bits(1)?;
                let mut update_segment_feature_data = 0;
                if segmentation_enabled == 1 {
                    if segmentation_enabled == 1 {
                        update_segment_feature_data = reader.get_bits(1)?; // L(1)
                        if reader.get_bits(1)? == 1 {
                            // update map data
                            for s in 0..NUM_MB_SEGMENTS {
                                quant[s] = if reader.get_bits(1)? == 1 {
                                    reader.get_signed_bits(7)? // signed
                                } else {
                                    0
                                };
                            }
                            for s in 0..NUM_MB_SEGMENTS {
                                filter[s] = if reader.get_bits(1)? == 1 {
                                    reader.get_signed_bits(6)? // signed
                                } else {
                                    0
                                };
                            }
                        }
                    }
                    if update_segment_feature_data == 1 {
                        for s in 0..MB_FEATURE_TREE_PROBS {
                            seg[s] = if reader.get_bits(1)? == 1 {
                                reader.get_bits(8)? // signed
                            } else {
                                0
                            };
                        }
                    }
                }

                // Paragraph 9.4
                // ParseFilterHeader
                // Paragraph 9.5

                // Paragraph 9.6 Quantize
                // idct

                // Paragraph 13.2 / 13.3
                webp_header.image = buf;
            }
            "VP8L" => {
                println!("This is Lossless");
                webp_header.lossy = false;
                webp_header.image_chunksize = size;
                let buf = reader.read_bytes_as_vec(size)?;
                webp_header.image = buf;
            }
            "VP8X" => {
                println!("This is Extended VP8");
                let flag = reader.read_byte()?;

                if flag & 0x20 > 0 {
                    // ICC PROFILE
                    webp_header.has_icc_profile = true;
                }
                if flag & 0x10 > 0 {
                    // has alpha
                    webp_header.has_alpha = true;
                }
                if flag & 0x08 > 0 {
                    // has metadata
                    webp_header.has_exif = true;
                }
                if flag & 0x04 > 0 {
                    // xmp
                    webp_header.has_xmp = true;
                }
                if flag & 0x02 > 0 {
                    // animation
                    webp_header.has_animation = true;
                }

                let _ = read_u24(reader)?; // Reserved
                webp_header.canvas_width = read_u24(reader)? as usize + 1;
                webp_header.canvas_height = read_u24(reader)? as usize + 1;
                if size > 10 {
                    reader.skip_ptr(size - 10)?;
                }
                println!("{} {}", webp_header.canvas_width, webp_header.canvas_height);
            }
            "ALPH" => {
                if webp_header.has_alpha == true {
                    let buf = reader.read_bytes_as_vec(size)?;
                    webp_header.alpha = Some(buf);
                } else {
                    reader.skip_ptr(size)?;
                }
            }
            "ANIM" => {
                if webp_header.has_animation == true {
                    let backgroud_color = reader.read_u32_le()?;
                    let loop_count = reader.read_u16_le()?;
                    if size > 8 {
                        reader.skip_ptr(size - 8)?;
                    }
                    let animation = AnimationControl {
                        backgroud_color,
                        loop_count,
                    };
                    webp_header.animation = Some(animation);
                } else {
                    reader.skip_ptr(size)?;
                }
            }
            "ANMF" | "ANIF" => {
                if webp_header.has_animation == true {
                    let frame_x = read_u24(reader)? as usize * 2;
                    let frame_y = read_u24(reader)? as usize * 2;
                    let width = read_u24(reader)? as usize + 1;
                    let height = read_u24(reader)? as usize + 1;
                    let duration = read_u24(reader)? as usize;
                    let flag = reader.read_byte()?;
                    let alpha_blending = if flag & 0x02 > 0 { false } else { true };
                    let disopse = if flag & 0x01 > 0 { true } else { false };

                    let buf = reader.read_bytes_as_vec(size - 16)?;
                    let (frame, alpha) = parse_animation_frame_payload(&buf)?;
                    let animation_frame = AnimationFrame {
                        frame_x,
                        frame_y,
                        width,
                        height,
                        duration,
                        alpha_blending,
                        disopse,
                        frame,
                        alpha,
                    };
                    if webp_header.animation_frame.as_ref().is_none() {
                        let frame = vec![animation_frame];
                        webp_header.animation_frame = Some(frame);
                    } else {
                        let frame = webp_header.animation_frame.as_mut().unwrap();
                        frame.push(animation_frame);
                    }
                } else {
                    reader.skip_ptr(size)?;
                }
            }
            "EXIF" => {
                if webp_header.has_exif == true {
                    let buf = reader.read_bytes_as_vec(size)?;
                    webp_header.exif = Some(buf);
                } else {
                    reader.skip_ptr(size)?;
                }
            }
            "XMP " => {
                if webp_header.has_xmp == true {
                    let buf = reader.read_bytes_as_vec(size)?;
                    webp_header.xmp = Some(buf);
                } else {
                    reader.skip_ptr(size)?;
                }
            }
            "ICCP" => {
                if webp_header.has_icc_profile == true {
                    let buf = reader.read_bytes_as_vec(size)?;
                    webp_header.icc_profile = Some(buf);
                } else {
                    reader.skip_ptr(size)?;
                }
            }
            _ => {
                reader.skip_ptr(size)?;
            }
        }
        if size & 1 == 1 {
            reader.skip_ptr(1)?;
        }
        if cksize <= padded_size + 8 {
            break;
        }
        cksize -= padded_size;
        cksize -= 8;
    }
    Ok(webp_header)
}
