//! Legacy RIFF parser kept for compatibility tests and chunk-oriented access.

use bin_rs::reader::{BinaryReader, BytesReader};

type Error = Box<dyn std::error::Error>;

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
        Self {
            buffer: data.to_vec(),
            last_byte: 0,
            ptr: 0,
            left_bits: 0,
            warning: false,
        }
    }

    fn look_bits(&mut self, size: usize) -> Result<usize, Error> {
        while self.left_bits < size {
            if self.ptr >= self.buffer.len() {
                self.warning = true;
                if size >= 12 {
                    return Ok(0x1);
                }
                return Ok(0x0);
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
        } else if self.look_bits(size).is_ok() && self.left_bits >= size {
            self.left_bits -= size;
        } else {
            self.left_bits = 0;
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

/// Global animation parameters stored in the `ANIM` chunk.
pub struct AnimationControl {
    /// Canvas background color in little-endian ARGB order.
    pub backgroud_color: u32,
    /// Loop count from the container. `0` means infinite loop.
    pub loop_count: u16,
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
    reader.read_exact(&mut b)?;
    Ok((b[0] as u32) | ((b[1] as u32) << 8) | ((b[2] as u32) << 16))
}

fn parse_animation_frame_payload(data: &[u8]) -> Result<(Vec<u8>, Option<Vec<u8>>), Error> {
    let mut reader = BytesReader::from(data.to_vec());
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

/// Parses the RIFF container and returns raw chunk-oriented metadata.
pub fn read_header<B: BinaryReader>(reader: &mut B) -> Result<WebpHeader, Error> {
    let riff = reader.read_ascii_string(4)?;
    if riff != "RIFF" {
        return Err(Box::new(std::io::Error::from(std::io::ErrorKind::Other)));
    }
    let mut cksize = reader.read_u32_le()? as usize;
    let webp = reader.read_ascii_string(4)?;
    if webp != "WEBP" {
        return Err(Box::new(std::io::Error::from(std::io::ErrorKind::Other)));
    }
    cksize -= 4;
    let mut webp_header = WebpHeader::new();

    loop {
        let vp8 = reader.read_ascii_string(4)?;
        let size = reader.read_u32_le()? as usize;
        let padded_size = size + (size & 1);
        match vp8.as_str() {
            "VP8 " => {
                webp_header.lossy = true;
                webp_header.image_chunksize = size;
                let buf = reader.read_bytes_as_vec(size)?;
                let flags = buf[0] as usize | ((buf[1] as usize) << 8) | ((buf[2] as usize) << 8);
                let key_frame = (flags & 0x0001) == 0;

                let w = buf[6] as usize | ((buf[7] as usize) << 8);
                webp_header.width = w & 0x3fff;
                let w = buf[8] as usize | ((buf[9] as usize) << 8);
                webp_header.height = w & 0x3fff;

                let mut reader = BitReader::new(&buf[10..]);
                if key_frame {
                    let _ = reader.get_bits(1)?;
                    let _ = reader.get_bits(1)?;
                }

                let mut quant = [0_isize; NUM_MB_SEGMENTS];
                let mut filter = [0_isize; NUM_MB_SEGMENTS];
                let mut seg = [0_usize; MB_FEATURE_TREE_PROBS];

                let segmentation_enabled = reader.get_bits(1)?;
                if segmentation_enabled == 1 {
                    let update_segment_feature_data = reader.get_bits(1)?;
                    if reader.get_bits(1)? == 1 {
                        for quant_item in quant.iter_mut().take(NUM_MB_SEGMENTS) {
                            *quant_item = if reader.get_bits(1)? == 1 {
                                reader.get_signed_bits(7)?
                            } else {
                                0
                            };
                        }
                        for filter_item in filter.iter_mut().take(NUM_MB_SEGMENTS) {
                            *filter_item = if reader.get_bits(1)? == 1 {
                                reader.get_signed_bits(6)?
                            } else {
                                0
                            };
                        }
                    }
                    if update_segment_feature_data == 1 {
                        for seg_item in seg.iter_mut().take(MB_FEATURE_TREE_PROBS) {
                            *seg_item = if reader.get_bits(1)? == 1 {
                                reader.get_bits(8)?
                            } else {
                                0
                            };
                        }
                    }
                }

                webp_header.image = buf;
            }
            "VP8L" => {
                webp_header.lossy = false;
                webp_header.image_chunksize = size;
                webp_header.image = reader.read_bytes_as_vec(size)?;
            }
            "VP8X" => {
                let flag = reader.read_byte()?;
                if flag & 0x20 > 0 {
                    webp_header.has_icc_profile = true;
                }
                if flag & 0x10 > 0 {
                    webp_header.has_alpha = true;
                }
                if flag & 0x08 > 0 {
                    webp_header.has_exif = true;
                }
                if flag & 0x04 > 0 {
                    webp_header.has_xmp = true;
                }
                if flag & 0x02 > 0 {
                    webp_header.has_animation = true;
                }

                let _ = read_u24(reader)?;
                webp_header.canvas_width = read_u24(reader)? as usize + 1;
                webp_header.canvas_height = read_u24(reader)? as usize + 1;
                if size > 10 {
                    reader.skip_ptr(size - 10)?;
                }
            }
            "ALPH" => {
                if webp_header.has_alpha {
                    webp_header.alpha = Some(reader.read_bytes_as_vec(size)?);
                } else {
                    reader.skip_ptr(size)?;
                }
            }
            "ANIM" => {
                if webp_header.has_animation {
                    let backgroud_color = reader.read_u32_le()?;
                    let loop_count = reader.read_u16_le()?;
                    if size > 8 {
                        reader.skip_ptr(size - 8)?;
                    }
                    webp_header.animation = Some(AnimationControl {
                        backgroud_color,
                        loop_count,
                    });
                } else {
                    reader.skip_ptr(size)?;
                }
            }
            "ANMF" | "ANIF" => {
                if webp_header.has_animation {
                    let frame_x = read_u24(reader)? as usize * 2;
                    let frame_y = read_u24(reader)? as usize * 2;
                    let width = read_u24(reader)? as usize + 1;
                    let height = read_u24(reader)? as usize + 1;
                    let duration = read_u24(reader)? as usize;
                    let flag = reader.read_byte()?;
                    let alpha_blending = (flag & 0x02) == 0;
                    let disopse = (flag & 0x01) != 0;

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
                    if let Some(frames) = webp_header.animation_frame.as_mut() {
                        frames.push(animation_frame);
                    } else {
                        webp_header.animation_frame = Some(vec![animation_frame]);
                    }
                } else {
                    reader.skip_ptr(size)?;
                }
            }
            "EXIF" => {
                if webp_header.has_exif {
                    webp_header.exif = Some(reader.read_bytes_as_vec(size)?);
                } else {
                    reader.skip_ptr(size)?;
                }
            }
            "XMP " => {
                if webp_header.has_xmp {
                    webp_header.xmp = Some(reader.read_bytes_as_vec(size)?);
                } else {
                    reader.skip_ptr(size)?;
                }
            }
            "ICCP" => {
                if webp_header.has_icc_profile {
                    webp_header.icc_profile = Some(reader.read_bytes_as_vec(size)?);
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
        cksize -= padded_size + 8;
    }
    Ok(webp_header)
}
