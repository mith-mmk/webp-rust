//! Images load to Buffer or save to buffer library
//! This library uses callback and decoder callbacks response initialize, draw, next, terminate function.
//! Drawer will use callback, flexisble drawing.
//! 0.0.10 using color space RGBA32 only
type Error = Box<dyn std::error::Error>;
use crate::color::RGBA;
use crate::error::ImgError;
use crate::error::ImgErrorKind;
use crate::metadata::DataMap;
use crate::util::format_check;
use crate::util::ImageFormat;
use crate::warning::ImgWarnings;
use bin_rs::reader::*;
use std::collections::HashMap;
#[cfg(not(target_family = "wasm"))]
use std::io::BufRead;
#[cfg(not(target_family = "wasm"))]
use std::io::BufReader;
#[cfg(not(target_family = "wasm"))]
use std::io::Seek;
use std::io::Write;

/* Dynamic Select Callback System */

/// Drawing Next Options using for Multi images format/ Animation images
#[derive(Debug)]
pub enum DrawNextOptions {
    Continue,
    NextImage,
    ClearNext,
    WaitTime(usize),
    None,
}

pub type Response = Result<Option<CallbackResponse>, Error>;

pub trait DrawCallback: Sync + Send {
    /// DrawCallback trait is callback template
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
        _term: Option<TerminateOptions>,
    ) -> Result<Option<CallbackResponse>, Error>;
    fn next(&mut self, _next: Option<NextOptions>) -> Result<Option<CallbackResponse>, Error>;
    fn verbose(
        &mut self,
        _verbose: &str,
        _: Option<VerboseOptions>,
    ) -> Result<Option<CallbackResponse>, Error>;
    /// fn set_metadata after 0.0.10
    fn set_metadata(
        &mut self,
        key: &str,
        value: DataMap,
    ) -> Result<Option<CallbackResponse>, Error>;
}

pub trait PickCallback: Sync + Send {
    /// PickCallback trait is callback template for saver
    fn encode_start(
        &mut self,
        option: Option<EncoderOptions>,
    ) -> Result<Option<ImageProfiles>, Error>;
    fn encode_pick(
        &mut self,
        start_x: usize,
        start_y: usize,
        width: usize,
        height: usize,
        option: Option<PickOptions>,
    ) -> Result<Option<Vec<u8>>, Error>;
    fn encode_end(&mut self, _: Option<EndOptions>) -> Result<(), Error>;
    /// fn metadata is after 0.0.10
    fn metadata(&mut self) -> Result<Option<HashMap<String, DataMap>>, Error>;
}

#[derive(Debug)]
pub struct EncoderOptions {}

#[derive(Debug)]
pub struct PickOptions {}

#[derive(Debug)]
pub struct EndOptions {}

#[allow(unused)]
/// InitOptions added infomations send for drawer allback function
/// loop_count uses animation images (not impl)
/// background if uses background with alpha channel images;
#[derive(Debug)]
pub struct InitOptions {
    pub loop_count: u32,
    pub background: Option<RGBA>, // RGBA
    pub animation: bool,
}

impl InitOptions {
    pub fn new() -> Option<Self> {
        Some(Self {
            loop_count: 1,
            background: None,
            animation: false,
        })
    }
}

#[derive(Debug)]
pub struct DrawOptions {}

#[derive(Debug)]
pub struct TerminateOptions {}

#[derive(Debug)]
pub struct VerboseOptions {}

#[derive(Debug)]
pub enum NextOption {
    Continue,
    Next,
    Dispose,
    ClearAbort,
    Terminate,
}

#[derive(Debug)]
pub enum NextDispose {
    None,
    Override,
    Background,
    Previous,
}

#[derive(Debug)]
pub enum NextBlend {
    Source,
    Override,
}

#[allow(unused)]
#[derive(Debug)]
pub struct ImageRect {
    pub start_x: i32,
    pub start_y: i32,
    pub width: usize,
    pub height: usize,
}

#[allow(unused)]
#[derive(Debug)]
pub struct NextOptions {
    pub flag: NextOption,
    pub await_time: u64,
    pub image_rect: Option<ImageRect>,
    pub dispose_option: Option<NextDispose>,
    pub blend: Option<NextBlend>,
}

impl NextOptions {
    pub fn new() -> Self {
        NextOptions {
            flag: NextOption::Continue,
            await_time: 0,
            image_rect: None,
            dispose_option: None,
            blend: None,
        }
    }

    pub fn wait(ms_time: u64) -> Self {
        NextOptions {
            flag: NextOption::Continue,
            await_time: ms_time,
            image_rect: None,
            dispose_option: None,
            blend: None,
        }
    }
}

#[derive(std::cmp::PartialEq, Debug)]
pub enum ResponseCommand {
    Abort,
    Continue,
}

#[derive(Debug)]
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

#[derive(Debug)]
pub struct ImageProfiles {
    pub width: usize,
    pub height: usize,
    pub background: Option<RGBA>,
    /// Image Buffer depend metadatas
    pub metadata: Option<HashMap<String, DataMap>>,
}

/// Using for Animation GIF/PNG/other
pub struct AnimationLayer {
    pub width: usize,
    pub height: usize,
    pub start_x: i32,
    pub start_y: i32,
    pub buffer: Vec<u8>,
    pub control: NextOptions,
}

#[allow(unused)]
pub struct ImageBuffer {
    /// ImageBuffer is default drawer
    /// Excample callback impliment
    /// ImageBuffer use only RGBA8888 color space model
    /// buffersize = width * height * 4 (8+8+8+8)/8
    /// other color space is implement feature.
    pub width: usize,
    pub height: usize,
    pub background_color: Option<RGBA>,
    pub buffer: Option<Vec<u8>>,
    pub animation: Option<Vec<AnimationLayer>>,
    pub current: Option<usize>,
    pub loop_count: Option<u32>,
    pub first_wait_time: Option<u64>,
    fnverbose: fn(&str) -> Result<Option<CallbackResponse>, Error>,
    pub metadata: Option<HashMap<String, DataMap>>,
}

fn default_verbose(_: &str) -> Result<Option<CallbackResponse>, Error> {
    Ok(None)
}

impl ImageBuffer {
    pub fn new() -> Self {
        Self {
            width: 0,
            height: 0,
            background_color: None,
            buffer: None,
            animation: None,
            current: None,
            loop_count: None,
            first_wait_time: None,
            fnverbose: default_verbose,
            metadata: None,
        }
    }

    pub fn from_buffer(width: usize, height: usize, buf: Vec<u8>) -> Self {
        Self {
            width,
            height,
            background_color: None,
            buffer: Some(buf),
            animation: None,
            current: None,
            loop_count: None,
            first_wait_time: None,
            fnverbose: default_verbose,
            metadata: None,
        }
    }

    /// ImageBuffer allow animation
    pub fn set_animation(&mut self, flag: bool) {
        if flag {
            self.animation = Some(Vec::new())
        } else {
            self.animation = None
        }
    }

    /// A funtion set_verbose uses also debug.
    pub fn set_verbose(&mut self, verbose: fn(&str) -> Result<Option<CallbackResponse>, Error>) {
        self.fnverbose = verbose;
    }
}

impl DrawCallback for ImageBuffer {
    /// initialized Image Buffer
    fn init(
        &mut self,
        width: usize,
        height: usize,
        option: Option<InitOptions>,
    ) -> Result<Option<CallbackResponse>, Error> {
        let buffersize = width * height * 4;
        self.width = width;
        self.height = height;
        if let Some(option) = option {
            self.background_color = option.background;
            if option.animation {
                self.set_animation(true);
            }
            self.loop_count = Some(option.loop_count);
        }
        if let Some(background) = &self.background_color {
            self.buffer = Some(
                (0..buffersize)
                    .map(|i| match i % 4 {
                        0 => background.red,
                        1 => background.green,
                        2 => background.blue,
                        _ => 0xff,
                    })
                    .collect(),
            );
        } else {
            self.buffer = Some((0..buffersize).map(|_| 0).collect());
        }

        Ok(None)
    }

    /// draw a part of image
    fn draw(
        &mut self,
        start_x: usize,
        start_y: usize,
        width: usize,
        height: usize,
        data: &[u8],
        _: Option<DrawOptions>,
    ) -> Result<Option<CallbackResponse>, Error> {
        if self.buffer.is_none() {
            return Err(Box::new(ImgError::new_const(
                ImgErrorKind::NotInitializedImageBuffer,
                "in draw".to_string(),
            )));
        }
        let buffer;
        let (w, h, raws);
        if self.current.is_none() {
            if start_x >= self.width || start_y >= self.height {
                return Ok(None);
            }
            w = if self.width < width + start_x {
                self.width - start_x
            } else {
                width
            };
            h = if self.height < height + start_y {
                self.height - start_y
            } else {
                height
            };
            raws = self.width;
            buffer = self.buffer.as_deref_mut().unwrap();
        } else if let Some(animation) = &mut self.animation {
            let current = self.current.unwrap();
            if start_x >= animation[current].width || start_y >= animation[current].height {
                return Ok(None);
            }
            w = if animation[current].width < width + start_x {
                animation[current].width - start_x
            } else {
                width
            };
            h = if animation[current].height < height + start_y {
                animation[current].height - start_y
            } else {
                height
            };
            raws = animation[current].width;
            buffer = &mut animation[current].buffer;
        } else {
            return Err(Box::new(ImgError::new_const(
                ImgErrorKind::NotInitializedImageBuffer,
                "in animation".to_string(),
            )));
        }

        for y in 0..h {
            let scanline_src = y * width * 4;
            let scanline_dest = (start_y + y) * raws * 4;
            for x in 0..w {
                let offset_src = scanline_src + x * 4;
                let offset_dest = scanline_dest + (x + start_x) * 4;
                if offset_src + 3 >= data.len() {
                    return Err(Box::new(ImgError::new_const(
                        ImgErrorKind::OutboundIndex,
                        "decoder buffer in draw".to_string(),
                    )));
                }
                buffer[offset_dest] = data[offset_src];
                buffer[offset_dest + 1] = data[offset_src + 1];
                buffer[offset_dest + 2] = data[offset_src + 2];
                buffer[offset_dest + 3] = data[offset_src + 3];
            }
        }
        Ok(None)
    }

    /// terminate ImageBuffer
    fn terminate(
        &mut self,
        _: Option<TerminateOptions>,
    ) -> Result<Option<CallbackResponse>, Error> {
        Ok(None)
    }

    /// Decoders tell ImageBuffer had next an image/images.
    fn next(&mut self, opt: Option<NextOptions>) -> Result<Option<CallbackResponse>, Error> {
        if self.animation.is_some() {
            if let Some(opt) = opt {
                if self.current.is_none() {
                    self.current = Some(0);
                    self.first_wait_time = Some(opt.await_time);
                } else {
                    self.current = Some(self.current.unwrap() + 1);
                }
                let (width, height, start_x, start_y);
                if let Some(ref rect) = opt.image_rect {
                    width = rect.width;
                    height = rect.height;
                    start_x = rect.start_x;
                    start_y = rect.start_y;
                } else {
                    width = self.width;
                    height = self.height;
                    start_x = 0;
                    start_y = 0;
                }
                let buffersize = width * height * 4;
                let buffer: Vec<u8> = (0..buffersize).map(|_| 0).collect();
                let layer = AnimationLayer {
                    width,
                    height,
                    start_x,
                    start_y,
                    buffer,
                    control: opt,
                };

                self.animation.as_mut().unwrap().push(layer);

                return Ok(Some(CallbackResponse::cont()));
            }
        }
        Ok(Some(CallbackResponse::abort()))
    }

    /// Decoder tell ImageBuffer verbose
    fn verbose(
        &mut self,
        str: &str,
        _: Option<VerboseOptions>,
    ) -> Result<Option<CallbackResponse>, Error> {
        (self.fnverbose)(str)
    }

    /// Decoder set metadata to ImageBuffer
    fn set_metadata(
        &mut self,
        key: &str,
        value: DataMap,
    ) -> Result<Option<CallbackResponse>, Error> {
        let hashmap = if let Some(ref mut hashmap) = self.metadata {
            hashmap
        } else {
            self.metadata = Some(HashMap::new());
            self.metadata.as_mut().unwrap()
        };
        hashmap.insert(key.to_string(), value);

        Ok(None)
    }
}

impl PickCallback for ImageBuffer {
    /// Encoder tell start ImageBuffer
    fn encode_start(&mut self, _: Option<EncoderOptions>) -> Result<Option<ImageProfiles>, Error> {
        let init = ImageProfiles {
            width: self.width,
            height: self.height,
            background: self.background_color.clone(),
            metadata: None,
        };
        Ok(Some(init))
    }

    /// Encoder pick a part of image from ImageBuffer
    fn encode_pick(
        &mut self,
        start_x: usize,
        start_y: usize,
        width: usize,
        height: usize,
        _: Option<PickOptions>,
    ) -> Result<Option<Vec<u8>>, Error> {
        if self.buffer.is_none() {
            return Err(Box::new(ImgError::new_const(
                ImgErrorKind::NotInitializedImageBuffer,
                "in pick".to_string(),
            )));
        }
        let buffersize = width * height * 4;
        let mut data = Vec::with_capacity(buffersize);
        let buffer = self.buffer.as_ref().unwrap();

        if start_x >= self.width || start_y >= self.height {
            return Ok(None);
        }
        let w = if self.width < width + start_x {
            self.width - start_x
        } else {
            width
        };
        let h = if self.height < height + start_y {
            self.height - start_y
        } else {
            height
        };

        for y in 0..h {
            let scanline_src = (start_y + y) * width * 4;
            for x in 0..w {
                let offset_src = scanline_src + (start_x + x) * 4;
                if offset_src + 3 >= buffer.len() {
                    return Err(Box::new(ImgError::new_const(
                        ImgErrorKind::OutboundIndex,
                        "Image buffer in pick".to_string(),
                    )));
                }
                data.push(buffer[offset_src]);
                data.push(buffer[offset_src + 1]);
                data.push(buffer[offset_src + 2]);
                data.push(buffer[offset_src + 3]);
            }
            for _ in w..width {
                // 0 fill
                data.push(0x00);
                data.push(0x00);
                data.push(0x00);
                data.push(0x00);
            }
        }
        for _ in h..height {
            // 0 fill
            for _ in 0..width {
                data.push(0x00);
                data.push(0x00);
                data.push(0x00);
                data.push(0x00);
            }
        }

        Ok(Some(data))
    }

    /// Encoder tell ending encode to ImageBuffer
    fn encode_end(&mut self, _: Option<EndOptions>) -> Result<(), Error> {
        Ok(())
    }

    /// Encoder get metadata from ImageBuffer
    fn metadata(&mut self) -> Result<Option<HashMap<String, DataMap>>, Error> {
        if let Some(hashmap) = &self.metadata {
            Ok(Some(hashmap.clone()))
        } else {
            Ok(None)
        }
    }
}

pub struct DecodeOptions<'a> {
    pub debug_flag: usize,
    pub drawer: &'a mut dyn DrawCallback,
}

pub struct EncodeOptions<'a> {
    pub debug_flag: usize,
    pub drawer: &'a mut dyn PickCallback,
    /// Endocer depend Options
    /// ex.)
    ///  "quality",DataMap::INT(80) // for JPEG
    ///  "compression",DataMap::ASCII("LZW") for TIFF
    ///  "color model",ASCII::ASCII("index") for TIFF/GIF/BMP
    pub options: Option<HashMap<String, DataMap>>,
}

pub fn image_from(buffer: &[u8]) -> Result<ImageBuffer, Error> {
    image_load(buffer)
}

/// load image from file
#[cfg(not(target_family = "wasm"))]
pub fn image_from_file(filename: String) -> Result<ImageBuffer, Error> {
    let f = std::fs::File::open(filename)?;
    let reader = BufReader::new(f);
    let mut image = ImageBuffer::new();
    let mut option = DecodeOptions {
        debug_flag: 0x00,
        drawer: &mut image,
    };
    let _ = image_reader(reader, &mut option)?;
    Ok(image)
}

/// save image to file
#[cfg(not(target_family = "wasm"))]
pub fn image_to_file(
    filename: String,
    image: &mut dyn PickCallback,
    format: ImageFormat,
) -> Result<(), Error> {
    let f = std::fs::File::create(filename)?;
    let mut option = EncodeOptions {
        debug_flag: 0x00,
        drawer: image,
        options: None,
    };
    image_writer(f, &mut option, format)?;
    Ok(())
}

/// load image from buffer
pub fn image_load(buffer: &[u8]) -> Result<ImageBuffer, Error> {
    let mut ib = ImageBuffer::new();
    let mut option = DecodeOptions {
        debug_flag: 0,
        drawer: &mut ib,
    };
    let mut reader = BytesReader::new(buffer);

    image_decoder(&mut reader, &mut option)?;
    Ok(ib)
}

pub fn image_loader(
    buffer: &[u8],
    option: &mut DecodeOptions,
) -> Result<Option<ImgWarnings>, Error> {
    let mut reader = BytesReader::new(buffer);

    let r = image_decoder(&mut reader, option)?;
    Ok(r)
}

#[cfg(not(target_family = "wasm"))]
pub fn image_reader<R: BufRead + Seek>(
    reader: R,
    option: &mut DecodeOptions,
) -> Result<Option<ImgWarnings>, Error> {
    let mut reader = StreamReader::new(reader);

    let r = image_decoder(&mut reader, option)?;
    Ok(r)
}

pub fn image_writer<W: Write>(
    mut writer: W,
    option: &mut EncodeOptions,
    format: ImageFormat,
) -> Result<Option<ImgWarnings>, Error> {
    let buffer = image_encoder(option, format)?;
    writer.write_all(&buffer)?;
    writer.flush()?;
    Ok(None)
}

pub fn image_decoder<B: BinaryReader>(
    reader: &mut B,
    option: &mut DecodeOptions,
) -> Result<Option<ImgWarnings>, Error> {
    let buffer = reader.read_bytes_no_move(128)?;
    let format = format_check(&buffer);

    use crate::util::ImageFormat::*;
    match format {
        Jpeg => {
            return crate::jpeg::decoder::decode(reader, option);
        }
        Bmp => {
            return crate::bmp::decoder::decode(reader, option);
        }
        Gif => {
            return crate::gif::decoder::decode(reader, option);
        }
        Png => {
            return crate::png::decoder::decode(reader, option);
        }
        Tiff => {
            return crate::tiff::decoder::decode(reader, option);
        }
        Mag => {
            return crate::mag::decoder::decode(reader, option);
        }
        #[cfg(not(feature = "noretoro"))]
        Maki => {
            return crate::maki::decoder::decode(reader, option);
        }
        #[cfg(not(feature = "noretoro"))]
        Pi => {
            return crate::pi::decoder::decode(reader, option);
        }
        #[cfg(not(feature = "noretoro"))]
        Pic => {
            return crate::pic::decoder::decode(reader, option);
        }
        #[cfg(not(feature = "noretoro"))]
        Vsp => {
            return crate::vsp::decoder::decode(reader, option);
        }
        _ => {}
    }

    #[cfg(not(feature = "noretoro"))]
    {
    let current = reader.seek(std::io::SeekFrom::Current(0))?;
    let pcd = (|| -> Result<bool, Error> {
        reader.seek(std::io::SeekFrom::Start(0x800))?;
        let mut id = [0u8; 7];
        reader.read_exact(&mut id)?;
        Ok(&id == b"PCD_IPI")
    })();
    reader.seek(std::io::SeekFrom::Start(current))?;
    if matches!(pcd, Ok(true)) {
        return crate::pcd::decoder::decode(reader, option);
    }
    }

    Err(Box::new(ImgError::new_const(
        ImgErrorKind::NoSupportFormat,
        "This buffer can not decode".to_string(),
    )))
}

pub fn image_encoder(option: &mut EncodeOptions, format: ImageFormat) -> Result<Vec<u8>, Error> {
    use crate::util::ImageFormat::*;

    match format {
        Bmp => {
            return crate::bmp::encoder::encode(option);
        }
        Jpeg => {
            return crate::jpeg::encoder::encode(option);
        }
        Png => {
            return crate::png::encoder::encode(option);
        }
        _ => Err(Box::new(ImgError::new_const(
            ImgErrorKind::NoSupportFormat,
            "This encoder no impl".to_string(),
        ))),
    }
}
