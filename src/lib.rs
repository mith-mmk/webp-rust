type Error = Box<dyn std::error::Error>;
use bin_rs::reader::BinaryReader;

pub fn read_chunkid<B:BinaryReader>(reader:&mut B) -> Result<String,Error> {

    let id = reader.read_ascii_string(4)?;

    Ok(id.to_string())
}
pub struct AnimationControl {
    pub backgroud_color: u32,
    pub loop_count: u16,
}

pub struct AnimationFrame {
    pub frame_x: usize,
    pub frame_y: usize,
    pub width: usize,
    pub height: usize,
    pub duration: usize,
    pub alpha_blending: bool,
    pub disopse: bool,
    pub frame: Vec<u8>,
}

pub struct WebpHeader {
    pub width: usize,
    pub height: usize,
    pub canvas_width: usize,
    pub canvas_height: usize,
    pub image_chunksize: usize,
    pub has_icc_profile:bool,
    pub has_alpha:bool,
    pub has_exif:bool,
    pub has_xmp:bool,
    pub has_animation:bool,
    pub lossy: bool,
    pub image:Vec<u8>,
    pub icc_profile: Option<Vec<u8>>,
    pub alpha: Option<Vec<u8>>,
    pub exif: Option<Vec<u8>>,
    pub xmp: Option<Vec<u8>>,
    pub animation: Option<AnimationControl>,
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
            has_icc_profile:false,
            has_alpha:false,
            has_exif:false,
            has_xmp:false,
            has_animation:false,
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



pub fn read_u24<B:BinaryReader>(reader:&mut B) -> Result<u32,Error> {
    let mut b = [0_u8;3];
    reader.read_bytes(&mut b)?;
    let val = (b[0] as u32) | ((b [1] as u32) << 8) | ((b[2] as u32) << 16) ;
    Ok(val)
}


pub fn read_header<B:BinaryReader>(reader:&mut B)-> Result<WebpHeader,Error>  {
    let riff = reader.read_ascii_string(4)?;
    if riff != "RIFF" {
        //Error
        return Err(Box::new(std::io::Error::from(std::io::ErrorKind::Other)))
    }
    let mut cksize = reader.read_u32_le()? as usize;
    let webp = reader.read_ascii_string(4)?;
    if webp != "WEBP" {
        //Error
        println!("This is not WEBP");
        return Err(Box::new(std::io::Error::from(std::io::ErrorKind::Other)))
    }
    cksize -= 4;
    let mut webp_header = WebpHeader::new();

    loop {
        let vp8:&str = &reader.read_ascii_string(4)?;
        println!("{}",vp8);
        let size = reader.read_u32_le()? as usize;
        match vp8 {
            "VP8 " => {
                println!("This is Lossy");
                webp_header.lossy = true;
                webp_header.image_chunksize = size;
                let buf = reader.read_bytes_as_vec(size)?;
                webp_header.image = buf;
            },
            "VP8L" => {
                println!("This is Lossless");
                webp_header.lossy = false;
                webp_header.image_chunksize = size;
                let buf = reader.read_bytes_as_vec(size)?;
                webp_header.image = buf;
            },
            "VP8X" => {
                println!("This is Extended VP8");
                let flag = reader.read_byte()?;
        
                if flag & 0x20 > 0 {  // ICC PROFILE
                    webp_header.has_icc_profile = true;
                }
                if flag & 0x10 > 0 {  // has alpha
                    webp_header.has_alpha = true;
                }
                if flag & 0x08 > 0 {  // has metadata
                    webp_header.has_exif = true;
                }
                if flag & 0x04 > 0 {  // xmp
                    webp_header.has_xmp = true;
                }
                if flag & 0x02 > 0 {  // animation
                    webp_header.has_animation = true;
                }
        
                let _ = read_u24(reader)?;  // Reserved
                webp_header.canvas_width = read_u24(reader)? as usize + 1;
                webp_header.canvas_height = read_u24(reader)? as usize  + 1;
                if size > 10 {
                    reader.skip_ptr(size - 10)?;
                }
                println!("{} {}",webp_header.canvas_width,webp_header.canvas_height);
            },
            "ALPH" => {
                if webp_header.has_alpha == true {
                    let buf = reader.read_bytes_as_vec(size)?;
                    webp_header.alpha = Some(buf);
                } else {
                    reader.skip_ptr(size)?;     
                }
            },
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
            },
            "ANIF" => {
                if webp_header.has_animation == true {
                    let frame_x = read_u24(reader)? as usize;
                    let frame_y = read_u24(reader)? as usize;
                    let width = read_u24(reader)? as usize + 1;
                    let height = read_u24(reader)? as usize + 1;
                    let duration = read_u24(reader)? as usize;
                    let flag = reader.read_byte()?;
                    let alpha_blending = if flag & 0x02 > 0 {
                            false
                        } else {true};
                    let disopse = if flag & 0x01 > 0 {
                            true
                        } else {false};

                    let buf = reader.read_bytes_as_vec(size - 16)?;
                    let animation_frame = AnimationFrame {
                        frame_x,
                        frame_y,
                        width,
                        height,
                        duration,
                        alpha_blending,
                        disopse,
                        frame: buf
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
            },
            "EXIF" => {
                if webp_header.has_exif == true {
                    let buf = reader.read_bytes_as_vec(size)?;
                    webp_header.exif = Some(buf);
                } else {
                    reader.skip_ptr(size)?;     
                }

            },
            "XMP " => {
                if webp_header.has_xmp == true {
                    let buf = reader.read_bytes_as_vec(size)?;
                    webp_header.xmp = Some(buf);
                } else {
                    reader.skip_ptr(size)?;     
                }

            },
            "ICCP" => {
                if webp_header.has_icc_profile == true {
                    let buf = reader.read_bytes_as_vec(size)?;
                    webp_header.icc_profile = Some(buf);
                } else {
                    reader.skip_ptr(size)?;     
                }

            },
            _ => {
                reader.skip_ptr(size)?;
            }
        }
        if cksize <= size + 8 {
            break
        }
        cksize -= size;
        cksize -= 8;
    }
    Ok(webp_header)
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
