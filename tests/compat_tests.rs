use bin_rs::reader::BytesReader;
use std::collections::HashMap;
use webp_rust::compat::{
    self, CallbackResponse, DataMap, DecodeOptions, DrawCallback, DrawOptions, InitOptions,
    NextOptions, ResponseCommand, TerminateOptions, VerboseOptions, RGBA,
};
use webp_rust::{read_header, read_u24, AnimationControl, AnimationFrame, ImageBuffer, WebpHeader};

type Error = Box<dyn std::error::Error>;

#[derive(Default)]
struct RecordingDrawer {
    init: Option<(usize, usize, InitOptions)>,
    draws: Vec<(usize, usize, usize, usize, Vec<u8>)>,
    nexts: Vec<NextOptions>,
    metadata: HashMap<String, DataMap>,
    terminated: bool,
}

impl DrawCallback for RecordingDrawer {
    fn init(
        &mut self,
        width: usize,
        height: usize,
        option: Option<InitOptions>,
    ) -> Result<Option<CallbackResponse>, Error> {
        self.init = Some((width, height, option.expect("init options")));
        Ok(Some(CallbackResponse::cont()))
    }

    fn draw(
        &mut self,
        start_x: usize,
        start_y: usize,
        width: usize,
        height: usize,
        data: &[u8],
        _option: Option<DrawOptions>,
    ) -> Result<Option<CallbackResponse>, Error> {
        self.draws
            .push((start_x, start_y, width, height, data.to_vec()));
        Ok(Some(CallbackResponse::cont()))
    }

    fn terminate(
        &mut self,
        _term: Option<TerminateOptions>,
    ) -> Result<Option<CallbackResponse>, Error> {
        self.terminated = true;
        Ok(Some(CallbackResponse::cont()))
    }

    fn next(&mut self, next: Option<NextOptions>) -> Result<Option<CallbackResponse>, Error> {
        self.nexts.push(next.expect("next options"));
        Ok(Some(CallbackResponse {
            response: ResponseCommand::Continue,
        }))
    }

    fn verbose(
        &mut self,
        _verbose: &str,
        _option: Option<VerboseOptions>,
    ) -> Result<Option<CallbackResponse>, Error> {
        Ok(Some(CallbackResponse::cont()))
    }

    fn set_metadata(
        &mut self,
        key: &str,
        value: DataMap,
    ) -> Result<Option<CallbackResponse>, Error> {
        self.metadata.insert(key.to_string(), value);
        Ok(Some(CallbackResponse::cont()))
    }
}

#[test]
fn image_buffer_keeps_legacy_accessors() {
    let image = ImageBuffer {
        width: 3,
        height: 2,
        rgba: vec![0, 1, 2, 3, 4, 5, 6, 7],
    };

    assert_eq!(image.width(), 3);
    assert_eq!(image.height(), 2);
    assert_eq!(image.get_width(), 3);
    assert_eq!(image.get_height(), 2);
    assert_eq!(image.rgba(), &[0, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(image.buffer(), vec![0, 1, 2, 3, 4, 5, 6, 7]);
}

#[test]
fn root_reexports_legacy_types() {
    let _header = WebpHeader::new();
    let _control = AnimationControl {
        backgroud_color: 0,
        loop_count: 0,
    };
    let _frame = AnimationFrame {
        frame_x: 0,
        frame_y: 0,
        width: 1,
        height: 1,
        duration: 0,
        alpha_blending: true,
        disopse: false,
        frame: vec![],
        alpha: None,
    };
    let mut reader = BytesReader::from(vec![0x56, 0x34, 0x12]);

    assert_eq!(read_u24(&mut reader).unwrap(), 0x12_34_56);

    let mut invalid = BytesReader::from(b"NOPE".to_vec());
    assert!(read_header(&mut invalid).is_err());
}

#[test]
fn compat_decode_decodes_still_images_via_callback_api() {
    let data = include_bytes!("../_testdata/sample.webp");
    let mut reader = BytesReader::from(data.to_vec());
    let mut drawer = RecordingDrawer::default();
    let mut options = DecodeOptions::new(&mut drawer);

    compat::decode(&mut reader, &mut options).unwrap();

    let (width, height, init) = drawer.init.as_ref().unwrap();
    assert_eq!((*width, *height), (1152, 896));
    assert!(!init.animation);
    assert_eq!(drawer.draws.len(), 1);
    assert!(drawer.nexts.is_empty());
    assert_eq!(
        drawer.metadata.get("Format"),
        Some(&DataMap::Ascii("WEBP".to_string()))
    );
    assert_eq!(drawer.metadata.get("width"), Some(&DataMap::UInt(1152)));
    assert_eq!(drawer.metadata.get("height"), Some(&DataMap::UInt(896)));
    assert!(drawer.terminated);
}

#[test]
fn compat_decode_decodes_animation_via_callback_api() {
    let data = include_bytes!("../_testdata/sample_animation.webp");
    let mut reader = BytesReader::from(data.to_vec());
    let mut drawer = RecordingDrawer::default();
    let mut options = DecodeOptions::new(&mut drawer);

    compat::decode(&mut reader, &mut options).unwrap();

    let (width, height, init) = drawer.init.as_ref().unwrap();
    assert_eq!((*width, *height), (1200, 1200));
    assert!(init.animation);
    assert_eq!(
        init.background,
        Some(RGBA {
            red: 0xb5,
            green: 0xee,
            blue: 0xf8,
            alpha: 0xff,
        })
    );
    assert_eq!(drawer.nexts.len(), 7);
    assert_eq!(
        drawer.metadata.get("Animation frames"),
        Some(&DataMap::UInt(7))
    );
    assert!(drawer.draws.len() >= 7);
    assert!(drawer.terminated);
}
