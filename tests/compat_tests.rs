use bin_rs::reader::{BinaryReader, BytesReader, StreamReader};
use std::collections::HashMap;
use std::io::{BufReader, Cursor};
use webp_rust::compat::{
    self, CallbackResponse, DataMap, DecodeOptions, DrawCallback, DrawOptions, InitOptions,
    NextOptions, TerminateOptions, VerboseOptions, RGBA,
};
use webp_rust::decoder::{get_features, parse_animation_webp};
use webp_rust::{read_header, read_u24, AnimationControl, AnimationFrame, ImageBuffer, WebpHeader};

type Error = Box<dyn std::error::Error>;

#[derive(Debug, Default, PartialEq, Eq)]
struct RecordingDrawer {
    init: Option<(usize, usize, InitOptions)>,
    draws: Vec<(usize, usize, usize, usize, Vec<u8>)>,
    nexts: Vec<NextOptions>,
    metadata: HashMap<String, DataMap>,
    terminated: bool,
    abort_at_next: Option<usize>,
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
        Ok(Some(if self.abort_at_next == Some(self.nexts.len()) {
            CallbackResponse::abort()
        } else {
            CallbackResponse::cont()
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
    let data = include_bytes!("../samples/sample.webp");
    let mut reader = BytesReader::from(data.to_vec());
    let mut drawer = RecordingDrawer::default();
    let mut options = DecodeOptions::new(&mut drawer);

    compat::decode(&mut reader, &mut options).unwrap();

    let (width, height, init) = drawer.init.as_ref().unwrap();
    let features = get_features(data).unwrap();
    assert_eq!((*width, *height), (features.width, features.height));
    assert!(!init.animation);
    assert_eq!(drawer.draws.len(), 1);
    assert!(drawer.nexts.is_empty());
    assert_eq!(
        drawer.metadata.get("Format"),
        Some(&DataMap::Ascii("WEBP".to_string()))
    );
    assert_eq!(
        drawer.metadata.get("width"),
        Some(&DataMap::UInt(features.width as u64))
    );
    assert_eq!(
        drawer.metadata.get("height"),
        Some(&DataMap::UInt(features.height as u64))
    );
    assert!(drawer.terminated);
}

#[test]
fn compat_decode_decodes_animation_via_callback_api() {
    let data = include_bytes!("../samples/sample_animation.webp");
    let mut reader = BytesReader::from(data.to_vec());
    let mut drawer = RecordingDrawer::default();
    let mut options = DecodeOptions::new(&mut drawer);

    compat::decode(&mut reader, &mut options).unwrap();

    let (width, height, init) = drawer.init.as_ref().unwrap();
    let features = get_features(data).unwrap();
    let parsed = parse_animation_webp(data).unwrap();
    assert_eq!((*width, *height), (features.width, features.height));
    assert!(init.animation);
    assert_eq!(
        init.background,
        Some(RGBA {
            red: (parsed.animation.background_color >> 16) as u8,
            green: (parsed.animation.background_color >> 8) as u8,
            blue: parsed.animation.background_color as u8,
            alpha: (parsed.animation.background_color >> 24) as u8,
        })
    );
    assert_eq!(drawer.nexts.len(), parsed.frames.len());
    assert_eq!(
        drawer.metadata.get("Animation frames"),
        Some(&DataMap::UInt(parsed.frames.len() as u64))
    );
    assert!(drawer.draws.len() >= parsed.frames.len());
    assert!(drawer.terminated);
}

fn record_decode<B: BinaryReader>(reader: &mut B, abort_at_next: Option<usize>) -> RecordingDrawer {
    let mut drawer = RecordingDrawer {
        abort_at_next,
        ..Default::default()
    };
    compat::decode(reader, &mut DecodeOptions::new(&mut drawer)).unwrap();
    drawer
}

#[test]
fn compat_stream_readers_match_bytes_with_small_and_default_buffers() {
    for data in [
        include_bytes!("../samples/sample_lossy.webp").as_slice(),
        include_bytes!("../samples/sample_lossless.webp").as_slice(),
        include_bytes!("../samples/sample_animation.webp").as_slice(),
    ] {
        let expected = record_decode(&mut BytesReader::new(data), None);
        for capacity in [1, 8, 16] {
            let mut reader =
                StreamReader::new(BufReader::with_capacity(capacity, Cursor::new(data)));
            assert_eq!(
                record_decode(&mut reader, None),
                expected,
                "capacity {capacity}"
            );
            assert_eq!(reader.offset().unwrap(), data.len() as u64);
        }
        let mut reader = StreamReader::new(BufReader::new(Cursor::new(data)));
        assert_eq!(record_decode(&mut reader, None), expected);
        assert_eq!(reader.offset().unwrap(), data.len() as u64);
    }
}

#[test]
fn compat_stream_animation_preserves_abort_callbacks() {
    let data = include_bytes!("../samples/sample_animation.webp").as_slice();
    assert!(parse_animation_webp(data).unwrap().frames.len() >= 2);
    for abort_at in [1, 2] {
        let expected = record_decode(&mut BytesReader::new(data), Some(abort_at));
        assert_eq!(expected.nexts.len(), abort_at);
        assert_eq!(expected.draws.len(), if abort_at == 1 { 1 } else { 2 });
        assert!(expected.terminated);
        assert!(!expected.metadata.is_empty());
        for capacity in [1, 8, 16, 8192] {
            let mut reader =
                StreamReader::new(BufReader::with_capacity(capacity, Cursor::new(data)));
            assert_eq!(record_decode(&mut reader, Some(abort_at)), expected);
        }
    }
}

#[test]
fn compat_stream_rejects_truncated_headers_and_payloads_before_callbacks() {
    let data = include_bytes!("../samples/sample_lossy.webp").as_slice();
    for length in [0, 4, 11, 12, data.len() - 1] {
        for capacity in [1, 8, 16, 8192] {
            let mut reader = StreamReader::new(BufReader::with_capacity(
                capacity,
                Cursor::new(&data[..length]),
            ));
            let mut drawer = RecordingDrawer::default();
            assert!(compat::decode(&mut reader, &mut DecodeOptions::new(&mut drawer)).is_err());
            assert_eq!(drawer, RecordingDrawer::default());
        }
    }
}
