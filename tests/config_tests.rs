use webp_rust::decoder::{get_features, WebpFormat};
use webp_rust::encoder::{
    encode_lossless_rgba_to_webp_with_config, encode_lossy_rgba_to_webp_with_config, AlphaFilter,
    LosslessEncodingConfig, LossyEncodingConfig, WebpPreset,
};
use webp_rust::{decode, EncoderError};

fn sample_rgba(width: usize, height: usize, alpha: bool) -> Vec<u8> {
    let mut rgba = vec![0u8; width * height * 4];
    for y in 0..height {
        for x in 0..width {
            let offset = (y * width + x) * 4;
            rgba[offset] = (x * 17 + y * 3) as u8;
            rgba[offset + 1] = (x * 5 + y * 19) as u8;
            rgba[offset + 2] = (x * 11 + y * 7) as u8;
            rgba[offset + 3] = if alpha {
                ((x * 23 + y * 13) & 0xff) as u8
            } else {
                0xff
            };
        }
    }
    rgba
}

#[test]
fn standard_lossless_config_round_trips_at_multiple_z_levels() {
    let width = 23;
    let height = 19;
    let rgba = sample_rgba(width, height, true);

    for z_level in [0, 4, 9] {
        let webp = encode_lossless_rgba_to_webp_with_config(
            width,
            height,
            &rgba,
            &LosslessEncodingConfig { z_level },
        )
        .unwrap();
        let decoded = decode(&webp).unwrap();
        assert_eq!((decoded.width, decoded.height), (width, height));
        assert_eq!(decoded.rgba, rgba);
    }
}

#[test]
fn lossy_config_preserves_alpha_and_supports_no_alpha() {
    let width = 17;
    let height = 13;
    let rgba = sample_rgba(width, height, true);
    let config = LossyEncodingConfig {
        method: 0,
        alpha_method: 1,
        alpha_filter: AlphaFilter::Fast,
        ..LossyEncodingConfig::default()
    };

    let webp = encode_lossy_rgba_to_webp_with_config(width, height, &rgba, &config).unwrap();
    let features = get_features(&webp).unwrap();
    assert_eq!(features.format, WebpFormat::Lossy);
    assert!(features.has_alpha);
    let decoded = decode(&webp).unwrap();
    assert_eq!(decoded.rgba.len(), rgba.len());
    assert_eq!(
        decoded
            .rgba
            .chunks_exact(4)
            .map(|p| p[3])
            .collect::<Vec<_>>(),
        rgba.chunks_exact(4).map(|p| p[3]).collect::<Vec<_>>()
    );

    let opaque = LossyEncodingConfig {
        no_alpha: true,
        ..config
    };
    let no_alpha_webp =
        encode_lossy_rgba_to_webp_with_config(width, height, &rgba, &opaque).unwrap();
    assert!(!get_features(&no_alpha_webp).unwrap().has_alpha);
}

#[test]
fn lossy_config_validates_libwebp_option_ranges() {
    let rgba = sample_rgba(2, 2, false);
    let invalid = [
        LossyEncodingConfig {
            quality: f32::NAN,
            ..LossyEncodingConfig::default()
        },
        LossyEncodingConfig {
            method: 7,
            ..LossyEncodingConfig::default()
        },
        LossyEncodingConfig {
            segments: 0,
            ..LossyEncodingConfig::default()
        },
        LossyEncodingConfig {
            alpha_method: 2,
            ..LossyEncodingConfig::default()
        },
    ];

    for config in invalid {
        assert!(matches!(
            encode_lossy_rgba_to_webp_with_config(2, 2, &rgba, &config),
            Err(EncoderError::InvalidParam(_))
        ));
    }
}

#[test]
fn presets_have_distinct_stable_defaults() {
    let photo = LossyEncodingConfig::preset(WebpPreset::Photo);
    let text = LossyEncodingConfig::preset(WebpPreset::Text);
    assert!(photo.sns_strength > text.sns_strength);
    assert!(text.quality > photo.quality);
    assert!(text.method >= photo.method);
}
