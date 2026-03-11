use std::fs;
use std::path::{Path, PathBuf};

use webp_rust::decoder::{
    decode_animation_webp_to_bmp_frames, decode_lossless_webp_to_bmp, decode_lossy_webp_to_bmp,
    get_features, WebpFormat,
};

type Error = Box<dyn std::error::Error>;

fn default_output_path(input: &Path) -> PathBuf {
    let mut output = input.to_path_buf();
    output.set_extension("bmp");
    output
}

fn default_animation_output_prefix(input: &Path) -> PathBuf {
    let mut output = input.to_path_buf();
    output.set_extension("");
    output
}

fn animation_frame_path(prefix: &Path, index: usize) -> PathBuf {
    let parent = prefix.parent().unwrap_or_else(|| Path::new(""));
    let stem = prefix
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or("frame");
    parent.join(format!("{stem}_{index:04}.bmp"))
}

fn main() -> Result<(), Error> {
    let mut args = std::env::args_os().skip(1);
    let input = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("_testdata/sample.webp"));
    let output = args.next().map(PathBuf::from).unwrap_or_else(|| {
        if input == PathBuf::from("_testdata/sample.webp") {
            PathBuf::from("target/sample.bmp")
        } else if input == PathBuf::from("_testdata/sample_animation.webp") {
            PathBuf::from("target/sample_animation")
        } else {
            default_output_path(&input)
        }
    });

    let data = fs::read(&input)?;
    let features = get_features(&data)?;

    if features.has_animation {
        let mut prefix = if output.is_dir() {
            output.join(
                input
                    .file_stem()
                    .and_then(|stem| stem.to_str())
                    .unwrap_or("frame"),
            )
        } else {
            output
        };
        if prefix.extension().is_some() {
            prefix.set_extension("");
        } else if prefix.as_os_str().is_empty() {
            prefix = default_animation_output_prefix(&input);
        }

        let frames = decode_animation_webp_to_bmp_frames(&data)?;
        let mut written_paths = Vec::with_capacity(frames.len());
        for (index, frame) in frames.into_iter().enumerate() {
            let path = animation_frame_path(&prefix, index);
            if let Some(parent) = path.parent() {
                if !parent.as_os_str().is_empty() {
                    fs::create_dir_all(parent)?;
                }
            }
            fs::write(&path, frame)?;
            written_paths.push(path);
        }
        for path in written_paths {
            println!("{}", path.display());
        }
        return Ok(());
    }

    let bmp = match features.format {
        WebpFormat::Lossy => decode_lossy_webp_to_bmp(&data)?,
        WebpFormat::Lossless => decode_lossless_webp_to_bmp(&data)?,
        WebpFormat::Undefined => return Err("unsupported WebP format".into()),
    };

    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    fs::write(&output, bmp)?;

    println!("{}", output.display());
    Ok(())
}
