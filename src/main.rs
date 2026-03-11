use std::fs;
use std::path::{Path, PathBuf};

use webp_rust::decoder::decode_lossy_webp_to_bmp;

type Error = Box<dyn std::error::Error>;

fn default_output_path(input: &Path) -> PathBuf {
    let mut output = input.to_path_buf();
    output.set_extension("bmp");
    output
}

pub fn main() -> Result<(), Error> {
    let mut args = std::env::args_os().skip(1);
    let input = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("_testdata/sample.webp"));
    let output = args.next().map(PathBuf::from).unwrap_or_else(|| {
        if input == PathBuf::from("_testdata/sample.webp") {
            PathBuf::from("target/sample.bmp")
        } else {
            default_output_path(&input)
        }
    });

    let data = fs::read(&input)?;
    let bmp = decode_lossy_webp_to_bmp(&data)?;

    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    fs::write(&output, bmp)?;

    println!("{}", output.display());
    Ok(())
}
