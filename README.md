# webp-rust

Pure Rust WebP decoder and partial encoder.

## Status

- Still image decode: `VP8` lossy, `VP8L` lossless
- Still image encode: lossy `VP8` and lossless `VP8L` from RGBA
- Alpha: `ALPH` for lossy still images and lossy animation frames
- Animation: compositing to RGBA frame sequence
- Library output: RGBA only
- BMP output: example only

## Library API

Top-level still-image decode:

```rust
let image = webp_rust::decode(&data)?;
println!("{}x{}", image.width, image.height);
```

Top-level still-image encode:

```rust
let webp = webp_rust::encode(&image, None)?;
let lossy = webp_rust::encode_lossy(&image, None)?;
let lossless = webp_rust::encode_lossless(&image, None)?;
```

To embed raw EXIF metadata, pass the chunk payload directly:

```rust
let webp = webp_rust::encode_lossless(&image, Some(exif_bytes))?;
```

Native file input:

```rust
#[cfg(not(target_family = "wasm"))]
let image = webp_rust::decode_file("input.webp")?;
```

Animated WebP is not accepted by `decode` / `decode_file`.
For animation, use the decoder module directly:

```rust
let animation = webp_rust::decoder::decode_animation_webp(&data)?;
println!("{}", animation.frames.len());
```

Advanced encoder tuning stays in the `encoder` module:

```rust
let lossy_options = webp_rust::LossyEncodingOptions {
    quality: 90,
    optimization_level: 4,
};
let lossy = webp_rust::encoder::encode_lossy_image_to_webp_with_options_and_exif(
    &image,
    &lossy_options,
    Some(exif_bytes),
)?;

let lossless_options = webp_rust::LosslessEncodingOptions {
    optimization_level: 2,
};
let lossless = webp_rust::encoder::encode_lossless_image_to_webp_with_options_and_exif(
    &image,
    &lossless_options,
    Some(exif_bytes),
)?;
```

Current encoder scope is still-image only. The lossy path currently targets
opaque RGBA input and emits a minimal intra-only `VP8` bitstream. Animated
encode is not implemented.

## Examples

`webp2bmp` converts still WebP to a BMP file and animated WebP to a BMP sequence.

Still image:

```bash
cargo run --example webp2bmp -- _testdata/sample.webp target/sample.bmp
```

Animation:

```bash
cargo run --example webp2bmp -- _testdata/sample_animation.webp target/sample_animation
```

This writes:

- `target/sample_animation_0000.bmp`
- `target/sample_animation_0001.bmp`
- `...`

`bmp2webp` converts an uncompressed 24bpp or 32bpp BMP file to a still WebP.

Lossless:

```bash
cargo run --example bmp2webp -- --opt-level 2 input.bmp output.webp
```

Lossy:

```bash
cargo run --example bmp2webp -- --lossy --quality 90 input.bmp output.webp
```

Heavier lossy search:

```bash
cargo run --example bmp2webp -- --lossy --quality 90 -z 9 input.bmp output.webp
```

## Tests

```bash
cargo test --tests
```

## License

- Project code: see `LICENSE`
- Bundled libwebp reference sources: see `LICENSE-LIBWEBP`

(C) MITH@mmk 2026
