# webp-rust 0.3.0

[English](README.md) | [日本語](README.ja.md) | [Overview (JA)](OVERVIEW.ja.md)

Pure Rust WebP decoder and partial encoder.

## Status

- Still image decode: `VP8` lossy, `VP8L` lossless
- Still image encode: lossy `VP8` and lossless `VP8L` from RGBA
- Alpha: `ALPH` for lossy still images and lossy animation frames
- Libwebp-style encode configuration for quality, method, filtering, segments,
  sharp YUV, partition limit, near-lossless, and alpha options
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
let webp = webp_rust::encode(
    &image,
    2,
    100,
    webp_rust::WebpEncoding::Lossless,
    None,
)?;
let lossy = webp_rust::encode_lossy(&image, 0, 90, None)?;
let lossless = webp_rust::encode_lossless(&image, 2, None)?;
```

To embed raw EXIF metadata, pass the chunk payload directly:

```rust
let webp = webp_rust::encode_lossless(&image, 2, Some(exif_bytes))?;
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

The standard encoder API uses libwebp-style configuration objects:

```rust
let lossy_config = webp_rust::LossyEncodingConfig {
    quality: 75.0,
    method: 4,
    ..Default::default()
};
let lossy = webp_rust::encode_lossy_with_config(&image, &lossy_config, Some(exif_bytes))?;

let lossless_config = webp_rust::LosslessEncodingConfig {
    z_level: 6,
};
let lossless = webp_rust::encode_lossless_with_config(&image, &lossless_config, Some(exif_bytes))?;
```

`LossyEncodingConfig::preset(WebpPreset::Photo)` and the other stable presets
provide convenient libwebp-style starting points. Exact output bytes are not
guaranteed to match a particular libwebp release.

The old `optimization_level` API is isolated behind the opt-in Cargo feature:

```toml
webp-rust = { version = "0.3", features = ["legacy"] }
```

The legacy feature is disabled by default. Animated encode is not implemented.

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
cargo run --example bmp2webp -- --opt-level 6 input.bmp output.webp
```

Lossy:

```bash
cargo run --example bmp2webp -- --lossy --quality 90 input.bmp output.webp
```

This default lossy path uses `-z 0` for fast encode speed.

Lossless effort also accepts `-z 0..9`. `-z 6` is the balanced preset.
`z7` is the current heavy preset, and `z8..9` currently reuse that path until a
better high-effort strategy lands.

Heavier lossy search:

```bash
cargo run --example bmp2webp -- --lossy --quality 90 -z 9 input.bmp output.webp
```

## Tests

```bash
cargo test --tests
cargo test --tests --features legacy
cargo run --example webp_bench
```

`webp_bench` writes temporary BMP/WebP files only below `.test-webp-bench`
and compares the Rust encoder with `cwebp` when it is available on `PATH`.

## License

- Project code: see `LICENSE`

(C) MITH@mmk 2026
