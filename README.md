# webp-rust

Pure Rust WebP decoder and partial encoder.

## Status

- Still image decode: `VP8` lossy, `VP8L` lossless
- Still image encode: lossless `VP8L` from RGBA
- Alpha: `ALPH` for lossy still images and lossy animation frames
- Animation: compositing to RGBA frame sequence
- Library output: RGBA only
- BMP output: example only

## Library API

Still images:

```rust
let image = webp_rust::image_from_bytes(&data)?;
println!("{}x{}", image.width, image.height);
```

Native file input:

```rust
#[cfg(not(target_family = "wasm"))]
let image = webp_rust::image_from_file("input.webp".to_string())?;
```

Animated WebP is not accepted by `image_from_bytes` / `image_from_file`.
For animation, use the decoder module directly:

```rust
let animation = webp_rust::decoder::decode_animation_webp(&data)?;
println!("{}", animation.frames.len());
```

Lossless encoding:

```rust
let webp = webp_rust::encode_lossless_rgba_to_webp(width, height, &rgba)?;
```

Current encoder scope is still-image lossless only. Lossy encode and animated
encode are not implemented.

## Example

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

## Tests

```bash
cargo test --tests
```

## License

- Project code: see `LICENSE`
- Bundled libwebp reference sources: see `LICENSE-LIBWEBP`

(C) MITH@mmk 2026
