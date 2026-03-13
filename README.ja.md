# webp-rust

[English](README.md) | [日本語](README.ja.md) | [実装概要](OVERVIEW.ja.md)

Pure Rust の WebP decoder / encoder です。

`OVERVIEW.ja.md` には、RFC 9649 ベースの WebP 技術解説と、この crate の実装方針をまとめています。

## 対応状況

- still image decode: lossy `VP8`, lossless `VP8L`
- still image encode: lossy `VP8`, lossless `VP8L`
- alpha: lossy still image と lossy animation frame の `ALPH`
- animation: RGBA frame sequence への compositing
- library 出力: RGBA
- BMP 出力: example のみ

## ライブラリ API

still image の decode:

```rust
let image = webp_rust::decode(&data)?;
println!("{}x{}", image.width, image.height);
```

still image の encode:

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

raw EXIF payload をそのまま埋め込む場合:

```rust
let webp = webp_rust::encode_lossless(&image, 2, Some(exif_bytes))?;
```

native 環境での file input:

```rust
#[cfg(not(target_family = "wasm"))]
let image = webp_rust::decode_file("input.webp")?;
```

`decode` / `decode_file` は animated WebP を受けません。animation は decoder module を直接使います。

```rust
let animation = webp_rust::decoder::decode_animation_webp(&data)?;
println!("{}", animation.frames.len());
```

高度な encode option は `encoder` module 側にあります。

```rust
let lossy_options = webp_rust::LossyEncodingOptions {
    quality: 90,
    optimization_level: 0,
};
let lossy = webp_rust::encoder::encode_lossy_image_to_webp_with_options_and_exif(
    &image,
    &lossy_options,
    Some(exif_bytes),
)?;

let lossless_options = webp_rust::LosslessEncodingOptions {
    optimization_level: 6,
};
let lossless = webp_rust::encoder::encode_lossless_image_to_webp_with_options_and_exif(
    &image,
    &lossless_options,
    Some(exif_bytes),
)?;
```

現在の encoder は still image のみです。lossy encode は opaque RGBA を前提にした intra-only `VP8` bitstream を出力します。animation encode は未実装です。

## Examples

`webp2bmp` は still WebP を BMP に、animated WebP を連番 BMP に変換します。

still image:

```bash
cargo run --example webp2bmp -- _testdata/sample.webp target/sample.bmp
```

animation:

```bash
cargo run --example webp2bmp -- _testdata/sample_animation.webp target/sample_animation
```

出力例:

- `target/sample_animation_0000.bmp`
- `target/sample_animation_0001.bmp`
- `...`

`bmp2webp` は uncompressed 24bpp / 32bpp BMP を still WebP に変換します。

lossless:

```bash
cargo run --example bmp2webp -- --opt-level 6 input.bmp output.webp
```

lossy:

```bash
cargo run --example bmp2webp -- --lossy --quality 90 input.bmp output.webp
```

デフォルトの lossy path は `-z 0` です。

lossless effort は `-z 0..9` を受けます。`-z 6` が balanced preset です。`z7` は current heavy preset、`z8..9` は現在この heavy path を再利用しています。

より重い lossy search:

```bash
cargo run --example bmp2webp -- --lossy --quality 90 -z 9 input.bmp output.webp
```

## テスト

```bash
cargo test --tests
```

## 関連文書

- [実装概要](OVERVIEW.ja.md)
- [英語 README](README.md)

## ライセンス

- project code: `LICENSE`
- bundled libwebp reference sources: `LICENSE-LIBWEBP`

(C) MITH@mmk 2026
