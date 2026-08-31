# webp-rust 0.3.1

[English](README.md) | [日本語](README.ja.md) | [実装概要](OVERVIEW.ja.md)

Pure Rust の WebP decoder / encoder です。

`OVERVIEW.ja.md` には、RFC 9649 ベースの WebP 技術解説と、この crate の実装方針をまとめています。

## 対応状況

- still image decode: lossy `VP8`, lossless `VP8L`
- still image encode: lossy `VP8`, lossless `VP8L`
- alpha: lossy still image と lossy animation frame の `ALPH`
- libwebp 互換の quality / method / filter / segment / sharp YUV /
  partition limit / near-lossless / alpha option
- animation: RGBA frame sequence への compositing
- library 出力: RGBA、および lower-level decoder API による VP8 の planar YUV420
- BMP 出力: example のみ

## 0.3.1 decoder 更新

- VP8 は macroblock を行単位で再構成し、loop filter 用の小型 metadata だけを保持します。
- VP8L は buffered bit reader、2段 Huffman table、in-place inverse transform、
  重なり対応の chunk copy を使用します。
- animation compositing に行コピーと透明・不透明 pixel の高速経路を追加しました。
- 公開 decode API、RGBA/YUV 出力、error 動作、callback compatibility layer は変更していません。

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

標準の encoder API は libwebp 互換の config を使います。

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

`LossyEncodingConfig::preset(WebpPreset::Photo)` などで libwebp 互換の
プリセットを利用できます。特定の libwebp 版との byte-perfect 一致は保証しません。

旧 `optimization_level` API は opt-in の Cargo feature に分離されています。

```toml
webp-rust = { version = "0.3", features = ["legacy"] }
```

`legacy` は既定無効です。animation encode は未実装です。

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
cargo test --tests --features legacy
cargo run --example webp_bench
cargo run --example webp_compare
cargo run --release --example webp_decode_bench -- --output target/webp_decode.csv
```

`webp_bench` は一時 BMP/WebP を `.test-webp-bench` 配下だけに作成し、
`PATH` 上の `cwebp` が利用できる場合は Rust encoder と比較します。
`webp_compare` は lossless の `z_level` と lossy の quality/method について、
画像ごとおよび平均の出力サイズ、圧縮率、RGB PSNR、エンコード時間をCSVで出力します。
`webp_decode_bench` は VP8 RGBA、VP8 YUV、VP8L RGBA、animation 全体を7バッチで測定し、
median、p95、MPixel/s をCSV出力します。

## 関連文書

- [実装概要](OVERVIEW.ja.md)
- [英語 README](README.md)

## ライセンス

- project code: `LICENSE`

(C) MITH@mmk 2026
