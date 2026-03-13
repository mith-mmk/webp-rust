# WebP 実装概要

この文書は、この crate の WebP 実装を読むための日本語 overview です。
目的は次の 3 つです。

- WebP のフォーマットを RFC 9649 ベースで整理する
- decoder / encoder をどの順で実装すべきかを示す
- このリポジトリのソース配置と責務を対応付ける

この crate は pure Rust の WebP 実装です。現状の大きな範囲は次のとおりです。

- decode: still lossy (`VP8`), still lossless (`VP8L`), alpha, animation compositing
- encode: still lossy (`VP8`), still lossless (`VP8L`)
- library の画像出力: RGBA
- example の出力: BMP

## 1. WebP とは

WebP は RIFF ベースの画像コンテナです。RFC 9649 は WebP container, animation, metadata, lossless bitstream (`VP8L`) を定義し、lossy payload としては VP8 bitstream を参照します。

WebP は大きく次の 3 形態に分かれます。

- Simple lossy: `RIFF` + `WEBP` + `VP8 `
- Simple lossless: `RIFF` + `WEBP` + `VP8L`
- Extended WebP: `RIFF` + `WEBP` + `VP8X` + optional chunks

optional chunks の代表は次です。

- `ALPH`: lossy image / frame 用の alpha plane
- `ICCP`: ICC profile
- `EXIF`: Exif metadata
- `XMP `: XMP metadata
- `ANIM`: animation 全体設定
- `ANMF`: animation の各 frame

## 2. RIFF / Chunk の基本

WebP file は RIFF container なので、まずここを正しく実装する必要があります。

- file header は `RIFF` + 32-bit little-endian size + `WEBP`
- 以降は chunk の列
- 各 chunk は `FourCC(4 bytes)` + `Chunk Size(32-bit little-endian)` + payload
- payload が奇数長なら 1 byte padding が入る

実装の第一段階は「chunk を安全に読むこと」です。codec 本体より先に、以下を必ずやるべきです。

- chunk size overflow の検査
- RIFF size と実 buffer 長の整合
- odd padding の吸収
- unknown chunk の skip
- mandatory / optional chunk の順序制約の確認

この crate では container 周りを次が担当します。

- `src/decoder/header.rs`
- `src/encoder/container.rs`
- `src/encoder/writer.rs`

## 3. Extended WebP (`VP8X`)

`VP8X` は extended header です。ここで feature flags と canvas size を持ちます。

- canvas width / height は 24-bit little-endian の `minus one` 表現
- alpha, EXIF, XMP, ICC, animation の有無を flag で持つ
- animation の場合は canvas 全体サイズの基準になる

実装上の注意点です。

- width / height は `stored + 1`
- `width * height <= 2^32 - 1` の制約がある
- `VP8X` がある場合は後続 chunk の feature と矛盾しないか確認する
- unknown flag は将来拡張に備えて無視できる形にする

この crate では `get_features()` と container parser がここを処理します。

- `src/decoder/header.rs`

## 4. Lossy WebP (`VP8 `)

lossy payload は `VP8 ` chunk に入ります。ここでの payload 自体は VP8 key frame bitstream です。

decode の流れは概ね次です。

1. frame header を読む
2. segmentation / filter / quantizer を読む
3. partition 0 と token partitions を読む
4. macroblock の intra mode を読む
5. residual coefficients を読む
6. inverse transform を適用する
7. intra prediction と合成して Y/U/V を復元する
8. loop filter をかける
9. YUV420 から RGBA に変換する

lossy 実装の要点です。

- VP8 bool coder が必要
- luma は `i16x16` または `i4x4`
- chroma は 8x8 の intra prediction
- residual は zig-zag 順の token 列
- dequant と inverse transform の順序を間違えると全面的に崩れる
- loop filter を省いても decode はできるが、参照実装と完全一致しにくい

この crate の対応箇所です。

- parser / low-level state:
  - `src/decoder/vp8.rs`
  - `src/decoder/quant.rs`
  - `src/decoder/tree.rs`
  - `src/decoder/vp8i.rs`
- lossy decode:
  - `src/decoder/lossy.rs`
- lossy encode:
  - `src/encoder/lossy/predict.rs`
  - `src/encoder/lossy/bitstream.rs`
  - `src/encoder/lossy/api.rs`

lossy encoder の実装順は、一般には次が安全です。

1. RGBA -> YUV420
2. intra prediction
3. forward transform / quantize
4. token encode
5. partition 0 encode
6. frame assemble
7. mode search
8. filter / segmentation 最適化

この crate でも `predict.rs` と `bitstream.rs` を分離して、この順序が追える構造にしています。

## 5. Alpha (`ALPH`)

`ALPH` は lossy still image または animation frame に付く alpha chunk です。

- `ALPH` は color payload とは別に alpha plane を持つ
- alpha plane には filtering がある
- alpha payload は raw または lossless-compressed の場合がある
- 実合成では color decode 後に alpha を RGBA の A channel に適用する

実装上は、`VP8 + ALPH` を `VP8L` と混同しないことが重要です。container 上は別 feature です。

この crate では次を使います。

- `src/decoder/alpha.rs`
- `src/decoder/lossy.rs`
- `src/decoder/animation.rs`

## 6. Lossless WebP (`VP8L`)

`VP8L` は lossy の `VP8 ` と完全に別の codec です。bitstream は 1-byte signature `0x2f` から始まり、その後に width / height などが続きます。

decode の高レベルな流れは次です。

1. 14-bit width / height を読む
2. transform chain を読む
3. color cache の有無を読む
4. Huffman / meta-Huffman を読む
5. token stream を decode する
6. backward references を展開する
7. transform を逆順に戻す
8. RGBA を得る

`VP8L` の重要要素は次です。

- subtract-green transform
- predictor transform
- cross-color transform
- color indexing transform (palette)
- color cache
- backward references
- Huffman / meta-Huffman

実装では「transform chain」「token stream」「entropy coding」を分けると保守しやすくなります。

この crate の対応は次です。

- decode:
  - `src/decoder/lossless.rs`
- encode:
  - `src/encoder/lossless/plans.rs`
  - `src/encoder/lossless/tokens.rs`
  - `src/encoder/lossless/entropy.rs`
  - `src/encoder/lossless/api.rs`

### 6.1 transform

lossless encode/decode で最初に理解すべきなのは transform です。

- subtract-green: R/B から G を差し引く
- predictor: 近傍画素から予測して residual を持つ
- cross-color: channel 間相関を減らす
- color indexing: palette 化して index を詰める

encoder では「どの transform を使うか」の探索が圧縮率に効きます。decode では「reverse order で正確に戻す」ことが重要です。

この crate では `plans.rs` がそれを担当します。

### 6.2 backward references / LZ77

lossless のサイズ効率を大きく左右するのは backward references です。

- literal を出すか
- color cache hit を出すか
- copy(distance, length) を出すか

この選択は単純 greedy でも動きますが、圧縮率を詰めるには cost model と traceback が必要になります。

この crate では `tokens.rs` に以下を集めています。

- match search
- window offsets
- lazy matching
- traceback
- cache-aware token build

### 6.3 Huffman / meta-Huffman

`VP8L` は単純な 1 本の Huffman tree だけではなく、tile ごとに histogram group を切る meta-Huffman を使えます。

ここで必要になるのは次です。

- token histogram の構築
- group ごとの tree 生成
- tile -> group assignment
- group 数や huffman bits の探索

この crate では `entropy.rs` にまとめています。

## 7. Animation (`ANIM` / `ANMF`)

animation は extended WebP の上に載ります。

- `ANIM` は background color と loop count
- `ANMF` は frame rectangle, duration, blend, dispose を持つ
- `Frame X`, `Frame Y` は half-pixel 単位で保存され、実座標は `2 * X`, `2 * Y`
- frame payload は `ALPH` + `VP8 ` または `VP8L` の組み合わせになれる

decode で重要なのは「frame を decode すること」より「canvas に正しく合成すること」です。

- blend: 上書きか alpha blend か
- dispose: 次 frame 前に背景へ戻すか
- frame rectangle が canvas 内に収まるか
- loop count の扱い

この crate では animation decode を次が扱います。

- `src/decoder/animation.rs`
- `src/decoder/header.rs`

現状の encoder は still image のみで、animation encode は未実装です。

## 8. Metadata (`EXIF`, `XMP `, `ICCP`)

extended WebP では metadata chunk を直接埋め込めます。

- `EXIF`
- `XMP `
- `ICCP`

この crate の top-level API は EXIF raw payload をそのまま受け取る設計です。

```rust
let webp = webp_rust::encode_lossless(&image, 6, Some(exif_bytes))?;
```

この設計にしている理由は単純です。

- metadata parser / serializer を encoder 本体から切り離せる
- caller 側が既に持っている Exif blob をそのまま流せる
- WebP container 実装は chunk を埋め込むだけで済む

対応箇所:

- `src/lib.rs`
- `src/encoder/container.rs`

## 9. この crate の読み方

実装を追う順番は次が分かりやすいです。

### 9.1 API 入口

- `src/lib.rs`
- `src/decoder/mod.rs`
- `src/encoder/mod.rs`

### 9.2 Container と feature 判定

- `src/decoder/header.rs`
- `src/encoder/container.rs`

### 9.3 Lossy decode / encode

- decode:
  - `src/decoder/vp8.rs`
  - `src/decoder/lossy.rs`
- encode:
  - `src/encoder/lossy/mod.rs`
  - `src/encoder/lossy/predict.rs`
  - `src/encoder/lossy/bitstream.rs`
  - `src/encoder/lossy/api.rs`

### 9.4 Lossless decode / encode

- decode:
  - `src/decoder/lossless.rs`
- encode:
  - `src/encoder/lossless/mod.rs`
  - `src/encoder/lossless/plans.rs`
  - `src/encoder/lossless/tokens.rs`
  - `src/encoder/lossless/entropy.rs`
  - `src/encoder/lossless/api.rs`

### 9.5 Animation / alpha

- `src/decoder/alpha.rs`
- `src/decoder/animation.rs`

## 10. 実装時の設計指針

WebP は container と codec の境界が比較的明確なので、以下の分割が保守しやすいです。

### 10.1 parser と codec を分ける

- parser は chunk 順序と feature flags を扱う
- codec は payload の bitstream を扱う

### 10.2 decode と encode を対称にしすぎない

decode は spec 準拠が最優先ですが、encode は探索と heuristic が大きいです。decode と encode を 1 対 1 に揃えようとすると設計が窮屈になります。

この crate でも次のように分けています。

- decode: codec ごとの module
- encode: search / token / entropy / frame assembly を分離

### 10.3 bit-level writer と byte-level writer を分ける

WebP 実装では両方必要です。

- byte-level:
  - RIFF
  - chunk header
  - frame header
- bit-level:
  - VP8 bool coder
  - VP8L bit writer

この crate では byte-oriented な部分を `bin-rs` ベースの `ByteWriter` に寄せています。

- `src/encoder/writer.rs`
- `src/encoder/container.rs`

一方で bit coder は codec 固有なので自前実装です。

- `src/encoder/bit_writer.rs`
- `src/encoder/vp8_bool_writer.rs`

### 10.4 先に正しさ、その後に最適化

特に encoder は、次の順で進めると安全です。

1. まず decode できる bitstream を出す
2. 参照 decoder と画素一致を取る
3. その後に mode search / transform search / traceback を入れる

lossless encode は最適化の影響範囲が広いので、以下の分割が有効です。

- transform 候補探索
- token build
- entropy coding
- candidate 比較

この構造はこの crate でもそのまま採っています。

## 11. 現状の制限

現状の主な制限です。

- library の decode 出力は RGBA のみ
- encoder は still image のみ
- lossy encoder は opaque RGBA 前提
- animation encode は未実装
- BMP は example でのみ扱う

この範囲でも container / lossy / lossless / alpha / animation compositing の主要部は読めます。

## 12. 参考資料

- RFC 9649: WebP Image Format
  - https://www.rfc-editor.org/rfc/rfc9649.html
- IETF Datatracker: RFC 9649
  - https://datatracker.ietf.org/doc/html/rfc9649
- VP8 bitstream reference mentioned by RFC 9649
  - RFC 6386: https://www.rfc-editor.org/rfc/rfc6386

## 13. 付記

WebP 実装を壊しやすい箇所は、lossy では prediction と transform、lossless では transform 順序と entropy coding、animation では canvas compositing です。バグが出たときは、まず container ではなくこの 3 層のどこでズレたかを切ると追いやすくなります。
