# Changelog

## 0.5.0

- Updated `bin-rs` from `0.0.11` to `0.1.0`, retaining the `stream` and `util` features.
- This is a breaking dependency-type change: `compat::decode`, `read_header`,
  and `read_u24` now accept the `bin-rs 0.1` `BinaryReader` trait.
  Callers passing readers from `bin-rs 0.0.11` must update their direct
  dependency to `bin-rs = "0.1"` together with `webp-rust = "0.5"`.
  Cargo treats these dependency versions as incompatible, so their reader
  types cannot be mixed at these public API boundaries.
- No codec implementation or byte-slice API changes are included. Animation
  callbacks, Abort handling, and the opt-in `legacy` feature are unchanged.
  Reader behavior follows the `bin-rs 0.1.0` contract.
- The `0.5.0` version keeps `webp-rust = "0.4"` consumers on the `0.4.x` series.
