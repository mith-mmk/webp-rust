pub const MB_FEATURE_TREE_PROBS: usize = 3;
pub const NUM_MB_SEGMENTS: usize = 4;
pub const NUM_REF_LF_DELTAS: usize = 4;
pub const NUM_MODE_LF_DELTAS: usize = 4;
pub const MAX_NUM_PARTITIONS: usize = 8;
pub const NUM_TYPES: usize = 4;
pub const NUM_BANDS: usize = 8;
pub const NUM_CTX: usize = 3;
pub const NUM_PROBAS: usize = 11;
pub const NUM_BMODES: usize = 10;

pub const B_DC_PRED: u8 = 0;
pub const B_TM_PRED: u8 = 1;
pub const B_VE_PRED: u8 = 2;
pub const B_HE_PRED: u8 = 3;
pub const B_RD_PRED: u8 = 4;
pub const B_VR_PRED: u8 = 5;
pub const B_LD_PRED: u8 = 6;
pub const B_VL_PRED: u8 = 7;
pub const B_HD_PRED: u8 = 8;
pub const B_HU_PRED: u8 = 9;

pub const DC_PRED: u8 = B_DC_PRED;
pub const TM_PRED: u8 = B_TM_PRED;
pub const V_PRED: u8 = B_VE_PRED;
pub const H_PRED: u8 = B_HE_PRED;
pub const B_PRED: u8 = NUM_BMODES as u8;

pub const TAG_SIZE: usize = 4;
pub const CHUNK_HEADER_SIZE: usize = 8;
pub const RIFF_HEADER_SIZE: usize = 12;
pub const VP8_FRAME_HEADER_SIZE: usize = 10;
pub const VP8L_FRAME_HEADER_SIZE: usize = 5;
pub const VP8X_CHUNK_SIZE: usize = 10;
pub const MAX_CHUNK_PAYLOAD: usize = u32::MAX as usize - CHUNK_HEADER_SIZE - 1;
pub const MAX_IMAGE_AREA: u64 = 1u64 << 32;

pub const ANIMATION_FLAG: u32 = 0x0000_0002;
pub const ALPHA_FLAG: u32 = 0x0000_0010;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebpFormat {
    Undefined = 0,
    Lossy = 1,
    Lossless = 2,
}
