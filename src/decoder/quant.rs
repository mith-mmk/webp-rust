use std::array::from_fn;

use crate::decoder::vp8::Vp8BoolDecoder;
use crate::decoder::vp8i::NUM_MB_SEGMENTS;
use crate::decoder::DecoderError;

use super::vp8::SegmentHeader;

pub const DC_TABLE: [u8; 128] = [
    4, 5, 6, 7, 8, 9, 10, 10, 11, 12, 13, 14, 15, 16, 17, 17, 18, 19, 20, 20, 21, 21, 22, 22, 23,
    23, 24, 25, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 37, 38, 39, 40, 41, 42, 43, 44,
    45, 46, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
    68, 69, 70, 71, 72, 73, 74, 75, 76, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 91,
    93, 95, 96, 98, 100, 101, 102, 104, 106, 108, 110, 112, 114, 116, 118, 122, 124, 126, 128, 130,
    132, 134, 136, 138, 140, 143, 145, 148, 151, 154, 157,
];

pub const AC_TABLE: [u16; 128] = [
    4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
    29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
    53, 54, 55, 56, 57, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94,
    96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 119, 122, 125, 128, 131, 134, 137, 140,
    143, 146, 149, 152, 155, 158, 161, 164, 167, 170, 173, 177, 181, 185, 189, 193, 197, 201, 205,
    209, 213, 217, 221, 225, 229, 234, 239, 245, 249, 254, 259, 264, 269, 274, 279, 284,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantIndices {
    pub base_q0: i32,
    pub y1_dc_delta: i32,
    pub y2_dc_delta: i32,
    pub y2_ac_delta: i32,
    pub uv_dc_delta: i32,
    pub uv_ac_delta: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantMatrix {
    pub y1: [u16; 2],
    pub y2: [u16; 2],
    pub uv: [u16; 2],
    pub uv_quant: i32,
}

impl Default for QuantMatrix {
    fn default() -> Self {
        Self {
            y1: [0; 2],
            y2: [0; 2],
            uv: [0; 2],
            uv_quant: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Quantization {
    pub indices: QuantIndices,
    pub matrices: [QuantMatrix; NUM_MB_SEGMENTS],
}

fn clip(v: i32, max: i32) -> usize {
    v.clamp(0, max) as usize
}

pub fn parse_quantization(
    br: &mut Vp8BoolDecoder<'_>,
    segment_header: &SegmentHeader,
) -> Result<Quantization, DecoderError> {
    let indices = QuantIndices {
        base_q0: br.get_value(7) as i32,
        y1_dc_delta: if br.get() == 1 {
            br.get_signed_value(4)
        } else {
            0
        },
        y2_dc_delta: if br.get() == 1 {
            br.get_signed_value(4)
        } else {
            0
        },
        y2_ac_delta: if br.get() == 1 {
            br.get_signed_value(4)
        } else {
            0
        },
        uv_dc_delta: if br.get() == 1 {
            br.get_signed_value(4)
        } else {
            0
        },
        uv_ac_delta: if br.get() == 1 {
            br.get_signed_value(4)
        } else {
            0
        },
    };

    let matrices = from_fn(|segment| {
        let q = if segment_header.use_segment {
            let mut q = segment_header.quantizer[segment] as i32;
            if !segment_header.absolute_delta {
                q += indices.base_q0;
            }
            q
        } else {
            indices.base_q0
        };

        let mut matrix = QuantMatrix::default();
        matrix.y1[0] = DC_TABLE[clip(q + indices.y1_dc_delta, 127)] as u16;
        matrix.y1[1] = AC_TABLE[clip(q, 127)];

        matrix.y2[0] = (DC_TABLE[clip(q + indices.y2_dc_delta, 127)] as u16) * 2;
        matrix.y2[1] =
            ((AC_TABLE[clip(q + indices.y2_ac_delta, 127)] as u32 * 101_581) >> 16).max(8) as u16;

        matrix.uv[0] = DC_TABLE[clip(q + indices.uv_dc_delta, 117)] as u16;
        matrix.uv[1] = AC_TABLE[clip(q + indices.uv_ac_delta, 127)];
        matrix.uv_quant = q + indices.uv_ac_delta;
        matrix
    });

    if br.eof() {
        return Err(DecoderError::Bitstream("cannot parse quantization"));
    }

    Ok(Quantization { indices, matrices })
}
