use crate::decoder::quant::{parse_quantization, Quantization};
use crate::decoder::tree::{
    parse_intra_mode_row, parse_probability_tables, parse_probability_updates, MacroBlockHeader,
    ProbabilityTables, ProbabilityUpdateSummary,
};
use crate::decoder::vp8i::{
    B_DC_PRED, MAX_NUM_PARTITIONS, MB_FEATURE_TREE_PROBS, NUM_MB_SEGMENTS, NUM_MODE_LF_DELTAS,
    NUM_REF_LF_DELTAS, VP8L_FRAME_HEADER_SIZE, VP8_FRAME_HEADER_SIZE,
};
use crate::decoder::DecoderError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Vp8FrameHeader {
    pub key_frame: bool,
    pub profile: u8,
    pub show: bool,
    pub partition_length: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Vp8PictureHeader {
    pub width: u16,
    pub height: u16,
    pub xscale: u8,
    pub yscale: u8,
    pub colorspace: u8,
    pub clamp_type: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegmentHeader {
    pub use_segment: bool,
    pub update_map: bool,
    pub absolute_delta: bool,
    pub quantizer: [i8; NUM_MB_SEGMENTS],
    pub filter_strength: [i8; NUM_MB_SEGMENTS],
    pub segment_probs: [u8; MB_FEATURE_TREE_PROBS],
}

impl Default for SegmentHeader {
    fn default() -> Self {
        Self {
            use_segment: false,
            update_map: false,
            absolute_delta: true,
            quantizer: [0; NUM_MB_SEGMENTS],
            filter_strength: [0; NUM_MB_SEGMENTS],
            segment_probs: [255; MB_FEATURE_TREE_PROBS],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterType {
    Off,
    Simple,
    Complex,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FilterHeader {
    pub simple: bool,
    pub level: u8,
    pub sharpness: u8,
    pub use_lf_delta: bool,
    pub ref_lf_delta: [i8; NUM_REF_LF_DELTAS],
    pub mode_lf_delta: [i8; NUM_MODE_LF_DELTAS],
    pub filter_type: FilterType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LosslessInfo {
    pub width: usize,
    pub height: usize,
    pub has_alpha: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LossyHeader {
    pub frame: Vp8FrameHeader,
    pub picture: Vp8PictureHeader,
    pub macroblock_width: usize,
    pub macroblock_height: usize,
    pub segment: SegmentHeader,
    pub filter: FilterHeader,
    pub token_partition_sizes: Vec<usize>,
    pub quantization: Quantization,
    pub probabilities: ProbabilityUpdateSummary,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MacroBlockHeaders {
    pub frame: LossyHeader,
    pub macroblocks: Vec<MacroBlockHeader>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MacroBlockData {
    pub header: MacroBlockHeader,
    pub coeffs: [i16; 384],
    pub non_zero_y: u32,
    pub non_zero_uv: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MacroBlockDataFrame {
    pub frame: LossyHeader,
    pub macroblocks: Vec<MacroBlockData>,
}

#[derive(Debug, Clone, Copy, Default)]
struct NonZeroContext {
    nz: u8,
    nz_dc: u8,
}

#[derive(Debug, Clone)]
pub struct Vp8BoolDecoder<'a> {
    data: &'a [u8],
    position: usize,
    value: u64,
    range: u32,
    bits: i32,
    eof: bool,
}

impl<'a> Vp8BoolDecoder<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        let mut reader = Self {
            data,
            position: 0,
            value: 0,
            range: 255 - 1,
            bits: -8,
            eof: false,
        };
        reader.load_new_bytes();
        reader
    }

    pub fn eof(&self) -> bool {
        self.eof
    }

    fn load_new_bytes(&mut self) {
        while self.bits < 0 {
            if self.position < self.data.len() {
                self.bits += 8;
                self.value = (self.value << 8) | self.data[self.position] as u64;
                self.position += 1;
            } else if !self.eof {
                self.value <<= 8;
                self.bits += 8;
                self.eof = true;
            } else {
                self.bits = 0;
            }
        }
    }

    pub fn get(&mut self) -> u32 {
        self.get_bit(0x80)
    }

    pub fn get_value(&mut self, num_bits: usize) -> u32 {
        let mut value = 0u32;
        for bit_index in (0..num_bits).rev() {
            value |= self.get() << bit_index;
        }
        value
    }

    pub fn get_signed_value(&mut self, num_bits: usize) -> i32 {
        let value = self.get_value(num_bits) as i32;
        if self.get() == 1 {
            -value
        } else {
            value
        }
    }

    pub fn get_signed(&mut self, value: i32) -> i32 {
        if self.get() == 1 {
            -value
        } else {
            value
        }
    }

    pub fn get_bit(&mut self, prob: u8) -> u32 {
        if self.bits < 0 {
            self.load_new_bytes();
        }

        let pos = self.bits as u32;
        let mut range = self.range;
        let split = (range * prob as u32) >> 8;
        let value = (self.value >> pos) as u32;
        let bit = (value > split) as u32;
        if bit == 1 {
            range -= split;
            self.value -= ((split + 1) as u64) << pos;
        } else {
            range = split + 1;
        }

        let shift = 7 ^ (31 - range.leading_zeros()) as i32;
        range <<= shift as u32;
        self.bits -= shift;
        self.range = range - 1;
        bit
    }
}

pub fn check_lossy_signature(data: &[u8]) -> bool {
    data.len() >= 3 && data[0] == 0x9d && data[1] == 0x01 && data[2] == 0x2a
}

pub fn get_info(data: &[u8], chunk_size: usize) -> Result<(usize, usize), DecoderError> {
    if data.len() < VP8_FRAME_HEADER_SIZE {
        return Err(DecoderError::NotEnoughData("VP8 frame header"));
    }
    if !check_lossy_signature(&data[3..]) {
        return Err(DecoderError::Bitstream("bad VP8 signature"));
    }

    let bits = data[0] as u32 | ((data[1] as u32) << 8) | ((data[2] as u32) << 16);
    let key_frame = (bits & 1) == 0;
    let profile = ((bits >> 1) & 0x07) as u8;
    let show = ((bits >> 4) & 1) == 1;
    let partition_length = (bits >> 5) as usize;
    let width = ((((data[7] as u16) << 8) | data[6] as u16) & 0x3fff) as usize;
    let height = ((((data[9] as u16) << 8) | data[8] as u16) & 0x3fff) as usize;

    if !key_frame {
        return Err(DecoderError::Unsupported("interframes are not supported"));
    }
    if profile > 3 {
        return Err(DecoderError::Bitstream("unknown VP8 profile"));
    }
    if !show {
        return Err(DecoderError::Unsupported("invisible VP8 frame"));
    }
    if partition_length >= chunk_size {
        return Err(DecoderError::Bitstream("bad VP8 partition length"));
    }
    if width == 0 || height == 0 {
        return Err(DecoderError::Bitstream("invalid VP8 dimensions"));
    }

    Ok((width, height))
}

pub fn check_lossless_signature(data: &[u8]) -> bool {
    data.len() >= VP8L_FRAME_HEADER_SIZE && data[0] == 0x2f && (data[4] >> 5) == 0
}

pub fn get_lossless_info(data: &[u8]) -> Result<LosslessInfo, DecoderError> {
    if data.len() < VP8L_FRAME_HEADER_SIZE {
        return Err(DecoderError::NotEnoughData("VP8L frame header"));
    }
    if !check_lossless_signature(data) {
        return Err(DecoderError::Bitstream("bad VP8L signature"));
    }

    let bits = u32::from_le_bytes([data[1], data[2], data[3], data[4]]);
    let width = ((bits & 0x3fff) + 1) as usize;
    let height = (((bits >> 14) & 0x3fff) + 1) as usize;
    let has_alpha = ((bits >> 28) & 1) == 1;
    let version = (bits >> 29) & 0x07;

    if version != 0 {
        return Err(DecoderError::Bitstream("unsupported VP8L version"));
    }

    Ok(LosslessInfo {
        width,
        height,
        has_alpha,
    })
}

fn parse_segment_header(br: &mut Vp8BoolDecoder<'_>) -> Result<SegmentHeader, DecoderError> {
    let mut header = SegmentHeader {
        use_segment: br.get() == 1,
        ..SegmentHeader::default()
    };
    if header.use_segment {
        header.update_map = br.get() == 1;
        if br.get() == 1 {
            header.absolute_delta = br.get() == 1;
            for value in &mut header.quantizer {
                *value = if br.get() == 1 {
                    br.get_signed_value(7) as i8
                } else {
                    0
                };
            }
            for value in &mut header.filter_strength {
                *value = if br.get() == 1 {
                    br.get_signed_value(6) as i8
                } else {
                    0
                };
            }
        }
        if header.update_map {
            for value in &mut header.segment_probs {
                *value = if br.get() == 1 {
                    br.get_value(8) as u8
                } else {
                    255
                };
            }
        }
    }

    if br.eof() {
        return Err(DecoderError::Bitstream("cannot parse segment header"));
    }

    Ok(header)
}

fn parse_filter_header(br: &mut Vp8BoolDecoder<'_>) -> Result<FilterHeader, DecoderError> {
    let simple = br.get() == 1;
    let level = br.get_value(6) as u8;
    let sharpness = br.get_value(3) as u8;
    let use_lf_delta = br.get() == 1;
    let mut header = FilterHeader {
        simple,
        level,
        sharpness,
        use_lf_delta,
        ref_lf_delta: [0; NUM_REF_LF_DELTAS],
        mode_lf_delta: [0; NUM_MODE_LF_DELTAS],
        filter_type: FilterType::Off,
    };

    if use_lf_delta && br.get() == 1 {
        for value in &mut header.ref_lf_delta {
            if br.get() == 1 {
                *value = br.get_signed_value(6) as i8;
            }
        }
        for value in &mut header.mode_lf_delta {
            if br.get() == 1 {
                *value = br.get_signed_value(6) as i8;
            }
        }
    }

    header.filter_type = if level == 0 {
        FilterType::Off
    } else if simple {
        FilterType::Simple
    } else {
        FilterType::Complex
    };

    if br.eof() {
        return Err(DecoderError::Bitstream("cannot parse filter header"));
    }

    Ok(header)
}

fn parse_token_partitions(
    br: &mut Vp8BoolDecoder<'_>,
    data: &[u8],
) -> Result<Vec<usize>, DecoderError> {
    let num_parts_minus_one = (1usize << br.get_value(2)) - 1;
    if num_parts_minus_one >= MAX_NUM_PARTITIONS {
        return Err(DecoderError::Bitstream("too many VP8 token partitions"));
    }

    let size_bytes = num_parts_minus_one * 3;
    if data.len() < size_bytes {
        return Err(DecoderError::NotEnoughData("VP8 token partition sizes"));
    }

    let mut partitions = Vec::with_capacity(num_parts_minus_one + 1);
    let mut size_left = data.len() - size_bytes;
    for chunk in data[..size_bytes].chunks_exact(3) {
        let stored = chunk[0] as usize | ((chunk[1] as usize) << 8) | ((chunk[2] as usize) << 16);
        let actual = stored.min(size_left);
        partitions.push(actual);
        size_left -= actual;
    }
    partitions.push(size_left);

    if data.len() == size_bytes {
        return Err(DecoderError::NotEnoughData("VP8 token partitions"));
    }

    Ok(partitions)
}

const CAT3: [u8; 4] = [173, 148, 140, 0];
const CAT4: [u8; 5] = [176, 155, 140, 135, 0];
const CAT5: [u8; 6] = [180, 157, 141, 134, 130, 0];
const CAT6: [u8; 12] = [254, 254, 243, 230, 196, 177, 153, 140, 133, 130, 129, 0];
const ZIGZAG: [usize; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];

fn transform_wht(input: &[i16; 16]) -> [i16; 16] {
    let mut tmp = [0i32; 16];
    for i in 0..4 {
        let a0 = input[i] as i32 + input[12 + i] as i32;
        let a1 = input[4 + i] as i32 + input[8 + i] as i32;
        let a2 = input[4 + i] as i32 - input[8 + i] as i32;
        let a3 = input[i] as i32 - input[12 + i] as i32;
        tmp[i] = a0 + a1;
        tmp[8 + i] = a0 - a1;
        tmp[4 + i] = a3 + a2;
        tmp[12 + i] = a3 - a2;
    }

    let mut out = [0i16; 16];
    for i in 0..4 {
        let base = i * 4;
        let dc = tmp[base] + 3;
        let a0 = dc + tmp[base + 3];
        let a1 = tmp[base + 1] + tmp[base + 2];
        let a2 = tmp[base + 1] - tmp[base + 2];
        let a3 = dc - tmp[base + 3];
        out[base] = ((a0 + a1) >> 3) as i16;
        out[base + 1] = ((a3 + a2) >> 3) as i16;
        out[base + 2] = ((a0 - a1) >> 3) as i16;
        out[base + 3] = ((a3 - a2) >> 3) as i16;
    }
    out
}

fn get_large_value(br: &mut Vp8BoolDecoder<'_>, p: &[u8; 11]) -> i32 {
    if br.get_bit(p[3]) == 0 {
        if br.get_bit(p[4]) == 0 {
            2
        } else {
            3 + br.get_bit(p[5]) as i32
        }
    } else if br.get_bit(p[6]) == 0 {
        if br.get_bit(p[7]) == 0 {
            5 + br.get_bit(159) as i32
        } else {
            7 + 2 * br.get_bit(165) as i32 + br.get_bit(145) as i32
        }
    } else {
        let (cat, table): (usize, &[u8]) = if br.get_bit(p[8]) == 0 {
            if br.get_bit(p[9]) == 0 {
                (0, &CAT3)
            } else {
                (1, &CAT4)
            }
        } else if br.get_bit(p[10]) == 0 {
            (2, &CAT5)
        } else {
            (3, &CAT6)
        };
        let mut value = 0i32;
        for &prob in table {
            if prob == 0 {
                break;
            }
            value = value + value + br.get_bit(prob) as i32;
        }
        value + 3 + (8 << cat) as i32
    }
}

fn get_coeffs(
    br: &mut Vp8BoolDecoder<'_>,
    probabilities: &ProbabilityTables,
    coeff_type: usize,
    ctx: usize,
    dq: [u16; 2],
    start: usize,
    out: &mut [i16],
) -> usize {
    let mut n = start;
    let mut p = probabilities.coeff_probs(coeff_type, n, ctx);
    while n < 16 {
        if br.get_bit(p[0]) == 0 {
            return n;
        }
        while br.get_bit(p[1]) == 0 {
            n += 1;
            if n == 16 {
                return 16;
            }
            p = probabilities.coeff_probs(coeff_type, n, 0);
        }

        let next_ctx;
        let value = if br.get_bit(p[2]) == 0 {
            next_ctx = 1;
            1
        } else {
            next_ctx = 2;
            get_large_value(br, p)
        };
        let dequant = if n > 0 { dq[1] } else { dq[0] } as i32;
        out[ZIGZAG[n]] = (br.get_signed(value) * dequant) as i16;
        n += 1;
        p = probabilities.coeff_probs(coeff_type, n, next_ctx);
    }
    16
}

fn nz_code_bits(nz_coeffs: u32, nz: usize, dc_nz: bool) -> u32 {
    (nz_coeffs << 2)
        | if nz > 3 {
            3
        } else if nz > 1 {
            2
        } else if dc_nz {
            1
        } else {
            0
        }
}

fn parse_residuals(
    header: MacroBlockHeader,
    top: &mut NonZeroContext,
    left: &mut NonZeroContext,
    token_br: &mut Vp8BoolDecoder<'_>,
    quantization: &Quantization,
    probabilities: &ProbabilityTables,
) -> MacroBlockData {
    let mut coeffs = [0i16; 384];
    if header.skip {
        top.nz = 0;
        left.nz = 0;
        if !header.is_i4x4 {
            top.nz_dc = 0;
            left.nz_dc = 0;
        }
        return MacroBlockData {
            header,
            coeffs,
            non_zero_y: 0,
            non_zero_uv: 0,
        };
    }

    let q = &quantization.matrices[header.segment as usize];
    let mut offset = 0usize;
    let first;
    let coeff_type;
    if !header.is_i4x4 {
        let mut dc = [0i16; 16];
        let ctx = (top.nz_dc + left.nz_dc) as usize;
        let nz = get_coeffs(token_br, probabilities, 1, ctx, q.y2, 0, &mut dc);
        let has_dc = nz > 0;
        top.nz_dc = has_dc as u8;
        left.nz_dc = has_dc as u8;
        if nz > 1 {
            let transformed = transform_wht(&dc);
            for (block, value) in transformed.into_iter().enumerate() {
                coeffs[block * 16] = value;
            }
        } else {
            let dc0 = ((dc[0] as i32 + 3) >> 3) as i16;
            for block in 0..16 {
                coeffs[block * 16] = dc0;
            }
        }
        first = 1;
        coeff_type = 0;
    } else {
        first = 0;
        coeff_type = 3;
    }

    let mut non_zero_y = 0u32;
    let mut tnz = top.nz & 0x0f;
    let mut lnz = left.nz & 0x0f;
    for _y in 0..4 {
        let mut l = lnz & 1;
        let mut nz_coeffs = 0u32;
        for _x in 0..4 {
            let ctx = (l + (tnz & 1)) as usize;
            let nz = get_coeffs(
                token_br,
                probabilities,
                coeff_type,
                ctx,
                q.y1,
                first,
                &mut coeffs[offset..offset + 16],
            );
            l = (nz > first) as u8;
            tnz = (tnz >> 1) | (l << 7);
            nz_coeffs = nz_code_bits(nz_coeffs, nz, coeffs[offset] != 0);
            offset += 16;
        }
        tnz >>= 4;
        lnz = (lnz >> 1) | (l << 7);
        non_zero_y = (non_zero_y << 8) | nz_coeffs;
    }

    let mut out_t_nz = tnz;
    let mut out_l_nz = lnz >> 4;
    let mut non_zero_uv = 0u32;
    for ch in [0usize, 2usize] {
        let mut nz_coeffs = 0u32;
        let mut tnz = top.nz >> (4 + ch);
        let mut lnz = left.nz >> (4 + ch);
        for _y in 0..2 {
            let mut l = lnz & 1;
            for _x in 0..2 {
                let ctx = (l + (tnz & 1)) as usize;
                let nz = get_coeffs(
                    token_br,
                    probabilities,
                    2,
                    ctx,
                    q.uv,
                    0,
                    &mut coeffs[offset..offset + 16],
                );
                l = (nz > 0) as u8;
                tnz = (tnz >> 1) | (l << 3);
                nz_coeffs = nz_code_bits(nz_coeffs, nz, coeffs[offset] != 0);
                offset += 16;
            }
            tnz >>= 2;
            lnz = (lnz >> 1) | (l << 5);
        }
        non_zero_uv |= nz_coeffs << (4 * ch);
        out_t_nz |= (tnz << 4) << ch;
        out_l_nz |= (lnz & 0xf0) << ch;
    }
    top.nz = out_t_nz;
    left.nz = out_l_nz;

    MacroBlockData {
        header,
        coeffs,
        non_zero_y,
        non_zero_uv,
    }
}

pub fn parse_lossy_headers(data: &[u8]) -> Result<LossyHeader, DecoderError> {
    if data.len() < VP8_FRAME_HEADER_SIZE {
        return Err(DecoderError::NotEnoughData("VP8 frame header"));
    }

    let frame_bits = data[0] as u32 | ((data[1] as u32) << 8) | ((data[2] as u32) << 16);
    let frame = Vp8FrameHeader {
        key_frame: (frame_bits & 1) == 0,
        profile: ((frame_bits >> 1) & 0x07) as u8,
        show: ((frame_bits >> 4) & 1) == 1,
        partition_length: (frame_bits >> 5) as usize,
    };
    if !frame.key_frame {
        return Err(DecoderError::Unsupported("interframes are not supported"));
    }
    if frame.profile > 3 {
        return Err(DecoderError::Bitstream("unknown VP8 profile"));
    }
    if !frame.show {
        return Err(DecoderError::Unsupported("invisible VP8 frame"));
    }
    if !check_lossy_signature(&data[3..]) {
        return Err(DecoderError::Bitstream("bad VP8 signature"));
    }

    let picture = Vp8PictureHeader {
        width: (((data[7] as u16) << 8) | data[6] as u16) & 0x3fff,
        height: (((data[9] as u16) << 8) | data[8] as u16) & 0x3fff,
        xscale: data[7] >> 6,
        yscale: data[9] >> 6,
        colorspace: 0,
        clamp_type: 0,
    };
    if picture.width == 0 || picture.height == 0 {
        return Err(DecoderError::Bitstream("invalid VP8 dimensions"));
    }

    let partition0_offset = VP8_FRAME_HEADER_SIZE;
    let partition0_end = partition0_offset + frame.partition_length;
    if partition0_end > data.len() {
        return Err(DecoderError::NotEnoughData("VP8 partition 0"));
    }

    let mut br = Vp8BoolDecoder::new(&data[partition0_offset..partition0_end]);
    let mut picture = picture;
    picture.colorspace = br.get() as u8;
    picture.clamp_type = br.get() as u8;

    let segment = parse_segment_header(&mut br)?;
    let filter = parse_filter_header(&mut br)?;
    let token_partition_sizes = parse_token_partitions(&mut br, &data[partition0_end..])?;
    let quantization = parse_quantization(&mut br, &segment)?;
    let _ = br.get();
    let probabilities = parse_probability_updates(&mut br)?;

    Ok(LossyHeader {
        frame,
        picture,
        macroblock_width: (picture.width as usize + 15) >> 4,
        macroblock_height: (picture.height as usize + 15) >> 4,
        segment,
        filter,
        token_partition_sizes,
        quantization,
        probabilities,
    })
}

pub fn parse_macroblock_headers(data: &[u8]) -> Result<MacroBlockHeaders, DecoderError> {
    let frame = parse_lossy_headers(data)?;

    let partition0_offset = VP8_FRAME_HEADER_SIZE;
    let partition0_end = partition0_offset + frame.frame.partition_length;
    let mut br = Vp8BoolDecoder::new(&data[partition0_offset..partition0_end]);

    let _ = br.get();
    let _ = br.get();
    let segment = parse_segment_header(&mut br)?;
    let _ = parse_filter_header(&mut br)?;
    let _ = parse_token_partitions(&mut br, &data[partition0_end..])?;
    let _ = parse_quantization(&mut br, &segment)?;
    let _ = br.get();
    let probabilities = parse_probability_updates(&mut br)?;

    let mut top_modes = vec![B_DC_PRED; frame.macroblock_width * 4];
    let mut macroblocks = Vec::with_capacity(frame.macroblock_width * frame.macroblock_height);
    for _mb_y in 0..frame.macroblock_height {
        let mut left_modes = [B_DC_PRED; 4];
        let row = parse_intra_mode_row(
            &mut br,
            frame.macroblock_width,
            segment.update_map,
            &segment.segment_probs,
            probabilities.use_skip_probability,
            probabilities.skip_probability.unwrap_or(0),
            &mut top_modes,
            &mut left_modes,
        )?;
        macroblocks.extend(row);
    }

    Ok(MacroBlockHeaders { frame, macroblocks })
}

pub fn parse_macroblock_data(data: &[u8]) -> Result<MacroBlockDataFrame, DecoderError> {
    let frame = parse_lossy_headers(data)?;
    let partition0_offset = VP8_FRAME_HEADER_SIZE;
    let partition0_end = partition0_offset + frame.frame.partition_length;
    let mut br = Vp8BoolDecoder::new(&data[partition0_offset..partition0_end]);

    let _ = br.get();
    let _ = br.get();
    let segment = parse_segment_header(&mut br)?;
    let _ = parse_filter_header(&mut br)?;
    let token_partition_sizes = parse_token_partitions(&mut br, &data[partition0_end..])?;
    let quantization = parse_quantization(&mut br, &segment)?;
    let _ = br.get();
    let probabilities = parse_probability_tables(&mut br)?;

    let partition_size_bytes = (token_partition_sizes.len() - 1) * 3;
    let mut token_offset = partition0_end + partition_size_bytes;
    let mut token_readers = Vec::with_capacity(token_partition_sizes.len());
    for size in &token_partition_sizes {
        let end = token_offset + *size;
        token_readers.push(Vp8BoolDecoder::new(&data[token_offset..end]));
        token_offset = end;
    }

    let mut top_modes = vec![B_DC_PRED; frame.macroblock_width * 4];
    let mut top_contexts = vec![NonZeroContext::default(); frame.macroblock_width];
    let part_mask = token_readers.len() - 1;
    let mut macroblocks = Vec::with_capacity(frame.macroblock_width * frame.macroblock_height);

    for mb_y in 0..frame.macroblock_height {
        let mut left_modes = [B_DC_PRED; 4];
        let row = parse_intra_mode_row(
            &mut br,
            frame.macroblock_width,
            segment.update_map,
            &segment.segment_probs,
            probabilities.summary.use_skip_probability,
            probabilities.summary.skip_probability.unwrap_or(0),
            &mut top_modes,
            &mut left_modes,
        )?;

        let token_br = &mut token_readers[mb_y & part_mask];
        let mut left_context = NonZeroContext::default();
        for (mb_x, header) in row.into_iter().enumerate() {
            let mb = parse_residuals(
                header,
                &mut top_contexts[mb_x],
                &mut left_context,
                token_br,
                &quantization,
                &probabilities,
            );
            macroblocks.push(mb);
        }
    }

    Ok(MacroBlockDataFrame { frame, macroblocks })
}
