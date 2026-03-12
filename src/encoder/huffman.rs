use crate::encoder::bit_writer::BitWriter;
use crate::encoder::EncoderError;

const MAX_ALLOWED_CODE_LENGTH: usize = 15;

#[derive(Debug, Clone)]
pub(crate) struct HuffmanCode {
    code_lengths: Vec<u8>,
    codes: Vec<u16>,
}

impl HuffmanCode {
    pub(crate) fn from_code_lengths(code_lengths: Vec<u8>) -> Result<Self, EncoderError> {
        let mut counts = [0u32; MAX_ALLOWED_CODE_LENGTH + 1];
        let num_symbols = code_lengths.iter().filter(|&&len| len != 0).count();

        if num_symbols == 0 {
            return Err(EncoderError::Bitstream("empty Huffman tree"));
        }

        for &len in &code_lengths {
            let bits = len as usize;
            if bits > MAX_ALLOWED_CODE_LENGTH {
                return Err(EncoderError::Bitstream("invalid Huffman code length"));
            }
            if bits > 0 {
                counts[bits] += 1;
            }
        }

        if num_symbols > 1 {
            let mut left = 1i32;
            for bits in 1..=MAX_ALLOWED_CODE_LENGTH {
                left = (left << 1) - counts[bits] as i32;
                if left < 0 {
                    return Err(EncoderError::Bitstream("oversubscribed Huffman tree"));
                }
            }
            if left != 0 {
                return Err(EncoderError::Bitstream("incomplete Huffman tree"));
            }
        }

        let mut next_code = [0u32; MAX_ALLOWED_CODE_LENGTH + 1];
        let mut code = 0u32;
        for bits in 1..=MAX_ALLOWED_CODE_LENGTH {
            code = (code + counts[bits - 1]) << 1;
            next_code[bits] = code;
        }

        let mut codes = vec![0u16; code_lengths.len()];
        for (symbol, &len) in code_lengths.iter().enumerate() {
            let bits = len as usize;
            if bits == 0 {
                continue;
            }
            let canonical = next_code[bits];
            next_code[bits] += 1;
            codes[symbol] = reverse_bits(canonical, bits);
        }

        Ok(Self {
            code_lengths,
            codes,
        })
    }

    pub(crate) fn write_symbol(
        &self,
        bw: &mut BitWriter,
        symbol: usize,
    ) -> Result<(), EncoderError> {
        let depth = *self
            .code_lengths
            .get(symbol)
            .ok_or(EncoderError::InvalidParam("Huffman symbol is out of range"))?
            as usize;
        if depth == 0 {
            return Err(EncoderError::Bitstream(
                "attempted to write unused Huffman symbol",
            ));
        }
        bw.put_bits(self.codes[symbol] as u32, depth)
    }
}

fn reverse_bits(mut code: u32, bits: usize) -> u16 {
    let mut out = 0u32;
    for _ in 0..bits {
        out = (out << 1) | (code & 1);
        code >>= 1;
    }
    out as u16
}
