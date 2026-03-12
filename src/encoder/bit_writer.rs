use bin_rs::io::write_byte;

use crate::encoder::EncoderError;

#[derive(Debug, Default, Clone)]
pub(crate) struct BitWriter {
    bytes: Vec<u8>,
    bit_pos: usize,
}

impl BitWriter {
    pub(crate) fn put_bits(&mut self, value: u32, num_bits: usize) -> Result<(), EncoderError> {
        if num_bits > 32 {
            return Err(EncoderError::Bitstream("bit write is too wide"));
        }
        for bit_index in 0..num_bits {
            let byte_index = self.bit_pos >> 3;
            if byte_index == self.bytes.len() {
                write_byte(0, &mut self.bytes);
            }
            let bit = ((value >> bit_index) & 1) as u8;
            self.bytes[byte_index] |= bit << (self.bit_pos & 7);
            self.bit_pos += 1;
        }
        Ok(())
    }

    pub(crate) fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}
