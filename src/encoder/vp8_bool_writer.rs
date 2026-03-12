use bin_rs::io::write_byte;

#[derive(Debug, Clone)]
pub(crate) struct Vp8BoolWriter {
    range: i32,
    value: i32,
    run: usize,
    nb_bits: i32,
    bytes: Vec<u8>,
}

impl Vp8BoolWriter {
    pub(crate) fn new(expected_size: usize) -> Self {
        Self {
            range: 255 - 1,
            value: 0,
            run: 0,
            nb_bits: -8,
            bytes: Vec::with_capacity(expected_size),
        }
    }

    fn flush(&mut self) {
        let shift = 8 + self.nb_bits;
        let bits = self.value >> shift;
        self.value -= bits << shift;
        self.nb_bits -= 8;
        if (bits & 0xff) != 0xff {
            let pos = self.bytes.len();
            if (bits & 0x100) != 0 && pos > 0 {
                self.bytes[pos - 1] = self.bytes[pos - 1].wrapping_add(1);
            }
            if self.run > 0 {
                let value = if (bits & 0x100) != 0 { 0x00 } else { 0xff };
                for _ in 0..self.run {
                    write_byte(value, &mut self.bytes);
                }
                self.run = 0;
            }
            write_byte((bits & 0xff) as u8, &mut self.bytes);
        } else {
            self.run += 1;
        }
    }

    pub(crate) fn put_bit(&mut self, bit: bool, prob: u8) -> bool {
        let split = (self.range * prob as i32) >> 8;
        if bit {
            self.value += split + 1;
            self.range -= split + 1;
        } else {
            self.range = split;
        }
        if self.range < 127 {
            let shift = 7 - ((self.range + 1) as u32).ilog2() as i32;
            self.range = ((self.range + 1) << shift) - 1;
            self.value <<= shift;
            self.nb_bits += shift;
            if self.nb_bits > 0 {
                self.flush();
            }
        }
        bit
    }

    pub(crate) fn put_bit_uniform(&mut self, bit: bool) -> bool {
        let split = self.range >> 1;
        if bit {
            self.value += split + 1;
            self.range -= split + 1;
        } else {
            self.range = split;
        }
        if self.range < 127 {
            self.range = ((self.range + 1) << 1) - 1;
            self.value <<= 1;
            self.nb_bits += 1;
            if self.nb_bits > 0 {
                self.flush();
            }
        }
        bit
    }

    pub(crate) fn put_bits(&mut self, value: u32, num_bits: usize) {
        for shift in (0..num_bits).rev() {
            self.put_bit_uniform(((value >> shift) & 1) != 0);
        }
    }

    pub(crate) fn put_signed_bits(&mut self, value: i32, num_bits: usize) {
        if !self.put_bit_uniform(value != 0) {
            return;
        }
        if value < 0 {
            self.put_bits(((-value as u32) << 1) | 1, num_bits + 1);
        } else {
            self.put_bits((value as u32) << 1, num_bits + 1);
        }
    }

    pub(crate) fn finish(mut self) -> Vec<u8> {
        self.put_bits(0, (9 - self.nb_bits) as usize);
        self.nb_bits = 0;
        self.flush();
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::Vp8BoolWriter;
    use crate::decoder::vp8::Vp8BoolDecoder;

    #[test]
    fn round_trips_boolean_stream() {
        let mut writer = Vp8BoolWriter::new(128);
        let mut expected = Vec::new();
        for i in 0..512usize {
            let prob = ((i * 73) % 255 + 1) as u8;
            let bit = ((i * 37 + 11) & 3) != 0;
            writer.put_bit(bit, prob);
            expected.push((bit, prob));
        }
        let bytes = writer.finish();

        let mut reader = Vp8BoolDecoder::new(&bytes);
        for (bit, prob) in expected {
            assert_eq!(reader.get_bit(prob) == 1, bit);
        }
    }
}
