use bin_rs::io::{write_byte, write_bytes, write_u16_le, write_u32_le};

#[derive(Debug, Default, Clone)]
pub(crate) struct ByteWriter {
    bytes: Vec<u8>,
}

impl ByteWriter {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(capacity),
        }
    }

    pub(crate) fn write_byte(&mut self, value: u8) {
        write_byte(value, &mut self.bytes);
    }

    pub(crate) fn write_bytes(&mut self, values: &[u8]) {
        write_bytes(values, &mut self.bytes);
    }

    pub(crate) fn write_u16_le(&mut self, value: u16) {
        write_u16_le(value, &mut self.bytes);
    }

    pub(crate) fn write_u24_le(&mut self, value: u32) {
        self.write_byte((value & 0xff) as u8);
        self.write_byte(((value >> 8) & 0xff) as u8);
        self.write_byte(((value >> 16) & 0xff) as u8);
    }

    pub(crate) fn write_u32_le(&mut self, value: u32) {
        write_u32_le(value, &mut self.bytes);
    }

    pub(crate) fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}
