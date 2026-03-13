//! Byte-oriented writer helpers backed by `bin-rs`.

use bin_rs::io::{write_byte, write_bytes, write_u16_le, write_u32_le};

#[derive(Debug, Default, Clone)]
pub(crate) struct ByteWriter {
    bytes: Vec<u8>,
}

impl ByteWriter {
    /// Creates a byte writer with reserved output capacity.
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(capacity),
        }
    }

    /// Appends a single byte.
    pub(crate) fn write_byte(&mut self, value: u8) {
        write_byte(value, &mut self.bytes);
    }

    /// Appends an arbitrary byte slice.
    pub(crate) fn write_bytes(&mut self, values: &[u8]) {
        write_bytes(values, &mut self.bytes);
    }

    /// Appends a little-endian 16-bit integer.
    pub(crate) fn write_u16_le(&mut self, value: u16) {
        write_u16_le(value, &mut self.bytes);
    }

    /// Appends a little-endian 24-bit integer.
    pub(crate) fn write_u24_le(&mut self, value: u32) {
        self.write_byte((value & 0xff) as u8);
        self.write_byte(((value >> 8) & 0xff) as u8);
        self.write_byte(((value >> 16) & 0xff) as u8);
    }

    /// Appends a little-endian 32-bit integer.
    pub(crate) fn write_u32_le(&mut self, value: u32) {
        write_u32_le(value, &mut self.bytes);
    }

    /// Finishes the writer and returns the accumulated bytes.
    pub(crate) fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}
