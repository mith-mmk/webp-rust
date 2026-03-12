/// RGBA pixel buffer for a decoded or to-be-encoded still image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageBuffer {
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
    /// Packed RGBA8 pixels in row-major order.
    pub rgba: Vec<u8>,
}

impl ImageBuffer {
    /// Returns the image width in pixels.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Returns the image height in pixels.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Returns the packed RGBA8 buffer.
    pub fn rgba(&self) -> &[u8] {
        &self.rgba
    }

    /// Consumes the image and returns the packed RGBA8 buffer.
    pub fn into_rgba(self) -> Vec<u8> {
        self.rgba
    }
}
