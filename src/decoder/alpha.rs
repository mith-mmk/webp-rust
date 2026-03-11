use crate::decoder::DecoderError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AlphaHeader {
    pub compression: u8,
    pub filter: u8,
    pub preprocessing: u8,
}

pub fn parse_alpha_header(data: &[u8]) -> Result<AlphaHeader, DecoderError> {
    let Some(&header) = data.first() else {
        return Err(DecoderError::NotEnoughData("ALPH header"));
    };

    let reserved = header >> 6;
    if reserved != 0 {
        return Err(DecoderError::Bitstream("ALPH reserved bits must be zero"));
    }

    Ok(AlphaHeader {
        compression: header & 0x03,
        filter: (header >> 2) & 0x03,
        preprocessing: (header >> 4) & 0x03,
    })
}
