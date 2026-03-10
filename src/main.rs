use webp_rust::read_header;
use std::fs::File;
use std::io::prelude::*;
type Error = Box<dyn std::error::Error>;



pub fn main() -> Result<(),Error>{
    let mut file = File::open("./_testdata/sample.webp")?;
    let mut buf = vec![];
    file.read_to_end(&mut buf)?;
    let mut reader = bin_rs::reader::BytesReader::from_vec(buf);
    let _ = read_header(&mut reader)?;
    Ok(())
}