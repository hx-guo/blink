use blink_core::error::Error;

pub struct SciFile {
    pub ccsds: Vec<[u8; 882]>,
}

impl SciFile {
    pub fn new(filename: &str) -> Result<Self, Error> {
        let fptr = fitsio_pure::compat::fitsfile::FitsFile::open(filename)?;

        // HDU 1: HE_Evt_Src
        let sci = fptr.hdu("HE_Evt_Src")?;
        let ccsds_raw: Vec<u8> = sci.read_col(&fptr, "CCSDS")?;
        let mut ccsds_array = Vec::with_capacity(ccsds_raw.len() / 882);
        for chunk in ccsds_raw.chunks_exact(882) {
            let mut array = [0; 882];
            array.copy_from_slice(chunk);
            ccsds_array.push(array);
        }
        Ok(Self { ccsds: ccsds_array })
    }
}
