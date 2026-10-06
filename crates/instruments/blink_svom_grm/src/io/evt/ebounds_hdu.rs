// pub(super) struct EboundsHdu {
//     channel: Vec<i16>,
//     e_min: Vec<f32>,
//     e_max: Vec<f32>,
// }

// impl EboundsHdu {
//     pub fn from_fptr(fptr: &fitsio_pure::compat::fitsfile::FitsFile) -> Result<Self, fitsio_pure::compat::errors::Error> {
//         let ebounds = fptr.hdu("EBOUNDS")?;

//         let channel = ebounds.read_col::<i16>(fptr, "CHANNEL")?;
//         let e_min = ebounds.read_col::<f32>(fptr, "E_MIN")?;
//         let e_max = ebounds.read_col::<f32>(fptr, "E_MAX")?;

//         Ok(Self {
//             channel,
//             e_min,
//             e_max,
//         })
//     }
// }
