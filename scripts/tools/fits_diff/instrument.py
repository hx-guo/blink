"""Add a test-only FITS digest harness to a blink tree (identical for the cfitsio and fitsio-pure trees).

    python3 instrument.py TREE

Nothing changes in non-test builds: Debug derives are cfg_attr(test), the extra constructors and the
digest test are cfg(test). The digest test reads BLINK_FITS_DIGEST (lines "kind<TAB>path"), opens
every file with the crate's reader, streams `{:?}` of the result into a hasher, and appends
"kind<TAB>path<TAB>OK<TAB>hash<TAB>length" or "kind<TAB>path<TAB>ERR<TAB>message" to
BLINK_FITS_DIGEST_OUT. With BLINK_FITS_DIGEST_DUMP=dir it also writes the full Debug text per file.
"""
import os, re, sys

T = os.path.abspath(sys.argv[1])

def edit(rel, fn):
    p = os.path.join(T, rel)
    s = open(p).read()
    t = fn(s)
    assert t != s, rel
    open(p, 'w').write(t)

def derive_debug(rel, *structs):
    def fn(s):
        for name in structs:
            rx = re.compile(r'^((?:#\[[^\n]*\]\n)*)((?:pub(?:\([a-z]+\))? )?struct ' + name + r'\b)', re.M)
            m = rx.search(s)
            assert m, (rel, name)
            attrs = m.group(1)
            if 'Debug' in attrs:
                continue
            s = s[:m.start(2)] + '#[cfg_attr(test, derive(Debug))]\n' + s[m.start(2):]
        return s
    edit(rel, fn)

def add_test_ctor(rel, ty):
    """Test-only public constructor for readers whose `new(filename)` is private."""
    edit(rel, lambda s: s + f'''
#[cfg(test)]
impl {ty} {{
    pub(crate) fn digest_open(filename: &str) -> Result<Self, blink_core::error::Error> {{
        Self::new(filename)
    }}
}}
''')

HARNESS = r'''//! Test-only FITS digest harness (see the comparison of the cfitsio and fitsio-pure builds).
use std::fmt::Write as _;
use std::hash::Hasher;

struct HashWriter {
    hasher: std::collections::hash_map::DefaultHasher,
    length: u64,
    dump: Option<std::fs::File>,
}

impl std::fmt::Write for HashWriter {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.hasher.write(s.as_bytes());
        self.length += s.len() as u64;
        if let Some(f) = self.dump.as_mut() {
            use std::io::Write as _;
            f.write_all(s.as_bytes()).map_err(|_| std::fmt::Error)?;
        }
        Ok(())
    }
}

pub(crate) fn digest<T: std::fmt::Debug>(kind: &str, path: &str, value: T) -> String {
    let dump = std::env::var("BLINK_FITS_DIGEST_DUMP").ok().map(|dir| {
        let name = format!("{kind}__{}", path.replace('/', "_"));
        std::fs::File::create(std::path::Path::new(&dir).join(name)).unwrap()
    });
    let mut w = HashWriter { hasher: Default::default(), length: 0, dump };
    write!(w, "{value:?}").unwrap();
    format!("{kind}\t{path}\tOK\t{:016x}\t{}", w.hasher.finish(), w.length)
}

pub(crate) fn error<E: std::fmt::Display>(kind: &str, path: &str, e: E) -> String {
    format!("{kind}\t{path}\tERR\t{}", e.to_string().replace(['\n', '\t'], " "))
}

pub(crate) fn run(open: impl Fn(&str, &str) -> Option<String>) {
    let Ok(list) = std::env::var("BLINK_FITS_DIGEST") else { return };
    let out = std::env::var("BLINK_FITS_DIGEST_OUT").expect("BLINK_FITS_DIGEST_OUT");
    let mut lines = Vec::new();
    for line in std::fs::read_to_string(list).unwrap().lines() {
        let Some((kind, path)) = line.split_once('\t') else { continue };
        if let Some(result) = open(kind, path) {
            lines.push(result);
        }
    }
    use std::io::Write as _;
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(out).unwrap();
    for l in lines {
        writeln!(f, "{l}").unwrap();
    }
}
'''

def add_harness(crate_dir, body):
    with open(os.path.join(T, crate_dir, 'src/fits_digest.rs'), 'w') as f:
        f.write(HARNESS + '\n#[test]\nfn fits_digest() {\n    run(|kind, path| {\n' + body + '    });\n}\n')
    edit(f'{crate_dir}/src/lib.rs', lambda s: s + '\n#[cfg(test)]\nmod fits_digest;\n')

def case(kind, expr):
    return (f'        if kind == "{kind}" {{\n'
            f'            return Some(match {expr} {{ Ok(v) => digest(kind, path, v), Err(e) => error(kind, path, e) }});\n'
            f'        }}\n')

# --- HXMT/HE -------------------------------------------------------------------------------------
H = 'crates/instruments/blink_hxmt_he'
derive_debug(f'{H}/src/io/level_1k/evt.rs', 'EventFile')
derive_debug(f'{H}/src/io/level_1k/orbit.rs', 'OrbitFile')
derive_debug(f'{H}/src/io/level_1k/att.rs', 'AttFile')
derive_debug(f'{H}/src/io/level_1b/sci.rs', 'SciFile')
for rel, ty in [('level_1k/evt.rs', 'EventFile'), ('level_1k/orbit.rs', 'OrbitFile'), ('level_1k/att.rs', 'AttFile')]:
    add_test_ctor(f'{H}/src/io/{rel}', ty)
add_harness(H,
    case('hxmt_evt', 'crate::io::level_1k::EventFile::digest_open(path)') +
    case('hxmt_orbit', 'crate::io::level_1k::OrbitFile::digest_open(path)') +
    case('hxmt_att', 'crate::io::level_1k::AttFile::digest_open(path)') +
    case('hxmt_sci', 'crate::io::level_1b::SciFile::new(path)') +
    case('hxmt_eng', 'crate::io::level_1b::read_stime_offset(path)') +
    '        None\n')

# --- Fermi/GBM -----------------------------------------------------------------------------------
G = 'crates/instruments/blink_fermi_gbm'
derive_debug(f'{G}/src/io/tte.rs', 'TteFile')
derive_debug(f'{G}/src/io/poshist.rs', 'PosHistFile')
add_harness(G,
    case('gbm_tte', 'crate::io::tte::TteFile::from_fits_file(path, crate::types::Detector::Nai, 0)') +
    case('gbm_poshist', 'crate::io::poshist::PosHistFile::from_fits_file(path)') +
    '        None\n')

# --- GECAM ---------------------------------------------------------------------------------------
C = 'crates/instruments/blink_gecam'
derive_debug(f'{C}/src/io/evt.rs', 'EvtFile')
derive_debug(f'{C}/src/io/evt/events_hdu.rs', 'EventsHdu', 'TimeReversals')
derive_debug(f'{C}/src/io/evt/gti_hdu.rs', 'GtiHdu')
derive_debug(f'{C}/src/io/evt/ebounds_hdu.rs', 'EboundsHdu')
derive_debug(f'{C}/src/io/cpd.rs', 'CpdFile')
derive_debug(f'{C}/src/io/posatt.rs', 'PosAttFile')
add_harness(C,
    case('gecam_evt', 'crate::io::evt::EvtFile::<crate::types::instrument::SatB>::from_fits_file(path)') +
    case('gecam_evt_c', 'crate::io::evt::EvtFile::<crate::types::instrument::SatC>::from_fits_file(path)') +
    case('gecam_cpd', 'crate::io::cpd::CpdFile::<crate::types::instrument::SatB>::from_fits_file(path)') +
    case('gecam_posatt', 'crate::io::posatt::PosAttFile::from_fits_file(path)') +
    '        None\n')

# --- SVOM/GRM ------------------------------------------------------------------------------------
V = 'crates/instruments/blink_svom_grm'
derive_debug(f'{V}/src/io/evt.rs', 'EvtFile')
derive_debug(f'{V}/src/io/evt/events_hdu.rs', 'EventsHdu', 'TimeReversals')
derive_debug(f'{V}/src/io/evt/gti_hdu.rs', 'GtiHdu')
derive_debug(f'{V}/src/io/att.rs', 'AttFile', 'QuaternionHdu')
derive_debug(f'{V}/src/io/orb.rs', 'OrbFile', 'OrbHdu')
add_harness(V,
    case('svom_evt', 'crate::io::evt::EvtFile::from_fits_file(path)') +
    case('svom_att', 'crate::io::att::AttFile::from_fits_file(path)') +
    case('svom_orb', 'crate::io::orb::OrbFile::from_fits_file(path)') +
    '        None\n')

# --- GRID ----------------------------------------------------------------------------------------
R = 'crates/instruments/blink_grid'
derive_debug(f'{R}/src/io/evt.rs', 'PassFile', 'DetectorHdu')
derive_debug(f'{R}/src/io/posatt.rs', 'PosAttFile')
add_harness(R,
    case('grid_evt', 'crate::io::evt::PassFile::from_fits_file(path)') +
    case('grid_gti', 'crate::io::evt::PassFile::gti_of_file(path)') +
    case('grid_posatt', 'crate::io::posatt::PosAttFile::from_fits_file(path)') +
    '        None\n')
print('instrumented', T)
