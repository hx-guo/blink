"""Synthetic FITS files for every blink reader, in many storage variants (written with astropy).

    python generate.py OUT_DIR       -> OUT_DIR/*.fits[.gz] and OUT_DIR/list.tsv ("kind<TAB>path")

Both builds read the same files; any difference between the cfitsio and the fitsio-pure build shows up
in the digests. astropy is a third implementation, so the values it writes are also the reference.
"""
import gzip, os, shutil, sys
import numpy as np
from astropy.io import fits

OUT = os.path.abspath(sys.argv[1])
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261021)
LIST = []

# reader type -> numpy dtype of the values we generate
def values(rtype, n, style):
    if rtype in ('f64', 'f32'):
        v = rng.normal(0, 1000, n)
        if style == 'special' and n >= 4:
            v[:4] = [np.nan, np.inf, -np.inf, -0.0]
        if style == 'scaled':
            v = np.round(v, 3)
        return v
    if rtype == 'time':                       # MET-like seconds with sub-microsecond digits
        return np.sort(5.0e8 + rng.uniform(0, 3600, n))
    if rtype == 'u8':
        return rng.integers(0, 256, n)
    if rtype == 'i16':
        return rng.integers(-32768, 32768, n) if style != 'scaled' else rng.integers(0, 32768, n)
    if rtype == 'i64':
        return rng.integers(0, 2**31 - 1, n)
    raise ValueError(rtype)

def column(name, rtype, v, style):
    """A fits.Column storing values v (for a reader type rtype) in the given storage style."""
    if rtype == 'time':
        if style == 'tzero':                  # like GBM TTE: TIME 1D with a non-integer TZERO
            return fits.Column(name=name, format='1D', bzero=611989422.9725, array=v)
        if style == 'narrow':
            return fits.Column(name=name, format='1D', array=v)
        return fits.Column(name=name, format='1D', array=v)
    if rtype == 'f64':
        if style == 'narrow':
            return fits.Column(name=name, format='1E', array=v)            # read wider than stored
        if style == 'scaled':
            SCALED[name] = 1e-3      # like HXMT Orbit: 1J with TSCAL, written raw, TSCAL set below
            return fits.Column(name=name, format='1J', array=np.round(np.asarray(v) / 1e-3).astype(np.int32))
        return fits.Column(name=name, format='1D', array=v)
    if rtype == 'f32':
        if style == 'narrow':
            return fits.Column(name=name, format='1D', array=v)            # read narrower than stored
        if style == 'scaled':
            SCALED[name] = 1e-3
            return fits.Column(name=name, format='1J', array=np.round(np.asarray(v) / 1e-3).astype(np.int32))
        return fits.Column(name=name, format='1E', array=v)
    if rtype == 'u8':
        if style == 'narrow':
            return fits.Column(name=name, format='1I', array=v)            # wider storage
        if style == 'scaled':
            return fits.Column(name=name, format='1J', array=v)
        return fits.Column(name=name, format='1B', array=v)
    if rtype == 'i16':
        if style == 'narrow':
            return fits.Column(name=name, format='1J', array=v)
        if style in ('scaled', 'unsigned'):    # like GBM poshist FLAGS: 1I with TZERO 32768
            return fits.Column(name=name, format='1I', bzero=32768, array=np.asarray(v, dtype=np.uint16))
        return fits.Column(name=name, format='1I', array=v)
    if rtype == 'i64':
        if style == 'narrow':
            return fits.Column(name=name, format='1J', array=v)
        if style in ('scaled', 'unsigned'):    # uint32 as 1J with TZERO 2^31
            return fits.Column(name=name, format='1J', bzero=2**31, array=np.asarray(v, dtype=np.uint32))
        return fits.Column(name=name, format='1K', array=v)
    raise ValueError(rtype)

SCALED = {}

def table(extname, cols, n, style, extra=False):
    SCALED.clear()
    cs = []
    for name, rtype in cols:
        v = values(rtype, n, style)
        if style == 'unsigned' and rtype == 'i16':
            v = rng.integers(0, 65536, n)       # beyond i16: what does each build do?
        if style == 'unsigned' and rtype == 'i64':
            v = rng.integers(2**31, 2**32, n)
        cs.append(column(name, rtype, v, style))
    if extra:                                   # unread columns, shuffled order
        cs.append(fits.Column(name='UNUSED_A', format='1D', array=rng.normal(size=n)))
        cs.insert(0, fits.Column(name='UNUSED_B', format='2J', array=rng.integers(0, 9, (n, 2))))
        cs = cs[::-1]
    hdu = fits.BinTableHDU.from_columns(cs, name=extname)
    for i, c in enumerate(hdu.columns, start=1):
        if c.name in SCALED:
            hdu.header[f'TSCAL{i}'] = SCALED[c.name]
            hdu.header[f'TZERO{i}'] = 0.0
    return hdu

def write(kind, stem, hdus, gz=False):
    path = os.path.join(OUT, f'{stem}.fits')
    fits.HDUList([fits.PrimaryHDU()] + hdus).writeto(path, overwrite=True)
    if gz:
        with open(path, 'rb') as a, gzip.open(path + '.gz', 'wb') as b:
            shutil.copyfileobj(a, b)
        os.remove(path)
        path += '.gz'
    LIST.append((kind, path))

STYLES = ['native', 'narrow', 'scaled', 'tzero', 'special', 'unsigned']

def variants(kind, make, n=1000, big=300_007):
    """make(n, style, extra, case) -> list of HDUs"""
    for style in STYLES:
        write(kind, f'{kind}__{style}', make(n, style, False, str))
    write(kind, f'{kind}__gz', make(n, 'native', False, str), gz=True)
    write(kind, f'{kind}__extra_cols', make(n, 'native', True, str))
    write(kind, f'{kind}__lowercase_extname', make(n, 'native', False, str.lower))
    write(kind, f'{kind}__empty', make(0, 'native', False, str))
    write(kind, f'{kind}__one_row', make(1, 'native', False, str))
    write(kind, f'{kind}__big', make(big, 'native', False, str))

# --- HXMT/HE 1K events, with the 18X ACD column ----------------------------------------------------
def hxmt_evt(n, style, extra, case):
    hdu = table(case('Events'), [('Time', 'time'), ('Det_ID', 'u8'), ('Channel', 'u8'),
                                 ('Pulse_Width', 'u8'), ('Event_Type', 'u8')], n, style, extra)
    acd = rng.integers(0, 2, (n, 18)).astype(bool)
    cols = hdu.columns + fits.ColDefs([fits.Column(name='ACD', format='18X', array=acd)])
    return [fits.BinTableHDU.from_columns(cols, name=case('Events'))]
variants('hxmt_evt', hxmt_evt)

def simple(kind, extname, cols, n=1000):
    variants(kind, lambda n, s, e, c: [table(c(extname), cols, n, s, e)], n)

simple('hxmt_orbit', 'Orbit', [('Time', 'time'), ('Lon', 'f64'), ('Lat', 'f64'), ('Alt', 'f64')])
simple('hxmt_att', 'ATT_Quater', [('Time', 'time'), ('Q1', 'f64'), ('Q2', 'f64'), ('Q3', 'f64')])
def hxmt_eng(n, style, extra, case):
    # one packet per second, constant UTC-sTime offset, the first fifth frozen (UTC stuck, sTime runs)
    base = 2**31 + 1000 if style == 'unsigned' else 100_000_000
    stime = base + np.arange(n)
    utc = stime + 387_527_433 if style != 'unsigned' else stime + 1_000_000
    if n:
        utc[: n // 5] = utc[0] + 5000
    SCALED.clear()
    cs = [column('UTC_Last_Bdc', 'i64', utc, style), column('sTime_Last_Bdc', 'i64', stime, style)]
    if extra:
        cs = [fits.Column(name='UNUSED_A', format='1D', array=rng.normal(size=n))] + cs[::-1]
    return [fits.BinTableHDU.from_columns(cs, name=case('HE_Eng'))]
variants('hxmt_eng', hxmt_eng, n=3600, big=300_007)

# HXMT 1B science packets: CCSDS is a vector column of 882 bytes per row
def hxmt_sci(n, style, extra, case):
    data = rng.integers(0, 256, (n, 882)).astype(np.uint8)
    cols = [fits.Column(name='CCSDS', format='882B', array=data)]
    if extra:
        cols = [fits.Column(name='Time', format='1D', array=rng.normal(size=n))] + cols
    return [fits.BinTableHDU.from_columns(cols, name=case('HE_Evt_Src'))]
for stem, n, extra, case, gz in [('native', 1000, False, str, False), ('extra_cols', 37, True, str, False),
                                 ('lowercase_extname', 5, False, str.lower, False), ('empty', 0, False, str, False),
                                 ('one_row', 1, False, str, False), ('gz', 300, False, str, True), ('big', 20_011, False, str, False)]:
    write('hxmt_sci', f'hxmt_sci__{stem}', hxmt_sci(n, 'native', extra, case), gz=gz)

# --- Fermi/GBM -----------------------------------------------------------------------------------
def gbm_tte(n, style, extra, case):
    return [table(case('EVENTS'), [('TIME', 'time'), ('PHA', 'i16')], n, style, extra),
            table(case('GTI'), [('START', 'time'), ('STOP', 'time')], 2, style)]
variants('gbm_tte', gbm_tte)
simple('gbm_poshist', 'GLAST POS HIST', [('SCLK_UTC', 'time'), ('QSJ_1', 'f64'), ('QSJ_2', 'f64'),
       ('QSJ_3', 'f64'), ('SC_LAT', 'f32'), ('SC_LON', 'f32'), ('FLAGS', 'i16'), ('POS_X', 'f32'),
       ('POS_Y', 'f32'), ('POS_Z', 'f32')])

# --- GECAM ---------------------------------------------------------------------------------------
def gecam_ebounds(case, n_channels=448, overflow=True):
    edges = np.geomspace(2.0, 10053.5, n_channels + 1)
    e_min, e_max = edges[:-1], edges[1:]
    if overflow:
        # overflow block like the 470-channel archive: E_MIN not monotonic, E_MAX the 20000 keV sentinel
        e_min = np.append(e_min, rng.uniform(50, 5000, 22)); e_max = np.append(e_max, np.full(22, 20000.0))
    ch = np.arange(len(e_min))
    return fits.BinTableHDU.from_columns([fits.Column(name='CHANNEL', format='1I', array=ch),
        fits.Column(name='E_MIN', format='1E', array=e_min), fits.Column(name='E_MAX', format='1E', array=e_max)],
        name=case('EBOUNDS'))
EV_COLS = [('TIME', 'time'), ('PI', 'i16'), ('GAIN_TYPE', 'u8'), ('DEAD_TIME', 'f32'), ('EVT_TYPE', 'u8'), ('FLAG', 'u8')]
def gecam_evt(detectors):
    def make(n, style, extra, case):
        hdus = [gecam_ebounds(case), table(case('GTI'), [('START', 'time'), ('STOP', 'time')], 3, style)]
        for d in range(1, detectors + 1):
            m = n if d % 7 else max(0, n // 10)        # uneven detectors, one nearly empty
            hdu = table(case(f'EVENTS{d:02d}'), EV_COLS, m, style, extra)
            if style != 'special':                    # PI must stay a valid i16 for the ladder check
                pass
            hdus.append(hdu)
        return hdus
    return make
variants('gecam_evt', gecam_evt(25), n=400, big=12_007)
variants('gecam_evt_c', gecam_evt(12), n=400, big=25_013)

def gecam_cpd(n, style, extra, case, stubs=False):
    hdus = []
    for d in range(1, 9):
        if stubs and d in (3, 6):                     # stub: a table without columns or rows
            h = fits.BinTableHDU.from_columns(fits.ColDefs([]), name=case(f'EVENTS{d:02d}'))
            hdus.append(h); continue
        hdus.append(table(case(f'EVENTS{d:02d}'), [('TIME', 'time'), ('EVT_TYPE', 'u8')], n, style, extra))
    return hdus
variants('gecam_cpd', gecam_cpd, big=50_021)
def gecam_cpd_all_stub(case=str):
    return [fits.BinTableHDU.from_columns(fits.ColDefs([]), name=case(f'EVENTS{d:02d}')) for d in range(1, 9)]
write('gecam_cpd', 'gecam_cpd__all_stub', gecam_cpd_all_stub())
write('gecam_cpd', 'err__cpd_mixed_stub', gecam_cpd(100, 'native', False, str, stubs=True))
simple('gecam_posatt', 'Orbit_Attitude', [('TIME', 'time'), ('Q1', 'f32'), ('Q2', 'f32'), ('Q3', 'f32'),
       ('X_WGS84', 'f64'), ('Y_WGS84', 'f64'), ('Z_WGS84', 'f64')])

# --- SVOM/GRM ------------------------------------------------------------------------------------
SV_COLS = [('TIME', 'time'), ('PI', 'i16'), ('GAIN_TYPE', 'u8'), ('DEAD_TIME', 'f32'), ('EVT_TYPE', 'u8'),
           ('ANTI_COIN', 'u8'), ('FLAG', 'u8')]
def svom_evt(n, style, extra, case):
    return [table(case('GTI'), [('START', 'time'), ('STOP', 'time')], 2, style)] + \
           [table(case(f'EVENTS0{d}'), SV_COLS, n, style, extra) for d in (1, 2, 3)]
variants('svom_evt', svom_evt, big=100_003)
simple('svom_att', 'Quaternion', [('TIME', 'time'), ('Q1', 'f32'), ('Q2', 'f32'), ('Q3', 'f32')])
simple('svom_orb', 'ORB', [('TIME', 'time'), ('LON', 'f32'), ('LAT', 'f32'), ('ALT', 'f32')])

# --- GRID ----------------------------------------------------------------------------------------
def grid_evt(n, style, extra, case):
    return [table(case('EBOUNDS'), [('E_MIN', 'f64')], 64, style),
            table(case('GTI'), [('START', 'time'), ('STOP', 'time')], 1, style)] + \
           [table(case(f'EVENTS{d}'), [('TIME', 'time'), ('PI', 'i16'), ('EVT_TYPE', 'u8')], n, style, extra) for d in range(4)]
variants('grid_evt', grid_evt, big=75_011)
for kind, path in [l for l in LIST if l[0] == 'grid_evt']:
    LIST.append(('grid_gti', path))
simple('grid_posatt', 'ORBIT_ATTITUDE', [('TIME', 'time'), ('Q1', 'f32'), ('Q2', 'f32'), ('Q3', 'f32'),
       ('Latitude', 'f32'), ('Longitude', 'f32'), ('Altitude', 'f32')])

# --- error paths: missing HDU or column -------------------------------------------------------------
def gecam_evt_broken(kind):
    """GECAM-B events where EVENTS05 lacks PI, or is missing altogether: both must fail the file."""
    hdus = gecam_evt(25)(50, 'native', False, str)
    bad = [h for h in hdus if h.name != 'EVENTS05']
    t = next(h for h in hdus if h.name == 'EVENTS05')
    no_pi = fits.BinTableHDU.from_columns([c for c in t.columns if c.name != 'PI'], name='EVENTS05')
    write(kind, f'err__{kind}_no_pi_in_05', [h if h.name != 'EVENTS05' else no_pi for h in hdus])
    write(kind, f'err__{kind}_no_05', bad)
gecam_evt_broken('gecam_evt')
write('hxmt_evt', 'err__no_events_hdu', [table('Other', [('Time', 'time')], 3, 'native')])
write('hxmt_orbit', 'err__no_lon_column', [table('Orbit', [('Time', 'time'), ('Lat', 'f64'), ('Alt', 'f64')], 3, 'native')])
write('gbm_tte', 'err__no_gti', [table('EVENTS', [('TIME', 'time'), ('PHA', 'i16')], 3, 'native')])
def acd_not_18x():
    t = table('Events', [('Time', 'time'), ('Det_ID', 'u8'), ('Channel', 'u8'), ('Pulse_Width', 'u8'), ('Event_Type', 'u8')], 3, 'native')
    return [fits.BinTableHDU.from_columns(t.columns + fits.ColDefs([fits.Column(name='ACD', format='1J', array=[1, 2, 3])]), name='Events')]
write('hxmt_evt', 'err__acd_is_1J', acd_not_18x())

with open(os.path.join(OUT, 'list.tsv'), 'w') as f:
    for kind, path in LIST:
        f.write(f'{kind}\t{path}\n')
print(len(LIST), 'files')
