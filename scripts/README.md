# scripts

Analysis, plotting and job scripts around the Rust crates. Run them from the repository root
(`python3 scripts/<dir>/<name>.py`); relative data paths such as `data/...` assume that.

| Directory | Contents |
|---|---|
| `pipeline/` | Real-time TGF pipeline: `realtime/` (cron run, trial run), `purity.py`, `validate_vet_v9.py`, `check_registry.py` |
| `hxmt_tgf/` | HXMT/HE TGF search and catalog studies (false-alarm distributions, REP/TGF separation, old search jobs) |
| `svom/`, `gecam/`, `grid/`, `gbm/` | TGF studies of the other instruments |
| `saturation/` | HXMT/HE saturation: 1B time reconstruction, gap filling, science-rate and ACD models, validation |
| `bursts/` | Bright bursts (GRB, XRB, FRB 200428) recovered by HE and compared with other instruments |
| `grb221009a/` | GRB 221009A |
| `talks/` | Figures for talks and slides |
| `lib/` | Modules shared across directories: figure styles (`pubstyle`, `talk20_style`, `cjk_font`) and model code (`engineering_prediction`, `unwrap_large_v2`) |
| `tools/` | General utilities |
| `cluster/` | Batch jobs for the computing farm |
| `he_nai_cal/`, `sql/` | HE NaI calibration; SQL of the old task database |
| `archive/` | Superseded versions kept for reference |

## Importing other scripts

A script imports modules of its own directory directly. To import from another directory, it puts that
directory on `sys.path` with one line before the import, relative to its own location:

```python
import sys as _sys; from pathlib import Path as _Path; _sys.path[:0] = [str(_Path(__file__).resolve().parent / _d) for _d in (".", "../lib",)]  # sibling and shared script modules
```

Modules used by scripts of more than one directory belong in `lib/`. Do not add `"scripts"` or an
absolute path to `sys.path`: it depends on the working directory or the machine.
