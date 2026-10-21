import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import subprocess
import os

center_time = 446726296.2
half_width = 4.0
binsize = 0.05 # 50 ms bins

vmin = center_time - half_width
vmax = center_time + half_width

print("Running blink_cli to dump 1B data...")
env = os.environ.copy()
env["HXMT_1B_DIR"] = "/Users/skyair/Developer/ihep/blink/data/1B"
cmd = [
    "/Users/skyair/Developer/ihep/blink/target/release/blink_cli",
    "2026-02-26T10",
    "--dump-events", str(center_time), str(half_width)
]
result = subprocess.run(cmd, env=env, capture_output=True, text=True)

lines = result.stdout.split('\n')
met_1b_A = []
met_1b_B = []
met_1b_C = []

for line in lines:
    if line.startswith('#') or not line.strip():
        continue
    parts = line.split(',')
    if len(parts) >= 6 and parts[3] == 'EVT':
        try:
            box = parts[0]
            met = float(parts[5])
            if box == 'A':
                met_1b_A.append(met)
            elif box == 'B':
                met_1b_B.append(met)
            elif box == 'C':
                met_1b_C.append(met)
        except:
            pass

print(f"Parsed {len(met_1b_A)} events for A, {len(met_1b_B)} for B, {len(met_1b_C)} for C")

print("Loading 1K data...")
f = fits.open('/Users/skyair/Developer/ihep/blink/data/1K/Y202602/20260226-3179/HXMT_20260226T10_HE-Evt_FFFFFF_V1_1K.FITS')
events = f[1].data
time_1k = events['Time']
det_id_1k = events['Det_ID']

met_1k_A = time_1k[det_id_1k < 6]
met_1k_B = time_1k[(det_id_1k >= 6) & (det_id_1k < 12)]
met_1k_C = time_1k[det_id_1k >= 12]

bins = np.arange(vmin, vmax, binsize)

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

def plot_box(ax, met_1b, met_1k, box_name):
    # histogram
    hist_1k, _ = np.histogram(met_1k, bins=bins)
    hist_1b, _ = np.histogram(met_1b, bins=bins)
    
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # subtract T0 for readability on x-axis
    t0 = 446726290.0
    x_centers = bin_centers - t0
    
    ax.step(x_centers, hist_1k, where='mid', label='1K Data', color='black', linewidth=1.5)
    ax.step(x_centers, hist_1b, where='mid', label='1B Data (Fixed)', color='blue', linewidth=1.5, linestyle='--')
    ax.fill_between(x_centers, hist_1k, hist_1b, where=(hist_1b > hist_1k), color='red', alpha=0.3, label='Excess 1B', step='mid')
    ax.fill_between(x_centers, hist_1b, hist_1k, where=(hist_1k > hist_1b), color='gray', alpha=0.3, label='Missing 1B', step='mid')
    
    ax.set_ylabel(f'Counts / {binsize}s')
    ax.set_title(f"Box {box_name} Lightcurve")
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right')

plot_box(axes[0], met_1b_A, met_1k_A, 'A')
plot_box(axes[1], met_1b_B, met_1k_B, 'B')
plot_box(axes[2], met_1b_C, met_1k_C, 'C')

axes[2].set_xlabel("Time (s) since 446726290.0")

plt.suptitle("GRB 260226A HE Lightcurve: 1K vs 1B (After Saturation Fix)", fontsize=16)
plt.tight_layout()

out_path = "/Users/skyair/Developer/ihep/blink/grb260226a_lightcurve_fixed.png"
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"Saved figure to {out_path}")
