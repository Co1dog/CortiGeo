"""
This script performs slice-level quality control for a 4D BOLD fMRI image.
It identifies voxels that are either zero across all time points or contain
NaN values, then computes the percentage of missing voxels for each axial
slice along the z-axis. The script visualizes the missing-data proportion
for all slices, highlights slices with severe missing data, and prints the
top slices with the highest missing-data ratios.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

BOLD_PATH = "fMRI/aligned_masked_sub-s102_ses-wk3_task-vid1_bold.nii.gz"

bold_img = nib.load(BOLD_PATH)
bold_data = bold_img.get_fdata()

zeros_all_time = np.logical_or(
    np.all(bold_data == 0, axis=3),
    np.any(np.isnan(bold_data), axis=3)
)

perc_missing = zeros_all_time.mean(axis=(0, 1)) * 100

plt.figure(figsize=(10, 4))
bars = plt.bar(range(len(perc_missing)), perc_missing, color="steelblue")
plt.xlabel("Slice index (z)")
plt.ylabel("Percentage of voxels always 0 / NaN")
plt.title("Per-slice missing-data proportion in the BOLD fMRI volume")

for bar, p in zip(bars, perc_missing):
    if p > 95:
        bar.set_color("crimson")

plt.tight_layout()
plt.show()

N = 5
worst = np.argsort(perc_missing)[-N:][::-1]

print(f"\nTop {N} slices with highest missing ratio:")
for z in worst:
    print(f"  z={z:2d}  ->  missing {perc_missing[z]:5.1f}%")