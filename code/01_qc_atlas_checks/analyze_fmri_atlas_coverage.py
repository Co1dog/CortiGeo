"""
This script evaluates ROI-level coverage of a BOLD fMRI image with respect to
the Harvard-Oxford cortical atlas. It resamples the atlas into the fMRI image
space, computes the number of atlas voxels, nonzero fMRI voxels, coverage
percentage, and mean signal intensity for each region, labels low-coverage
regions as empty, saves the results as a CSV file, generates a coverage
distribution plot, exports a NIfTI coverage map, and writes a text summary report.
"""

import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from nilearn import datasets, image


FMRI_FILE = (
    "ThinkLikeExperts/sub-s102/ses-wk2/func/"
    "aligned_masked_sub-s102_ses-wk2_task-vid1_bold.nii.gz"
)

OUTPUT_CSV = "fmri_coverage_analysis.csv"
OUTPUT_DISTRIBUTION_FIG = "coverage_distribution.png"
OUTPUT_COVERAGE_MAP = "coverage_map.nii.gz"
OUTPUT_REPORT = "coverage_report.txt"

fmri_img = nib.load(FMRI_FILE)
fmri_data = fmri_img.get_fdata()

atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
atlas_img = atlas.maps
atlas_labels = atlas.labels[1:]

atlas_aligned = image.resample_to_img(
    source_img=atlas_img,
    target_img=fmri_img,
    interpolation="nearest",
    force_resample=True,
    copy_header=True
)

atlas_data = atlas_aligned.get_fdata()

results = []

for region_index, label in enumerate(atlas_labels, start=1):
    region_mask = atlas_data == region_index
    total_voxels = np.sum(region_mask)

    if total_voxels > 0:
        region_data = fmri_data[region_mask]

        nonzero_voxels = np.sum(np.any(region_data > 0, axis=1))
        coverage_percentage = nonzero_voxels / total_voxels * 100
        mean_signal = np.mean(region_data)

        is_empty = coverage_percentage < 5 or mean_signal < 1e-6

        results.append(
            {
                "Region_Index": region_index,
                "Region": label,
                "Total_Voxels": total_voxels,
                "Nonzero_Voxels": nonzero_voxels,
                "Coverage_Percentage": coverage_percentage,
                "Mean_Signal": mean_signal,
                "Status": "Empty" if is_empty else "Normal"
            }
        )

df = pd.DataFrame(results)
df = df.sort_values("Coverage_Percentage", ascending=False)

pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda value: "%.2f" % value)

print("\nfMRI data coverage analysis results:")
print(df)

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nResults saved to: {OUTPUT_CSV}")

print("\nSummary statistics:")
print(f"Number of normal regions: {len(df[df['Status'] == 'Normal'])}")
print(f"Number of empty regions: {len(df[df['Status'] == 'Empty'])}")

plt.figure(figsize=(15, 6))

sns.histplot(
    data=df,
    x="Coverage_Percentage",
    bins=20
)

plt.title("Brain region coverage distribution")
plt.xlabel("Coverage percentage (%)")
plt.ylabel("Number of regions")
plt.tight_layout()
plt.savefig(OUTPUT_DISTRIBUTION_FIG, dpi=300, bbox_inches="tight")
plt.close()

coverage_map = np.zeros_like(atlas_data)

for _, row in df.iterrows():
    region_index = int(row["Region_Index"])
    region_mask = atlas_data == region_index
    coverage_map[region_mask] = row["Coverage_Percentage"]

coverage_img = nib.Nifti1Image(coverage_map, fmri_img.affine)
nib.save(coverage_img, OUTPUT_COVERAGE_MAP)

with open(OUTPUT_REPORT, "w", encoding="utf-8") as file:
    file.write("fMRI Data Coverage Analysis Report\n")
    file.write("=" * 50 + "\n\n")

    file.write("File information:\n")
    file.write(f"Analyzed file: {FMRI_FILE}\n")
    file.write(f"Data shape: {fmri_data.shape}\n\n")

    file.write("High-coverage regions (>80%):\n")
    for _, row in df[df["Coverage_Percentage"] > 80].iterrows():
        file.write(f"{row['Region']}: {row['Coverage_Percentage']:.2f}%\n")

    file.write("\nLow-coverage regions (<20%):\n")
    for _, row in df[df["Coverage_Percentage"] < 20].iterrows():
        file.write(f"{row['Region']}: {row['Coverage_Percentage']:.2f}%\n")

    file.write("\nEmpty regions:\n")
    for _, row in df[df["Status"] == "Empty"].iterrows():
        file.write(f"{row['Region']}\n")

print(f"\nDetailed report saved to: {OUTPUT_REPORT}")
print(f"Coverage map saved to: {OUTPUT_COVERAGE_MAP}")