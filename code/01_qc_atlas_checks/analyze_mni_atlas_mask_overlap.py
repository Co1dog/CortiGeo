"""
This script evaluates the spatial overlap between a 3 mm MNI152 brain mask and
Harvard-Oxford cortical atlas regions. It resamples the atlas into the MNI mask
space, computes the total voxel count, overlapping voxel count, and overlap
percentage for each atlas region, saves the results as a CSV file, visualizes
the distribution of overlap percentages, and exports a NIfTI overlap map for
further inspection.
"""

import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from nilearn import datasets, image


MNI_BRAIN_MASK_FILE = "MNI152_T1_3mm_brain_mask.nii"
OUTPUT_CSV = "mni_harvard_oxford_overlap.csv"
OUTPUT_DISTRIBUTION_FIG = "overlap_distribution.png"
OUTPUT_OVERLAP_MAP = "overlap_map.nii.gz"

mni_mask = nib.load(MNI_BRAIN_MASK_FILE)
mni_mask_data = mni_mask.get_fdata().astype(bool)

atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
atlas_img = atlas.maps
atlas_labels = atlas.labels[1:]

atlas_3mm = image.resample_to_img(
    source_img=atlas_img,
    target_img=mni_mask,
    interpolation="nearest",
    force_resample=True,
    copy_header=True
)

atlas_data = atlas_3mm.get_fdata()

results = []

for region_index, label in enumerate(atlas_labels, start=1):
    region_mask = atlas_data == region_index

    total_voxels = np.sum(region_mask)
    overlap_voxels = np.sum(region_mask & mni_mask_data)

    overlap_percentage = (
        overlap_voxels / total_voxels * 100
        if total_voxels > 0
        else 0
    )

    results.append(
        {
            "Region": label,
            "Total_Voxels": total_voxels,
            "Overlap_Voxels": overlap_voxels,
            "Overlap_Percentage": overlap_percentage
        }
    )

df = pd.DataFrame(results)
df = df.sort_values("Overlap_Percentage", ascending=False)

print("\nMNI152 mask and Harvard-Oxford atlas overlap analysis:\n")

pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda value: "%.2f" % value)

print(df)

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nResults saved to: {OUTPUT_CSV}")

print("\nSummary statistics:")
print(f"Total voxels in MNI mask: {np.sum(mni_mask_data)}")
print(f"Regions with high overlap (>95%): {len(df[df['Overlap_Percentage'] > 95])}")
print(
    "Regions with partial overlap (50-95%): "
    f"{len(df[(df['Overlap_Percentage'] > 50) & (df['Overlap_Percentage'] <= 95)])}"
)
print(f"Regions with low overlap (<50%): {len(df[df['Overlap_Percentage'] <= 50])}")

plt.figure(figsize=(15, 6))

sns.histplot(
    data=df,
    x="Overlap_Percentage",
    bins=20
)

plt.title("Overlap distribution between Harvard-Oxford atlas regions and MNI mask")
plt.xlabel("Overlap percentage (%)")
plt.ylabel("Number of regions")
plt.tight_layout()
plt.savefig(OUTPUT_DISTRIBUTION_FIG, dpi=300, bbox_inches="tight")
plt.close()

overlap_map = np.zeros_like(atlas_data)

for region_index, _ in enumerate(atlas_labels, start=1):
    region_mask = atlas_data == region_index
    overlap_voxels = region_mask & mni_mask_data
    overlap_map[overlap_voxels] = 1

overlap_img = nib.Nifti1Image(overlap_map, mni_mask.affine)
nib.save(overlap_img, OUTPUT_OVERLAP_MAP)

print(f"Overlap map saved to: {OUTPUT_OVERLAP_MAP}")