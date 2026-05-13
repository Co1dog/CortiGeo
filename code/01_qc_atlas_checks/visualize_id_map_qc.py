"""
This script performs quality control and visualization for the fMRI intrinsic
dimension map. It loads the precomputed intrinsic dimension results, reconstructs
them as a NIfTI image using a reference fMRI volume, and computes basic voxel-level
diagnostics such as nonzero voxels, missing values, NaN values, low-value voxels,
and voxels outside the brain mask. The script also estimates a percentile-based
display range and visualizes the mean intrinsic dimension map in the original
fMRI space using nilearn.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from nilearn.datasets import load_mni152_template
from nilearn.image import resample_to_img, mean_img
from nilearn.plotting import plot_stat_map

id_result_path = "dimension_diff.npy"
fmri_ref_path = "fMRI/aligned_masked_sub-s102_ses-wk3_task-vid1_bold.nii.gz"
brain_mask_path = "MNI152_T1_3mm_brain_mask.nii"

id_result = np.load(id_result_path)
fmri_img = nib.load(fmri_ref_path)
brain_mask = nib.load(brain_mask_path).get_fdata().astype(bool)

id_img = nib.Nifti1Image(id_result, affine=fmri_img.affine, header=fmri_img.header)

mean_id_img = mean_img(id_img)
mean_data = mean_id_img.get_fdata()


vmin = np.percentile(id_result[id_result > 0], 2)
vmax = np.percentile(id_result, 98)

total_voxels = np.prod(mean_data.shape)
nonzero_mask = mean_data > 0
nan_mask = np.isnan(mean_data)
low_mask = mean_data < vmin
outside_brain_mask = ~brain_mask

diagnosis = {
    "Total voxels": total_voxels,
    "Nonzero voxels": np.count_nonzero(nonzero_mask),
    "Zero or missing voxels": total_voxels - np.count_nonzero(nonzero_mask),
    "NaN voxels": np.count_nonzero(nan_mask),
    f"Voxels < vmin ({vmin:.2f})": np.count_nonzero(low_mask),
    "Voxels outside brain mask": np.count_nonzero(outside_brain_mask & nonzero_mask),
    "Voxels likely contributing to blue color": np.count_nonzero(low_mask & nonzero_mask & ~nan_mask)
}

for k, v in diagnosis.items():
    print(f"{k:<40}: {v}")

plot_stat_map(mean_id_img, title="Mean Intrinsic Dimension (original space)", cut_coords=(0, 0, 0))
plt.show()
