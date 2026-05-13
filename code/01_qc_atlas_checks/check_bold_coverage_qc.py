"""
This script checks the spatial coverage of a 4D BOLD fMRI image against a
3 mm MNI152 brain template. It resamples the template into the BOLD image
space, constructs a valid coverage mask from nonzero and non-NaN BOLD voxels,
and identifies template brain voxels that are missing from the acquired BOLD
data. The script reports the number and percentage of missing brain voxels
and visualizes the coverage mask in sagittal, coronal, and axial views.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from nilearn.image import reorder_img, resample_to_img, mean_img

BOLD_4D = "fMRI/aligned_masked_sub-s102_ses-wk3_task-vid1_bold.nii.gz"
TPL3_PATH = "MNI152_T1_3mm_brain.nii.gz"

bold = reorder_img(nib.load(BOLD_4D), resample="continuous")
tpl3 = reorder_img(nib.load(TPL3_PATH), resample="nearest")

tpl3_on_bold = resample_to_img(tpl3, bold, interpolation="nearest")
tpl_mask = tpl3_on_bold.get_fdata() > 0

bold_data = bold.get_fdata()
coverage_mask = np.logical_and(
    ~np.all(bold_data == 0, axis=3),
    ~np.any(np.isnan(bold_data), axis=3)
)

missing_mask = np.logical_and(tpl_mask, ~coverage_mask)
n_missing = int(np.count_nonzero(missing_mask))
n_tpl_brain = int(np.count_nonzero(tpl_mask))

print(f"Template brain voxels      : {n_tpl_brain}")
print(f"Missing brain voxels       : {n_missing}")
print(f"Missing ratio              : {n_missing / n_tpl_brain * 100:.2f}%")

fig, axes = plt.subplots(1, 3, figsize=(9, 3))
titles = ["sagittal (x)", "coronal (y)", "axial (z)"]
slices = [
    coverage_mask.shape[0] // 2,
    coverage_mask.shape[1] // 2,
    coverage_mask.shape[2] // 2
]

for ax, slc, title in zip(axes, slices, titles):
    if title.startswith("sagittal"):
        img = coverage_mask[slc, :, :] * 1.0
        tmpl = tpl_mask[slc, :, :]
    elif title.startswith("coronal"):
        img = coverage_mask[:, slc, :] * 1.0
        tmpl = tpl_mask[:, slc, :]
    else:
        img = coverage_mask[:, :, slc] * 1.0
        tmpl = tpl_mask[:, :, slc]

    img_missing = img.copy()
    img_missing[~img.astype(bool) & tmpl] = -0.5

    ax.imshow(
        img_missing.T,
        origin="lower",
        cmap="Greys",
        vmin=-0.5,
        vmax=1
    )
    ax.set_title(title)
    ax.axis("off")

plt.suptitle("Coverage mask")
plt.tight_layout()
plt.show()