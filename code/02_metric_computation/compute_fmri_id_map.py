"""
This script computes a local intrinsic dimension map from a 4D BOLD fMRI image.
It first converts the BOLD time series into temporal-difference signals, then
slides a 4 x 4 x 4 voxel cube across the brain volume and estimates the intrinsic
dimension of each local activation pattern using the MLE estimator. The resulting
intrinsic dimension values are assigned back to the corresponding voxel blocks,
saved as a NumPy array, reconstructed as a NIfTI image, resampled to the MNI152
template space, averaged across time, and visualized on the cortical surface.
"""

import numpy as np
import nibabel as nib
from nilearn.image import mean_img, resample_to_img
from nilearn.plotting import plot_img_on_surf
from nilearn.datasets import load_mni152_template
from skdim.id import MLE
import multiprocessing as mp
import matplotlib.pyplot as plt

shared_data = None

def init_worker(data_np):
    global shared_data
    shared_data = data_np

def process_cube(coord):
    x, y, z = coord
    cube = shared_data[x:x + 4, y:y + 4, z:z + 4, :]

    if np.count_nonzero(cube[:, :, :, 0] == 0) > 60:
        return None

    cube_reordered = np.transpose(cube, (3, 0, 1, 2))
    cube_flattened = cube_reordered.reshape(182, -1)

    try:
        dim = MLE().fit_transform(cube_flattened, n_neighbors=10)
        return (x, y, z, dim)
    except Exception:
        return None

def main():
    img = nib.load("fMRI/aligned_masked_sub-s102_ses-wk3_task-vid1_bold.nii.gz")
    data = img.get_fdata()
    data = data[:, :, :, 1:] - data[:, :, :, :-1]

    id_result = np.zeros_like(data)

    coords = [
        (x, y, z)
        for x in range(0, 61 - 3)
        for y in range(0, 73 - 3)
        for z in range(0, 61 - 3)
    ]

    with mp.Pool(
        processes=mp.cpu_count(),
        initializer=init_worker,
        initargs=(data,)
    ) as pool:
        results = pool.map(process_cube, coords)

    for item in results:
        if item is not None:
            x, y, z, dim = item
            id_result[x:x + 4, y:y + 4, z:z + 4, :] = dim

    np.save("dimension_diff.npy", id_result)
    print("Intrinsic dimension results saved: dimension_diff.npy")

    new_img = nib.Nifti1Image(id_result, affine=img.affine, header=img.header)

    mni152_2mm_template = load_mni152_template(resolution=2)
    new_img_resampled = resample_to_img(
        new_img,
        target_img=mni152_2mm_template,
        interpolation="continuous"
    )

    mean_modified = mean_img(new_img_resampled)

    vmin = np.percentile(id_result, 2)
    vmax = np.percentile(id_result, 98)

    print(f"Plotting range: vmin={vmin:.2f}, vmax={vmax:.2f}")

    plot_img_on_surf(
        stat_map=mean_modified,
        views=["lateral", "medial", "dorsal"],
        hemispheres=["left", "right"],
        colorbar=False,
        inflate=False,
        bg_on_data=True,
        cmap="RdYlBu_r",
        vmin=vmin,
        vmax=vmax
    )

    plt.show()

if __name__ == "__main__":
    main()