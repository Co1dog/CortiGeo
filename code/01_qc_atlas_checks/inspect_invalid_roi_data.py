"""
This script inspects ROI-level fMRI time-series files that produced invalid
intrinsic dimension estimates. For each specified NumPy file, it reports the
raw data shape, value range, mean, standard deviation, NaN and Inf status,
then applies feature-wise standardization, computes temporal-difference signals,
and attempts to estimate intrinsic dimension using the MLE estimator. The script
also saves the original and temporal-difference data into CSV files for further
manual inspection.
"""

import os
import numpy as np
import skdim

from sklearn.preprocessing import StandardScaler


def check_invalid_data(file_path):
    print(f"\nChecking file: {file_path}")

    voxel_data = np.load(file_path)

    print(f"Raw data shape: {voxel_data.shape}")
    print(f"Raw data range: [{np.min(voxel_data)}, {np.max(voxel_data)}]")
    print(f"Raw data mean: {np.mean(voxel_data)}")
    print(f"Raw data standard deviation: {np.std(voxel_data)}")
    print(f"Contains NaN: {np.any(np.isnan(voxel_data))}")
    print(f"Contains Inf: {np.any(np.isinf(voxel_data))}")

    scaler = StandardScaler()
    voxel_data_norm = np.zeros_like(voxel_data)

    for feature_index in range(voxel_data.shape[1]):
        if np.std(voxel_data[:, feature_index]) > 1e-10:
            voxel_data_norm[:, feature_index] = scaler.fit_transform(
                voxel_data[:, feature_index].reshape(-1, 1)
            ).ravel()

    print("\nAfter standardization:")
    print(f"Data range: [{np.min(voxel_data_norm)}, {np.max(voxel_data_norm)}]")
    print(f"Data mean: {np.mean(voxel_data_norm)}")
    print(f"Data standard deviation: {np.std(voxel_data_norm)}")

    voxel_diff = voxel_data_norm[1:] - voxel_data_norm[:-1]

    print("\nAfter temporal differencing:")
    print(f"Difference data shape: {voxel_diff.shape}")
    print(f"Difference data range: [{np.min(voxel_diff)}, {np.max(voxel_diff)}]")
    print(f"Difference data mean: {np.mean(voxel_diff)}")
    print(f"Difference data standard deviation: {np.std(voxel_diff)}")

    try:
        n_neighbors = min(10, voxel_diff.shape[0] - 1)
        dimension = skdim.id.MLE().fit_transform(
            voxel_diff,
            n_neighbors=n_neighbors
        )
        print(f"\nIntrinsic dimension estimate: {dimension}")
    except Exception as error:
        print(f"\nIntrinsic dimension estimation error: {error}")

    print("\nAdditional statistics:")
    print(f"Number of nonzero entries: {np.sum(np.abs(voxel_data) > 1e-10)}")
    print(f"Number of time points: {voxel_data.shape[0]}")
    print(f"Number of voxels: {voxel_data.shape[1]}")

    return voxel_data, voxel_diff


invalid_files = [
    "ThinkLikeExpertsROIs/sub-s122/ses-wk2/func/regions/CunealCortex_sub-s122_ses-wk2_task-vid1_bold.npy",
    "ThinkLikeExpertsROIs/sub-s122/ses-wk2/func/regions/CunealCortex_sub-s122_ses-wk2_task-vid2_bold.npy",
    "ThinkLikeExpertsROIs/sub-s122/ses-wk2/func/regions/CunealCortex_sub-s122_ses-wk2_task-vid3_bold.npy",
    "ThinkLikeExpertsROIs/sub-s114/ses-wk2/func/regions/InferiorTemporalGyrusanteriordivision_sub-s114_ses-wk2_task-vid1_bold.npy",
    "ThinkLikeExpertsROIs/sub-s125/ses-wk2/func/regions/InferiorTemporalGyrusanteriordivision_sub-s125_ses-wk2_task-vid1_bold.npy",
    "ThinkLikeExpertsROIs/sub-s114/ses-wk2/func/regions/InferiorTemporalGyrusanteriordivision_sub-s114_ses-wk2_task-vid2_bold.npy",
    "ThinkLikeExpertsROIs/sub-s125/ses-wk2/func/regions/InferiorTemporalGyrusanteriordivision_sub-s125_ses-wk2_task-vid2_bold.npy",
    "ThinkLikeExpertsROIs/sub-s114/ses-wk2/func/regions/InferiorTemporalGyrusanteriordivision_sub-s114_ses-wk2_task-vid3_bold.npy",
    "ThinkLikeExpertsROIs/sub-s125/ses-wk2/func/regions/InferiorTemporalGyrusanteriordivision_sub-s125_ses-wk2_task-vid3_bold.npy",
    "ThinkLikeExpertsROIs/sub-s102/ses-wk2/func/regions/OccipitalPole_sub-s102_ses-wk2_task-vid1_bold.npy",
    "ThinkLikeExpertsROIs/sub-s105/ses-wk2/func/regions/OccipitalPole_sub-s105_ses-wk2_task-vid1_bold.npy",
    "ThinkLikeExpertsROIs/sub-s106/ses-wk2/func/regions/OccipitalPole_sub-s106_ses-wk2_task-vid1_bold.npy",
    "ThinkLikeExpertsROIs/sub-s107/ses-wk2/func/regions/OccipitalPole_sub-s107_ses-wk2_task-vid1_bold.npy",
    "ThinkLikeExpertsROIs/sub-s110/ses-wk2/func/regions/OccipitalPole_sub-s110_ses-wk2_task-vid1_bold.npy",
    "ThinkLikeExpertsROIs/sub-s120/ses-wk2/func/regions/OccipitalPole_sub-s120_ses-wk2_task-vid1_bold.npy",
    "ThinkLikeExpertsROIs/sub-s121/ses-wk2/func/regions/OccipitalPole_sub-s121_ses-wk2_task-vid1_bold.npy",
    "ThinkLikeExpertsROIs/sub-s122/ses-wk2/func/regions/OccipitalPole_sub-s122_ses-wk2_task-vid1_bold.npy",
    "ThinkLikeExpertsROIs/sub-s125/ses-wk2/func/regions/OccipitalPole_sub-s125_ses-wk2_task-vid1_bold.npy",
    "ThinkLikeExpertsROIs/sub-s129/ses-wk2/func/regions/OccipitalPole_sub-s129_ses-wk2_task-vid1_bold.npy",
    "ThinkLikeExpertsROIs/sub-s102/ses-wk2/func/regions/OccipitalPole_sub-s102_ses-wk2_task-vid2_bold.npy",
    "ThinkLikeExpertsROIs/sub-s105/ses-wk2/func/regions/OccipitalPole_sub-s105_ses-wk2_task-vid2_bold.npy",
    "ThinkLikeExpertsROIs/sub-s106/ses-wk2/func/regions/OccipitalPole_sub-s106_ses-wk2_task-vid2_bold.npy",
    "ThinkLikeExpertsROIs/sub-s107/ses-wk2/func/regions/OccipitalPole_sub-s107_ses-wk2_task-vid2_bold.npy",
    "ThinkLikeExpertsROIs/sub-s110/ses-wk2/func/regions/OccipitalPole_sub-s110_ses-wk2_task-vid2_bold.npy",
    "ThinkLikeExpertsROIs/sub-s120/ses-wk2/func/regions/OccipitalPole_sub-s120_ses-wk2_task-vid2_bold.npy",
    "ThinkLikeExpertsROIs/sub-s121/ses-wk2/func/regions/OccipitalPole_sub-s121_ses-wk2_task-vid2_bold.npy",
    "ThinkLikeExpertsROIs/sub-s122/ses-wk2/func/regions/OccipitalPole_sub-s122_ses-wk2_task-vid2_bold.npy",
    "ThinkLikeExpertsROIs/sub-s125/ses-wk2/func/regions/OccipitalPole_sub-s125_ses-wk2_task-vid2_bold.npy",
    "ThinkLikeExpertsROIs/sub-s129/ses-wk2/func/regions/OccipitalPole_sub-s129_ses-wk2_task-vid2_bold.npy",
    "ThinkLikeExpertsROIs/sub-s102/ses-wk2/func/regions/OccipitalPole_sub-s102_ses-wk2_task-vid3_bold.npy",
    "ThinkLikeExpertsROIs/sub-s105/ses-wk2/func/regions/OccipitalPole_sub-s105_ses-wk2_task-vid3_bold.npy",
    "ThinkLikeExpertsROIs/sub-s106/ses-wk2/func/regions/OccipitalPole_sub-s106_ses-wk2_task-vid3_bold.npy",
    "ThinkLikeExpertsROIs/sub-s107/ses-wk2/func/regions/OccipitalPole_sub-s107_ses-wk2_task-vid3_bold.npy",
    "ThinkLikeExpertsROIs/sub-s110/ses-wk2/func/regions/OccipitalPole_sub-s110_ses-wk2_task-vid3_bold.npy",
    "ThinkLikeExpertsROIs/sub-s120/ses-wk2/func/regions/OccipitalPole_sub-s120_ses-wk2_task-vid3_bold.npy",
    "ThinkLikeExpertsROIs/sub-s121/ses-wk2/func/regions/OccipitalPole_sub-s121_ses-wk2_task-vid3_bold.npy",
    "ThinkLikeExpertsROIs/sub-s122/ses-wk2/func/regions/OccipitalPole_sub-s122_ses-wk2_task-vid3_bold.npy",
    "ThinkLikeExpertsROIs/sub-s125/ses-wk2/func/regions/OccipitalPole_sub-s125_ses-wk2_task-vid3_bold.npy",
    "ThinkLikeExpertsROIs/sub-s129/ses-wk2/func/regions/OccipitalPole_sub-s129_ses-wk2_task-vid3_bold.npy",
    "ThinkLikeExpertsROIs/sub-s102/ses-wk2/func/regions/TemporalFusiformCortexanteriordivision_sub-s102_ses-wk2_task-vid1_bold.npy",
    "ThinkLikeExpertsROIs/sub-s114/ses-wk2/func/regions/TemporalFusiformCortexanteriordivision_sub-s114_ses-wk2_task-vid1_bold.npy",
    "ThinkLikeExpertsROIs/sub-s125/ses-wk2/func/regions/TemporalFusiformCortexanteriordivision_sub-s125_ses-wk2_task-vid1_bold.npy",
    "ThinkLikeExpertsROIs/sub-s102/ses-wk2/func/regions/TemporalFusiformCortexanteriordivision_sub-s102_ses-wk2_task-vid2_bold.npy",
    "ThinkLikeExpertsROIs/sub-s114/ses-wk2/func/regions/TemporalFusiformCortexanteriordivision_sub-s114_ses-wk2_task-vid2_bold.npy",
    "ThinkLikeExpertsROIs/sub-s125/ses-wk2/func/regions/TemporalFusiformCortexanteriordivision_sub-s125_ses-wk2_task-vid2_bold.npy",
    "ThinkLikeExpertsROIs/sub-s102/ses-wk2/func/regions/TemporalFusiformCortexanteriordivision_sub-s102_ses-wk2_task-vid3_bold.npy",
    "ThinkLikeExpertsROIs/sub-s114/ses-wk2/func/regions/TemporalFusiformCortexanteriordivision_sub-s114_ses-wk2_task-vid3_bold.npy",
    "ThinkLikeExpertsROIs/sub-s125/ses-wk2/func/regions/TemporalFusiformCortexanteriordivision_sub-s125_ses-wk2_task-vid3_bold.npy"
]

for file_path in invalid_files:
    if os.path.exists(file_path):
        voxel_data, voxel_diff = check_invalid_data(file_path)

        output_dir = "invalid_data_analysis"
        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(file_path))[0]

        np.savetxt(
            os.path.join(output_dir, f"{base_name}_original.csv"),
            voxel_data,
            delimiter=","
        )

        np.savetxt(
            os.path.join(output_dir, f"{base_name}_diff.csv"),
            voxel_diff,
            delimiter=","
        )

        print(f"\nData saved to the {output_dir} folder\n")
        print("-" * 80)
    else:
        print(f"\nFile does not exist: {file_path}")
        print("-" * 80)