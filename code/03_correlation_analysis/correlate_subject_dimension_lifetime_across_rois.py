"""
This script computes subject-level correlations between ROI intrinsic dimension
and H1 persistent-homology lifetime. For each subject in a selected recording
week, it scans all available ROI time-series files, averages intrinsic dimension
and mean positive H1 lifetime across multiple videos for each ROI, and then
correlates these two quantities across ROIs within the same subject. A scatter
plot is saved for each subject with enough valid ROI measurements.
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import skdim

from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from ripser import ripser


ROOT_DIR = "ThinkLikeExpertsROIs"
WEEK_ID = 2
VIDEOS = [1, 2, 3, 4, 5]
OUTPUT_ROOT = f"dimension_lifetime_subjectwise_week{WEEK_ID}"

os.makedirs(OUTPUT_ROOT, exist_ok=True)


def safe_standardize_time_series(data):
    data = np.asarray(data, dtype=float)
    normalized_data = np.zeros_like(data)

    scaler = StandardScaler()

    for voxel_index in range(data.shape[1]):
        column = data[:, voxel_index]

        if np.isfinite(column).all() and np.std(column) > 1e-10:
            normalized_data[:, voxel_index] = scaler.fit_transform(
                column.reshape(-1, 1)
            ).ravel()
        else:
            normalized_data[:, voxel_index] = 0.0

    return normalized_data


def compute_dimension(normalized_data):
    if normalized_data.shape[0] < 10 or normalized_data.shape[1] < 2:
        return None

    temporal_diff = normalized_data[1:] - normalized_data[:-1]

    if not np.isfinite(temporal_diff).all():
        return None

    try:
        n_neighbors = max(2, min(10, temporal_diff.shape[0] - 1))
        dimension = skdim.id.MLE().fit_transform(
            temporal_diff,
            n_neighbors=n_neighbors
        )
        dimension = float(np.asarray(dimension).mean())

        return dimension if np.isfinite(dimension) else None

    except Exception:
        return None


def compute_mean_h1_lifetime(normalized_data):
    if normalized_data.shape[0] < 10 or normalized_data.shape[1] < 2:
        return None

    temporal_diff = normalized_data[1:] - normalized_data[:-1]

    if not np.isfinite(temporal_diff).all():
        return None

    try:
        diagrams = ripser(temporal_diff)["dgms"]

        if len(diagrams) < 2:
            return None

        h1_diagram = diagrams[1]

        if h1_diagram.size == 0:
            return None

        lifetimes = h1_diagram[:, 1] - h1_diagram[:, 0]
        lifetimes = lifetimes[np.isfinite(lifetimes)]
        lifetimes = lifetimes[lifetimes > 0]

        if lifetimes.size == 0:
            return None

        return float(np.mean(lifetimes))

    except Exception:
        return None


def list_subject_ids(root_dir):
    subject_ids = []

    for name in os.listdir(root_dir):
        match = re.match(r"^sub-(.+)$", name)

        if match and os.path.isdir(os.path.join(root_dir, name)):
            subject_ids.append(match.group(1))

    return sorted(subject_ids)


def get_roi_names(root_dir, subject_id, week_id):
    roi_dir = os.path.join(
        root_dir,
        f"sub-{subject_id}",
        f"ses-wk{week_id}",
        "func",
        "regions"
    )

    if not os.path.isdir(roi_dir):
        return []

    return sorted(
        {
            file_name.split("_sub-")[0]
            for file_name in os.listdir(roi_dir)
            if file_name.endswith(".npy")
        }
    )


def get_roi_file_path(root_dir, subject_id, week_id, video_id, roi_name):
    return os.path.join(
        root_dir,
        f"sub-{subject_id}",
        f"ses-wk{week_id}",
        "func",
        "regions",
        f"{roi_name}_sub-{subject_id}_ses-wk{week_id}_task-vid{video_id}_bold.npy"
    )


subject_ids = list_subject_ids(ROOT_DIR)

if not subject_ids:
    raise RuntimeError(f"No sub-* directories were found under {ROOT_DIR}")

for subject_id in subject_ids:
    roi_names = get_roi_names(ROOT_DIR, subject_id, WEEK_ID)

    if not roi_names:
        print(f"[Skipped] sub-{subject_id}: no ROI files found")
        continue

    roi_dimensions = []
    roi_lifetimes = []

    for roi_name in roi_names:
        video_dimensions = []
        video_lifetimes = []

        for video_id in VIDEOS:
            file_path = get_roi_file_path(
                ROOT_DIR,
                subject_id,
                WEEK_ID,
                video_id,
                roi_name
            )

            if not os.path.exists(file_path):
                continue

            try:
                voxel_data = np.load(file_path)

                if voxel_data.ndim != 2 or voxel_data.shape[0] < 10 or voxel_data.shape[1] < 2:
                    continue

                if not np.isfinite(voxel_data).all():
                    continue

                normalized_data = safe_standardize_time_series(voxel_data)

                dimension_value = compute_dimension(normalized_data)
                lifetime_value = compute_mean_h1_lifetime(normalized_data)

                if dimension_value is not None:
                    video_dimensions.append(dimension_value)

                if lifetime_value is not None:
                    video_lifetimes.append(lifetime_value)

            except Exception:
                continue

        if video_dimensions and video_lifetimes:
            roi_dimensions.append(float(np.mean(video_dimensions)))
            roi_lifetimes.append(float(np.mean(video_lifetimes)))

    roi_dimensions = np.asarray(roi_dimensions, dtype=float)
    roi_lifetimes = np.asarray(roi_lifetimes, dtype=float)

    valid_mask = np.isfinite(roi_dimensions) & np.isfinite(roi_lifetimes)
    roi_dimensions = roi_dimensions[valid_mask]
    roi_lifetimes = roi_lifetimes[valid_mask]

    if len(roi_dimensions) >= 3 and len(roi_lifetimes) >= 3:
        r_value, p_value = pearsonr(roi_dimensions, roi_lifetimes)

        print(
            f"[OK] sub-{subject_id}: "
            f"r = {r_value:.3f}, p = {p_value:.3g}, n_ROIs = {len(roi_dimensions)}"
        )

        subject_output_dir = os.path.join(OUTPUT_ROOT, f"sub-{subject_id}")
        os.makedirs(subject_output_dir, exist_ok=True)

        plt.figure(figsize=(6, 5))

        sns.regplot(
            x=roi_dimensions,
            y=roi_lifetimes,
            scatter_kws={"alpha": 0.8}
        )

        plt.xlabel("Intrinsic dimension per ROI")
        plt.ylabel("Mean H1 lifetime per ROI")
        plt.title(
            f"sub-{subject_id} Week {WEEK_ID} across ROIs\n"
            f"r = {r_value:.3f}, p = {p_value:.3g}, n = {len(roi_dimensions)}"
        )

        plt.tight_layout()

        output_path = os.path.join(
            subject_output_dir,
            f"sub-{subject_id}_week{WEEK_ID}_dimension_vs_lifetime_scatter.png"
        )

        plt.savefig(output_path, dpi=300)
        plt.close()

    else:
        print(
            f"[Skipped] sub-{subject_id}: too few valid ROIs "
            f"(dimension={len(roi_dimensions)}, lifetime={len(roi_lifetimes)})"
        )

print(f"\nAll done. Output directory: {OUTPUT_ROOT}")