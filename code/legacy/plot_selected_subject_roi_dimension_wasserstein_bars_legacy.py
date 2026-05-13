"""
This script provides a simple legacy visualization of ROI-level intrinsic
dimension and within-subject video-to-video Wasserstein distance for selected
subjects. For each subject and ROI, it loads multiple video-specific ROI
time-series files, computes temporal-difference intrinsic dimension, computes
Wasserstein distances between video pairs, summarizes each ROI by mean and
standard deviation, and saves full-region bar plots for both metrics.
"""

import os
import re
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skdim
import ot

from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


BASE_DIR = "ThinkLikeExpertsROIs"
SELECTED_SUBJECTS = ["s107", "s122"]
WEEK_ID = 1
VIDEOS = range(1, 6)
N_NEIGHBORS = 10
OUTPUT_DIR = "figure_results_legacy"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def add_space_before_caps(text):
    if not isinstance(text, str):
        return text

    text = text.strip()
    text = re.sub(r"(?<!^)([A-Z])", r" \1", text)

    corrections = {
        r"[Ss]uperiordivision": "Superior Division",
        r"[Ii]nferiordivision": "Inferior Division",
        r"[Aa]nteriordivision": "Anterior Division",
        r"[Pp]osteriordivision": "Posterior Division",
        r"([Cc]ortex)\s*[Ii]nferior": "Cortex Inferior",
        r"([Cc]ortex)\s*[Ss]uperior": "Cortex Superior",
        r"([Cc]ortex)\s*[Aa]nterior": "Cortex Anterior",
        r"([Cc]ortex)\s*[Pp]osterior": "Cortex Posterior",
        r"([Gg]yrus)\s*[Aa]nterior": "Gyrus Anterior",
        r"([Gg]yrus)\s*[Pp]osterior": "Gyrus Posterior",
        r"([Gg]yrus)\s*[Mm]iddle": "Gyrus Middle",
        r"([Pp]ole)\s*[Oo]ccipital": "Pole Occipital",
        r"([Pp]ole)\s*[Tt]emporal": "Pole Temporal",
        r"([Pp]arahippocampal)\s*[Gg]yrus": "Parahippocampal Gyrus",
        r"([Ss]uperior)\s*[Tt]emporal": "Superior Temporal",
        r"([Mm]iddle)\s*[Tt]emporal": "Middle Temporal",
        r"([Ii]nferior)\s*[Tt]emporal": "Inferior Temporal",
        r"[Hh]eschl.?s\s*[Gg]yrus.*[Ii]ncludes\s*H1.*H2": (
            "Heschls Gyrus (includes H1 and H2)"
        ),
        r"[Ii]nferior\s*[Tt]emporal\s*[Gg]yrus.*[Tt]emporo.*[Oo]ccipital.*[Pp]art": (
            "Inferior Temporal Gyrus Temporo Occipital Part"
        )
    }

    for pattern, replacement in corrections.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return re.sub(r"\s+", " ", text).strip()


def wasserstein_distance(data_1, data_2):
    combined_data = np.vstack([data_1, data_2])
    combined_data = StandardScaler().fit_transform(combined_data)

    normalized_1 = combined_data[:len(data_1)]
    normalized_2 = combined_data[len(data_1):]

    weights_1 = np.ones(len(normalized_1)) / len(normalized_1)
    weights_2 = np.ones(len(normalized_2)) / len(normalized_2)

    cost_matrix = ot.dist(normalized_1, normalized_2)
    return float(ot.emd2(weights_1, weights_2, cost_matrix))


def get_roi_file_path(subject_id, roi_name, video_id):
    return os.path.join(
        BASE_DIR,
        f"sub-{subject_id}",
        f"ses-wk{WEEK_ID}",
        "func",
        "regions",
        f"{roi_name}_sub-{subject_id}_ses-wk{WEEK_ID}_task-vid{video_id}_bold.npy"
    )


first_subject = SELECTED_SUBJECTS[0]
first_roi_dir = os.path.join(
    BASE_DIR,
    f"sub-{first_subject}",
    f"ses-wk{WEEK_ID}",
    "func",
    "regions"
)

roi_names = sorted(
    {
        file_name.split("_sub-")[0]
        for file_name in os.listdir(first_roi_dir)
        if file_name.endswith(".npy")
    }
)

print(f"[Week {WEEK_ID}] Found {len(roi_names)} ROIs")

for subject_id in SELECTED_SUBJECTS:
    print(f"\n===== Processing sub-{subject_id} =====")

    dimension_records = []
    wasserstein_records = []

    for roi_name in tqdm(roi_names, desc=f"[sub-{subject_id}]"):
        dimensions = []
        wasserstein_values = []

        for video_id in VIDEOS:
            file_path = get_roi_file_path(subject_id, roi_name, video_id)

            if not os.path.exists(file_path):
                continue

            try:
                voxel_data = np.load(file_path)

                if voxel_data.shape[0] < 5 or voxel_data.shape[1] < 2:
                    continue

                voxel_norm = np.zeros_like(voxel_data)
                scaler = StandardScaler()

                for voxel_index in range(voxel_data.shape[1]):
                    column = voxel_data[:, voxel_index]

                    if np.std(column) > 1e-10:
                        voxel_norm[:, voxel_index] = scaler.fit_transform(
                            column.reshape(-1, 1)
                        ).ravel()

                voxel_diff = voxel_norm[1:] - voxel_norm[:-1]

                dimension = skdim.id.MLE().fit_transform(
                    voxel_diff,
                    n_neighbors=N_NEIGHBORS
                )
                dimension = float(np.asarray(dimension).mean())

                if np.isfinite(dimension):
                    dimensions.append(dimension)

            except Exception:
                continue

        video_files = {
            video_id: get_roi_file_path(subject_id, roi_name, video_id)
            for video_id in VIDEOS
        }

        existing_videos = [
            video_id
            for video_id, path in video_files.items()
            if os.path.exists(path)
        ]

        for video_1, video_2 in combinations(existing_videos, 2):
            try:
                voxel_data_1 = np.load(video_files[video_1])
                voxel_data_2 = np.load(video_files[video_2])

                if voxel_data_1.shape[0] < 3 or voxel_data_2.shape[0] < 3:
                    continue

                diff_1 = voxel_data_1[1:] - voxel_data_1[:-1]
                diff_2 = voxel_data_2[1:] - voxel_data_2[:-1]

                distance = wasserstein_distance(diff_1, diff_2)

                if np.isfinite(distance):
                    wasserstein_values.append(distance)

            except Exception:
                continue

        if dimensions:
            dimension_records.append(
                {
                    "region": add_space_before_caps(roi_name),
                    "mean": float(np.mean(dimensions)),
                    "std": float(np.std(dimensions))
                }
            )

        if wasserstein_values:
            wasserstein_records.append(
                {
                    "region": add_space_before_caps(roi_name),
                    "mean": float(np.mean(wasserstein_values)),
                    "std": float(np.std(wasserstein_values))
                }
            )

    if dimension_records:
        dimension_df = pd.DataFrame(dimension_records).sort_values(
            "mean",
            ascending=False
        )

        plt.figure(figsize=(10, 6))

        plt.bar(
            dimension_df["region"],
            dimension_df["mean"],
            yerr=dimension_df["std"],
            capsize=3,
            alpha=0.8
        )

        plt.xticks(rotation=75, ha="right")
        plt.ylabel("Intrinsic Dimension")
        plt.title(f"Intrinsic Dimension across ROIs (sub-{subject_id}, Week {WEEK_ID})")
        plt.tight_layout()

        output_path = os.path.join(
            OUTPUT_DIR,
            f"sub-{subject_id}_week{WEEK_ID}_dimension_bar.png"
        )

        plt.savefig(output_path, dpi=300)
        plt.close()

    if wasserstein_records:
        wasserstein_df = pd.DataFrame(wasserstein_records).sort_values(
            "mean",
            ascending=False
        )

        plt.figure(figsize=(10, 6))

        plt.bar(
            wasserstein_df["region"],
            wasserstein_df["mean"],
            yerr=wasserstein_df["std"],
            capsize=3,
            alpha=0.8
        )

        plt.xticks(rotation=75, ha="right")
        plt.ylabel("Wasserstein Distance")
        plt.title(f"Wasserstein Distance across ROIs (sub-{subject_id}, Week {WEEK_ID})")
        plt.tight_layout()

        output_path = os.path.join(
            OUTPUT_DIR,
            f"sub-{subject_id}_week{WEEK_ID}_wasserstein_bar.png"
        )

        plt.savefig(output_path, dpi=300)
        plt.close()

print(f"\nFigures have been saved to: {OUTPUT_DIR}")