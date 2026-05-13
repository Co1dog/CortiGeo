"""
This script visualizes ROI-level intrinsic dimension and within-subject
video-to-video Wasserstein distance for selected subjects. For each selected
subject and ROI, it loads multiple video-specific ROI time-series files,
standardizes voxel activity, computes temporal-difference activation patterns,
applies PCA to harmonize feature dimensionality, estimates intrinsic dimension
with the MLE estimator, and computes Wasserstein distances between video pairs.
The script then selects the top and bottom ROIs by mean value and saves boxplots
based on the actual per-video and per-video-pair measurements.
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
from sklearn.decomposition import PCA
from tqdm import tqdm


BASE_DIR = "ThinkLikeExpertsROIs"
SELECTED_SUBJECTS = ["s107", "s122"]
WEEK_ID = 1
VIDEOS = range(1, 6)
N_NEIGHBORS = 100
OUTPUT_DIR = "figure_results"

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


def wrap_label(text, max_length=25):
    if len(text) <= max_length:
        return text

    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        if len(current_line) + len(word) + 1 <= max_length:
            current_line += (" " + word if current_line else word)
        else:
            lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return "\n".join(lines)


def filter_regions(df):
    excluded_regions = [
        "inferior temporal gyrus anterior division",
        "temporal fusiform cortex anterior division",
        "occipital pole"
    ]

    def is_excluded(region_name):
        region_name = str(region_name).lower().replace(",", "").strip()
        return any(region in region_name for region in excluded_regions)

    return df[~df["region"].apply(is_excluded)].copy()


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


voxel_counts = []
time_counts = []

for subject_id in SELECTED_SUBJECTS:
    subject_dir = os.path.join(
        BASE_DIR,
        f"sub-{subject_id}",
        f"ses-wk{WEEK_ID}",
        "func",
        "regions"
    )

    if not os.path.exists(subject_dir):
        continue

    for file_name in os.listdir(subject_dir):
        if not file_name.endswith(".npy"):
            continue

        data = np.load(os.path.join(subject_dir, file_name))

        if data.ndim == 2:
            voxel_counts.append(data.shape[1])
            time_counts.append(data.shape[0])

if not voxel_counts:
    raise RuntimeError("No valid ROI data was found.")

n_components_auto = int(np.min(voxel_counts))
time_steps_auto = int(np.min(time_counts))

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
print(
    f"[Week {WEEK_ID}] Automatic settings: "
    f"n_components={n_components_auto}, time_steps={time_steps_auto}"
)


def plot_top_bottom_boxplots(records, value_label, output_path, title):
    if not records:
        return

    df = pd.DataFrame(records)
    df = filter_regions(df)
    df = df.sort_values("mean", ascending=False)

    selected_df = pd.concat([df.head(5), df.tail(5)])
    selected_df = selected_df.drop_duplicates(subset=["region"])
    selected_df = selected_df.sort_values("mean", ascending=True)

    selected_df["region_wrapped"] = selected_df["region"].apply(wrap_label)

    box_values = selected_df["values"].tolist()

    plt.figure(figsize=(8, 6))

    boxplot = plt.boxplot(
        box_values,
        vert=False,
        patch_artist=True,
        labels=selected_df["region_wrapped"]
    )

    cmap = plt.get_cmap("RdYlBu_r")
    colors = [
        cmap(index / max(len(selected_df) - 1, 1))
        for index in range(len(selected_df))
    ]

    for patch, color in zip(boxplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
        patch.set_alpha(0.85)

    plt.xlabel(value_label)
    plt.ylabel("")
    plt.title(title)

    if len(selected_df) >= 10:
        plt.text(
            np.mean(plt.xlim()),
            len(selected_df) / 2,
            "... remaining regions ...",
            ha="center",
            va="center",
            fontsize=10,
            color="gray",
            alpha=0.8
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


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

                voxel_norm = StandardScaler().fit_transform(voxel_data)
                voxel_diff = voxel_norm[1:] - voxel_norm[:-1]

                n_components = min(
                    n_components_auto,
                    voxel_diff.shape[0],
                    voxel_diff.shape[1]
                )

                if n_components < 1:
                    continue

                voxel_diff_pca = PCA(n_components=n_components).fit_transform(
                    voxel_diff
                )

                time_steps = min(time_steps_auto, voxel_diff_pca.shape[0])

                if time_steps < 3:
                    continue

                voxel_diff_pca = voxel_diff_pca[:time_steps]
                n_neighbors = max(2, min(N_NEIGHBORS, voxel_diff_pca.shape[0] - 1))

                dimension = skdim.id.MLE().fit_transform(
                    voxel_diff_pca,
                    n_neighbors=n_neighbors
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

                voxel_data_1 = StandardScaler().fit_transform(voxel_data_1)
                voxel_data_2 = StandardScaler().fit_transform(voxel_data_2)

                diff_1 = voxel_data_1[1:] - voxel_data_1[:-1]
                diff_2 = voxel_data_2[1:] - voxel_data_2[:-1]

                n_components = min(
                    n_components_auto,
                    diff_1.shape[1],
                    diff_2.shape[1],
                    diff_1.shape[0] + diff_2.shape[0]
                )

                if n_components < 1:
                    continue

                stacked_diff = np.vstack([diff_1, diff_2])
                stacked_pca = PCA(n_components=n_components).fit_transform(
                    stacked_diff
                )

                diff_1_pca = stacked_pca[:diff_1.shape[0]]
                diff_2_pca = stacked_pca[diff_1.shape[0]:]

                time_steps = min(
                    time_steps_auto,
                    diff_1_pca.shape[0],
                    diff_2_pca.shape[0]
                )

                if time_steps < 3:
                    continue

                diff_1_pca = diff_1_pca[:time_steps]
                diff_2_pca = diff_2_pca[:time_steps]

                distance = wasserstein_distance(diff_1_pca, diff_2_pca)

                if np.isfinite(distance):
                    wasserstein_values.append(distance)

            except Exception:
                continue

        if dimensions:
            dimension_records.append(
                {
                    "region": add_space_before_caps(roi_name),
                    "mean": float(np.mean(dimensions)),
                    "std": float(np.std(dimensions)),
                    "values": dimensions
                }
            )

        if wasserstein_values:
            wasserstein_records.append(
                {
                    "region": add_space_before_caps(roi_name),
                    "mean": float(np.mean(wasserstein_values)),
                    "std": float(np.std(wasserstein_values)),
                    "values": wasserstein_values
                }
            )

    plot_top_bottom_boxplots(
        dimension_records,
        "Intrinsic Dimension",
        os.path.join(
            OUTPUT_DIR,
            f"sub-{subject_id}_week{WEEK_ID}_dimension_boxplot.png"
        ),
        f"Intrinsic Dimension (sub-{subject_id}, Week {WEEK_ID})"
    )

    plot_top_bottom_boxplots(
        wasserstein_records,
        "Wasserstein Distance",
        os.path.join(
            OUTPUT_DIR,
            f"sub-{subject_id}_week{WEEK_ID}_wasserstein_boxplot.png"
        ),
        f"Wasserstein Distance (sub-{subject_id}, Week {WEEK_ID})"
    )

print(f"\nFigures have been saved to: {OUTPUT_DIR}")