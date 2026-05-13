"""
This script compares Angular Gyrus intrinsic dimension between initial learning
sessions and later review sessions. For each subject and learning week, it loads
the Angular Gyrus fMRI time-series data from the original lecture video and the
corresponding Week 6 recap session, standardizes voxel activity, applies PCA for
dimensionality reduction, computes temporal-difference activation patterns, and
estimates intrinsic dimension using the MLE estimator. The script visualizes
paired first-review intrinsic dimension changes for each subject, with line color
indicating exam score, and overlays the group-level mean trend.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import skdim

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


SCORE_PATH = "subject_weekly_data2.csv"
DATA_DIR = "ThinkLikeExpertsAngularGyrus"
VIDEO_ID = 2
N_COMPONENTS = 80
N_NEIGHBORS = 10
N_TIMEPOINTS = 90

df = pd.read_csv(SCORE_PATH)

scores = df["score"].values
subject_id = df["participant_id"].values

norm = colors.Normalize(vmin=min(scores), vmax=max(scores))
cmap = cm.Blues

for week_id in range(1, 6):
    dimension_pairs = []

    fig, ax = plt.subplots(figsize=(4, 6))

    for i in range(len(subject_id)):
        first_path = (
            f"{DATA_DIR}/sub-{subject_id[i]}_ses-wk{week_id}"
            f"_func_angular_gyrus_sub-{subject_id[i]}_ses-wk{week_id}"
            f"_task-vid{VIDEO_ID}_bold.npy"
        )

        review_path = (
            f"{DATA_DIR}/sub-{subject_id[i]}_ses-wk6"
            f"_func_angular_gyrus_sub-{subject_id[i]}_ses-wk6"
            f"_task-wk{week_id}recap_bold.npy"
        )

        if os.path.exists(first_path) and os.path.exists(review_path):
            first_data = np.load(first_path)[:N_TIMEPOINTS]
            first_data = StandardScaler().fit_transform(first_data)
            first_data = PCA(n_components=N_COMPONENTS).fit_transform(first_data)
            first_diff = first_data[1:] - first_data[:-1]

            first_dimension = skdim.id.MLE().fit_transform(
                first_diff,
                n_neighbors=N_NEIGHBORS
            )

            review_data = np.load(review_path)[:N_TIMEPOINTS]
            review_data = StandardScaler().fit_transform(review_data)
            review_data = PCA(n_components=N_COMPONENTS).fit_transform(review_data)
            review_diff = review_data[1:] - review_data[:-1]

            review_dimension = skdim.id.MLE().fit_transform(
                review_diff,
                n_neighbors=N_NEIGHBORS
            )

            dimension_pairs.append((first_dimension, review_dimension))

            color = cmap(norm(scores[i]))
            ax.plot(
                (0, 1),
                (first_dimension, review_dimension),
                color=color
            )

    if not dimension_pairs:
        plt.close(fig)
        continue

    mean_first = np.mean([pair[0] for pair in dimension_pairs])
    mean_review = np.mean([pair[1] for pair in dimension_pairs])

    ax.plot(
        (0, 1),
        (mean_first, mean_review),
        color="brown",
        linewidth=2,
        label="Mean trend"
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["First", "Review"])
    ax.set_ylabel("Intrinsic dimension")
    ax.set_title(f"Week {week_id}: Video {VIDEO_ID} vs. recap")
    ax.legend()

    scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_map.set_array([])
    fig.colorbar(scalar_map, ax=ax, label="Score")

    fig.tight_layout()
    plt.show()