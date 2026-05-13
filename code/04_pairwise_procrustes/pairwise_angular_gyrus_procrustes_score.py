"""
This script performs pairwise Procrustes alignment analysis on Angular Gyrus
fMRI representations across students. For each recording week and available
lecture task, it loads each subject's Angular Gyrus time-series data, standardizes
voxel activity, computes temporal-difference activation patterns, and measures
pairwise representational similarity between subjects using Procrustes disparity.
The script then relates pairwise neural similarity to the summed exam scores of
each subject pair, saves similarity and score matrices, and generates regression
plots for each task as well as an aggregated analysis across all available data.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from scipy.spatial import procrustes
from scipy.stats import pearsonr


FIG_DIR = "pairwise_student_diff_fmri_procrustes_figs"
os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_csv("subject_weekly_data2.csv")
scores = df["score"].values
subject_id = df["participant_id"].values
n_subjects = len(subject_id)

all_similarity_values = []
all_score_values = []

for week_id in range(1, 7):
    task_set = set()

    for sid in subject_id:
        pattern = (
            f"ThinkLikeExpertsAngularGyrus/sub-{sid}_ses-wk{week_id}"
            f"_func_angular_gyrus_sub-{sid}_ses-wk{week_id}_task-*bold.npy"
        )

        for path in glob.glob(pattern):
            task = path.split("_task-")[1].replace("_bold.npy", "")
            task_set.add(task)

    tasks = sorted(list(task_set))

    if not tasks:
        continue

    for task in tasks:
        similarity_matrix = np.zeros((n_subjects, n_subjects))
        score_matrix = np.zeros((n_subjects, n_subjects))

        for i in range(n_subjects):
            for j in range(i + 1, n_subjects):
                path_1 = (
                    f"ThinkLikeExpertsAngularGyrus/sub-{subject_id[i]}_ses-wk{week_id}"
                    f"_func_angular_gyrus_sub-{subject_id[i]}_ses-wk{week_id}"
                    f"_task-{task}_bold.npy"
                )

                path_2 = (
                    f"ThinkLikeExpertsAngularGyrus/sub-{subject_id[j]}_ses-wk{week_id}"
                    f"_func_angular_gyrus_sub-{subject_id[j]}_ses-wk{week_id}"
                    f"_task-{task}_bold.npy"
                )

                if not (os.path.exists(path_1) and os.path.exists(path_2)):
                    continue

                voxel_data_1 = StandardScaler().fit_transform(np.load(path_1))
                voxel_data_2 = StandardScaler().fit_transform(np.load(path_2))

                diff_data_1 = voxel_data_1[1:] - voxel_data_1[:-1]
                diff_data_2 = voxel_data_2[1:] - voxel_data_2[:-1]

                min_length = min(diff_data_1.shape[0], diff_data_2.shape[0])
                diff_data_1 = diff_data_1[:min_length]
                diff_data_2 = diff_data_2[:min_length]

                _, _, disparity = procrustes(diff_data_1.T, diff_data_2.T)
                similarity_value = 1.0 / (disparity + 1e-12)

                similarity_matrix[i, j] = similarity_value
                similarity_matrix[j, i] = similarity_value

                pair_score = scores[i] + scores[j]
                score_matrix[i, j] = pair_score
                score_matrix[j, i] = pair_score

        for matrix, tag in [
            (similarity_matrix, "similarity"),
            (score_matrix, "score")
        ]:
            nonzero_values = matrix[matrix != 0]

            if nonzero_values.size == 0:
                continue

            plt.imshow(
                matrix,
                cmap="RdYlBu_r",
                vmin=nonzero_values.min(),
                vmax=nonzero_values.max()
            )
            plt.colorbar()
            plt.title(f"{tag.capitalize()} - Week {week_id} - {task}")

            output_path = f"{FIG_DIR}/week{week_id}_{task}_{tag}.png"
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.show()

        upper_triangle_indices = np.triu_indices_from(similarity_matrix, k=1)

        similarity_values = similarity_matrix[upper_triangle_indices].flatten()
        score_values = score_matrix[upper_triangle_indices].flatten()

        valid_mask = similarity_values != 0
        similarity_values = similarity_values[valid_mask]
        score_values = score_values[valid_mask]

        if similarity_values.size < 2:
            continue

        r_value, p_value = pearsonr(similarity_values, score_values)

        sns.regplot(
            x=similarity_values,
            y=score_values,
            scatter_kws={"alpha": 0.7},
            line_kws={"color": "navy"}
        )

        plt.xlabel("Pairwise neural similarity")
        plt.ylabel("Pairwise summed score")
        plt.title(
            f"Week {week_id} - {task}\n"
            f"r = {r_value:.3f}, p = {p_value:.3g}"
        )

        output_path = f"{FIG_DIR}/week{week_id}_{task}_scatter.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.show()

        all_similarity_values.append(similarity_values)
        all_score_values.append(score_values)

if all_similarity_values:
    all_similarity_values = np.concatenate(all_similarity_values)
    all_score_values = np.concatenate(all_score_values)

    r_value, p_value = pearsonr(all_similarity_values, all_score_values)

    sns.regplot(
        x=all_similarity_values,
        y=all_score_values,
        scatter_kws={"alpha": 0.6},
        line_kws={"color": "navy"}
    )

    plt.xlabel("Pairwise neural similarity")
    plt.ylabel("Pairwise summed score")
    plt.title(
        "All data\n"
        f"r = {r_value:.3f}, p = {p_value:.3g}"
    )

    output_path = f"{FIG_DIR}/scatter_all.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()