"""
This script analyzes the relationship between Angular Gyrus intrinsic dimension
and learning performance across multiple fMRI recording weeks. It loads subject
scores and region-specific fMRI time-series data, standardizes each subject's
voxel activity, computes temporal-difference activation patterns, and estimates
the intrinsic dimension of the Angular Gyrus representation using the MLE
estimator. For each week, the script performs a permutation-based Pearson
correlation test between intrinsic dimension and exam score, then visualizes the
relationship with a regression plot and reports the permutation correlation
statistics.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import skdim

from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler


def angle_between(a, b):
    cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return np.arccos(np.clip(cos_theta, -1, 1))


def permutation_test(x, y, n_permutations=10000, seed=42):
    np.random.seed(seed)

    x = np.array(x)
    y = np.array(y)

    observed_r, _ = pearsonr(x, y)

    r_permuted = []
    for _ in range(n_permutations):
        y_perm = np.random.permutation(y)
        r_perm, _ = pearsonr(x, y_perm)
        r_permuted.append(r_perm)

    r_permuted = np.array(r_permuted)
    p_val = np.mean(np.abs(r_permuted) >= np.abs(observed_r))

    return observed_r, p_val, r_permuted


file_path = "subject_weekly_data2.csv"
df = pd.read_csv(file_path)

scores = df["score"].values
subject_id = df["participant_id"].values

print(scores)
print(subject_id)

score_result = np.zeros(len(subject_id))
flatness_result = np.zeros(len(subject_id))

for week_id in range(1, 6):
    for i in range(len(subject_id)):
        file_path = (
            "ThinkLikeExpertsAngularGyrus/"
            + "sub-"
            + str(subject_id[i])
            + "_ses-wk"
            + str(week_id)
            + "_func_angular_gyrus_sub-"
            + str(subject_id[i])
            + "_ses-wk"
            + str(week_id)
            + "_task-vid3_bold.npy"
        )

        if os.path.exists(file_path):
            voxel_data = np.load(file_path)

            scaler = StandardScaler()
            voxel_data = scaler.fit_transform(voxel_data)

            voxel_diff = voxel_data[1:] - voxel_data[:-1]

            dimension = skdim.id.MLE().fit_transform(
                voxel_diff,
                n_neighbors=10
            )

            flatness_result[i] = dimension
            score_result[i] = scores[i]

    r, p, _ = permutation_test(
        flatness_result,
        score_result,
        n_permutations=10000,
        seed=42
    )

    sns.regplot(
        x=flatness_result,
        y=score_result,
        scatter_kws={"alpha": 0.7},
        line_kws={"color": "navy"}
    )

    plt.title(f"Week {week_id}", fontsize=15)
    plt.xlabel("Intrinsic dimension")
    plt.ylabel("Score")

    plt.text(
        0.05,
        0.95,
        f"Permutation r = {r:.3f}, p = {p:.3g}",
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    plt.tight_layout()
    plt.show()