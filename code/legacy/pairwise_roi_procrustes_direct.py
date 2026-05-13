"""
This script performs a direct whole-brain ROI-level pairwise Procrustes analysis
for one selected fMRI session and video. For each Harvard-Oxford cortical region,
it extracts ROI-specific BOLD time-series data from each subject, computes
temporal-difference activation patterns, measures pairwise subject similarity
using Procrustes disparity, and correlates the resulting pairwise neural
similarity with pairwise summed exam scores. A regression plot is saved for each
ROI with enough valid subject pairs.
"""

import os
import warnings
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns

from nilearn import datasets, image
from nilearn.input_data import NiftiMasker
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from scipy.spatial import procrustes


warnings.filterwarnings("ignore")


harvard_oxford = datasets.fetch_atlas_harvard_oxford(
    "cort-maxprob-thr25-2mm"
)

atlas_filename = harvard_oxford.maps
atlas_labels = harvard_oxford.labels

mni_template_3mm = "MNI152_T1_3mm_brain.nii.gz"

atlas_3mm = image.resample_to_img(
    source_img=atlas_filename,
    target_img=mni_template_3mm,
    interpolation="nearest"
)


def process_roi_data(fmri_file, roi_masker):
    fmri_img = nib.load(fmri_file)

    fmri_aligned = image.resample_to_img(
        source_img=fmri_img,
        target_img=mni_template_3mm,
        interpolation="linear"
    )

    roi_timeseries = roi_masker.fit_transform(fmri_aligned)
    return roi_timeseries


def calculate_similarity(data_1, data_2):
    scaler = StandardScaler()

    data_1 = scaler.fit_transform(data_1)
    data_2 = scaler.fit_transform(data_2)

    diff_1 = data_1[1:] - data_1[:-1]
    diff_2 = data_2[1:] - data_2[:-1]

    min_time = min(diff_1.shape[0], diff_2.shape[0])
    diff_1 = diff_1[:min_time]
    diff_2 = diff_2[:min_time]

    n_features = min(300, diff_1.shape[1], diff_2.shape[1])
    sample_indices = np.random.choice(
        diff_1.shape[1],
        size=n_features,
        replace=False
    )

    _, _, disparity = procrustes(
        diff_1[:, sample_indices],
        diff_2[:, sample_indices]
    )

    return 1.0 / (disparity + 1e-12)


score_table_path = "subject_weekly_data2.csv"
df = pd.read_csv(score_table_path)

scores = df["score"].values
subject_ids = df["participant_id"].values


def analyze_brain_region(region_index, region_name):
    region_mask = image.math_img(
        f"img == {region_index}",
        img=atlas_3mm
    )

    roi_masker = NiftiMasker(
        mask_img=region_mask,
        standardize=True,
        target_affine=nib.load(mni_template_3mm).affine
    )

    n_subjects = len(subject_ids)

    similarity_matrix = np.zeros((n_subjects, n_subjects))
    score_matrix = np.zeros((n_subjects, n_subjects))

    for i in range(n_subjects):
        for j in range(i + 1, n_subjects):
            file_path_1 = (
                f"ThinkLikeExperts/sub-{subject_ids[i]}/ses-wk5/func/"
                f"aligned_masked_sub-{subject_ids[i]}_ses-wk5_task-vid4_bold.nii.gz"
            )

            file_path_2 = (
                f"ThinkLikeExperts/sub-{subject_ids[j]}/ses-wk5/func/"
                f"aligned_masked_sub-{subject_ids[j]}_ses-wk5_task-vid4_bold.nii.gz"
            )

            if os.path.exists(file_path_1) and os.path.exists(file_path_2):
                try:
                    data_1 = process_roi_data(file_path_1, roi_masker)
                    data_2 = process_roi_data(file_path_2, roi_masker)

                    if data_1.shape[1] > 0 and data_2.shape[1] > 0:
                        similarity = calculate_similarity(data_1, data_2)

                        similarity_matrix[i, j] = similarity
                        similarity_matrix[j, i] = similarity

                        pair_score = scores[i] + scores[j]
                        score_matrix[i, j] = pair_score
                        score_matrix[j, i] = pair_score

                except Exception as error:
                    print(
                        f"Error processing {region_name} for subjects "
                        f"{subject_ids[i]} and {subject_ids[j]}: {error}"
                    )
                    continue

    upper_indices = np.triu_indices_from(similarity_matrix, k=1)

    similarity_values = similarity_matrix[upper_indices].flatten()
    score_values = score_matrix[upper_indices].flatten()

    valid_mask = (similarity_values != 0) & (score_values != 0)

    similarity_values = similarity_values[valid_mask]
    score_values = score_values[valid_mask]

    if len(similarity_values) > 0:
        r_value, p_value = pearsonr(similarity_values, score_values)

        plt.figure(figsize=(10, 6))

        sns.regplot(
            x=similarity_values,
            y=score_values,
            scatter_kws={"alpha": 0.7},
            line_kws={"color": "navy"}
        )

        plt.xlabel("Pairwise neural similarity")
        plt.ylabel("Pairwise summed score")
        plt.title(f"Region: {region_name}\nWeek 5, Video 4")

        text = f"Pearson r = {r_value:.3f}, p = {p_value:.3g}"

        plt.text(
            0.05,
            0.95,
            text,
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )

        plt.tight_layout()

        safe_region_name = str(region_name).replace(" ", "_").replace("/", "_")
        plt.savefig(f"similarity_{safe_region_name}_week5_vid4.png")
        plt.close()


def main():
    for region_index, label in enumerate(atlas_labels[1:], 1):
        if label:
            print(f"Processing region: {label}")
            analyze_brain_region(region_index, label)


if __name__ == "__main__":
    main()