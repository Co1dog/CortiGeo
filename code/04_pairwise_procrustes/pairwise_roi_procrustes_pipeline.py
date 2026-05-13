"""
This script performs whole-brain ROI-level pairwise Procrustes alignment analysis
for fMRI representations. It first extracts ROI-specific time-series data from
Harvard-Oxford cortical atlas regions and saves the processed ROI matrices as
NumPy files. It then computes pairwise representational similarity between
subjects using temporal-difference activation patterns and Procrustes disparity.
Finally, it correlates pairwise neural similarity with pairwise summed exam
scores, saves region-wise correlation results, generates scatter plots for each
ROI, and creates a brain-level correlation map.
"""

import os
import warnings
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from nilearn import datasets, image
from nilearn.input_data import NiftiMasker
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from scipy.spatial import procrustes


warnings.filterwarnings("ignore")


def process_roi_data(fmri_file, roi_masker, mni_template_3mm):
    fmri_img = nib.load(fmri_file)

    fmri_aligned = image.resample_to_img(
        source_img=fmri_img,
        target_img=mni_template_3mm,
        interpolation="linear",
        force_resample=True,
        copy_header=True
    )

    roi_timeseries = roi_masker.fit_transform(fmri_aligned)
    return roi_timeseries


def calculate_similarity(data_1, data_2):
    scaler = StandardScaler()

    data_1 = scaler.fit_transform(data_1)
    data_2 = scaler.fit_transform(data_2)

    diff_1 = np.diff(data_1, axis=0)
    diff_2 = np.diff(data_2, axis=0)

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


def preprocess_data(subject_ids, mni_template_3mm, atlas_3mm, atlas_labels):
    print("Starting data preprocessing...")

    os.makedirs("processed_data", exist_ok=True)

    for region_index, label in enumerate(
        tqdm(
            atlas_labels[1:],
            desc="Processing regions",
            total=len(atlas_labels[1:])
        ),
        1
    ):
        if not label:
            continue

        region_mask = image.math_img(
            f"img == {region_index}",
            img=atlas_3mm
        )

        roi_masker = NiftiMasker(
            mask_img=region_mask,
            standardize=True,
            target_affine=nib.load(mni_template_3mm).affine
        )

        for subject_id in subject_ids:
            fmri_file = (
                f"ThinkLikeExperts/sub-{subject_id}/ses-wk5/func/"
                f"aligned_masked_sub-{subject_id}_ses-wk5_task-vid4_bold.nii.gz"
            )

            if os.path.exists(fmri_file):
                try:
                    data = process_roi_data(
                        fmri_file,
                        roi_masker,
                        mni_template_3mm
                    )

                    if data.shape[1] > 0:
                        safe_label = str(label).replace(" ", "_").replace("/", "_")
                        save_path = (
                            f"processed_data/"
                            f"region_{region_index}_{safe_label}_sub-{subject_id}.npy"
                        )
                        np.save(save_path, data)

                except Exception as error:
                    print(
                        f"Error processing {label} for subject "
                        f"{subject_id}: {error}"
                    )

    print("Data preprocessing completed")


def analyze_region_data(region_prefix, scores, subject_ids):
    region_name = os.path.basename(region_prefix).split("_sub-")[0]
    n_subjects = len(subject_ids)

    similarity_matrix = np.zeros((n_subjects, n_subjects))
    subject_data = {}

    for subject_id in subject_ids:
        file_path = f"{region_prefix}_sub-{subject_id}.npy"

        if os.path.exists(file_path):
            subject_data[str(subject_id)] = np.load(file_path)

    for i in range(n_subjects):
        for j in range(i + 1, n_subjects):
            subject_i = str(subject_ids[i])
            subject_j = str(subject_ids[j])

            if subject_i in subject_data and subject_j in subject_data:
                try:
                    similarity = calculate_similarity(
                        subject_data[subject_i],
                        subject_data[subject_j]
                    )

                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity

                except Exception as error:
                    print(
                        f"Error analyzing {region_name} for subjects "
                        f"{subject_ids[i]} and {subject_ids[j]}: {error}"
                    )

    upper_indices = np.triu_indices_from(similarity_matrix, k=1)
    similarity_values = similarity_matrix[upper_indices]

    score_pairs = np.array(
        [
            scores[i] + scores[j]
            for i, j in zip(*upper_indices)
        ]
    )

    valid_mask = similarity_values != 0
    similarity_values = similarity_values[valid_mask]
    score_pairs = score_pairs[valid_mask]

    if len(similarity_values) > 0:
        r_value, p_value = pearsonr(similarity_values, score_pairs)

        return {
            "region_name": region_name,
            "correlation": r_value,
            "p_value": p_value,
            "similarity_matrix": similarity_matrix,
            "similarity_values": similarity_values,
            "score_pairs": score_pairs
        }

    return None


def visualize_results(results, output_dir="analysis_results"):
    print("Starting visualization...")

    os.makedirs(output_dir, exist_ok=True)

    correlations = {
        result["region_name"]: result["correlation"]
        for result in results
        if result is not None
    }

    correlation_df = pd.DataFrame.from_dict(
        correlations,
        orient="index",
        columns=["correlation"]
    )

    correlation_csv = f"{output_dir}/correlations_procrustes.csv"
    correlation_df.to_csv(correlation_csv)

    print(f"Saved correlations to {correlation_csv}")

    for result in tqdm(results, desc="Creating scatter plots"):
        if result is None:
            continue

        plt.figure(figsize=(10, 6))

        sns.regplot(
            x=result["similarity_values"],
            y=result["score_pairs"],
            scatter_kws={"alpha": 0.7},
            line_kws={"color": "navy"}
        )

        plt.xlabel("Pairwise neural similarity")
        plt.ylabel("Pairwise summed score")
        plt.title(f"Region: {result['region_name']}\nWeek 5, Video 4")

        text = (
            f"Pearson r = {result['correlation']:.3f}, "
            f"p = {result['p_value']:.3g}"
        )

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

        safe_region_name = result["region_name"].replace("/", "_")
        plt.savefig(f"{output_dir}/similarity_{safe_region_name}.png")
        plt.close()

    print("Creating correlation map...")

    harvard_oxford = datasets.fetch_atlas_harvard_oxford(
        "cort-maxprob-thr25-2mm"
    )

    atlas_img = (
        harvard_oxford.maps
        if isinstance(harvard_oxford.maps, nib.Nifti1Image)
        else nib.load(harvard_oxford.maps)
    )

    atlas_data = atlas_img.get_fdata()
    correlation_map = np.zeros_like(atlas_data)

    for region_name, correlation in correlations.items():
        try:
            region_index = int(region_name.split("_")[1])
            correlation_map[atlas_data == region_index] = correlation
        except Exception:
            continue

    mid_z = correlation_map.shape[2] // 2

    plt.figure(figsize=(15, 10))
    plt.imshow(
        correlation_map[:, :, mid_z].T,
        cmap="RdBu_r",
        aspect="equal",
        origin="lower"
    )
    plt.colorbar(label="Correlation")
    plt.title("Brain region correlations")
    plt.savefig(f"{output_dir}/correlation_map.png")
    plt.close()

    np.save(f"{output_dir}/correlation_map.npy", correlation_map)

    print("Visualization completed")


def main():
    print("Loading data...")

    df = pd.read_csv("subject_weekly_data2.csv")
    scores = df["score"].values
    subject_ids = df["participant_id"].values

    harvard_oxford = datasets.fetch_atlas_harvard_oxford(
        "cort-maxprob-thr25-2mm"
    )

    atlas_filename = harvard_oxford.maps
    atlas_labels = harvard_oxford.labels

    mni_template_3mm = "MNI152_T1_3mm_brain.nii.gz"

    atlas_3mm = image.resample_to_img(
        source_img=atlas_filename,
        target_img=mni_template_3mm,
        interpolation="nearest",
        force_resample=True,
        copy_header=True
    )

    if not os.path.exists("processed_data") or not os.listdir("processed_data"):
        print("No preprocessed data found. Starting preprocessing...")
        preprocess_data(subject_ids, mni_template_3mm, atlas_3mm, atlas_labels)
    else:
        print("Found preprocessed data. Skipping preprocessing.")

    print("Starting analysis...")

    region_prefixes = set()

    for file_name in os.listdir("processed_data"):
        if file_name.endswith(".npy"):
            prefix = "_".join(file_name.split("_")[:-1])
            region_prefixes.add(os.path.join("processed_data", prefix))

    region_prefixes = sorted(list(region_prefixes))

    results = []

    for prefix in tqdm(region_prefixes, desc="Analyzing regions"):
        result = analyze_region_data(prefix, scores, subject_ids)

        if result is not None:
            results.append(result)

    visualize_results(results)

    print("Analysis completed successfully")


if __name__ == "__main__":
    main()