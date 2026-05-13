"""
This script analyzes the relationship between pairwise topological differences
in ROI-level fMRI representations and learning performance. For each week and
ROI, it loads subject-specific fMRI time-series data across available lecture
videos, computes temporal-difference activation patterns, extracts H1 persistence
diagrams using persistent homology, and measures pairwise Wasserstein distances
between subjects' persistence diagrams. The script then correlates the mean
pairwise topological distance with the average exam score of each subject pair,
saves ROI-level weekly correlation results, and generates regression plots for
regions with enough valid subject pairs.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ripser import ripser
from persim import wasserstein
from scipy.stats import pearsonr


plt.rcParams["text.usetex"] = False

ROOT_DIR = "ThinkLikeExpertsROIs"
WEEKS = [1, 2, 3, 4, 5]
VIDEOS = [1, 2, 3, 4, 5, 6]
SCORES_CSV = "subject_weekly_data2.csv"
OUTPUT_DIR = "pairwise_persistence_wasserstein_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_roi_name(filename):
    return filename.split("_sub-")[0]


def load_score_table(scores_csv):
    df = pd.read_csv(scores_csv)

    scores = df["score"].astype(float).values
    subject_ids = df["participant_id"].astype(str).values
    score_map = dict(zip(subject_ids, scores))

    return subject_ids, score_map


def get_roi_names_for_week(root_dir, subject_id, week_id):
    roi_dir = os.path.join(
        root_dir,
        f"sub-{subject_id}",
        f"ses-wk{week_id}",
        "func",
        "regions"
    )

    if not os.path.isdir(roi_dir):
        print(f"[Week {week_id}] ROI directory not found: {roi_dir}")
        return []

    roi_names = sorted(
        {
            extract_roi_name(filename)
            for filename in os.listdir(roi_dir)
            if filename.endswith(".npy")
        }
    )

    if not roi_names:
        print(f"[Week {week_id}] No ROI .npy files found in: {roi_dir}")

    return roi_names


def get_roi_file_path(root_dir, subject_id, week_id, video_id, roi_name):
    return os.path.join(
        root_dir,
        f"sub-{subject_id}",
        f"ses-wk{week_id}",
        "func",
        "regions",
        f"{roi_name}_sub-{subject_id}_ses-wk{week_id}_task-vid{video_id}_bold.npy"
    )


def compute_h1_diagram(voxel_data):
    temporal_diff = voxel_data[1:] - voxel_data[:-1]
    diagrams = ripser(temporal_diff, maxdim=1)["dgms"]

    if len(diagrams) <= 1:
        return np.empty((0, 2))

    return diagrams[1]


def compute_pairwise_persistence_distance(
    root_dir,
    subject_id_1,
    subject_id_2,
    week_id,
    roi_name,
    videos
):
    distances = []

    for video_id in videos:
        path_1 = get_roi_file_path(
            root_dir,
            subject_id_1,
            week_id,
            video_id,
            roi_name
        )
        path_2 = get_roi_file_path(
            root_dir,
            subject_id_2,
            week_id,
            video_id,
            roi_name
        )

        if not (os.path.exists(path_1) and os.path.exists(path_2)):
            continue

        voxel_data_1 = np.load(path_1)
        voxel_data_2 = np.load(path_2)

        if voxel_data_1.shape[0] < 3 or voxel_data_2.shape[0] < 3:
            continue

        h1_diagram_1 = compute_h1_diagram(voxel_data_1)
        h1_diagram_2 = compute_h1_diagram(voxel_data_2)

        if h1_diagram_1.size == 0 or h1_diagram_2.size == 0:
            continue

        try:
            distance = wasserstein(h1_diagram_1, h1_diagram_2)

            if np.isfinite(distance):
                distances.append(distance)

        except Exception:
            continue

    if not distances:
        return None

    return float(np.mean(distances))


def plot_pairwise_correlation(x, y, roi_name, week_id, r_value, p_value, n_pairs, output_path):
    plt.figure(figsize=(6, 5))
    sns.set(style="white")

    sns.regplot(
        x=x,
        y=y,
        scatter_kws={"s": 80, "alpha": 0.8},
        line_kws={"linewidth": 3}
    )

    plt.xlabel("Persistence Wasserstein distance", fontsize=12)
    plt.ylabel("Pair average score", fontsize=12)

    p_text = "p < 0.05" if p_value < 0.05 else f"p = {p_value:.3f}"

    plt.text(
        0.03,
        0.97,
        f"r = {r_value:.3f}\n{p_text}\nN = {n_pairs}",
        transform=plt.gca().transAxes,
        fontsize=11,
        va="top",
        bbox=dict(
            facecolor="#f0f0f0",
            edgecolor="none",
            boxstyle="round,pad=0.3"
        )
    )

    plt.title(f"{roi_name} - Week {week_id}", fontsize=13, weight="bold")
    sns.despine(top=True, right=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    subject_ids, score_map = load_score_table(SCORES_CSV)

    if len(subject_ids) == 0:
        print("[Error] No subjects found in the score table")
        return

    first_subject = subject_ids[0]
    all_records = []

    for week_id in WEEKS:
        roi_names = get_roi_names_for_week(
            ROOT_DIR,
            first_subject,
            week_id
        )

        if not roi_names:
            continue

        week_records = []

        for roi_name in roi_names:
            roi_output_dir = os.path.join(OUTPUT_DIR, roi_name)
            os.makedirs(roi_output_dir, exist_ok=True)

            pair_distances = []
            pair_scores = []

            n_subjects = len(subject_ids)

            for i in range(n_subjects):
                subject_id_1 = subject_ids[i]

                for j in range(i + 1, n_subjects):
                    subject_id_2 = subject_ids[j]

                    pair_distance = compute_pairwise_persistence_distance(
                        ROOT_DIR,
                        subject_id_1,
                        subject_id_2,
                        week_id,
                        roi_name,
                        VIDEOS
                    )

                    if pair_distance is None:
                        continue

                    pair_score = (
                        score_map[subject_id_1]
                        + score_map[subject_id_2]
                    ) / 2.0

                    pair_distances.append(pair_distance)
                    pair_scores.append(pair_score)

            r_value = np.nan
            p_value = np.nan
            n_pairs = 0

            if len(pair_distances) >= 3:
                x = np.asarray(pair_distances, dtype=float)
                y = np.asarray(pair_scores, dtype=float)

                valid_mask = np.isfinite(x) & np.isfinite(y)
                x = x[valid_mask]
                y = y[valid_mask]

                if x.size >= 3 and y.size >= 3:
                    r_value, p_value = pearsonr(x, y)
                    n_pairs = x.size

                    output_png = os.path.join(
                        roi_output_dir,
                        f"{roi_name}_week{week_id}_pairwise_persistence_wasserstein.png"
                    )

                    plot_pairwise_correlation(
                        x,
                        y,
                        roi_name,
                        week_id,
                        r_value,
                        p_value,
                        n_pairs,
                        output_png
                    )

            record = {
                "region": roi_name,
                "week": week_id,
                "r": None if pd.isna(r_value) else round(float(r_value), 4),
                "p": None if pd.isna(p_value) else f"{p_value:.4g}",
                "n_pairs": int(n_pairs)
            }

            week_records.append(record)
            all_records.append(record)

        week_df = pd.DataFrame(
            week_records,
            columns=["region", "week", "r", "p", "n_pairs"]
        )

        week_csv_path = os.path.join(
            OUTPUT_DIR,
            f"correlations_persistence_wasserstein_pairs_week{week_id}.csv"
        )

        week_df.to_csv(week_csv_path, index=False)
        print(f"[Week {week_id}] Results saved to: {week_csv_path}")

    result_df = pd.DataFrame(
        all_records,
        columns=["region", "week", "r", "p", "n_pairs"]
    )

    result_csv_path = os.path.join(
        OUTPUT_DIR,
        "correlations_persistence_wasserstein_pairs_all_weeks.csv"
    )

    result_df.to_csv(result_csv_path, index=False)
    print(f"Done. Summary results saved to: {result_csv_path}")


if __name__ == "__main__":
    main()