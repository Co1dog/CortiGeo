"""
This script performs a combined voxel-wise intrinsic dimension and Wasserstein
distance analysis for multi-week fMRI data. For each subject, week, and available
lecture video, it computes temporal-difference BOLD signals, estimates local
intrinsic dimension using sliding 4 x 4 x 4 voxel cubes, and aggregates the
results within Harvard-Oxford cortical atlas regions. It also computes local
Wasserstein distances between all pairs of videos from the same week, aggregates
them at the ROI level, and then calculates three region-wise correlations:
intrinsic dimension versus exam score, Wasserstein distance versus exam score,
and intrinsic dimension versus Wasserstein distance. The final outputs are
weekly CSV files containing correlation coefficients, p-values, and sample counts.
"""

import os
import traceback
import warnings
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import nibabel as nib
import ot

from tqdm import tqdm
from nilearn.image import resample_to_img
from nilearn.datasets import fetch_atlas_harvard_oxford
from sklearn.preprocessing import StandardScaler
from skdim.id import MLE
from scipy.stats import pearsonr
import multiprocessing as mp


shared_data = None
shared_data_pairs = None


def init_worker_dimension(data_np):
    global shared_data
    shared_data = data_np


def init_worker_wasserstein(data_pairs):
    global shared_data_pairs
    shared_data_pairs = data_pairs


def process_cube_dimension(coord):
    x, y, z = coord

    try:
        cube = shared_data[x:x + 4, y:y + 4, z:z + 4, :]

        if np.count_nonzero(cube[:, :, :, 0] == 0) > 60:
            return None

        cube_reordered = np.transpose(cube, (3, 0, 1, 2))
        time_points = cube_reordered.shape[0]
        cube_flattened = cube_reordered.reshape(time_points, -1)

        try:
            mle = MLE()
            values = mle.fit_transform(cube_flattened, n_neighbors=100)
            dimension_value = float(np.median(values))
        except Exception:
            return None

        return x, y, z, dimension_value

    except Exception:
        return None


def wasserstein_distance_between_cubes(cube_1, cube_2):
    try:
        if cube_1.shape[0] < 3 or cube_2.shape[0] < 3:
            return np.nan

        combined_data = np.vstack([cube_1, cube_2])
        combined_data = StandardScaler().fit_transform(combined_data)

        if np.isnan(combined_data).any():
            return np.nan

        cube_1_norm = combined_data[:cube_1.shape[0]]
        cube_2_norm = combined_data[cube_1.shape[0]:]

        diff_1 = cube_1_norm[1:] - cube_1_norm[:-1]
        diff_2 = cube_2_norm[1:] - cube_2_norm[:-1]

        cost_matrix = ot.dist(diff_1, diff_2)

        weights_1 = np.ones(len(diff_1)) / len(diff_1)
        weights_2 = np.ones(len(diff_2)) / len(diff_2)

        wasserstein_distance = float(ot.emd2(weights_1, weights_2, cost_matrix))
        return wasserstein_distance

    except Exception:
        traceback.print_exc()
        return np.nan


def process_cube_pair_wasserstein(args):
    coord, _ = args
    x, y, z = coord

    try:
        data_pairs = shared_data_pairs

        cube_1 = data_pairs[0][x:x + 4, y:y + 4, z:z + 4, :]
        cube_2 = data_pairs[1][x:x + 4, y:y + 4, z:z + 4, :]

        cube_1 = np.transpose(cube_1, (3, 0, 1, 2))
        cube_2 = np.transpose(cube_2, (3, 0, 1, 2))

        cube_1 = cube_1.reshape(cube_1.shape[0], -1)
        cube_2 = cube_2.reshape(cube_2.shape[0], -1)

        wasserstein_distance = wasserstein_distance_between_cubes(cube_1, cube_2)

        if not np.isfinite(wasserstein_distance):
            return None

        return x, y, z, wasserstein_distance

    except Exception:
        traceback.print_exc()
        return None


def calculate_dimension(data, coords):
    dimension_result = np.zeros_like(data, dtype=np.float32)

    cpu_count = max(1, mp.cpu_count() - 1)

    with mp.Pool(
        processes=cpu_count,
        initializer=init_worker_dimension,
        initargs=(data,)
    ) as pool:
        results = pool.imap_unordered(process_cube_dimension, coords)

        for result in tqdm(
            results,
            total=len(coords),
            desc="    Computing intrinsic dimension",
            leave=False
        ):
            if result is not None:
                x, y, z, dimension_value = result

                try:
                    dimension_result[x:x + 4, y:y + 4, z:z + 4, :] = dimension_value
                except Exception:
                    continue

    return dimension_result


def calculate_wasserstein_distance(data_1, data_2, coords, n_workers=None):
    if n_workers is None:
        n_workers = os.cpu_count() or 4

    wasserstein_result = np.zeros_like(data_1)

    init_worker_wasserstein([data_1, data_2])

    args_list = [(coord, 0) for coord in coords]

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = executor.map(process_cube_pair_wasserstein, args_list)

        for result in tqdm(
            results,
            total=len(args_list),
            desc="Computing Wasserstein"
        ):
            if result is not None:
                x, y, z, wasserstein_distance = result
                wasserstein_result[x:x + 4, y:y + 4, z:z + 4, :] = wasserstein_distance

    return wasserstein_result


def calculate_roi_dimensions_for_file(dimension_result, atlas_img, atlas_labels, reference_img):
    if isinstance(atlas_img, str):
        atlas_nii = nib.load(atlas_img)
    else:
        atlas_nii = atlas_img

    atlas_data = atlas_nii.get_fdata()

    if atlas_data.shape != dimension_result.shape[:3]:
        atlas_resampled = resample_to_img(
            atlas_nii,
            reference_img,
            interpolation="nearest"
        )
        atlas_data = atlas_resampled.get_fdata()

    results = {}
    unique_labels = np.unique(atlas_data)
    unique_labels = unique_labels[unique_labels != 0]

    for label in unique_labels:
        roi_mask = atlas_data == label

        if roi_mask.sum() == 0:
            continue

        roi_dimensions = dimension_result[roi_mask]
        valid_dimensions = roi_dimensions[np.isfinite(roi_dimensions)]
        valid_dimensions = valid_dimensions[valid_dimensions != 0]

        if valid_dimensions.size > 0:
            mean_dimension = float(np.mean(valid_dimensions))
            label_index = int(label) - 1

            if 0 <= label_index < len(atlas_labels):
                region_name = atlas_labels[label_index]
            else:
                region_name = f"Region_{int(label)}"

            results[region_name] = mean_dimension

    return results


def calculate_roi_wasserstein(wasserstein_result, atlas_img, atlas_labels, reference_img):
    if not hasattr(atlas_img, "get_fdata"):
        atlas_img = nib.load(atlas_img)

    atlas_shape = atlas_img.get_fdata().shape

    if atlas_shape != wasserstein_result.shape[:3]:
        atlas_img = resample_to_img(
            atlas_img,
            reference_img,
            interpolation="nearest"
        )
        atlas_data = atlas_img.get_fdata()
    else:
        atlas_data = atlas_img.get_fdata()

    atlas_data = np.rint(atlas_data).astype(np.int32)
    unique_labels = np.unique(atlas_data)

    results = {}

    for label in unique_labels:
        if label == 0:
            continue

        roi_mask = atlas_data == label
        roi_wasserstein = wasserstein_result[roi_mask]

        if roi_wasserstein.size == 0:
            continue

        finite_values = roi_wasserstein[np.isfinite(roi_wasserstein)]

        if finite_values.size == 0:
            continue

        mean_wasserstein = float(np.mean(finite_values))
        label_index = int(label) - 1

        if 0 <= label_index < len(atlas_labels):
            region_name = atlas_labels[label_index]
        else:
            region_name = f"Region_{int(label)}"

        results[region_name] = mean_wasserstein

    return results


def get_existing_subjects(base_path, start=102, end=129):
    subjects = []

    for subject_id in range(start, end + 1):
        subject_path = os.path.join(base_path, f"sub-s{subject_id}")

        if os.path.exists(subject_path):
            subjects.append(f"sub-s{subject_id}")

    return sorted(subjects)


def get_fmri_file_path(base_path, subject, week, video):
    return os.path.join(
        base_path,
        subject,
        f"ses-wk{week}",
        "func",
        f"{subject}_ses-wk{week}_task-vid{video}_bold.nii.gz"
    )


def save_correlation_csv(rows, output_path):
    df = pd.DataFrame(rows)

    if len(df) > 0 and "r" in df.columns:
        df = df.sort_values("r", ascending=False)

    df.to_csv(output_path, index=False, encoding="utf-8")
    return df


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = r"E:\ThinkLikeExperts"
    output_dir = os.path.join(script_dir, "voxel_wise_k100_results")
    os.makedirs(output_dir, exist_ok=True)

    subject_score_csv = "subject_weekly_data2.csv"

    print("=" * 80)
    print("Combined voxel-wise intrinsic dimension and Wasserstein analysis")
    print("Weeks 1-5, neighborhood size k=100")
    print("=" * 80)
    print(f"Script directory: {script_dir}")
    print(f"Data root: {base_path}")
    print(f"Output directory: {output_dir}\n")

    if not os.path.exists(subject_score_csv):
        print(f"[Error] Score CSV not found: {subject_score_csv}")
        return

    score_df = pd.read_csv(subject_score_csv)

    if "participant_id" not in score_df.columns or "score" not in score_df.columns:
        print("[Error] The score CSV must contain 'participant_id' and 'score' columns")
        return

    score_map = dict(
        zip(
            score_df["participant_id"].astype(str).values,
            score_df["score"].astype(float).values
        )
    )

    print("Scanning available subjects...")
    subjects = get_existing_subjects(base_path, start=102, end=129)
    print(f"Found {len(subjects)} subjects: {subjects}\n")

    if len(subjects) == 0:
        print("[Error] No subject data found")
        return

    print("Loading Harvard-Oxford Atlas...")
    atlas = fetch_atlas_harvard_oxford(
        "cort-maxprob-thr0-2mm",
        symmetric_split=False
    )
    atlas_maps = atlas.maps
    atlas_labels = atlas.labels
    atlas_img = nib.load(atlas_maps) if isinstance(atlas_maps, str) else atlas_maps

    print(f"Number of atlas labels: {len(atlas_labels)}\n")

    processed_count = 0
    skipped_count = 0

    for week in range(1, 6):
        print(f"\n{'=' * 80}")
        print(f"Processing Week {week}")
        print(f"{'=' * 80}")

        week_processed = 0
        week_skipped = 0

        week_dimension_score_data = {}
        week_wasserstein_score_data = {}
        week_dimension_wasserstein_data = {}

        for subject in subjects:
            print(f"\nProcessing subject: {subject}")
            print("-" * 80)

            subject_videos = {}

            for video in range(1, 6):
                fmri_file = get_fmri_file_path(
                    base_path,
                    subject,
                    week=week,
                    video=video
                )

                if os.path.exists(fmri_file):
                    subject_videos[video] = fmri_file

            if len(subject_videos) < 2:
                print(
                    f"  [Skipped] Subject {subject}, Week {week}: "
                    f"fewer than two available videos"
                )
                week_skipped += 1
                continue

            reference_img = nib.load(list(subject_videos.values())[0])
            data_shape = reference_img.shape

            coords = [
                (x, y, z)
                for x in range(0, data_shape[0] - 3, 4)
                for y in range(0, data_shape[1] - 3, 4)
                for z in range(0, data_shape[2] - 3, 4)
            ]

            subject_dimension_values = {}
            subject_wasserstein_values = {}

            for video, fmri_file in subject_videos.items():
                try:
                    img = nib.load(fmri_file)
                    data = img.get_fdata()

                    if data.shape[3] < 2:
                        continue

                    data = data[:, :, :, 1:] - data[:, :, :, :-1]

                    dimension_result = calculate_dimension(data, coords)

                    roi_dimensions = calculate_roi_dimensions_for_file(
                        dimension_result,
                        atlas_img,
                        atlas_labels,
                        img
                    )

                    subject_dimension_values[video] = roi_dimensions

                except Exception as error:
                    print(f"    Error during dimension computation for video {video}: {error}")
                    continue

            pair_count = 0

            for (video_1, path_1), (video_2, path_2) in combinations(
                subject_videos.items(),
                2
            ):
                try:
                    data_1 = nib.load(path_1).get_fdata()
                    data_2 = nib.load(path_2).get_fdata()

                    if (
                        data_1.ndim < 4
                        or data_2.ndim < 4
                        or data_1.shape[3] < 2
                        or data_2.shape[3] < 2
                    ):
                        continue

                    data_1 = data_1[:, :, :, 1:] - data_1[:, :, :, :-1]
                    data_2 = data_2[:, :, :, 1:] - data_2[:, :, :, :-1]

                    wasserstein_map = calculate_wasserstein_distance(
                        data_1,
                        data_2,
                        coords
                    )

                    roi_wasserstein = calculate_roi_wasserstein(
                        wasserstein_map,
                        atlas_img,
                        atlas_labels,
                        reference_img
                    )

                    subject_wasserstein_values[(video_1, video_2)] = roi_wasserstein
                    pair_count += 1

                except Exception as error:
                    print(
                        f"    Error during Wasserstein computation for "
                        f"video pair {video_1}-{video_2}: {error}"
                    )
                    continue

            if pair_count == 0:
                print(
                    f"  [Skipped] Subject {subject}, Week {week}: "
                    f"no successful distance computations"
                )
                week_skipped += 1
                continue

            participant_id = subject.replace("sub-", "")

            for video in range(1, 6):
                if video not in subject_dimension_values:
                    continue

                for region, dimension_value in subject_dimension_values[video].items():
                    if participant_id in score_map:
                        week_dimension_score_data.setdefault(
                            region,
                            {"pids": [], "dimensions": [], "scores": []}
                        )
                        week_dimension_score_data[region]["pids"].append(participant_id)
                        week_dimension_score_data[region]["dimensions"].append(dimension_value)
                        week_dimension_score_data[region]["scores"].append(
                            score_map[participant_id]
                        )

            for (video_1, video_2), region_wasserstein_dict in (
                subject_wasserstein_values.items()
            ):
                for region, wasserstein_value in region_wasserstein_dict.items():
                    has_dimension_pair = (
                        video_1 in subject_dimension_values
                        and video_2 in subject_dimension_values
                        and region in subject_dimension_values[video_1]
                        and region in subject_dimension_values[video_2]
                    )

                    if not has_dimension_pair:
                        continue

                    pair_mean_dimension = (
                        subject_dimension_values[video_1][region]
                        + subject_dimension_values[video_2][region]
                    ) / 2

                    if participant_id in score_map:
                        week_wasserstein_score_data.setdefault(
                            region,
                            {"pids": [], "wasserstein": [], "scores": []}
                        )
                        week_wasserstein_score_data[region]["pids"].append(
                            participant_id
                        )
                        week_wasserstein_score_data[region]["wasserstein"].append(
                            wasserstein_value
                        )
                        week_wasserstein_score_data[region]["scores"].append(
                            score_map[participant_id]
                        )

                        week_dimension_wasserstein_data.setdefault(
                            region,
                            {"dimensions": [], "wasserstein": []}
                        )
                        week_dimension_wasserstein_data[region]["dimensions"].append(
                            pair_mean_dimension
                        )
                        week_dimension_wasserstein_data[region]["wasserstein"].append(
                            wasserstein_value
                        )

            week_processed += 1
            print(f"  Completed Subject {subject}, Week {week}")

        print(
            f"\nWeek {week} summary: "
            f"{week_processed} subjects processed, {week_skipped} subjects skipped"
        )

        print(f"\n{'=' * 60}")
        print(f"Week {week}: computing correlations and saving results")
        print(f"{'=' * 60}")

        print("\nComputing intrinsic dimension-score correlations...")
        dimension_score_rows = []

        for region, data in week_dimension_score_data.items():
            if len(data["dimensions"]) < 2:
                dimension_score_rows.append(
                    {
                        "region": region,
                        "r": np.nan,
                        "p": np.nan,
                        "n_samples": len(data["dimensions"])
                    }
                )
                continue

            try:
                r_value, p_value = pearsonr(data["dimensions"], data["scores"])
            except Exception:
                r_value, p_value = np.nan, np.nan

            dimension_score_rows.append(
                {
                    "region": region,
                    "r": r_value,
                    "p": p_value,
                    "n_samples": len(data["dimensions"])
                }
            )

        dimension_score_csv = os.path.join(
            output_dir,
            f"week{week}_k100_dimension_score_correlation.csv"
        )
        dimension_score_df = save_correlation_csv(
            dimension_score_rows,
            dimension_score_csv
        )
        print(f"Saved Week {week} dimension-score correlations: {dimension_score_csv}")

        print("\nComputing Wasserstein-score correlations...")
        wasserstein_score_rows = []

        for region, data in week_wasserstein_score_data.items():
            if len(data["wasserstein"]) < 2:
                wasserstein_score_rows.append(
                    {
                        "region": region,
                        "r": np.nan,
                        "p": np.nan,
                        "n_samples": len(data["wasserstein"])
                    }
                )
                continue

            try:
                r_value, p_value = pearsonr(data["wasserstein"], data["scores"])
            except Exception:
                r_value, p_value = np.nan, np.nan

            wasserstein_score_rows.append(
                {
                    "region": region,
                    "r": r_value,
                    "p": p_value,
                    "n_samples": len(data["wasserstein"])
                }
            )

        wasserstein_score_csv = os.path.join(
            output_dir,
            f"week{week}_k100_wasserstein_score_correlation.csv"
        )
        wasserstein_score_df = save_correlation_csv(
            wasserstein_score_rows,
            wasserstein_score_csv
        )
        print(f"Saved Week {week} Wasserstein-score correlations: {wasserstein_score_csv}")

        print("\nComputing intrinsic dimension-Wasserstein correlations...")
        dimension_wasserstein_rows = []

        for region, data in week_dimension_wasserstein_data.items():
            if len(data["dimensions"]) < 2:
                dimension_wasserstein_rows.append(
                    {
                        "region": region,
                        "r": np.nan,
                        "p": np.nan,
                        "n_samples": len(data["dimensions"])
                    }
                )
                continue

            try:
                r_value, p_value = pearsonr(data["dimensions"], data["wasserstein"])
            except Exception:
                r_value, p_value = np.nan, np.nan

            dimension_wasserstein_rows.append(
                {
                    "region": region,
                    "r": r_value,
                    "p": p_value,
                    "n_samples": len(data["dimensions"])
                }
            )

        dimension_wasserstein_csv = os.path.join(
            output_dir,
            f"week{week}_k100_dimension_wasserstein_correlation.csv"
        )
        dimension_wasserstein_df = save_correlation_csv(
            dimension_wasserstein_rows,
            dimension_wasserstein_csv
        )
        print(
            f"Saved Week {week} dimension-Wasserstein correlations: "
            f"{dimension_wasserstein_csv}"
        )

        if len(dimension_score_df) > 0:
            print(f"\nTop 5 regions for Week {week} dimension-score correlation:")
            print(dimension_score_df.head(5).to_string(index=False))
        else:
            print(f"\nWeek {week}: not enough data for correlation analysis")

        processed_count += week_processed
        skipped_count += week_skipped

    print(f"\n{'=' * 80}")
    print("All weekly analyses completed")
    print(f"{'=' * 80}")
    print("Summary:")
    print(f"  - Successfully processed subjects: {processed_count}")
    print(f"  - Skipped subjects: {skipped_count}")
    print("  - Output files: 3 CSV files x 5 weeks")
    print(f"\nAll weekly results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()