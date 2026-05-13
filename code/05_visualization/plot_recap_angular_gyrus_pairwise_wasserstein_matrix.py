"""
This script computes and visualizes a subject-by-subject Wasserstein distance
matrix for Week 6 recap Angular Gyrus fMRI data. It scans recap-specific Angular
Gyrus NumPy files, loads each subject's wk1-wk5 recap time-series data,
standardizes each recap file separately, concatenates all recap sessions for the
same subject, computes temporal-difference representations, and calculates
pairwise Wasserstein OT costs between subjects. Subjects are sorted by total
exam score, and the final distance matrix is saved as both a NumPy file and a
heatmap.
"""

import os
import re

import numpy as np
import matplotlib.pyplot as plt
import ot

from sklearn.preprocessing import StandardScaler


ROOT_DIR = r"D:\Projects\thinkLikeExperts\ThinkLikeExpertsAngularGyrus"
OUTPUT_DIR = os.path.join(ROOT_DIR, "ws_subject_matrix_out")

os.makedirs(OUTPUT_DIR, exist_ok=True)

FILE_PATTERN = re.compile(
    r"^(sub-(s\d+))_ses-(wk\d+)_func_angular_gyrus_.*?_task-wk(\d+)recap_bold\.npy$",
    re.IGNORECASE
)

SCORES_BY_SUBJECT = {
    "s102": [0.5, 1.5, 1.5, 1, 0, 0.5, 1, 3, 1, 1.5, 1, 1.5, 1.5, 1, 3, 1.5],
    "s103": [1.5, 2, 1.5, 0.5, 0, 2, 0, 1.5, 0.5, 0.5, 0, 2.5, 0.5, 0, 0.5, 1],
    "s105": [2, 2, 1.5, 0, 0, 2, 2, 3, 2, 1.5, 2, 2, 1, 0.5, 2.5, 2],
    "s106": [2.5, 3, 1, 1, 0, 2, 2.5, 3, 2.5, 1.5, 0, 2.5, 0, 2, 2.5, 2],
    "s107": [1.5, 2, 1.5, 1.5, 2, 2, 3, 3, 2, 1.5, 3, 3, 2.5, 2, 3, 3],
    "s108": [2.5, 1.5, 1.5, 1, 2.5, 3, 2.5, 3, 2, 1.5, 1, 3, 2.5, 1.5, 3, 0],
    "s110": [2.5, 1.5, 0.5, 1.5, 2, 2, 1, 3, 2, 1.5, 0, 2, 0, 1.5, 3, 2.5],
    "s111": [2, 3, 2, 3, 0, 2, 3, 3, 2.5, 2, 0, 3, 1.5, 1.5, 3, 1.5],
    "s112": [1, 1, 1.5, 1, 1, 1, 0, 0, 0, 1, 0, 1.5, 0, 0.5, 0.5, 0.5],
    "s113": [2.5, 3, 1.5, 0.5, 2, 3, 3, 3, 2, 1.5, 3, 2, 1.5, 1, 3, 1],
    "s114": [2.5, 1.5, 2.5, 0, 0, 1.5, 0, 2, 1.5, 0.5, 0, 2, 0, 0, 0.5, 1],
    "s116": [2.5, 2.5, 1.5, 0, 1, 2, 2, 2.5, 1, 1.5, 2, 2.5, 1.5, 2, 2.5, 0],
    "s118": [1.5, 2, 1, 0, 0, 1.5, 0, 2.5, 1, 1.5, 1, 1.5, 0, 0, 3, 0],
    "s120": [1, 2, 1.5, 0.5, 0.5, 1.5, 1, 2.5, 1, 0.5, 1, 1.5, 0, 0.5, 3, 0],
    "s121": [1, 0.5, 1.5, 0.5, 0, 0, 0, 1, 0, 1.5, 0, 2, 0.5, 0.5, 2.5, 0],
    "s122": [1.5, 3, 1.5, 0, 0, 1, 0, 1, 0, 1.5, 0, 0, 0.5, 0, 0.5, 0],
    "s125": [1.5, 2, 1.5, 0.5, 0, 2, 1, 3, 2.5, 1, 1.5, 1, 0, 2, 3, 3],
    "s126": [2.5, 3, 0, 0, 2.5, 2, 2.5, 2.5, 3, 3, 2, 2.5, 1, 2, 3, 0],
    "s127": [2, 2, 1.5, 0, 0, 1, 2.5, 2.5, 2, 1.5, 0, 2, 0, 0.5, 2.5, 0],
    "s129": [2.5, 2, 2, 2, 2, 1.5, 0.5, 2, 2, 1.5, 1, 2.5, 0, 1, 3, 0],
    "s201": [None] * 16,
    "s213": [3] * 16,
    "s214": [2, 3, 2, 2, 2, 3, 1, 3, 2, 0, 2, 2, 0, 3, 3, 3],
    "s215": [3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    "s216": [1, 2, 3, 3, 0, 3, 3, 2, 2, 0, 2, 3, 3, 2, 3, 3]
}


def total_score(subject_key):
    scores = SCORES_BY_SUBJECT.get(subject_key)

    if scores is None:
        return np.nan

    valid_scores = [
        value
        for value in scores
        if value is not None and np.isfinite(value)
    ]

    if not valid_scores:
        return np.nan

    return float(np.sum(valid_scores))


def wasserstein_distance_ot_cost(data_1, data_2):
    if data_1.ndim != 2 or data_2.ndim != 2:
        return np.nan

    if data_1.shape[0] < 3 or data_2.shape[0] < 3:
        return np.nan

    try:
        combined_data = np.vstack([data_1, data_2])
        combined_data = StandardScaler().fit_transform(combined_data)

        normalized_1 = combined_data[:len(data_1)]
        normalized_2 = combined_data[len(data_1):]

        if np.isnan(normalized_1).any() or np.isnan(normalized_2).any():
            return np.nan

        weights_1 = np.ones(len(normalized_1)) / len(normalized_1)
        weights_2 = np.ones(len(normalized_2)) / len(normalized_2)

        cost_matrix = ot.dist(normalized_1, normalized_2)
        return float(ot.emd2(weights_1, weights_2, cost_matrix))

    except Exception:
        return np.nan


def subject_number(subject_name):
    numbers = re.findall(r"\d+", subject_name)
    return int(numbers[0]) if numbers else 10**9


def get_subject_score(subject_name, subject_key_map):
    subject_key = subject_key_map.get(subject_name)

    if subject_key is None:
        return np.nan

    return total_score(subject_key)


files = [
    file_name
    for file_name in os.listdir(ROOT_DIR)
    if file_name.lower().endswith(".npy")
]

subject_to_week_files = {}
subject_to_key = {}

for file_name in files:
    match = FILE_PATTERN.match(file_name)

    if not match:
        continue

    subject_full = match.group(1)
    subject_key = match.group(2)
    week_index = int(match.group(4))

    subject_to_week_files.setdefault(subject_full, {})
    subject_to_week_files[subject_full][week_index] = os.path.join(ROOT_DIR, file_name)
    subject_to_key[subject_full] = subject_key

if not subject_to_week_files:
    raise RuntimeError(f"No matching recap NumPy files were found in: {ROOT_DIR}")

subjects = sorted(subject_to_week_files.keys(), key=subject_number)
print(f"Found subjects in folder: {len(subjects)}")

subject_diff_data = {}

for subject_full in subjects:
    week_map = subject_to_week_files[subject_full]
    missing_weeks = [
        week_index
        for week_index in range(1, 6)
        if week_index not in week_map
    ]

    if missing_weeks:
        print(f"[Warning] {subject_full} missing recap weeks {missing_weeks}. Skipping.")
        continue

    subject_arrays = []
    is_valid = True

    for week_index in range(1, 6):
        data = np.load(week_map[week_index])

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        if data.ndim != 2:
            print(
                f"[Warning] {subject_full} week {week_index} has shape {data.shape}. "
                "Expected a 2D array. Skipping subject."
            )
            is_valid = False
            break

        data = StandardScaler().fit_transform(data)
        subject_arrays.append(data)

    if not is_valid:
        continue

    concatenated_data = np.concatenate(subject_arrays, axis=0)

    if concatenated_data.shape[0] < 4:
        print(
            f"[Warning] {subject_full} is too short after concatenation: "
            f"{concatenated_data.shape}. Skipping."
        )
        continue

    temporal_diff = concatenated_data[1:] - concatenated_data[:-1]

    if temporal_diff.shape[0] < 3:
        print(
            f"[Warning] {subject_full} is too short after temporal differencing: "
            f"{temporal_diff.shape}. Skipping."
        )
        continue

    subject_diff_data[subject_full] = temporal_diff

usable_subjects = list(subject_diff_data.keys())
print(f"Usable subjects with valid recap arrays: {len(usable_subjects)}")

if len(usable_subjects) < 2:
    raise RuntimeError("Fewer than two usable subjects are available.")

scored_subjects = [
    subject
    for subject in usable_subjects
    if np.isfinite(get_subject_score(subject, subject_to_key))
]

subjects_sorted = sorted(
    scored_subjects,
    key=lambda subject: get_subject_score(subject, subject_to_key)
)

print("Sorted subjects by total score from low to high:")

for subject in subjects_sorted:
    print(f"  {subject} total={get_subject_score(subject, subject_to_key):.3f}")

if len(subjects_sorted) < 2:
    raise RuntimeError("Fewer than two subjects with valid total scores are available.")

n_subjects = len(subjects_sorted)
distance_matrix = np.full((n_subjects, n_subjects), np.nan, dtype=np.float64)

for i in range(n_subjects):
    distance_matrix[i, i] = 0.0
    data_i = subject_diff_data[subjects_sorted[i]]

    for j in range(i + 1, n_subjects):
        data_j = subject_diff_data[subjects_sorted[j]]
        distance = wasserstein_distance_ot_cost(data_i, data_j)

        distance_matrix[i, j] = distance
        distance_matrix[j, i] = distance

matrix_path = os.path.join(
    OUTPUT_DIR,
    "angular_gyrus_wasserstein_matrix_sorted.npy"
)

np.save(matrix_path, distance_matrix)

valid_values = distance_matrix[
    np.isfinite(distance_matrix) & (distance_matrix > 0)
]

vmin = 2950.0

if valid_values.size == 0:
    vmax = vmin + 100.0
else:
    vmax = float(np.percentile(valid_values, 80))
    max_value = float(np.max(valid_values))

    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = max_value if max_value > vmin else vmin + 50.0

fig, axis = plt.subplots(
    figsize=(0.6 * n_subjects + 4, 0.6 * n_subjects + 4)
)

image = axis.imshow(
    distance_matrix,
    interpolation="nearest",
    vmin=vmin,
    vmax=vmax,
    cmap="RdYlBu_r"
)

axis.set_title("Angular Gyrus Wasserstein distance")

axis.set_xticks(np.arange(n_subjects))
axis.set_yticks(np.arange(n_subjects))
axis.set_xticklabels(subjects_sorted, rotation=90)
axis.set_yticklabels(subjects_sorted)

if n_subjects <= 40:
    for i in range(n_subjects):
        for j in range(n_subjects):
            value = distance_matrix[i, j]

            if np.isfinite(value):
                axis.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7
                )

colorbar = fig.colorbar(image, ax=axis)
colorbar.set_label("Wasserstein OT cost")

plt.tight_layout()

figure_path = os.path.join(
    OUTPUT_DIR,
    "angular_gyrus_wasserstein_heatmap_sorted_by_total_score.png"
)

plt.savefig(figure_path, dpi=300)
plt.show()

print(f"\n[Done] Sorted Wasserstein matrix saved to: {matrix_path}")
print(f"[Done] Heatmap saved to: {figure_path}")