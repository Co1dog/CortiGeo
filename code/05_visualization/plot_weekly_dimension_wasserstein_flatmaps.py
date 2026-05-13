"""
This script visualizes weekly ROI-level dimension, Wasserstein, and
dimension-Wasserstein correlation results as cortical flatmaps and compact
correlation plots. For each week, it reads three CSV files produced by the
within-subject video-variability analysis, filters predefined unreliable regions,
maps significant ROI-level correlations back to the Harvard-Oxford cortical
atlas, resamples them into Pycortex subject space, and saves both Pycortex
flatmaps and top-bottom regional correlation plots for dimension-score,
Wasserstein-score, and Wasserstein-dimension relationships.
"""

import os
import re
import tempfile

import cortex
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nilearn import datasets, image
from scipy.ndimage import generic_filter


def fill_nan_with_nearest(values):
    center = values[len(values) // 2]

    if not np.isnan(center) and center != 0:
        return center

    non_nan = values[~np.isnan(values)]
    non_nan = non_nan[non_nan != 0]

    return non_nan[0] if len(non_nan) > 0 else np.nan


def normalize_name(text):
    return re.sub(r"[^a-z]", "", str(text).lower())


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
        r"([Cc]ortex)\s*[Ff]ormerly": "Cortex Formerly",
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
        ),
        r"[Gg]yrus\s*[Tt]emporo\s*[Oo]ccipital\s*[Pp]art": (
            "Gyrus Temporo Occipital Part"
        )
    }

    for pattern, replacement in corrections.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return re.sub(r"\s+", " ", text).strip()


def wrap_label_two_lines(text):
    text = text.strip()
    words = text.split()

    if len(words) <= 4:
        special_cases = [
            "parahippocampal gyrus posterior division",
            "parahippocampal gyrus anterior division",
            "supramarginal gyrus posterior division",
            "supramarginal gyrus anterior division"
        ]

        if text.lower() in special_cases:
            return " ".join(words[:2]) + "\n" + " ".join(words[2:])

        return text

    midpoint = len(words) // 2
    return " ".join(words[:midpoint]) + "\n" + " ".join(words[midpoint:])


def filter_regions(df):
    excluded_regions = [
        "InferiorTemporalGyrusanteriordivision",
        "TemporalFusiformCortexanteriordivision",
        "OccipitalPole"
    ]

    df = df.copy()
    df["region_stripped"] = (
        df["region"]
        .astype(str)
        .str.replace(r"[^A-Za-z]", "", regex=True)
    )

    filtered_df = df[~df["region_stripped"].isin(excluded_regions)].copy()
    return filtered_df.drop(columns=["region_stripped"])


def make_corr_plot(df, correlation_column, output_path):
    df = df.copy()
    df["short_region"] = df["region"].astype(str).str.strip()

    df_sorted = df.sort_values(by=correlation_column)
    bottom_regions = df_sorted.head(5)
    top_regions = df_sorted.tail(5)

    df_plot = pd.concat([bottom_regions, top_regions]).reset_index(drop=True)

    bar_colors = [
        "lightsteelblue" if value < 0 else "lightcoral"
        for value in df_plot[correlation_column]
    ]

    df_plot["label_processed"] = (
        df_plot["short_region"]
        .apply(add_space_before_caps)
        .apply(wrap_label_two_lines)
    )

    plt.figure(figsize=(10, 6), dpi=300)
    axis = plt.gca()
    axis.set_facecolor("whitesmoke")

    y_positions = range(len(df_plot))

    plt.hlines(
        y=y_positions,
        xmin=0,
        xmax=df_plot[correlation_column],
        color=bar_colors,
        linewidth=4
    )

    plt.plot(
        df_plot[correlation_column],
        y_positions,
        "o",
        color="dimgrey",
        markersize=8
    )

    for x_value, y_value, label in zip(
        df_plot[correlation_column],
        y_positions,
        df_plot["label_processed"]
    ):
        if x_value > 0:
            plt.text(
                -0.02,
                y_value,
                label,
                va="center",
                ha="right",
                fontsize=15
            )
        else:
            plt.text(
                0.02,
                y_value,
                label,
                va="center",
                ha="left",
                fontsize=15
            )

    plt.yticks([])
    plt.xlabel("Correlation", fontsize=15)
    plt.xticks(fontsize=13)
    plt.xlim(-0.8, 0.9)

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_visible(False)
    axis.spines["bottom"].set_visible(True)
    axis.tick_params(axis="y", length=0)
    axis.tick_params(axis="x", length=5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


template = r'inkscape \"{svg}\" --export-type=png --export-filename=\"{png}\" --export-area-drawing'
os.environ["CORTEX_INKSCAPE_CMD"] = template
os.environ["CORTEX_SVGTEMP"] = tempfile.gettempdir()

SUBJECT = "subj01"
XFM = "full"
DB_ROOT = r"C:\Users\Admin\anaconda3\envs\myenv\share\pycortex\db"
REF_IMG_PATH = os.path.join(DB_ROOT, SUBJECT, "transforms", XFM, "reference.nii.gz")

os.environ["CORTEXFOLDER"] = DB_ROOT

atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr0-2mm")
atlas_img = atlas.maps if isinstance(atlas.maps, nib.Nifti1Image) else nib.load(atlas.maps)
atlas_data = atlas_img.get_fdata()
atlas_labels = list(atlas.labels)
name_to_index = {
    normalize_name(label): index
    for index, label in enumerate(atlas_labels)
}

ref_img = nib.load(REF_IMG_PATH)
xfm_ref_shape = cortex.db.get_xfm(SUBJECT, XFM).shape


def make_flatmap(correlation_df, correlation_column, p_column, output_path):
    correlation_map = np.zeros_like(atlas_data, dtype=np.float32)

    for _, row in correlation_df.iterrows():
        region = str(row["region"]).strip()
        correlation = row[correlation_column]
        p_value = row[p_column]

        if pd.isna(correlation) or pd.isna(p_value) or float(p_value) > 0.05:
            value = np.nan
        else:
            value = float(correlation)

        match = re.match(r"(?:region_)?(\d+)", region)

        if match:
            atlas_index = int(match.group(1))
        else:
            region_clean = re.sub(r"^region_\d+_", "", region)
            region_clean = re.sub(r"^region_", "", region_clean)
            atlas_index = name_to_index.get(normalize_name(region_clean), None)

        if atlas_index is not None:
            correlation_map[atlas_data == atlas_index] = value

    mni_correlation_img = nib.Nifti1Image(
        correlation_map,
        atlas_img.affine
    )

    subject_img = image.resample_img(
        mni_correlation_img,
        target_affine=ref_img.affine,
        target_shape=ref_img.shape,
        interpolation="continuous"
    )

    data = subject_img.get_fdata().astype(np.float32)

    if data.shape != xfm_ref_shape:
        if data.shape[::-1] == xfm_ref_shape:
            data = data.transpose(2, 1, 0)
        elif data.transpose(1, 0, 2).shape == xfm_ref_shape:
            data = data.transpose(1, 0, 2)

    data[data == 0] = np.nan
    nan_mask = np.isnan(data)

    filled_data = generic_filter(
        data,
        fill_nan_with_nearest,
        size=3,
        mode="mirror"
    )
    filled_data = generic_filter(
        filled_data,
        fill_nan_with_nearest,
        size=3,
        mode="mirror"
    )
    filled_data[nan_mask] = np.nan

    vmax = np.nanmax(np.abs(filled_data)) if np.isfinite(filled_data).any() else 1.0

    fig = plt.figure(figsize=(14, 6), dpi=300)

    volume = cortex.Volume(
        filled_data,
        SUBJECT,
        XFM,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax
    )

    cortex.quickflat.make_figure(
        volume,
        with_curvature=True,
        with_sulci=False,
        with_rois=False,
        with_colorbar=False,
        curvature_brightness=0.9,
        curvature_contrast=0.04,
        nanmean=True,
        fig=fig,
        linewidth=3,
        labelsize="20pt",
        linecolor="black"
    )

    plt.subplots_adjust(top=0.9, bottom=0.1)

    scalar_map = plt.cm.ScalarMappable(
        cmap="RdBu_r",
        norm=plt.Normalize(vmin=-vmax, vmax=vmax)
    )
    scalar_map.set_array([])

    colorbar = fig.colorbar(
        scalar_map,
        ax=fig.axes,
        orientation="horizontal",
        fraction=0.046,
        pad=0.08
    )
    colorbar.ax.tick_params(labelsize=18)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


for week_id in range(1, 7):
    input_dir = f"wasserstein_rois_week{week_id}_video_with_scores_log"
    output_dir = f"flatmaps_week{week_id}"

    os.makedirs(output_dir, exist_ok=True)

    dimension_csv = os.path.join(
        input_dir,
        f"roi_dimension_correlations_week{week_id}.csv"
    )
    wasserstein_csv = os.path.join(
        input_dir,
        f"roi_wasserstein_correlations_week{week_id}.csv"
    )
    wasserstein_dimension_csv = os.path.join(
        input_dir,
        f"roi_wasserstein_dimension_log_correlations_week{week_id}.csv"
    )

    if not (
        os.path.exists(dimension_csv)
        and os.path.exists(wasserstein_csv)
        and os.path.exists(wasserstein_dimension_csv)
    ):
        print(f"[Week {week_id}] Missing input CSV files. Skipping.")
        continue

    dimension_df = pd.read_csv(dimension_csv)
    wasserstein_df = pd.read_csv(wasserstein_csv)
    wasserstein_dimension_df = pd.read_csv(wasserstein_dimension_csv)

    dimension_df.rename(
        columns={
            "r": "r_dimension_score",
            "p": "p_dimension_score",
            "mean_dimension": "dimension"
        },
        inplace=True
    )

    wasserstein_df.rename(
        columns={
            "r": "r_wasserstein_score",
            "p": "p_wasserstein_score",
            "mean_wasserstein": "wasserstein"
        },
        inplace=True
    )

    wasserstein_dimension_df.rename(
        columns={
            "r": "r_wasserstein_dimension",
            "p": "p_wasserstein_dimension"
        },
        inplace=True
    )

    dimension_df = filter_regions(dimension_df)
    wasserstein_df = filter_regions(wasserstein_df)
    wasserstein_dimension_df = filter_regions(wasserstein_dimension_df)

    make_flatmap(
        dimension_df,
        "r_dimension_score",
        "p_dimension_score",
        os.path.join(output_dir, f"dimension_score_flatmap_week{week_id}.png")
    )

    make_corr_plot(
        dimension_df,
        "r_dimension_score",
        os.path.join(output_dir, f"dimension_score_corrplot_week{week_id}.png")
    )

    make_flatmap(
        wasserstein_df,
        "r_wasserstein_score",
        "p_wasserstein_score",
        os.path.join(output_dir, f"wasserstein_score_flatmap_week{week_id}.png")
    )

    make_corr_plot(
        wasserstein_df,
        "r_wasserstein_score",
        os.path.join(output_dir, f"wasserstein_score_corrplot_week{week_id}.png")
    )

    make_flatmap(
        wasserstein_dimension_df,
        "r_wasserstein_dimension",
        "p_wasserstein_dimension",
        os.path.join(output_dir, f"wasserstein_dimension_flatmap_week{week_id}.png")
    )

    make_corr_plot(
        wasserstein_dimension_df,
        "r_wasserstein_dimension",
        os.path.join(output_dir, f"wasserstein_dimension_corrplot_week{week_id}.png")
    )