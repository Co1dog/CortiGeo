"""
This script generates publication-style bar plots for FDR-corrected regional
correlation results from fMRI analyses. It reads region-wise correlation CSV
files for distance-score, dimension-score, and dimension-distance relationships,
removes predefined regions that should be excluded from visualization, selects
the five strongest negative and five strongest positive correlations, formats
region labels for readability, and saves compact correlation plots for each
analysis type and recording week.
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt


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
        r"([Gg]yrus)\s*[Pp]osteriordivision": "Gyrus Posterior Division",
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

    mid = len(words) // 2
    return " ".join(words[:mid]) + "\n" + " ".join(words[mid:])


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

    df = df[~df["region_stripped"].isin(excluded_regions)].copy()
    return df.drop(columns=["region_stripped"])


def make_corr_plot(df, correlation_column, output_path):
    df = df.copy()
    df["short_region"] = df["region"].astype(str).str.strip().str.title()

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
    ax = plt.gca()
    ax.set_facecolor("whitesmoke")

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
                fontsize=14
            )
        else:
            plt.text(
                0.02,
                y_value,
                label,
                va="center",
                ha="left",
                fontsize=14
            )

    plt.yticks([])
    plt.xlabel("Correlation", fontsize=16)
    plt.xticks(fontsize=16)
    plt.xlim(-0.65, 0.65)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    base_dir = "fdr_corrected_correlation_results/voxel_wise_k10_results_fdr"
    os.makedirs(base_dir, exist_ok=True)

    for week_id in range(1, 2):
        week_tag = f"week{week_id}"

        output_dir = f"region_correlation_plots_{week_tag}"
        os.makedirs(output_dir, exist_ok=True)

        analysis_files = [
            {
                "csv_path": os.path.join(
                    base_dir,
                    f"{week_tag}_distance_score_correlation_fdr.csv"
                ),
                "output_path": os.path.join(
                    output_dir,
                    f"{week_tag}_k10_distance_score_corrplot_fdr.png"
                )
            },
            {
                "csv_path": os.path.join(
                    base_dir,
                    f"{week_tag}_dimension_score_correlation_fdr.csv"
                ),
                "output_path": os.path.join(
                    output_dir,
                    f"{week_tag}_k10_dimension_score_corrplot_fdr.png"
                )
            },
            {
                "csv_path": os.path.join(
                    base_dir,
                    f"{week_tag}_dimension_distance_correlation_fdr.csv"
                ),
                "output_path": os.path.join(
                    output_dir,
                    f"{week_tag}_k10_dimension_distance_corrplot_fdr.png"
                )
            }
        ]

        for analysis in analysis_files:
            corr_df = pd.read_csv(analysis["csv_path"])
            corr_df = filter_regions(corr_df)

            make_corr_plot(
                df=corr_df,
                correlation_column="r",
                output_path=analysis["output_path"]
            )


if __name__ == "__main__":
    main()