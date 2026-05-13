"""
This script visualizes ROI-level intrinsic dimension values as a sorted bar
plot. It loads a CSV file containing region-wise intrinsic dimension estimates,
removes predefined regions from the visualization, extracts readable region
names, sorts regions by intrinsic dimension, and displays a bar plot showing the
relative dimensionality of cortical regions.
"""

import pandas as pd
import matplotlib.pyplot as plt


CSV_PATH = "brain_region_dimensions/region_dimensions.csv"

EXCLUDED_REGIONS = [
    "InferiorTemporalGyrusanteriordivision",
    "TemporalFusiformCortexanteriordivision",
    "OccipitalPole"
]

df = pd.read_csv(CSV_PATH, index_col=0)

df = df[~df.index.isin(EXCLUDED_REGIONS)]

df["short_region"] = df.index.str.replace(
    r"region_\d+_",
    "",
    regex=True
)

df_sorted = df.sort_values(by="Intrinsic_Dimension")

bar_colors = [
    "lightsteelblue" if value < 0 else "navajowhite"
    for value in df_sorted["Intrinsic_Dimension"]
]

plt.figure(figsize=(18, 6))

plt.bar(
    df_sorted["short_region"],
    df_sorted["Intrinsic_Dimension"],
    color=bar_colors
)

plt.ylabel("Intrinsic dimension")
plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.show()