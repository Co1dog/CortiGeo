# Code Organization

This folder groups the release scripts by workflow stage and analysis function.
Version-specific scripts keep their version or week markers in the filename, for
example `week1`, `week2`, `week6`, `all_weeks`, `pca`, `fdr`, and `legacy`.

## Directory Guide

- `01_qc_atlas_checks/`: data quality control, atlas coverage checks, invalid ROI inspection, and ID-map visualization.
- `02_metric_computation/`: scripts that compute primary neural metrics, including ROI dimensions, Wasserstein distances, ID maps, autocorrelation, and voxelwise maps.
- `03_correlation_analysis/`: scripts that correlate computed neural metrics with behavioral or learning scores across ROIs, subjects, weeks, and method variants.
- `04_pairwise_procrustes/`: pairwise subject/ROI Procrustes analysis pipelines and related correlation scripts.
- `05_visualization/`: flatmaps, surface maps, bar plots, boxplots, and matrix visualizations for computed or corrected results.
- `06_statistics_utilities/`: reusable statistical helpers such as Pearson confidence intervals and Benjamini-Hochberg FDR correction.
- `legacy/`: older versions retained for provenance or comparison.
