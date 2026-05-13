# CortiGeo: Cortical Geometry

Code release for the fMRI analyses in **Low-Dimensional Representations Support Efficient Learning Across Brains and AI**.

## Overview

**CortiGeo** stands for **Cortical Geometry**. This repository contains the code used to analyze the fMRI component of the study *Low-Dimensional Representations Support Efficient Learning Across Brains and AI*. The analyses focus on geometric properties of cortical representations, including intrinsic dimensionality, Wasserstein distances between neural representations, ROI-level correlations with learning outcomes, and cortical visualization of the resulting statistics.

The repository is organized as a public archival release. It includes the main scripts used for the reported fMRI analyses as well as additional exploratory analyses that were developed during the project but were not all included in the final manuscript.

## Data Source

The fMRI data used in these analyses come from the public OpenNeuro dataset associated with:

Meshulam, M., Hasenfratz, L., Hillman, H. et al. **Neural alignment predicts learning outcomes in students taking an introduction to computer science course.** *Nature Communications* 12, 1922 (2021).  
https://doi.org/10.1038/s41467-021-22202-3

Dataset:

- OpenNeuro: https://openneuro.org/datasets/ds003233/versions/1.2.0

Raw fMRI data are not included in this repository. Users should download the dataset directly from OpenNeuro and adapt the paths in the scripts as needed.

## Repository Structure

- `code/`: curated analysis, statistics, QC, and visualization scripts.

This GitHub release is code-only. Raw data, derived tables, large result files, and figure outputs are not included.

## Code Directory Guide

The scripts in `code/` are grouped by workflow stage and analysis function.

### `code/01_qc_atlas_checks/`

Quality-control and atlas-inspection scripts. These scripts check BOLD coverage, inspect Harvard-Oxford atlas overlap, identify invalid or missing ROI data, compare angular gyrus review-session data, and visualize fMRI intrinsic-dimension maps for QC.

Typical uses:

- checking whether fMRI images cover the MNI template or atlas mask;
- inspecting missing or invalid ROI time-series files;
- validating ROI extraction and atlas alignment before downstream analyses.

### `code/02_metric_computation/`

Scripts for computing primary neural metrics. These include intrinsic-dimension maps, ROI-level dimension values, weekly ROI Wasserstein distances, fMRI ID maps, lag-1 autocorrelation, whole-brain cube-based dimension maps, and voxelwise dimension-Wasserstein correlation summaries.

Typical uses:

- computing intrinsic dimensionality from ROI or voxelwise fMRI representations;
- computing Wasserstein distances between representations;
- generating intermediate metric tables that are later used in correlation analyses.

### `code/03_correlation_analysis/`

Scripts for relating neural metrics to behavioral or learning-performance scores. This folder contains ROI-level, subject-level, week-specific, all-week, raw-value, PCA-controlled, FDR-related, persistent-homology-related, and angular-gyrus-specific correlation analyses.

Typical uses:

- correlating ROI intrinsic dimension with learning scores;
- correlating Wasserstein distances with learning scores;
- comparing week-specific and all-week analyses;
- running method variants such as raw-dimension, PCA-controlled, persistent-entropy, and lifetime-based analyses.

### `code/04_pairwise_procrustes/`

Scripts for pairwise subject/ROI Procrustes analyses. These scripts were developed to test whether pairwise neural alignment or representational similarity based on Procrustes distance related to learning outcomes.

Typical uses:

- extracting ROI-level fMRI matrices for pairwise comparison;
- computing Procrustes-based representational similarity;
- correlating pairwise neural similarity with pairwise score measures.

### `code/05_visualization/`

Plotting scripts for cortical maps and summary figures. This folder includes flatmaps, surface maps, ROI correlation maps, FDR-corrected bar plots, dimension and Wasserstein flatmaps, selected-subject boxplots, and pairwise Wasserstein matrix visualizations.

Typical uses:

- visualizing ROI-level correlations on cortical surfaces or flatmaps;
- generating publication-style summary plots;
- comparing significant and non-significant regions after correction.

### `code/06_statistics_utilities/`

Small statistical utility scripts. These include Benjamini-Hochberg FDR correction for CSV files and Pearson-correlation confidence-interval calculation using Fisher's z transform.

Typical uses:

- applying FDR correction to batches of result CSV files;
- reporting confidence intervals for Pearson correlations.

### `code/legacy/`

Older versions retained for provenance and comparison. These scripts reflect earlier implementations or exploratory versions of analyses that were later revised, replaced, or generalized.

## Historical Note on the Analyses

This project developed over an extended period. The earliest analyses focused on **week 2** of the dataset and extracted fMRI signals from the **angular gyrus**. As the project progressed, the analysis expanded from this focused ROI/week setting to a broader whole-dataset analysis across multiple weeks and brain regions.

For transparency and archival completeness, this release includes more than only the final scripts used in the manuscript. Some scripts correspond to exploratory analyses that were informative during development but were not ultimately included in the paper, including analyses based on **Procrustes distance**, **persistent homology**, persistent entropy, lifetime summaries, and other intermediate variants.

Because the project spans a long development period, fully reducing the repository to only a minimal final pipeline would require substantial additional reconstruction and could remove useful provenance. Therefore, this release keeps both the final analysis scripts and the organized exploratory scripts so that unused but relevant experimental attempts remain documented.

## Usage Notes

The scripts were written as research-analysis scripts rather than a single installable Python package. Many scripts define dataset paths, output paths, week numbers, ROI names, or CSV filenames near the top of the file. Before running them, users should:

1. Download the source dataset from OpenNeuro.
2. Arrange local data paths to match the expected script inputs, or edit the path constants in each script.
3. Install the required scientific Python dependencies used by the relevant script.
4. Run QC and metric-computation scripts before downstream correlation and visualization scripts.

The exact execution order depends on the analysis variant being reproduced.

## Contact

Questions about the code, file organization, or analysis variants can be raised through this repository or by email:

- Primary email: 12211804@mail.sustech.edu.cn
- Backup email: 1613173740@qq.com

## Citation

If you use this repository, please cite the associated manuscript:

**Low-Dimensional Representations Support Efficient Learning Across Brains and AI**

Please also cite the original fMRI dataset paper:

Meshulam, M., Hasenfratz, L., Hillman, H. et al. **Neural alignment predicts learning outcomes in students taking an introduction to computer science course.** *Nature Communications* 12, 1922 (2021). https://doi.org/10.1038/s41467-021-22202-3

---

# CortiGeo: 皮层几何

本文库是 **Low-Dimensional Representations Support Efficient Learning Across Brains and AI** 一文中 fMRI 实验部分的代码发布版本。

## 项目简介

**CortiGeo** 意为 **Cortical Geometry**，即“皮层几何”。这个代码库整理了论文 *Low-Dimensional Representations Support Efficient Learning Across Brains and AI* 中 fMRI 相关实验所使用的分析代码。分析内容主要围绕皮层表征的几何性质展开，包括内在维度、神经表征之间的 Wasserstein 距离、ROI 水平指标与学习结果之间的相关性，以及这些统计结果在皮层上的可视化。

这个仓库是面向公开发布和归档的版本。它不仅包含论文中主要使用的 fMRI 分析代码，也保留了一些项目探索阶段开发、但最终没有全部写入正文的实验代码。

## 数据来源

本项目使用的 fMRI 数据来自以下公开 OpenNeuro 数据集，对应论文为：

Meshulam, M., Hasenfratz, L., Hillman, H. et al. **Neural alignment predicts learning outcomes in students taking an introduction to computer science course.** *Nature Communications* 12, 1922 (2021).  
https://doi.org/10.1038/s41467-021-22202-3

数据集地址：

- OpenNeuro: https://openneuro.org/datasets/ds003233/versions/1.2.0

本仓库不包含原始 fMRI 数据。使用者需要从 OpenNeuro 下载数据，并根据本地目录结构调整脚本中的路径。

## 仓库结构

- `code/`：整理后的分析、统计、质控和可视化脚本。

这个 GitHub 发布版本只上传代码。原始数据、衍生表格、大型结果文件和图像输出均不包含在仓库中。

## 代码目录说明

`code/` 中的脚本按照分析流程和功能进行了分类。

### `code/01_qc_atlas_checks/`

质控和脑区模板检查相关代码。该目录中的脚本用于检查 BOLD 图像覆盖情况、Harvard-Oxford atlas 与数据的重叠情况、无效或缺失 ROI 数据、angular gyrus 复习周数据对比，以及 fMRI intrinsic-dimension map 的质控可视化。

主要用途包括：

- 检查 fMRI 图像是否覆盖 MNI 模板或 atlas mask；
- 检查缺失或无效的 ROI 时间序列文件；
- 在后续分析前验证 ROI 提取和 atlas 对齐情况。

### `code/02_metric_computation/`

神经表征指标计算代码。该目录包含内在维度图、ROI 水平维度值、每周 ROI Wasserstein 距离、fMRI ID map、lag-1 自相关、全脑 cube-based 维度图，以及 voxelwise 维度-Wasserstein 相关性结果的计算脚本。

主要用途包括：

- 从 ROI 或 voxelwise fMRI 表征中计算内在维度；
- 计算不同神经表征之间的 Wasserstein 距离；
- 生成后续相关性分析所需要的中间指标表。

### `code/03_correlation_analysis/`

神经指标与行为或学习成绩之间的相关性分析代码。该目录包含 ROI 水平、被试水平、特定周次、全周、原始值、PCA 控制、FDR 相关、persistent homology 相关，以及 angular gyrus 特定分析脚本。

主要用途包括：

- 分析 ROI 内在维度与学习成绩之间的相关性；
- 分析 Wasserstein 距离与学习成绩之间的相关性；
- 比较不同周次和全周分析结果；
- 运行 raw dimension、PCA-controlled、persistent entropy、lifetime 等不同方法变体。

### `code/04_pairwise_procrustes/`

两两被试或 ROI 的 Procrustes 分析代码。这些脚本用于探索基于 Procrustes 距离的神经对齐或表征相似性是否与学习结果相关。

主要用途包括：

- 提取用于两两比较的 ROI 水平 fMRI 矩阵；
- 计算基于 Procrustes 的表征相似性；
- 分析两两神经相似性与两两成绩指标之间的相关性。

### `code/05_visualization/`

绘图和可视化代码。该目录包含 cortical flatmap、surface map、ROI correlation map、FDR 校正后的柱状图、维度与 Wasserstein flatmap、被试 boxplot，以及 pairwise Wasserstein matrix 等可视化脚本。

主要用途包括：

- 将 ROI 水平相关性结果展示在皮层表面或 flatmap 上；
- 生成用于论文或补充材料的统计图；
- 比较校正后显著与不显著脑区的结果。

### `code/06_statistics_utilities/`

小型统计工具脚本。该目录包含对 CSV 文件批量进行 Benjamini-Hochberg FDR 校正的脚本，以及基于 Fisher z 变换计算 Pearson 相关置信区间的脚本。

主要用途包括：

- 对结果 CSV 文件批量进行 FDR 校正；
- 为 Pearson 相关系数报告置信区间。

### `code/legacy/`

旧版本代码归档。该目录中的脚本保留了较早的实现方式或探索阶段版本，用于追溯分析过程和进行版本对照。

## 关于分析过程的说明

这个项目持续时间较长。最早的实验从数据集中的 **week 2** 开始，并提取 fMRI 中的 **angular gyrus** 区域进行分析。随着项目推进，分析逐渐从这个特定脑区和周次扩展到多周次、多脑区的完整数据分析。

为了保证透明性和归档完整性，本发布版本不只包含最终写入论文的脚本，也包含了一些探索阶段的实验代码。其中包括后来没有实际写入文章正文的 **Procrustes distance**、**persistent homology**、persistent entropy、lifetime summary 等分析，以及其他中间方法变体。

由于项目时间跨度较长，如果只整理出一个极简的最终版本，需要重新追溯和重构大量历史代码，也可能丢失有价值的实验记录。因此，这里选择将最终分析脚本和经过整理的探索性脚本一起发布，方便之后阅读、复查和归档。

## 使用说明

这些代码是研究分析脚本，而不是一个单独封装好的 Python package。很多脚本会在文件开头定义数据路径、输出路径、周次、ROI 名称或 CSV 文件名。运行前建议使用者：

1. 从 OpenNeuro 下载原始数据集。
2. 根据本地目录结构调整各脚本中的路径常量。
3. 安装相应脚本需要的科学计算 Python 依赖。
4. 先运行质控和指标计算脚本，再运行后续相关性分析和可视化脚本。

具体运行顺序取决于希望复现的分析版本。

## 联系方式

如果在使用或阅读本仓库代码时有疑问，可以在本仓库下留言，也可以通过邮件联系：

- 主要邮箱：12211804@mail.sustech.edu.cn
- 备用邮箱：1613173740@qq.com

## 引用

如果使用本仓库，请引用相关论文：

**Low-Dimensional Representations Support Efficient Learning Across Brains and AI**

同时请引用原始 fMRI 数据集论文：

Meshulam, M., Hasenfratz, L., Hillman, H. et al. **Neural alignment predicts learning outcomes in students taking an introduction to computer science course.** *Nature Communications* 12, 1922 (2021). https://doi.org/10.1038/s41467-021-22202-3
