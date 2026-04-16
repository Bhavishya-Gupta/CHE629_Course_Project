# CHE629 Course Project — DeepGRNCS

A polished, end-to-end implementation of **DeepGRNCS**, a deep learning framework for **jointly inferring gene regulatory networks (GRNs) across cell subpopulations**. This repository contains the complete project notebook, Python implementation, final report, and the reference research paper used for the study.

---

## Overview

DeepGRNCS is designed to infer regulatory relationships between transcription factors and target genes by leveraging information from multiple cell subpopulations. The project reproduces the core idea of the paper through a dual-stream neural architecture and benchmarks it against widely used GRN inference methods.

This repository is structured to be easy to run, easy to review, and easy to present. It includes:

- a fully reproducible Python/Colab workflow,
- synthetic data generation utilities,
- support for real BEELINE mHSC-L data,
- benchmark implementations of **GENIE3** and **GRNBoost2**,
- evaluation metrics and visualization utilities,
- exported results and figures for analysis.

---

## Key Features

- **Dual-stream DeepGRNCS architecture** for subpopulation-aware GRN inference
- **Synthetic Gaussian dataset experiments** to validate the method in controlled settings
- **BoolODE-based network experiments** for testing on diverse graph topologies
- **Ablation study on the number of subpopulations**
- **Real-data support** for the BEELINE **mHSC-L** benchmark
- **Baseline comparisons** against GENIE3 and GRNBoost2
- **Comprehensive evaluation** using AUROC, AUPRC, EPR, F1-score, precision, recall, and accuracy
- **Publication-style plots** and CSV summaries saved to disk

---

## Repository Structure

```text
CHE629_Course_Project/
├── CHE629 Project Report.pdf
├── Project_Colab_Notebook.ipynb
├── Project_Python_Code.py
├── Reference Research Paper.pdf
└── README.md
```

### File Summary

- **`Project_Python_Code.py`** — the main executable Python script / Colab-ready notebook export
- **`Project_Colab_Notebook.ipynb`** — notebook version of the project for interactive execution
- **`CHE629 Project Report.pdf`** — final project report
- **`Reference Research Paper.pdf`** — original research paper used as the project reference

---

## Methodology

The project follows the workflow below:

1. **Generate or load expression data**
   - Gaussian simulated subpopulations
   - BoolODE-style synthetic networks
   - Real BEELINE mHSC-L dataset

2. **Infer GRNs**
   - DeepGRNCS
   - GENIE3
   - GRNBoost2

3. **Evaluate predicted networks**
   - AUROC
   - AUPRC
   - EPR
   - F1-score
   - Precision
   - Recall
   - Accuracy

4. **Visualize results**
   - ROC and PR curves
   - Metric comparison bars
   - Confusion matrices
   - Heatmaps
   - Architecture diagram
   - Ablation plots

---

## Experiments Included

### 1. Gaussian Simulated Dataset
A multi-subpopulation synthetic benchmark used to test whether the model can recover regulatory structure under controlled conditions.

### 2. BoolODE Network Benchmarks
A set of Boolean-network-based simulations used to compare performance across different graph topologies.

### 3. Subpopulation Ablation Study
An ablation experiment that studies how performance changes as the number of available cell subpopulations increases.

### 4. Real BEELINE mHSC-L Dataset
A real-world benchmark for TF-to-target inference using the BEELINE mHSC-L dataset.

---

## Requirements

The project is built around standard scientific Python libraries:

- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- SciPy
- scikit-learn
- XGBoost

The notebook also creates output folders automatically for figures and results.

---

## Quick Start

### Option 1: Run in Google Colab
1. Open the notebook in Colab.
2. Run the cells from top to bottom.
3. The notebook will install dependencies, generate data, train models, evaluate results, and save outputs.

### Option 2: Run the Python script locally
```bash
python Project_Python_Code.py
```

The script is written in a Colab-friendly format, so it can also be executed by copying the cells into a notebook environment.

---

## Output Files

The project saves generated artifacts in the following folders:

- **`figures/`** — plots, architecture diagram, comparison charts, heatmaps
- **`results/`** — CSV summaries, inferred weight matrices, and experiment outputs

---

## Why This Project Stands Out

This repository is more than a code submission. It demonstrates:

- a clear translation of a recent research idea into working code,
- structured experimentation across synthetic and real datasets,
- careful comparison with established baselines,
- reproducibility through script-based execution,
- presentation-ready outputs for academic review.

---

## Citation / Reference

If you use or build upon this work, please cite the original research paper referenced in the repository:

**DeepGRNCS: deep learning-based framework for jointly inferring gene regulatory networks across cell subpopulations**  
Lei et al., *Briefings in Bioinformatics*, 2024

---

## Authors

Project authors listed in the code:
- Bhavishya Gupta
- Ayush Bokad
- Harshit Gupta
- Anas Ali

---

## Acknowledgements

- Original DeepGRNCS research paper
- BEELINE benchmark resources
- The open-source Python scientific computing ecosystem

---

## Contact

For questions, improvements, or collaboration, feel free to open an issue or update the repository documentation as needed.
