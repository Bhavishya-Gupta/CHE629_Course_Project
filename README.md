# 🧬 CHE629: Artificial Intelligence in Systems Biology
**Course Project: DeepGRNCs - Inferring Gene Regulatory Networks**

[![Author](https://img.shields.io/badge/Author-Bhavishya_Gupta-blue.svg)](https://github.com/Bhavishya-Gupta)
[![Institution](https://img.shields.io/badge/Institution-IIT_Kanpur-orange.svg)](https://www.iitk.ac.in/)
[![Course](https://img.shields.io/badge/Course-CHE629-success.svg)](#)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](#)

## 📌 Project Overview
This repository contains the code, analysis, and final documentation for the **CHE629 (Artificial Intelligence in Systems Biology)** course project completed at the Indian Institute of Technology (IIT) Kanpur.

The project explores **DeepGRNCs**, a robust deep learning framework engineered to infer complex Gene Regulatory Networks (GRNs) from biological expression datasets. By leveraging advanced data processing pipelines and specialized neural network architectures, this work aims to decode and map the intricate regulatory interactions between genes with high precision.

## 📂 Repository Structure

| File | Description |
| :--- | :--- |
| **`Project_Colab_Notebook.ipynb`** | An interactive Google Colab notebook containing the end-to-end implementation, including dataset loading, preprocessing, model training, and evaluation. |
| **`Project_Python_Code.py`** | A modular, standalone Python script version of the deep learning pipeline, optimized for local execution or cluster environments. |
| **`CHE629 Project Report.pdf`** | The comprehensive final project report, compiled in LaTeX following the official **ICLR format**. It details the methodology, mathematical foundations, and empirical results. |
| **`Reference Research Paper.pdf`** | The foundational literature and primary reference material that guided the conceptual and architectural implementation of the framework. |

## 🛠️ Technical Implementation
The computational pipeline is built to handle the complexities of biological datasets. Key components of the implementation include:
- **Dataset Processing:** Custom loaders and preprocessing scripts designed to normalize, filter, and structure raw gene expression matrices.
- **Deep Learning Framework:** Neural network modules tailored to capture the non-linear, high-dimensional dynamics of gene regulation.
- **Evaluation & Metrics:** Rigorous benchmarking to validate the predicted networks against ground-truth biological interactions.

## 🚀 Getting Started

### Prerequisites
To run the scripts locally, ensure you have a Python environment set up with standard scientific computing and machine learning libraries (e.g., PyTorch/TensorFlow, NumPy, Pandas, Scikit-learn).

### Running via Google Colab (Recommended)
1. Open `Project_Colab_Notebook.ipynb` in Google Colab.
2. Ensure your runtime is set to utilize a GPU hardware accelerator if available.
3. Upload the required datasets as directed in the initial cells.
4. Execute the cells sequentially. The notebook is entirely self-contained and handles necessary visualizations and outputs.

### Running Locally
To execute the pipeline directly from your terminal, run:
```bash
python Project_Python_Code.py
