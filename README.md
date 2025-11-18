# Explainable Foundation Model-Based Approach for Dementia Screening and Risk Stratification Using Retinal Fundus Images

This repository contains the official implementation of the study:
**"Explainable foundation model-based approach for dementia screening and risk stratification using retinal fundus images"**

The repository provides code for data preprocessing, model training, inference, performance evaluation, and explainability analysis used in the manuscript.

---

## 🧩 Installation
This codebase requires Python 3.11.8.
Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📂 Data Preparation

Please refer to **`data_preparation.ipynb`** for details on data preprocessing and cohort construction.

---

## 🧠 Model Training

Model training can be initiated using the provided shell script:

```bash
bash script/script.sh
```

The training process relies on the following key components:

* `trainer.py` — main training loop and optimization logic
* `data.py` — dataset loading and augmentation
* `models/` — implementation of foundation model architectures and fine-tuning strategies

---

## 🔍 Inference and Performance Evaluation

For model inference and evaluation of predictive performance, refer to:

* **`save_inference_outputs.py`** — generate and save model predictions
* **`evaluate_performance_regression.ipynb`** — assess performance metrics (AUROC, C-index, OR/HR, etc.)

Bootstrap-based performance confidence intervals are implemented in:

* **`bootstrap.ipynb`**

---

## 💡 Explainability Analysis

For saliency map generation, regional saliency quantification, and visualization, see:

* **`saliency_analysis.ipynb`**

---

## 📜 Other Manuscript Codes

Additional scripts used for figure generation and supplementary analyses are provided in:

* **`Other codes for manuscript.ipynb`**

---

## 📘 Citation

If you use this code or reproduce results from the study, please cite our paper:

```
to be updated
```

