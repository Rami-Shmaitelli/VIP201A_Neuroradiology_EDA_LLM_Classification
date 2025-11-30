# VIP201A Neuroradiology EDA LLM Classification

## Overview

This repository contains implementations of **Large Language Models (LLMs)** applied to **classification tasks in medical and general domains**. The main goal of this project is to classify **neuroradiology reports** from patients with **Multiple Sclerosis (MS)** into **Evidence of Disease Activity (EDA)** or **No Evidence of Disease Activity (NEDA)**. Additionally, a publicly available **SMS spam dataset** is used to verify model robustness and performance on larger datasets.  

The project demonstrates:  
- Preprocessing and class balancing using **undersampling**.  
- Fine-tuning **LLaMA-based models** with **LoRA adapters** for memory-efficient training.  
- Evaluation using standard metrics: Accuracy, Precision, Recall, and F1 Score.  

---

## Purpose

The primary aim is to develop a **robust AI model capable of classifying MS neuroradiology reports** into clinically meaningful categories (EDA / NEDA) based on textual input. Secondary objectives include:  
- Validating that the model performs well on **larger datasets** by testing it on the **SMS spam classification task**.  
- Demonstrating **efficient fine-tuning strategies** (LoRA adapters) on limited GPU resources.  
- Ensuring reproducibility of preprocessing, training, and evaluation pipelines.  

---

## Datasets

### 1. Multiple Sclerosis (MS) Neuroradiology Dataset
- **Purpose:** Classify evidence of disease activity in MRI/clinical reports.  
- **Columns:**
  - `MSC research database ID` – Unique identifier for each report.  
  - `text` – Original report content.  
  - `output` – Structured textual output (used for reference).  
  - `label` – Ground-truth label: `EDA` or `NEDA`.  
- **Preprocessing:**
  - Undersampling of the majority class (`NEDA`) to balance the dataset.  
  - Split into **train / validation / test sets** (70% / 15% / 15%).  

### 2. SMS Spam Dataset (UC Irvine Repository)
- **Purpose:** Provide a secondary, larger dataset to verify model performance and ensure generalization beyond small medical datasets.  
- **Columns:**
  - `sms` – Text message content.  
  - `label` – Class label: `HAM` or `SPAM`.  

---

## Repository Structure

```
VIP201A_Neuroradiology_EDA_LLM_Classification/
│
├── data/                     # Preprocessed datasets
│   ├── train.json
│   ├── val.json
│   ├── test.json
│   ├── train_balanced.json
│   └── test_balanced.json
│
├── notebooks/                # Colab notebooks for different training/evaluation cases
│   ├── Case1_EDA_LLM.ipynb
│   ├── Case2_EDA_LLM.ipynb
│   └── ...
│
├── preprocessing/            # Dataset balancing and preparation scripts
│   └── dataset_balancing.ipynb
│
├── outputs/                  # Model predictions, metrics, logs
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md
```

---

## How to Run

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/VIP201A_Neuroradiology_EDA_LLM_Classification.git
cd VIP201A_Neuroradiology_EDA_LLM_Classification
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Open notebooks** in `notebooks/` or `preprocessing/` using **Google Colab** or **Jupyter**.  

4. Ensure dataset files are in the `data/` folder.  

5. Run cells sequentially for **preprocessing, training, evaluation, and analysis**.  

---

## Training & Evaluation

- Fine-tuned **LLaMA-based models** with **LoRA adapters** to reduce GPU memory usage.  
- Evaluated using metrics: **Accuracy, Precision, Recall, and F1 Score**.  
- Predictions and metrics are saved in **`outputs/`** as `.csv` or `.xlsx` for further analysis.  

---

## Methodology

- **Undersampling:** The MS dataset is imbalanced (more `NEDA` than `EDA` samples). Undersampling the majority class creates a balanced dataset for training and testing.  
- **Validation Set:** A separate validation set is created to monitor performance during training.  
- **Secondary Dataset Testing:** The SMS spam dataset tests model performance on larger, general datasets to ensure robustness.  
- **Inference:** Models are prepared for inference using `FastLanguageModel.for_inference()` and predictions are decoded using the tokenizer.  

---

## Outputs

- **Metrics:** Accuracy, Precision, Recall, F1 Score for each dataset.  
- **Predictions:** Full predictions stored in `.xlsx` for inspection and reporting.  
- **Logs:** Training and evaluation logs captured in `outputs/`.  

---

## Author

- **Rami Shmaitelli

---

## References

-MS Neuroradiology Dataset (ms_text_data_cleaned) – Provided by the American University of Beirut Medical Center (AUBMC) for the VIP201A project. This cleaned dataset was used for training and evaluating models for Evidence of Disease Activity (EDA) / No Evidence of Disease Activity (NEDA) classification.
- UC Irvine Machine Learning Repository: [SMS Spam Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)  
- Unsloth LoRA Hyperparameters Guide – [https://docs.unsloth.ai/get-started/fine-tuning-llms-guide](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide)
- imbalanced-learn library for dataset balancing (undersampling)  
- scikit-learn for train/test/validation splitting and metrics computation

