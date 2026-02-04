# MP-CADMS: M-protein Computer-Aided Diagnosis and monitoring System
We developed an M-protein Computer-Aided Diagnosis and monitoring System (MP-CADMS) based on deep learning, which employs two modal data of immunofixation electrophoresis (IFE) and structured clinical information (SCI) to achieve the diagnosis and monitoring of M-protein. The SCI involves 2 demographics (age and gender) and 34 test items (sFLC-κ, sFLC-λ, sFLC-κ/λ, α1, α2, Alb%, β1, β2, γ, A/G, F-κ, F-λ, M-Pro, PT, PT%, INR, Fbg, APTT, APTT-R, TT, D-Dimer, Cr(E), Ca, HGB, CK, NT-proBNP, LD, UA, CRP/hsCRP, Alb, 24hUPr, U-Pro, 24hU-V, β2-MG). The diagnosis tasks contain the M-protein presence detection (task 1), isotype classification (task 2) and severity grading. And the monitoring tasks contain the recurrence-recovery prediction (taks 4)and severity-progression prediction (task 5).
Here we release our Python codes for research reading. The code will be improved over time. 

## Contents

- [System requirements](#system-requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Usage Instructions](#usage-instructions)
- [Datasets](#datasets)
- [Expected runtime](#expected-runtime)
- [Repository structure](#repository-structure)
- [Citation](#citation)
- [Contact](#contact)

## System Requirements
This section lists (i) software dependencies and OS (with versions), (ii) versions tested, and (iii) any required non-standard hardware.

### Operating system
- **Tested on:** **Windows 10/11** (64-bit)

### Software dependencies
- **Python:** 3.10.5  
- **PyTorch:** 1.12.0  
- **Scientific stack:**  NumPy 1.23.1, pandas 1.4.4, SciPy 1.8.1, scikit-learn 1.1.1, matplotlib 3.5.2  

> Note: Additional dependencies may be required for specific modules (e.g., NeuralODE solvers). Please install from `requirements.txt` provided in this repository.

### Versions tested
- The codebase has been tested on **Windows 10/11** with **Python 3.10.5** and **PyTorch 1.12.0**, together with the package versions listed above.

### Hardware requirements
- **Recommended / non-standard hardware for training:**  
  - **GPU:** NVIDIA GPU recommended. Our experiments were run on **3 × NVIDIA RTX 3060 Ti** GPUs.  
  - **CPU/RAM:** Intel Core i7-8700 CPU and **256 GB RAM** (used in our experiments).
- **Minimum (approximate):**  
  - Training can run on a **single NVIDIA GPU** (with reduced throughput).  
  - Inference can be performed on a **single GPU**.  
  - CPU-only execution is possible for limited testing but will be substantially slower.

---

## Installation

### 1) Create a clean environment
We recommend using Conda (or venv) to avoid dependency conflicts.

```bash
conda create -n mpcadms python=3.10.5 -y
conda activate mpcadms
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) (Optional) Install PyTorch with CUDA
Install a CUDA-enabled build matching your NVIDIA driver/CUDA environment.  
Please refer to the official PyTorch installation selector: https://pytorch.org/get-started/locally/

### Typical install time on a “normal” desktop computer
On a typical desktop (e.g., 8–16 CPU cores, 16–32 GB RAM, stable broadband), installing dependencies usually takes:
- **~5-15 minutes** if using pre-built wheels (most common), excluding the time to download large packages (notably PyTorch/CUDA).
- **~10-30 minutes** if you need to reinstall/resolve CUDA-related packages or if network bandwidth is limited.
 
---

## Datasets
- The datasets from four participating hospitals cannot be shared publicly due to privacy restrictions. Here we released the self-collected (SC) dataset, which is collected from scientific publications (English and Chinese), and various social media platforms. The SC dataset contains 271 images, i.e., 45 negative and 226 positive (25 IgAκ, 28 IgAλ, 66 IgGκ, 39 IgGλ, 27 IgMκ, 14 IgMλ, 9 κ and 18 λ). You can use this dataset for the IFE-modality validation towards presence and PI diagnosis.

- We have also provised a demo dataset containing 50 samples (`demo_dataset/`), i.e., 10 negative and 40 positive (6 IgAκ, 6 IgAλ, 6 IgGκ, 6 IgGλ, 5 IgMκ, 5 IgMλ, 3 κ and 3 λ; 8 WP, 20 P(+), and 12 SP(++)).

---

## Quick start

The fastest way to verify MP-CADMS is to use our **Hugging Face (HF) Spaces** with the provided **demo dataset**. This requires **no local environment setup**.

### 1) Download the demo dataset or prepare private datasets
Download the lightweight demo dataset (50 samples) from `demo_dataset/`, and crop IFE images to remove lane marks and albumin region;

Or prepare your provate datasets containing two modalities of IFE and SCI

### 2) Evaluate the performance of MP-CADMS on HF Spaces
Open the diagnosis demo, then upload the cropped IFE images and fill in the corresponding SCI values for interactive verification:

- **HF (Diagnosis):** https://huggingface.co/spaces/THUxiaoming/MP-CADMS-Diagnosis
- **HF (Monitoring):** https://huggingface.co/spaces/THUxiaoming/MP-CADMS-Monitoring

---

## Usage Instructions
### Model Development and evaluation
- **Pretrain:** The model architecture of IFE extractor employs the self-supervised learning paradigm. In the pretraining stage, both 7,689 unlabeled and 8,943 labeled IFE images are used to train a variational autoencoder (VAE), through completing the pretext task of reconstructing the input iamge pixels. Then the encoder and bottleneck in pretrained VAE are fine-tuned to perform the downstream tasks of diagnosis and monitoring. You can run `IFE_pretrain.py` (or `Pretrain_IFE_extractor.py` in Model_Develop folder) to accomplish the pretraining task.

- **M-protein Presence Detection:** For the IFE-modality, the pretrained encoder and bottleneck are used to extract the features. And for the SCI-modality, one-dimensional SCI is first transformred into two-dimensional matrix, where the diagonal elements correspond to the original values, and the non-diagonal elements correspond to the correlation between any two test items. Then a ResNet is employed to extract the features of SCI matrix. Two modal features are fused via several stacked transformer blocks and projected to two probabilities of negative and positive classes. The training and evaluation codes using multimodal and single-modality data are included in Model_Develop folder:
  - **Multimodal diagnosis:** You can run PresDet_Multimodality_train.py to accomplish the training process of negative-positive classifier, and PresDet_Multimodality_eva.py to evaluate the performance of trained classifier.
  - **IFE-modality diagnosis:** You can run `PresDet_IFE_train.py` to accomplish the training process, and PresDet_IFE_eva.py to evaluate the performance of trained classifier.
  - **SCI-modality diagnosis:** You can run `PresDet_SCI_train.py` to accomplish the training process, and PresDet_SCI_eva.py to evaluate the performance of trained classifier.  

- **Multitask of Positive-Isotype (PI) Classification and Positive-Severity (PS) Grading:** The extraction and fusion of two modal features are the same as the above description. Since a positive sample possesses both isotype and severity attributes simultaneously, we introduce a multitask learning paradigm to complete the two diagnosis tasks. One branch (PS) projects the fused features into three probabilities corresponding to WP, P(+) and SP(++) classes, while the other branch (PI) projects the fused features into eight probabilities corresponding to IgAκ, IgAλ, IgGκ, IgGλ, IgMκ, IgMλ, κ and λ isotypes. The training and evaluation codes using multimodal and single-modality data are included in Model_Develop folder:
  - **Multimodal diagnosis:** You can run `Multitask_Multimodality_train.py` to accomplish the training process of multitask classifier, and `Multitask_Multimodality_eva.py` to evaluate the performance of trained classifier.
  - **IFE-modality diagnosis:** You can run `Multitask_IFE_train.py` to accomplish the training process, and `Multitask_IFE_eva.py` to evaluate the performance of trained classifier.
  - **SCI-modality diagnosis:** You can run `Multitask_SCI_train.py` to accomplish the training process, and `Multitask_SCI_eva.py` to evaluate the performance of trained classifier.

- **Monitoring:** For a patient's temporal sequence, the extraction and fusion of IFE and SCI features at each time remain consistent with the above description, except that the information of diagnosis labels is also introduced. The three-modality fusion features of all historical time are integrated through a time-aware LSTM. Then the integrated features serve as the initial value conditions of Neural ODE for continuous prediction, including the prediction of recurrence-recovery and severity-progression. The training and evaluation codes are also included in Model_Develop folder: You can run `Trans_TSMonitor_train.py` to achieve the training process of two nmonitors, and `Trans_TSMonitor_eva.py` to evaluate the performance of trained monitors.

### Statistical Analysis

We provide all scripts used for statistical significance testing in the folder `Statistical_Analysis/`. Briefly, we use **DeLong’s test** for comparing correlated AUROCs, and **permutation tests** for all other evaluation metrics (AUPRC, sensitivity, specificity, top-1 accuracy, F1-score, precision). The scripts are organized by task:

- `pd_AUROC_delong_pvalues.py`: DeLong’s test for AUROC in **presence detection** (or **recurrence–recovery monitoring**) tasks.
- `pd_others_permutation_pvalues.py`: permutation tests for **AUPRC, sensitivity, specificity, top-1 accuracy, F1-score, precision** in presence detection (or recurrence–recovery monitoring) tasks.
- `pi_AUROC_delong_pvalues.py`: DeLong’s test for AUROC in **8-class isotype classification** (PI) task.
- `pi_others_permutation_pvalues.py`: permutation tests for **AUPRC, sensitivity, specificity, top-1 accuracy, F1-score, precision** in 8-class isotype classification task.
- `ps_AUROC_delong_pvalues.py`: DeLong’s test for AUROC in **3-class severity grading** (PS) and/or **severity-progression monitoring** tasks.
- `ps_others_permutation_pvalues.py`: permutation tests for **AUPRC, sensitivity, specificity, top-1 accuracy, F1-score, precision** in 3-class severity grading and/or severity-progression monitoring tasks.

### Data imputation (Remasker)

The folder `Data_Imputer/` contains the implementation of **Remasker**, a **label-agnostic** SCI missing-value imputer based on a **masked autoencoder**. Both the encoder and decoder adopt Transformer-based architectures. Remasker learns the dependencies among SCI variables by reconstructing randomly masked entries from the observed (non-missing) values, and then uses the learned cross-variable correlations to impute missing indicators.

- `tony_eva_example.py`: training script for Remasker and imputation of missing values in the **training set**.
- `remasker_eva2.py`: imputation of missing values in the **validation sets** (e.g., internal/external validation) using a trained Remasker model.

> Note: Remasker does not use outcome labels at any stage; it models the joint distribution of SCI variables only, which helps avoid label leakage in the imputation procedure.
---

## Expected runtime

All reported runtimes are **approximate wall-clock times** observed on our workstation (**3 × RTX 3060 Ti**, 256 GB RAM, i7-8700). Actual time may vary depending on early stopping, CUDA/driver versions, and I/O performance.

- **IFE pretraining (VAE; 16,883 images at 128×128):** ~4 hours  
- **SCI imputation training (Remasker; 6,922 records, 36 indicators):** ~6 hours  
- **Each diagnosis model training:** ~5–7 hours per model  
- **Each monitoring model training:** ~9–12 hours per model  
- **Inference (internal + external validation):** ~5–15 minutes on a single GPU  
- **Full end-to-end reproduction (all modules + evaluations):** ~24–48 hours

---

## Repository structure

```
.
├── requirements.txt
├── Model Card.md
├── demo_dataset/
├── SC_dataset/
├── Model_Develop/
├── Data_Imputer/
└── Statistical_Analysis/
```
---

## Citation

If you use this code, please cite:

```bibtex
@article{MP-CADMS,
  title={A Computer-Aided System for Enhanced M-protein Diagnosis and Monitoring via Multimodal Learning},
  author={Xuming An, Pengchang Li, Chen Zhang et al.},
  journal={submitted to Nature Communications for peer review},
  year={2026}
}
```

## Contact
For any questions, please contact: **axm24@mails.tsinghua.edu.cn**, **zhangchen01@tsinghua.edu.cn**.
