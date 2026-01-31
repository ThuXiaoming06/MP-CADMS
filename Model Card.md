# Model Card for MP-CADMS
MP-CADMS is a multimodal deep-learning system that integrates immunofixation electrophoresis (IFE) images and structured clinical information (SCI) to (i) diagnose M-protein status (presence detection, isotype classification, severity grading) and (ii) predictively monitor longitudinal outcomes (recurrence–recovery and severity progression) from patient follow-up sequences.
## Model Details
### Model Description
MP-CADMS comprises (1) a multimodal feature extraction and fusion module, (2) a multimodal diagnosis module for three diagnostic tasks, and (3) a longitudinal monitoring module that models irregularly sampled follow-up visits to forecast future status and severity trajectories. It was trained and internally validated on a large multimodal dataset from a top-tier hospital in China (PUMCH), and externally validated on multi-institution datasets, demonstrating strong performance across diagnostic and monitoring endpoints.
- **Developed by:** Xuming An, Pengchang Li, Jianhua Han, Wei Ji, Qian Di, Yonggang Liang, Yanjun Hou, Yanhua Sha, Wei Su, Ling Qiu, Chen Zhang 
- **Shared by:** Xuming An
- **Model type:** Multimodal deep learning (image + tabular) for classification and longitudinal sequence forecasting in clinical laboratory medicine
- **Language(s) (NLP):** N/A
- **Finetuned from model:** Not a single base model; uses multiple backbones (e.g., ResNet18; transformer blocks) and a self-supervised pretext model for IFE feature learning.
### Model Sources
- **Repository:** https://github.com/ThuXiaoming06/MP-CADMS ; https://gitee.com/xuminan/MP-CADMS
- **Paper:** A Computer-Aided System for Enhanced M-protein Diagnosis and Monitoring via Multimodal Learning
- **Demo:** Interactive demos are provided via Hugging Face Spaces for diagnosis and monitoring modules: https://huggingface.co/spaces/THUxiaoming/MP-CADMS-Diagnosis ; https://huggingface.co/spaces/THUxiaoming/MP-CADMS-Monitoring
### Direct Use
- Clinical decision support in laboratory medicine workflows: given preprocessed IFE images and SCI inputs from a patient visit, output probabilities for (1) M-protein presence, (2) isotype among eight categories, and (3) severity among three ordered levels.
- Predictive monitoring for patients with multiple follow-up visits: given a historical sequence of visits, forecast future recurrence/recovery and severity progression at a target time.
Intended direct users include clinical laboratory physicians/technologists, hematology care teams, and clinical informatics teams evaluating CAD tools. Affected stakeholders include patients undergoing M-protein-related testing and follow-up planning.
### Downstream Use
- Integration into hospital LIS/EHR pipelines as a triage/second-reader system for IFE + lab panels.
- Fine-tuning/adaptation to additional institutions, scanner settings, or extended label sets (e.g., institution-specific severity definitions), subject to local validation and governance.
### Out-of-Scope Use
- Not a standalone diagnostic device: outputs should not replace clinician judgment or established diagnostic criteria.
- Not designed for diseases/biomarkers beyond M-protein disorders, or for non-IFE modalities (e.g., mass spectrometry MRD) without redevelopment.
- Monitoring module expects sufficient longitudinal context (e.g., sequences with fewer than five encounters were excluded in the study setup), so performance may degrade with sparse histories.
- Not guaranteed to generalize to drastically different preprocessing, labeling practices, or patient populations without validation.
## Bias, Risks, and Limitations
- **Population and site bias:** training/internal validation data are from a single top-tier hospital in China, which may bias representations toward its case mix and testing practices, even though external validation includes multiple institutions.
- **Label subjectivity:** severity grading is clinically subtle and showed lower performance than presence detection/isotype classification, reflecting intrinsic ambiguity and borderline cases.
- **Monitoring asymmetry:** recurrence–recovery monitoring achieved high sensitivity but comparatively lower specificity in internal validation, implying a risk of false positives (unnecessary alerts/follow-ups) if deployed without calibration.
- **Data quality shift:** publicly sourced SC IFE images can be lower quality and heterogeneous (compression artifacts, variable color balance), which can affect model robustness; the study treated SC as a supplementary robustness test.
- **Preprocessing dependence:** the model relies on a specific IFE lane segmentation/alignment and resizing pipeline (including dynamic time warping alignment and lane-wise resizing); mismatched preprocessing may reduce accuracy.
### Recommendations
- Validate and calibrate the model prospectively at each deployment site (including threshold tuning for monitoring alerts), and continuously audit subgroup performance (age/sex, platform type, institution).
- Keep a human-in-the-loop workflow, especially for borderline severity and for positive monitoring predictions with low specificity.
- Enforce consistent preprocessing and QC checks; log preprocessing failures and “out-of-distribution” indicators for review.
## How to Get Started with the Model
Use the code below to get started with the model.

Repositories (as stated in the manuscript):
- GitHub: https://github.com/ThuXiaoming06/MP-CADMS/tree/main
- Gitee:  https://gitee.com/xuminan/MP-CADMS

Interactive demos (Hugging Face Spaces):
- Diagnosis:  https://huggingface.co/spaces/THUxiaoming/MP-CADMS-Diagnosis
- Monitoring: https://huggingface.co/spaces/THUxiaoming/MP-CADMS-Monitoring
## Training Details

### Training Data
- **Primary (train + internal validation):** a multimodal dataset from Peking Union Medical College Hospital (PUMCH), including 16,883 IFE images and 7,869 SCI records. 
- **External validation:** IFE-SCI pairs collected from four independent hospitals (2,018 pairs). 
- **Additional robustness evaluation:** SC dataset with 271 publicly sourced IFE images (no severity labels; no paired SCI).
### Training Procedure
#### Preprocessing
- **IFE images:** removed non-informative regions; aligned lanes via dynamic time warping; split into six lanes; resized lanes (SP lane to 23×128; other lanes to 21×128); augmentation via random lane flipping to address class imbalance; cross-platform harmonization by flipping Sebia images to match Helena orientation. 
- **SCI tabular data:** missing values completed using Remasker (a masked autoencoder for tabular imputation), trained on the training split before imputation.
#### Training Hyperparameters
All modules in MP-CADMS were uniformly trained with a learning rate of $2.5\times{10}^{-3}$, a maximum epoch of 500, and an Adam optimizer. The diagnosis module used a batch size of 128, while the monitoring module used a batch size of 64. The early-stopping patience of 20 and dropout regularization of 0.2 were also introduced.
## Evaluation
### Testing Data, Factors & Metrics
#### Testing Data
- Internal validation dataset (IVD) from PUMCH (held-out). 
- External validation datasets from multiple hospitals (FIDGL and SC mentioned; SC is IFE-only).
#### Factors
- **Modality:** multimodal vs single-modality ablations (IFE-only, SCI-only).
- **Platform robustness:** Helena vs Sebia IFE systems (no significant performance differences reported for key tasks).
- **Task type:** 3 diagnosis tasks and 2 monitoring tasks.
#### Metrics
Primary metrics include AUROC and AUPRC, with additional metrics such as sensitivity, specificity, top-1 accuracy, F1-score, and precision (reported across tasks).
### Results
Key reported performance (internal / external where specified in the abstract):
- Presence detection AUROC: 0.997 / 0.979
- Isotype classification AUROC: 0.998 / 0.992
- Severity grading AUROC: 0.861 / 0.869
- Recurrence–recovery monitoring AUROC: 0.908 / 0.874
- Severity-progression monitoring AUROC: 0.875 / 0.861
#### Summary
Across all five tasks, MP-CADMS demonstrated strong internal and multi-institution external validation performance and was benchmarked as matching or exceeding clinical experts for the evaluated tasks (as stated by the authors).
## Technical Specifications
### Model Architecture and Objective
- **IFE feature extraction:** self-supervised pretext model (a VAE-style objective with reconstruction loss + KL divergence) pretrains an encoder; the pretrained encoder/bottleneck are used with MLP layers to construct the IFE feature extractor. 
- **SCI feature extraction:** constructs a 36-node graph (SCI items as nodes; edges from clinical domain knowledge) and builds a feature map combining normalized values and pairwise similarity (using a frozen Remasker encoder for similarity); then applies a ResNet18 backbone to extract SCI features. 
- **Feature fusion:** three-layer stacked transformer block with eight attention heads to model cross-modality interactions and output fused features. 
- **Diagnosis heads:**
(1) Negative/positive classifier: 3-layer MLP + softmax; trained with BCE loss. 
(2) Multitask classifier: isotype branch (8 one-layer MLPs + weighting) and severity branch (3 one-layer MLPs + weighting), with cross-branch interaction via transformer blocks; severity uses ordinal regression with a tailored output conversion; weighted multitask loss emphasizes severity.
- **Monitoring module:** time-aware LSTM (T-LSTM) to encode irregular visit intervals, followed by a NeuralODE solver for forecasting; produces recurrence–recovery and severity-progression predictions.
### Compute Infrastructure
#### Hardware
All model training, validation and data analysis were performed on a workstation platform with three NVIDIA RTX 3060Ti GPUs, 256G RAM, and an Intel Core i7-8700 CPU.
#### Software
Data analysis was conducted in Python 3.10 using the libraries NumPy 1.23.1, pandas 1.4.4, matplotlib 3.5.2, SciPy 1.8.1, and scikit-learn 1.1.1. And model development was based on PyTorch framework. 
