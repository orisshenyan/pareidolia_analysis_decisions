# Pareidolia task: analysis & thresholding

Analysis pipeline for a pareidolia experiment investigating face perception in dynamic Gaussian noise. The following analysis addresses two key design decisions:

1. **Should face opacity in the main experiment be set using a group-level or individual-level perceptual threshold?**
2. **How should opacity be adaptively adjusted across blocks during the main experiment?**

The repository contains an **R Markdown** analysis document, a **Python** and **R** script for Bayesian psychometric threshold estimation, and a **Python** Jupyter notebook for visualisation of this Bayesian psychometric threshold.

---

## Table of contents

- [Background](#background)
- [Project structure](#project-structure)
- [Task overview](#task-overview)
  - [Training experiment](#training-experiment)
  - [Main experiment](#main-experiment)
- [Analysis overview](#analysis-overview)
  - [Question 1 — group vs individual thresholds](#question-1--group-vs-individual-thresholds)
  - [Question 2 — block-wise opacity adjustment](#question-2--block-wise-opacity-adjustment)
  - [Bayesian threshold estimation](#bayesian-threshold-estimation)
- [Data](#data)
  - [Data structure](#data-structure)
  - [Training data columns](#training-data-columns)
  - [Main data columns](#main-data-columns)
- [Requirements](#requirements)
  - [R dependencies](#r-dependencies)
  - [Python dependencies](#python-dependencies)

---

## Background

 Participants view dynamic Gaussian noise and must either **discriminate** (training) or **detect** (main experiment) embedded face stimuli at varying opacity levels. We are considering calibrating stimulus difficulty so that the task is neither too easy nor too hard for each individual, while preserving the key dependent variable of interest: **high-confidence false alarms (HCFA)** (where participants report seeing a face with high confidence when none is present).

---

## Project structure
```
├── Pipeline.Rmd # R Markdown: pilot analysis & design decisions
├── Pipeline.html # Knitted R markdown
├── Bayesian_thresholds_estimation.py # Python: per-participant Bayesian threshold estimation using PyMC
├── Bayesian_thresholds_estimation_BRMS.r # R: per-participant Bayesian threshold estimation using BRMS 
├── Bayesian_Comparison.Rmd # R:  visualiation and comparison of Bayesian thresholds from PyMC and BRMS
├── Training.png # Schematic of the training experiment
├── Main.png # Schematic of the main experiment
└── Data/
├──── training_data.csv # Aggregated training data (all participants)
├──── main_data.csv # Aggregated main experiment data (all participants)
├──── threshold_estimates.csv # Output: per-participant threshold estimates generated using PyMC
├──── threshold_estimates_BRMS.csv # Output: per-participant threshold estimates generated using BRMS
└──── individual_data/ # Per-participant training and main experiment block CSVs (for blocks 0-6) and opacity levels
├────── sub001/
│ ├────── sub001_trainingblock001.csv
│ ├────── sub001_mainblock001.csv
│ ├────── sub001_final_opac.csv
│ └── ...
├────── sub002/
└── ...
```

---

## Task overview

### Training experiment

- **Task type:** Two-alternative forced choice (2AFC)
- **Trials:** 120 trials across 7 blocks
- **Stimulus:** Faces embedded in 4-second dynamic Gaussian noise clips
- **Opacity range:** 10%–70%
- **Responses:** Direction of the face (left/right) + confidence (1–4)
- **Purpose:** Estimate each participant's psychometric function and perceptual threshold (opacity at 65% accuracy)

### Main experiment

- **Task type:** Detection (present/absent)
- **Duration:** ~5.5 minutes of continuous dynamic Gaussian noise
- **Pseudo-trial structure:** 120 trials × 2.5 seconds (not visible to participants)
- **Face-present trials:** 30 out of 120
- **Responses:** Face detection + confidence (1–4)
- **Adaptive opacity:** Starting opacity is set at the group threshold for 65% accuracy. Block-wise adjustments are made:
  - If **< 10%** of real faces detected → opacity **increases by 1%**
  - If **> 90%** of real faces detected → opacity **decreases by 1%**

---

## Analysis overview

### Question 1 — Group vs individual thresholds

The R Markdown document investigates whether a single group-level opacity threshold adequately calibrates task difficulty for all participants, or whether individual thresholds are needed.

**Analyses include:**

- **Group psychometric curve** — Multi-level logistic regression (`glmer`) with bootstrapped 95% CI for the 65% accuracy threshold
- **Individual psychometric curves** — Per-participant GLM fits with individual threshold extraction
- **Threshold distribution** — Histogram of individual thresholds with classification into three groups:
  - *Below CI* — individual threshold below the group threshold 95% CI (task likely too easy)
  - *Within CI* — appropriately thresholded
  - *Above CI* — individual threshold above the group threshold 95% CI (task likely too hard)
- **Opacity drift across blocks** — How starting opacity changes over the main experiment for each threshold group
- **HCFA by threshold group** — Whether high-confidence false alarms (the key DV) differ across threshold groups

**Key finding:** Participants within the group threshold CI show the most HCFA, suggesting that individual thresholding may be beneficial. Those who find the task too hard tend not to see anything, while those who find it too easy can distinguish faces from noise too readily.

### Question 2 — Block-wise opacity adjustment

The analysis examines whether the current detection-based adjustment rule is sufficient, or whether adjustment should be based on **signal detection sensitivity (d')**.

**Analyses include:**

- **d' distribution** — Experiment-wide sensitivity across participants
- **d' by block and threshold group** — Block-wise sensitivity trajectories
- **HCFA by d' bin** — Relationship between perceptual sensitivity and high-confidence false alarms

**Key finding:** HCFA generally decreases as d' increases, with a suggested adjustment target of d' ≈ 0.5–0.75.

### Bayesian threshold estimation

This is a follow-up to question 1.

Both scripts estimate per-participant opacity thresholds (the opacity at which
accuracy = 65%) using two methods: a Bayesian model with an informative prior
and a classical frequentist logistic regression (GLM). 

---

#### Common model structure

| Component | Detail |
|-----------|--------|
| **Psychometric function** | `p(correct) = sigmoid(β × (Opacity − threshold) + logit(0.65))` |
| **Threshold parameter** | Directly represents the 65% accuracy point |
| **Threshold prior** | Normal(0.264, 0.085) — from pilot data |
| **Slope (β) prior** | Normal(7.6, 3), lower-bounded at 0 — from pilot data |
| **Likelihood** | Bernoulli (trial-level) |
| **MCMC settings** | 4 chains × 4000 iterations (2000 warmup + 2000 sampling), target acceptance = 0.99 |
| **Point estimate** | Posterior median |
| **Uncertainty interval** | 95% equal-tailed credible interval (2.5th–97.5th percentiles) |
| **Frequentist GLM** | Standard logistic regression freely estimating both intercept and slope; threshold derived algebraically |

---

#### Python script (`Bayesian_threshold_estimation.py`)

Uses **PyMC** (Abril-Pla et al., 2023) for Bayesian estimation. The slope is
estimated via a `TruncatedNormal` prior (lower bound = 0), and logit(0.65) is
embedded directly in the sigmoid so that the threshold parameter represents the
65% accuracy point without post-hoc transformation. The frequentist GLM uses
**statsmodels**.

Generates `threshold_estimates.csv`.

#### R script (`Bayesian_threshold_estimation_BRMS.R`)

Uses **brms** (Bürkner, 2017) for Bayesian estimation with an identical
non-linear formula. The slope (β) is estimated with a Normal(7.6, 3) prior
constrained to be positive (`lb = 0`), and logit(0.65) is baked into the model
formula so that the threshold directly represents the 65% accuracy point. The
frequentist GLM uses base R's `glm()`.

Generates `threshold_estimates_BRMS.csv`.

---

#### Output columns (both scripts)

| Column | Description |
|--------|-------------|
| `subject` / `subnumber` | Participant identifier |
| `threshold_freq` | Frequentist GLM threshold (65% accuracy point) |
| `threshold_bayes` | Posterior median of the Bayesian threshold |
| `bayes_lower` | 2.5th percentile of the posterior (lower bound of 95% CI) |
| `bayes_upper` | 97.5th percentile of the posterior (upper bound of 95% CI) |
| `n_trials` | Total number of training trials for the participant |


### Threshold Visualisation

The R Markdown document (`Bayesian_Comparison.Rmd`) provides a visual comparison of the threshold estimates produced by both the Python (`Bayesian_threshold_estimation.py`) and R (`brms`) Bayesian estimation pipelines. It imports and merges threshold estimates from both frameworks alongside the final adaptive opacity reached per participant in the main experiment, producing three sets of comparison plots:

- **Plot 1 — PyMC vs brms:** A direct comparison of the Bayesian threshold estimates generated by PyMC (Python) and brms (R) for each participant.

- **Plot2 2 — GLM vs Bayesian (R and Python):** Two side-by-side plots showing, for each implementation, how much the informative Bayesian prior pulls each participant's threshold estimate away from the pfrequentist GLM.

- **Plot 3 — Bayesian vs final Opacity (R and Python):** Two side-by-side plots comparing each participant's Bayesian threshold estimate (brms on the left, PyMC on the right) to the final opacity reached by the adaptive staircase procedure in the main experiment. 

This document requires:
- `Data/threshold_estimates.csv` (output from `Bayesian_threshold_estimation.py`)
- `Data/threshold_estimates_BRMS.csv` (output from the brms estimation pipeline)
- `Data/individual_data/*/final_opac.csv` (per-participant final opacity from the main experiment)

It should be run **after** both `Bayesian_threshold_estimation.py` and the brms estimation script.
---

## Data

### Data structure

The pilot dataset consists of **41 participants**:
- First half started at **26% opacity** (based on earlier pilot data)
- Second half started at **28% opacity** (re-adjusted group threshold)

### Training Data Columns

| Column | Description |
|--------|-------------|
| `Face_response` | Whether the participant responded (should always be 1) |
| `Face_confidence` | Confidence in direction judgment (1–4) |
| `Face_Onset_Time` | Time in the block when the face was presented |
| `Face_Position` | Position in degrees of visual angle (negative = left, positive = right) |
| `Noise_Number` | Label of the noise .png shown |
| `Face_Size` | Size of the face in visual angle |
| `Opacity` | Opacity of the face (0.1–0.7) |
| `Direction_Report` | Reported direction: left (1) or right (2) |
| `subnumber` | Subject identifier |
| `block_number` | Block number |

### Main Data Columns

| Column | Description |
|--------|-------------|
| `Realface_response` | Face-present trial: 0 (miss) or 1 (hit); Face-absent trial: 80 |
| `Realface_confidence` | Confidence for detected faces (1–4); NA if missed |
| `Realface_Image_Onset_Time` | Time of face onset (0 if face-absent) |
| `RealFace_Image_Response_Time` | Time of face detection response (0 if face-absent) |
| `Hallucination_response` | Face-present trial: 90; Face-absent: 0 (correct rejection) or 1 (false alarm) |
| `Hallucination_confidence` | Confidence for false alarms (1–4) |
| `Noise_Onset_Time` | Timestamp of face-absent trial onset |
| `Hallucination_Response_Time` | Time of false alarm response |
| `ConfidenceScreen_Onset` | Time the confidence screen appeared after a hit or false alarm |
| `Face_Position` | Position in degrees of visual angle |
| `Noise_Number` | Noise image label |
| `Face_Size` | Face size in visual angle |
| `subnumber` | Subject identifier |
| `block_number` | Block number |
| `Opacity` | Opacity of the face stimulus |

---

## Requirements

### R dependencies

install.packages(c(
  "tidyverse",
  "lme4",
  "boot",
  "gridExtra",
  "patchwork",
  "grid",
  "ggpubr"
  "brms"
  "purrr"
))

### Python dependencies

pip install pandas numpy pymc arviz statsmodels
