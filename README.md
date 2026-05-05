# Membership Inference Attack (MIA) using LiRA-style Shadow Models

This project implements a **Membership Inference Attack (MIA)** based on the idea of the **Likelihood Ratio Attack (LiRA)**. The goal of the attack is to estimate whether a given sample was part of the target model's training set or not.

The implementation trains several shadow models on different splits of the available dataset, records confidence scores for samples that were included and excluded from training, fits per-sample statistics for both cases, and then compares the target model's confidence score against these distributions.

---

## How to Run

Run the attack using:

```bash
python run_attack.py
```

The script calls the attack pipeline and produces a `submission.csv` file containing membership scores for each sample.

---

## Main Idea

The core assumption behind membership inference is:

> A model usually behaves differently on data it has seen during training compared to data it has not seen.

For a given sample, the target model's confidence on the true class is compared against two distributions:

- `conf_in`: confidence values when the sample was used to train a shadow model
- `conf_out`: confidence values when the sample was not used to train a shadow model

The final LiRA score is based on the log-likelihood ratio:

```text
log p(score | in) - log p(score | out)
```

A higher score means the sample is more likely to be a member of the target model's training set.

---
## Important Output Files

| File | Description |
|---|---|
| `conf_in.csv` | Confidence scores from shadow models when samples were included in training |
| `conf_out.csv` | Confidence scores from shadow models when samples were excluded from training |
| `submission.csv` | Final normalized membership scores |

---

## Main Functions

| Function | Purpose |
|---|---|
| `lira_attack` | Runs the full attack pipeline on the target model |
| `create_conf_csv` | Trains shadow models and collects in/out confidence scores |
| `create_shadow_split` | Creates class-balanced in/out splits |
| `train_model` | Trains a shadow ResNet-18 model |
| `create_shadow_model` | Builds the modified ResNet-18 architecture |
| `get_confidence` | Extracts true-label confidence scores and converts them to log-odds |
| `save_conf` | Saves confidence values to CSV |
| `get_stats` | Computes mean and standard deviation per sample ID |
| `gaussian_logpdf` | Computes Gaussian log probability density |
| `compute_lira_scores` | Computes log-likelihood ratio scores |
| `compute_tpr_at_fpr` | Evaluates TPR at a fixed FPR when public labels are available |
| `normalize_scores` | Converts raw scores to the range `[0, 1]` |
| `validate_submission` | Checks whether the final submission is valid |
| `save_submission` | Saves the final `submission.csv` file |

---

