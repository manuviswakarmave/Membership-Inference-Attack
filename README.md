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
