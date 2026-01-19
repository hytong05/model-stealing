# Metrics Explanation Guide

## Overview

This document provides detailed explanations of all metrics used in the MIR framework for evaluating model extraction attacks. Each metric measures a different aspect of the attack process and surrogate model quality.

---

## 1. Configuration

**Definition:** Name of the extraction attack configuration, indicating the maximum number of queries used.

**Examples:**
- `max_queries_10000_H5`: Configuration with maximum 10,000 queries, using H5 (Keras) model
- `max_queries_5000_H5`: Configuration with maximum 5,000 queries
- `max_queries_2000_H5`: Configuration with maximum 2,000 queries

**Meaning:** Helps distinguish different experiments and compare effectiveness by query count.

---

## 2. Queries

**Definition:** Actual number of queries sent to the target model (oracle) during active learning.

**Formula:** `Queries = Total number of samples queried from oracle (excluding seed and validation set)`

**Examples:**
- `max_queries_10000_H5`: 9,000 queries (expected 10,000 but actual 9,000)
- `max_queries_5000_H5`: 4,000 queries
- `max_queries_2000_H5`: 1,000 queries

**Meaning:**
- **Important in model extraction:** More queries mean attacker has more information about target model
- **Cost:** Each query is one interaction with target model (may be expensive or detectable)
- **Efficiency:** Need to balance between query count and extraction quality

**Note:** Actual queries may be lower than expected due to pool data exhaustion or class balancing.

---

## 3. Labels

**Definition:** Total number of labels used to train surrogate model, including:
- **Seed set:** Initial data (typically 2,000 samples)
- **Validation set:** Validation data (typically 1,000 samples)
- **Queries:** Labels from oracle through active learning rounds

**Formula:** `Labels = Seed size + Val size + Actual queries`

**Examples:**
- `max_queries_10000_H5`: 12,000 labels = 2,000 (seed) + 1,000 (val) + 9,000 (queries)
- `max_queries_5000_H5`: 7,000 labels = 2,000 + 1,000 + 4,000
- `max_queries_2000_H5`: 4,000 labels = 2,000 + 1,000 + 1,000

**Meaning:**
- **Total training data:** More labels may allow model to learn better
- **Comparison with Queries:** Labels = Queries + Seed + Val, so always larger than Queries

---

## 4. Accuracy

**Definition:** Accuracy of surrogate model when compared with **ground truth labels** (actual data labels).

**Formula:** 
```
Accuracy = (Correct predictions) / (Total samples)
         = (TP + TN) / (TP + TN + FP + FN)
```

Where:
- **TP (True Positive):** Correctly predicted positive (malware)
- **TN (True Negative):** Correctly predicted negative (benign)
- **FP (False Positive):** Incorrectly predicted positive (false alarm)
- **FN (False Negative):** Incorrectly predicted negative (missed malware)

**Examples:**
- `max_queries_10000_H5`: 0.4338 = 43.38% accuracy
- `max_queries_5000_H5`: 0.4065 = 40.65% accuracy
- `max_queries_2000_H5`: 0.3950 = 39.50% accuracy

**Meaning:**
- **Evaluates actual performance:** Accuracy measures surrogate model's ability to classify correctly with ground truth
- **Low in this case:** Values ~40% indicate:
  - Surrogate model didn't learn well from oracle
  - **OR** Oracle (target model) is not accurate with ground truth
  - **OR** Severe class imbalance (need to check Balanced Accuracy)

**Important Note:**
- Accuracy is calculated with **ground truth**, not oracle labels
- If oracle is inaccurate, accuracy will be low even with high agreement
- Low accuracy (~40%) but high agreement (~90%) indicates: **Oracle is not accurate with ground truth**

---

## 5. Balanced Accuracy

**Definition:** Balanced accuracy, calculated as average of per-class accuracies. Important when there is **class imbalance** (one class dominates).

**Formula:**
```
Balanced Accuracy = (Sensitivity + Specificity) / 2
                  = (TP/(TP+FN) + TN/(TN+FP)) / 2
```

Where:
- **Sensitivity (Recall):** Ratio of correctly predicted positive among all actual positives
- **Specificity:** Ratio of correctly predicted negative among all actual negatives

**Examples:**
- `max_queries_10000_H5`: 0.4320 = 43.20% balanced accuracy
- `max_queries_5000_H5`: 0.4049 = 40.49% balanced accuracy
- `max_queries_2000_H5`: 0.3935 = 39.35% balanced accuracy

**Meaning:**
- **Important with class imbalance:** If one class dominates 90% of data, accuracy will be high but doesn't reflect true classification ability
- **Comparison with Accuracy:** 
  - If Balanced Accuracy ≈ Accuracy: No severe class imbalance
  - If Balanced Accuracy << Accuracy: Class imbalance exists, model biased toward majority class
- **In this case:** Balanced Accuracy ≈ Accuracy (~40%), indicating no severe class imbalance, but model still doesn't learn well

**When Important:**
- Unbalanced data (e.g., 90% benign, 10% malware)
- Need fair evaluation for both classes
- Regular accuracy can be misleading

---

## 6. F1 Score

**Definition:** Harmonic mean of Precision and Recall, balancing accuracy and coverage.

**Formula:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

Where:
- **Precision:** Ratio of correct positive predictions among all positive predictions
- **Recall:** Ratio of correct positive predictions among all actual positives

**Examples:**
- `max_queries_10000_H5`: 0.1391 = 13.91% F1 score
- `max_queries_5000_H5`: 0.1194 = 11.94% F1 score
- `max_queries_2000_H5`: 0.1276 = 12.76% F1 score

**Meaning:**
- **Comprehensive evaluation:** F1 balances Precision and Recall
- **Low in this case:** F1 ~12-14% is very low, indicating:
  - Low Precision: Many false positives
  - Low Recall: Missing many malware samples
  - Model doesn't learn well for accurate classification

**When Important:**
- Need balance between false positives and false negatives
- In malware detection: Both errors are important (false alarms and missed malware)
- With class imbalance, F1 often better than accuracy

**Relationship:**
- Low F1 → Low Precision or Recall (or both)
- With Precision = 0.2855 and Recall = 0.0920 (max_queries_10000_H5):
  - F1 = 2 × (0.2855 × 0.0920) / (0.2855 + 0.0920) ≈ 0.1391 ✓

---

## 7. Agreement

**Definition:** Agreement rate between surrogate model predictions and target model (oracle) predictions. **This is the most important metric in model extraction attacks.**

**Formula:**
```
Agreement = (Matching predictions) / (Total samples)
          = (Surrogate predictions == Oracle predictions).mean()
```

**Examples:**
- `max_queries_10000_H5`: 0.9133 = 91.33% agreement
- `max_queries_5000_H5`: 0.9275 = 92.75% agreement
- `max_queries_2000_H5`: 0.9028 = 90.28% agreement

**Meaning:**
- **Goal of model extraction:** High agreement = Surrogate model learned well to mimic target model
- **Not accuracy:** Agreement compares with oracle, not ground truth
- **In this case:** Agreement ~90% is high, but Accuracy ~40% is low → **Oracle is not accurate with ground truth**

**Comparison with Accuracy:**
- **High Agreement + Low Accuracy:** 
  - Surrogate learned well from oracle ✓
  - But oracle is not accurate with ground truth ✗
  - → Model extraction successful, but target model not good

- **High Agreement + High Accuracy:**
  - Surrogate learned well from oracle ✓
  - Oracle also accurate with ground truth ✓
  - → Model extraction successful and target model good

**When Important:**
- **Evaluating attack success:** Agreement is main metric for measuring extraction
- **No ground truth needed:** Agreement only needs oracle predictions
- **In practice:** In model extraction, attacker has no ground truth, only oracle responses

---

## 8. Threshold (Optimal Threshold)

**Definition:** Optimal threshold for converting probabilities to binary predictions (0 or 1).

**Formula:** Threshold found by optimizing F1-score on validation set:
```
For each threshold in [0.1, 0.2, ..., 0.9]:
    predictions = (probabilities >= threshold).astype(int)
    f1 = calculate_f1_score(ground_truth, predictions)
    
optimal_threshold = threshold with highest F1
```

**Examples:**
- All configurations: 0.100 = 10% threshold

**Meaning:**
- **Low (0.1):** Model needs only 10% probability to predict positive (malware)
- **Indicates:**
  - Model tends to predict low probabilities
  - May be due to class imbalance (many negative, few positive)
  - Or model not confident with predictions

**Comparison:**
- **Threshold = 0.5 (default):** Balanced, predict positive when probability ≥ 50%
- **Threshold = 0.1 (in this case):** Very low, predict positive when probability ≥ 10%
  - → Model needs very little confidence to predict malware
  - → May be due to model not learning well or having bias

**When Important:**
- **Optimize performance:** Threshold affects Precision, Recall, F1
- **Class imbalance:** Low threshold often better with many negatives
- **Cost-sensitive:** If false positive more expensive than false negative, increase threshold

---

## 9. AUC (Area Under ROC Curve)

**Definition:** Area under ROC (Receiver Operating Characteristic) curve, measuring model's ability to distinguish between 2 classes.

**Formula:**
```
AUC = ∫ ROC_curve d(False Positive Rate)
```

ROC curve plots:
- **X-axis:** False Positive Rate (FPR) = FP / (FP + TN)
- **Y-axis:** True Positive Rate (TPR/Recall) = TP / (TP + FN)

**Values:**
- **AUC = 1.0:** Perfect classifier (perfect distinction)
- **AUC = 0.5:** Random classifier (no better than random guessing)
- **AUC < 0.5:** Worse than random (may reverse predictions)

**Examples:**
- `max_queries_10000_H5`: 0.4258 = 42.58% AUC
- `max_queries_5000_H5`: 0.2613 = 26.13% AUC
- `max_queries_2000_H5`: 0.2634 = 26.34% AUC

**Meaning:**
- **Very low in this case:** AUC ~26-43% < 50% (random)
- **Indicates:**
  - Model cannot distinguish between malware and benign
  - May be due to:
    - Model not learning well from oracle
    - Oracle not accurate with ground truth
    - Insufficient or unrepresentative data

**Comparison with other metrics:**
- **Low AUC + Low Accuracy:** Model not learning well
- **Low AUC + High Agreement:** 
  - Surrogate mimics oracle well
  - But oracle doesn't distinguish well between classes
  - → Oracle has issues, not surrogate

**When Important:**
- **Evaluate discrimination ability:** AUC independent of threshold
- **Class imbalance:** AUC better than accuracy with imbalance
- **Compare models:** AUC allows model comparison independent of threshold

---

## Summary and Analysis

### Observed Patterns:

1. **High Agreement (~90%):** Surrogate model learned well to mimic oracle
2. **Low Accuracy (~40%):** Oracle not accurate with ground truth
3. **Very Low AUC (~26-43%):** Oracle doesn't distinguish well between malware and benign
4. **Low F1 (~12-14%):** Model doesn't classify well with ground truth
5. **Low Threshold (0.1):** Model needs little confidence to predict positive

### Conclusion:

**Model extraction attack successful** (Agreement ~90%), but:
- **Target model (oracle) not accurate** with ground truth
- **Surrogate learned well** to mimic oracle, but because oracle is not good, surrogate is also not good
- **Need to check:** Oracle accuracy vs ground truth to confirm

### Recommendations:

1. **Check Oracle:** Calculate oracle accuracy vs ground truth to confirm oracle has issues
2. **Improve Oracle:** If oracle inaccurate, need to retrain target model
3. **Evaluate Extraction:** High agreement shows extraction successful, but need good oracle for good surrogate
4. **Increase Queries:** May try increasing queries to see if improvement (but based on results, doesn't seem to be query count issue)

---

## References

- **Accuracy:** Overall correct prediction rate
- **Balanced Accuracy:** Balanced accuracy per class
- **Precision:** Accuracy of positive predictions
- **Recall:** Coverage of positive predictions
- **F1 Score:** Balance between Precision and Recall
- **AUC:** Ability to distinguish between 2 classes
- **Agreement:** Agreement between surrogate and oracle (main metric in model extraction)
