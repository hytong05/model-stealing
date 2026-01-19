# Surrogate Model Comparison Report

## Summary

Model extraction experiments were run with 3 different query budget configurations to evaluate the impact of query count on extraction quality.

## Comparison Table

| Configuration | Queries | Labels | Accuracy | Balanced Acc | F1 | Agreement | Threshold | AUC |
|---------------|---------|--------|----------|--------------|----|-----------|-----------|-----|
| max_queries_10000_H5 | 9,000 | 12,000 | 0.5022 | 0.4998 | 0.0000 | 0.9998 | 0.100 | 0.6630 |
| max_queries_5000_H5 | 4,000 | 7,000 | 0.5025 | 0.5000 | 0.0010 | 0.9995 | 0.750 | 0.5746 |
| max_queries_2000_H5 | 1,000 | 4,000 | 0.5025 | 0.5000 | 0.0010 | 0.9995 | 0.150 | 0.6365 |

## Configuration Details

### Max Queries 10000 H5

**Description:** Total 10,000 queries (2000 queries/round × 5 rounds)

- Query batch: 2,000
- Number of rounds: 5
- Expected queries: 10,000
- Actual queries: 9,000
- Total labels used (including seed+val): 12,000

**Final Metrics:**

- Accuracy: 0.5022 (50.22%)
- Balanced Accuracy: 0.4998 (49.98%) [important with class imbalance]
- F1-score: 0.0000
- Optimal Threshold: 0.1000
- Agreement: 0.9998 (99.98%)
- AUC: 0.6630
- Precision: 0.0000
- Recall: 0.0000

**Files:**

- Metrics CSV: `output/max_queries_10000_H5/extraction_metrics.csv`
- Surrogate model: `output/max_queries_10000_H5/surrogate_model.h5`
- Output directory: `output/max_queries_10000_H5`

### Max Queries 5000 H5

**Description:** Total 5,000 queries (1250 queries/round × 4 rounds)

- Query batch: 1,250
- Number of rounds: 4
- Expected queries: 5,000
- Actual queries: 4,000
- Total labels used (including seed+val): 7,000

**Final Metrics:**

- Accuracy: 0.5025 (50.25%)
- Balanced Accuracy: 0.5000 (50.00%) [important with class imbalance]
- F1-score: 0.0010
- Optimal Threshold: 0.7500
- Agreement: 0.9995 (99.95%)
- AUC: 0.5746
- Precision: 0.5000
- Recall: 0.0005

**Files:**

- Metrics CSV: `output/max_queries_5000_H5/extraction_metrics.csv`
- Surrogate model: `output/max_queries_5000_H5/surrogate_model.h5`
- Output directory: `output/max_queries_5000_H5`

### Max Queries 2000 H5

**Description:** Total 2,000 queries (2000 queries/round × 1 round)

- Query batch: 2,000
- Number of rounds: 1
- Expected queries: 2,000
- Actual queries: 1,000
- Total labels used (including seed+val): 4,000

**Final Metrics:**

- Accuracy: 0.5025 (50.25%)
- Balanced Accuracy: 0.5000 (50.00%) [important with class imbalance]
- F1-score: 0.0010
- Optimal Threshold: 0.1500
- Agreement: 0.9995 (99.95%)
- AUC: 0.6365
- Precision: 0.5000
- Recall: 0.0005

**Files:**

- Metrics CSV: `output/max_queries_2000_H5/extraction_metrics.csv`
- Surrogate model: `output/max_queries_2000_H5/surrogate_model.h5`
- Output directory: `output/max_queries_2000_H5`

## Analysis

### Key Observations

1. **High Agreement**: All configurations achieved >99% agreement, indicating successful model extraction
2. **Low Accuracy**: Accuracy around 50% suggests the target model may not be accurate with ground truth
3. **Low F1-Score**: Very low F1-scores indicate poor classification performance
4. **Query Efficiency**: Higher query counts don't necessarily improve accuracy in this case

### Recommendations

- Investigate target model accuracy with ground truth
- Consider different sampling strategies
- Evaluate with different target models
- Analyze feature quality and dataset distribution

## See Also

- [metrics_explanation.md](metrics_explanation.md) - Detailed metric explanations
- [README.md](../../README.md) - Framework overview
