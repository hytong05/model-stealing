# Báo Cáo Đánh Giá Độ Tương Đồng Giữa Target và Surrogate Models

## Thông Tin Dữ Liệu Test

- **File**: `/home/hytong/Documents/model_extraction_malware/src/train_ember_2018_v2_features_label_minus1.parquet`
- **Số samples**: 10,000
- **Phân bố nhãn từ target model**: {0: 121, 1: 9879}

## Kết Quả Đánh Giá

| Model | Accuracy | Agreement | AUC | Precision | Recall | F1 |
|-------|----------|-----------|-----|-----------|--------|----|
| max_queries_10000 | 0.9861 | 0.9861 | 0.9338 | 0.9949 | 0.9910 | 0.9930 |
| max_queries_5000 | 0.9937 | 0.9937 | 0.9911 | 0.9949 | 0.9988 | 0.9968 |
| max_queries_2000 | 0.9901 | 0.9901 | 0.7744 | 0.9907 | 0.9994 | 0.9950 |

## Chi Tiết Từng Model

### Max Queries 10000

- **Accuracy**: 0.9861 (98.61%)
- **Agreement**: 0.9861 (98.61%)
- **AUC**: 0.9338
- **Precision**: 0.9949
- **Recall**: 0.9910
- **F1-score**: 0.9930

**Confusion Matrix:**

| | Predicted 0 | Predicted 1 |
|------|------------|-------------|
| Actual 0 | 71 | 50 |
| Actual 1 | 89 | 9790 |

- **Target distribution**: {0: 121, 1: 9879}
- **Surrogate distribution**: {0: 160, 1: 9840}

### Max Queries 5000

- **Accuracy**: 0.9937 (99.37%)
- **Agreement**: 0.9937 (99.37%)
- **AUC**: 0.9911
- **Precision**: 0.9949
- **Recall**: 0.9988
- **F1-score**: 0.9968

**Confusion Matrix:**

| | Predicted 0 | Predicted 1 |
|------|------------|-------------|
| Actual 0 | 70 | 51 |
| Actual 1 | 12 | 9867 |

- **Target distribution**: {0: 121, 1: 9879}
- **Surrogate distribution**: {0: 82, 1: 9918}

### Max Queries 2000

- **Accuracy**: 0.9901 (99.01%)
- **Agreement**: 0.9901 (99.01%)
- **AUC**: 0.7744
- **Precision**: 0.9907
- **Recall**: 0.9994
- **F1-score**: 0.9950

**Confusion Matrix:**

| | Predicted 0 | Predicted 1 |
|------|------------|-------------|
| Actual 0 | 28 | 93 |
| Actual 1 | 6 | 9873 |

- **Target distribution**: {0: 121, 1: 9879}
- **Surrogate distribution**: {0: 34, 1: 9966}

