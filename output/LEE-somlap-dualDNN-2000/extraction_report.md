# Báo Cáo Extraction Attack: LEE-somlap-dualDNN-2000

## Thông Tin Cấu Hình

- **Mô tả:** Tổng 2,000 queries (2000 queries/round × 1 round)
- **Query batch:** 2,000
- **Số rounds:** 1
- **Queries dự kiến:** 2,000
- **Queries thực tế:** 2,000
- **Ghi chú queries:** on_target
- **Tổng labels sử dụng (bao gồm seed+val):** 4,000

## Kết Quả Metrics

- **Accuracy:** 0.7212 (72.12%)
- **Balanced Accuracy:** 0.7421 (74.21%) [quan trọng với class imbalance]
- **F1-score:** 0.6905
- **Optimal Threshold:** 0.1300
- **Agreement:** 0.9988 (99.88%)
- **AUC:** 0.8054
- **Precision:** 0.5929
- **Recall:** 0.8266

## Files

- **Metrics CSV:** `/home/hytong/Documents/model_extraction_malware/output/LEE-somlap-dualDNN-2000/extraction_metrics.csv`
- **Surrogate model:** `/home/hytong/Documents/model_extraction_malware/output/LEE-somlap-dualDNN-2000/surrogate_model.h5`
- **Output directory:** `/home/hytong/Documents/model_extraction_malware/output/LEE-somlap-dualDNN-2000`
