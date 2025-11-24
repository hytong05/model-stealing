# Báo Cáo Extraction Attack: LSE-somlap-dualDNN-10000

## Thông Tin Cấu Hình

- **Mô tả:** Tổng 10,000 queries (2000 queries/round × 5 rounds)
- **Query batch:** 2,000
- **Số rounds:** 5
- **Queries dự kiến:** 10,000
- **Queries thực tế:** 10,000
- **Ghi chú queries:** on_target
- **Tổng labels sử dụng (bao gồm seed+val):** 12,000

## Kết Quả Metrics

- **Accuracy:** 0.9012 (90.12%)
- **Balanced Accuracy:** 0.9166 (91.66%) [quan trọng với class imbalance]
- **F1-score:** 0.8818
- **Optimal Threshold:** 0.7800
- **Agreement:** 0.9990 (99.90%)
- **AUC:** 0.9544
- **Precision:** 0.8023
- **Recall:** 0.9787

## Files

- **Metrics CSV:** `/home/hytong/Documents/model_extraction_malware/output/LSE-somlap-dualDNN-10000/extraction_metrics.csv`
- **Surrogate model:** `/home/hytong/Documents/model_extraction_malware/output/LSE-somlap-dualDNN-10000/surrogate_model.h5`
- **Output directory:** `/home/hytong/Documents/model_extraction_malware/output/LSE-somlap-dualDNN-10000`
