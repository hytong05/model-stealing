# Báo Cáo Extraction Attack: LSE-ember-dualDNN-10000

## Thông Tin Cấu Hình

- **Mô tả:** Tổng 10,000 queries (2000 queries/round × 5 rounds)
- **Query batch:** 2,000
- **Số rounds:** 5
- **Queries dự kiến:** 10,000
- **Queries thực tế:** 10,000
- **Ghi chú queries:** on_target
- **Tổng labels sử dụng (bao gồm seed+val):** 12,000

## Kết Quả Metrics

- **Accuracy:** 0.4975 (49.75%)
- **Balanced Accuracy:** 0.5000 (50.00%) [quan trọng với class imbalance]
- **F1-score:** 0.6644
- **Optimal Threshold:** 0.1000
- **Agreement:** 1.0000 (100.00%)
- **AUC:** 0.4203
- **Precision:** 0.4975
- **Recall:** 1.0000

## Files

- **Metrics CSV:** `/home/hytong/Documents/model_extraction_malware/output/LSE-ember-dualDNN-10000/extraction_metrics.csv`
- **Surrogate model:** `/home/hytong/Documents/model_extraction_malware/output/LSE-ember-dualDNN-10000/surrogate_model.h5`
- **Output directory:** `/home/hytong/Documents/model_extraction_malware/output/LSE-ember-dualDNN-10000`
