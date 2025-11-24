# Báo Cáo Extraction Attack: LEE-somlap-dualDNN-10000

## Thông Tin Cấu Hình

- **Mô tả:** Tổng 10,000 queries (2000 queries/round × 5 rounds)
- **Query batch:** 2,000
- **Số rounds:** 5
- **Queries dự kiến:** 10,000
- **Queries thực tế:** 10,000
- **Ghi chú queries:** on_target
- **Tổng labels sử dụng (bao gồm seed+val):** 12,000

## Kết Quả Metrics

- **Accuracy:** 0.6947 (69.47%)
- **Balanced Accuracy:** 0.7246 (72.46%) [quan trọng với class imbalance]
- **F1-score:** 0.6757
- **Optimal Threshold:** 0.3800
- **Agreement:** 0.9990 (99.90%)
- **AUC:** 0.8467
- **Precision:** 0.5628
- **Recall:** 0.8452

## Files

- **Metrics CSV:** `/home/hytong/Documents/model_extraction_malware/output/LEE-somlap-dualDNN-10000/extraction_metrics.csv`
- **Surrogate model:** `/home/hytong/Documents/model_extraction_malware/output/LEE-somlap-dualDNN-10000/surrogate_model.h5`
- **Output directory:** `/home/hytong/Documents/model_extraction_malware/output/LEE-somlap-dualDNN-10000`
