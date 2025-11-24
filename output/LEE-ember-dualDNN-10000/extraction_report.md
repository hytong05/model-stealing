# Báo Cáo Extraction Attack: LEE-ember-dualDNN-10000

## Thông Tin Cấu Hình

- **Mô tả:** Tổng 10,000 queries (2000 queries/round × 5 rounds)
- **Query batch:** 2,000
- **Số rounds:** 5
- **Queries dự kiến:** 10,000
- **Queries thực tế:** 10,000
- **Ghi chú queries:** on_target
- **Tổng labels sử dụng (bao gồm seed+val):** 12,000

## Kết Quả Metrics

- **Accuracy:** 0.9227 (92.27%)
- **Balanced Accuracy:** 0.9229 (92.29%) [quan trọng với class imbalance]
- **F1-score:** 0.9246
- **Optimal Threshold:** 0.2500
- **Agreement:** 0.9912 (99.12%)
- **AUC:** 0.9591
- **Precision:** 0.8989
- **Recall:** 0.9518

## Files

- **Metrics CSV:** `/home/hytong/Documents/model_extraction_malware/output/LEE-ember-dualDNN-10000/extraction_metrics.csv`
- **Surrogate model:** `/home/hytong/Documents/model_extraction_malware/output/LEE-ember-dualDNN-10000/surrogate_model.h5`
- **Output directory:** `/home/hytong/Documents/model_extraction_malware/output/LEE-ember-dualDNN-10000`
