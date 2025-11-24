# Báo Cáo Extraction Attack: LEE-ember-dualDNN-5000

## Thông Tin Cấu Hình

- **Mô tả:** Tổng 5,000 queries (1250 queries/round × 4 rounds)
- **Query batch:** 1,250
- **Số rounds:** 4
- **Queries dự kiến:** 5,000
- **Queries thực tế:** 5,000
- **Ghi chú queries:** on_target
- **Tổng labels sử dụng (bao gồm seed+val):** 7,000

## Kết Quả Metrics

- **Accuracy:** 0.9207 (92.07%)
- **Balanced Accuracy:** 0.9209 (92.09%) [quan trọng với class imbalance]
- **F1-score:** 0.9222
- **Optimal Threshold:** 0.4000
- **Agreement:** 0.9842 (98.42%)
- **AUC:** 0.9558
- **Precision:** 0.9016
- **Recall:** 0.9437

## Files

- **Metrics CSV:** `/home/hytong/Documents/model_extraction_malware/output/LEE-ember-dualDNN-5000/extraction_metrics.csv`
- **Surrogate model:** `/home/hytong/Documents/model_extraction_malware/output/LEE-ember-dualDNN-5000/surrogate_model.h5`
- **Output directory:** `/home/hytong/Documents/model_extraction_malware/output/LEE-ember-dualDNN-5000`
