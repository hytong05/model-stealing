# Báo Cáo Extraction Attack: LEE-ember-LGB-10000

## Thông Tin Cấu Hình

- **Mô tả:** Tổng 10,000 queries (2000 queries/round × 5 rounds)
- **Query batch:** 2,000
- **Số rounds:** 5
- **Queries dự kiến:** 10,000
- **Queries thực tế:** 10,000
- **Ghi chú queries:** on_target
- **Tổng labels sử dụng (bao gồm seed+val):** 12,000

## Kết Quả Metrics

- **Accuracy:** 0.9125 (91.25%)
- **Balanced Accuracy:** 0.9126 (91.26%) [quan trọng với class imbalance]
- **F1-score:** 0.9143
- **Optimal Threshold:** 0.5400
- **Agreement:** 0.9670 (96.70%)
- **AUC:** 0.9731
- **Precision:** 0.8912
- **Recall:** 0.9387

## Files

- **Metrics CSV:** `/home/hytong/Documents/model_extraction_malware/output/LEE-ember-LGB-10000/extraction_metrics.csv`
- **Surrogate model:** `/home/hytong/Documents/model_extraction_malware/output/LEE-ember-LGB-10000/surrogate_model.txt`
- **Output directory:** `/home/hytong/Documents/model_extraction_malware/output/LEE-ember-LGB-10000`
