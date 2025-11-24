# Báo Cáo Extraction Attack: LEE-ember-LGB-2000

## Thông Tin Cấu Hình

- **Mô tả:** Tổng 2,000 queries (2000 queries/round × 1 round)
- **Query batch:** 2,000
- **Số rounds:** 1
- **Queries dự kiến:** 2,000
- **Queries thực tế:** 2,000
- **Ghi chú queries:** on_target
- **Tổng labels sử dụng (bao gồm seed+val):** 4,000

## Kết Quả Metrics

- **Accuracy:** 0.9050 (90.50%)
- **Balanced Accuracy:** 0.9049 (90.49%) [quan trọng với class imbalance]
- **F1-score:** 0.9033
- **Optimal Threshold:** 0.8000
- **Agreement:** 0.9475 (94.75%)
- **AUC:** 0.9692
- **Precision:** 0.9149
- **Recall:** 0.8920

## Files

- **Metrics CSV:** `/home/hytong/Documents/model_extraction_malware/output/LEE-ember-LGB-2000/extraction_metrics.csv`
- **Surrogate model:** `/home/hytong/Documents/model_extraction_malware/output/LEE-ember-LGB-2000/surrogate_model.txt`
- **Output directory:** `/home/hytong/Documents/model_extraction_malware/output/LEE-ember-LGB-2000`
