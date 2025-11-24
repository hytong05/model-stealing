# Báo Cáo Extraction Attack: LEE-somlap-LGB-5000

## Thông Tin Cấu Hình

- **Mô tả:** Tổng 5,000 queries (1250 queries/round × 4 rounds)
- **Query batch:** 1,250
- **Số rounds:** 4
- **Queries dự kiến:** 5,000
- **Queries thực tế:** 5,000
- **Ghi chú queries:** on_target
- **Tổng labels sử dụng (bao gồm seed+val):** 7,000

## Kết Quả Metrics

- **Accuracy:** 0.6693 (66.92%)
- **Balanced Accuracy:** 0.7007 (70.07%) [quan trọng với class imbalance]
- **F1-score:** 0.6532
- **Optimal Threshold:** 0.1000
- **Agreement:** 0.9980 (99.80%)
- **AUC:** 0.8275
- **Precision:** 0.5394
- **Recall:** 0.8279

## Files

- **Metrics CSV:** `/home/hytong/Documents/model_extraction_malware/output/LEE-somlap-LGB-5000/extraction_metrics.csv`
- **Surrogate model:** `/home/hytong/Documents/model_extraction_malware/output/LEE-somlap-LGB-5000/surrogate_model.txt`
- **Output directory:** `/home/hytong/Documents/model_extraction_malware/output/LEE-somlap-LGB-5000`
