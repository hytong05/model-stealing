# Báo Cáo Extraction Attack: LSE-somlap-LGB-5000

## Thông Tin Cấu Hình

- **Mô tả:** Tổng 5,000 queries (1250 queries/round × 4 rounds)
- **Query batch:** 1,250
- **Số rounds:** 4
- **Queries dự kiến:** 5,000
- **Queries thực tế:** 5,000
- **Ghi chú queries:** on_target
- **Tổng labels sử dụng (bao gồm seed+val):** 7,000

## Kết Quả Metrics

- **Accuracy:** 0.9032 (90.33%)
- **Balanced Accuracy:** 0.9176 (91.76%) [quan trọng với class imbalance]
- **F1-score:** 0.8835
- **Optimal Threshold:** 0.8000
- **Agreement:** 0.9885 (98.85%)
- **AUC:** 0.9567
- **Precision:** 0.8075
- **Recall:** 0.9754

## Files

- **Metrics CSV:** `/home/hytong/Documents/model_extraction_malware/output/LSE-somlap-LGB-5000/extraction_metrics.csv`
- **Surrogate model:** `/home/hytong/Documents/model_extraction_malware/output/LSE-somlap-LGB-5000/surrogate_model.txt`
- **Output directory:** `/home/hytong/Documents/model_extraction_malware/output/LSE-somlap-LGB-5000`
