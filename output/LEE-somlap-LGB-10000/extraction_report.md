# Báo Cáo Extraction Attack: LEE-somlap-LGB-10000

## Thông Tin Cấu Hình

- **Mô tả:** Tổng 10,000 queries (2000 queries/round × 5 rounds)
- **Query batch:** 2,000
- **Số rounds:** 5
- **Queries dự kiến:** 10,000
- **Queries thực tế:** 10,000
- **Ghi chú queries:** on_target
- **Tổng labels sử dụng (bao gồm seed+val):** 12,000

## Kết Quả Metrics

- **Accuracy:** 0.6697 (66.97%)
- **Balanced Accuracy:** 0.7011 (70.11%) [quan trọng với class imbalance]
- **F1-score:** 0.6536
- **Optimal Threshold:** 0.8400
- **Agreement:** 0.9985 (99.85%)
- **AUC:** 0.8329
- **Precision:** 0.5399
- **Recall:** 0.8279

## Files

- **Metrics CSV:** `/home/hytong/Documents/model_extraction_malware/output/LEE-somlap-LGB-10000/extraction_metrics.csv`
- **Surrogate model:** `/home/hytong/Documents/model_extraction_malware/output/LEE-somlap-LGB-10000/surrogate_model.txt`
- **Output directory:** `/home/hytong/Documents/model_extraction_malware/output/LEE-somlap-LGB-10000`
