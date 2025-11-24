# Báo Cáo Extraction Attack: LSE-somlap-LGB-2000

## Thông Tin Cấu Hình

- **Mô tả:** Tổng 2,000 queries (2000 queries/round × 1 round)
- **Query batch:** 2,000
- **Số rounds:** 1
- **Queries dự kiến:** 2,000
- **Queries thực tế:** 2,000
- **Ghi chú queries:** on_target
- **Tổng labels sử dụng (bao gồm seed+val):** 4,000

## Kết Quả Metrics

- **Accuracy:** 0.9055 (90.55%)
- **Balanced Accuracy:** 0.9179 (91.79%) [quan trọng với class imbalance]
- **F1-score:** 0.8852
- **Optimal Threshold:** 0.8900
- **Agreement:** 0.9832 (98.32%)
- **AUC:** 0.9529
- **Precision:** 0.8153
- **Recall:** 0.9681

## Files

- **Metrics CSV:** `/home/hytong/Documents/model_extraction_malware/output/LSE-somlap-LGB-2000/extraction_metrics.csv`
- **Surrogate model:** `/home/hytong/Documents/model_extraction_malware/output/LSE-somlap-LGB-2000/surrogate_model.txt`
- **Output directory:** `/home/hytong/Documents/model_extraction_malware/output/LSE-somlap-LGB-2000`
