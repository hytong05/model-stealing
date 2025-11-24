# Báo Cáo Extraction Attack: LSE-somlap-dualDNN-5000

## Thông Tin Cấu Hình

- **Mô tả:** Tổng 5,000 queries (1250 queries/round × 4 rounds)
- **Query batch:** 1,250
- **Số rounds:** 4
- **Queries dự kiến:** 5,000
- **Queries thực tế:** 5,000
- **Ghi chú queries:** on_target
- **Tổng labels sử dụng (bao gồm seed+val):** 7,000

## Kết Quả Metrics

- **Accuracy:** 0.9005 (90.05%)
- **Balanced Accuracy:** 0.9159 (91.59%) [quan trọng với class imbalance]
- **F1-score:** 0.8809
- **Optimal Threshold:** 0.4800
- **Agreement:** 0.9992 (99.92%)
- **AUC:** 0.9393
- **Precision:** 0.8013
- **Recall:** 0.9781

## Files

- **Metrics CSV:** `/home/hytong/Documents/model_extraction_malware/output/LSE-somlap-dualDNN-5000/extraction_metrics.csv`
- **Surrogate model:** `/home/hytong/Documents/model_extraction_malware/output/LSE-somlap-dualDNN-5000/surrogate_model.h5`
- **Output directory:** `/home/hytong/Documents/model_extraction_malware/output/LSE-somlap-dualDNN-5000`
