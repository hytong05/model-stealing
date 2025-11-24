# Báo Cáo Extraction Attack: LEE-somlap-dualDNN-5000

## Thông Tin Cấu Hình

- **Mô tả:** Tổng 5,000 queries (1250 queries/round × 4 rounds)
- **Query batch:** 1,250
- **Số rounds:** 4
- **Queries dự kiến:** 5,000
- **Queries thực tế:** 5,000
- **Ghi chú queries:** on_target
- **Tổng labels sử dụng (bao gồm seed+val):** 7,000

## Kết Quả Metrics

- **Accuracy:** 0.7262 (72.62%)
- **Balanced Accuracy:** 0.7386 (73.86%) [quan trọng với class imbalance]
- **F1-score:** 0.6843
- **Optimal Threshold:** 0.4100
- **Agreement:** 0.9972 (99.72%)
- **AUC:** 0.8129
- **Precision:** 0.6044
- **Recall:** 0.7887

## Files

- **Metrics CSV:** `/home/hytong/Documents/model_extraction_malware/output/LEE-somlap-dualDNN-5000/extraction_metrics.csv`
- **Surrogate model:** `/home/hytong/Documents/model_extraction_malware/output/LEE-somlap-dualDNN-5000/surrogate_model.h5`
- **Output directory:** `/home/hytong/Documents/model_extraction_malware/output/LEE-somlap-dualDNN-5000`
