# model_extraction_malware
[![DOI:10.1016/j.cose.2023.103192](http://img.shields.io/badge/DOI-10.1016/j.cose.2023.103192-1AA7EC.svg)](https://doi.org/10.1016/j.cose.2023.103192)

Repository for the paper [Stealing Malware Classifiers and antivirus at Low False Positive Conditions](https://www.sciencedirect.com/science/article/pii/S0167404823001025)


## Setup

Xem file [docs/SETUP.md](docs/SETUP.md) để biết hướng dẫn chi tiết về cách setup môi trường ảo và cài đặt dependencies.

Tóm tắt nhanh:
```bash
# Tạo và setup môi trường ảo
./setup_venv.sh

# Kích hoạt môi trường ảo
source venv/bin/activate
```

## Cấu Trúc Thư Mục

```
model_extraction_malware/
├── docs/                          # Documentation & reports
│   ├── BUGFIX_SUMMARY.md
│   ├── SETUP.md
│   └── reports/                   # Generated summaries (metrics, comparisons, …)
├── notebooks/
│   └── targets/                   # Training notebooks (CEE.ipynb, …)
├── artifacts/
│   └── targets/                   # Packaged target models + normalization stats
├── scripts/                       # Executable utilities
│   ├── attacks/                   # Extraction / evaluation pipelines
│   │   ├── model_extraction.py
│   │   ├── extract_final_model.py
│   │   ├── run_multiple_extractions.py
│   │   └── evaluate_surrogate_similarity.py
│   ├── oracle/                    # Black-box API server & clients
│   ├── data/                      # Data preprocessing helpers
│   └── examples/                  # Sample/legacy attack flows
├── src/                           # Core library (attackers, targets, utils, …)
├── data/                          # Vectorized datasets (EMBER/SOREL/AV)
├── storage/                       # Training checkpoints (ignored)
├── output/                        # Experiment outputs (ignored)
├── logs/                          # Log files
├── config/                        # Configuration files
└── venv/                          # Virtual environment (tự động tạo)
```

## Documentation

- `docs/SETUP.md`: Hướng dẫn chi tiết cài đặt môi trường
- `docs/BUGFIX_SUMMARY.md`: Nhật ký sửa lỗi và quyết định thiết kế
- `docs/reports/`: Các báo cáo kết quả (ví dụ `metrics_explanation.md`, `extraction_comparison_report.md`)

## Usage 

In order to generate a surrogate model you need to specify the target, the surrogate type, the sampling method and the dataset. Please see the details below for the allowed values for each parameter.

```bash
python scripts/attacks/model_extraction.py -h                               
usage: Model extraction using active learning techniques [-h] -d DATA_DIR [-s SEED] [-m METHOD] [-n NUM_QUERIES] [-b BUDGET]
                                                         [-e NUM_EPOCHS] [-t {DNN,dualDNN,LGB,SVM}] [-l LOG_DIR]
                                                         [-tg {ember,sorel-FCNN,sorel-LGB,AV1,AV2,AV3,AV4}]
                                                         [-f {top10families,Adload,WannaCry,Pykse,Azorult,Bancteian,Emotet,Swisyn,Vobfus}]
                                                         [--dataset {ember,sorel,AV}] [--fpr FPR]

optional arguments:
  -h, --help            show this help message and exit
  -d DATA_DIR, --data_dir DATA_DIR
                        Directory that holds the data
  -s SEED, --seed SEED  Seed for random states
  -m METHOD, --method METHOD
                        entropy, random, medoids, mc_dropout, k-center, ensemble
  -n NUM_QUERIES, --num_queries NUM_QUERIES
                        Number of query rounds
  -b BUDGET, --budget BUDGET
                        Total query budget
  -e NUM_EPOCHS, --num_epochs NUM_EPOCHS
                        Number of training epochs per round
  -t {DNN,dualDNN,LGB,SVM}, --type {DNN,dualDNN,LGB,SVM}
                        Type of surrogate model
  -l LOG_DIR, --log_dir LOG_DIR
                        Where to store the log files with the results
  -tg {ember,sorel-FCNN,sorel-LGB,AV1,AV2,AV3,AV4}, --target_model {ember,sorel-FCNN,sorel-LGB,AV1,AV2,AV3,AV4}
                        Target model
  -f {top10families,Adload,WannaCry,Pykse,Azorult,Bancteian,Emotet,Swisyn,Vobfus}, --family {top10families,Adload,WannaCry,Pykse,Azorult,Bancteian,Emotet,Swisyn,Vobfus}
                        Select top10 families or one specific malware family
  --dataset {ember,sorel,AV}
                        Thief and test dataset
  --fpr FPR             FPR level for surrogate merics.
```

## Oracle Query Module

Để đảm bảo tấn công và truy vấn target model tách biệt hoàn toàn, sử dụng module `scripts/oracle/query_labels.py`:

1. **Sinh nhãn từ target model (không cần server)**
   ```bash
   python scripts/oracle/query_labels.py \
     --input-path data/pool_features.npy \
     --output-path cache/pool_labels.npy \
     --model-type h5 \
     --model-path artifacts/targets/CEE.h5
   ```
   Với LightGBM:
   ```bash
   python scripts/oracle/query_labels.py \
     --input-path data/pool.parquet \
     --output-path cache/pool_labels.npy \
     --model-type lgb \
     --model-path artifacts/targets/LEE.lgb \
     --normalization-stats-path artifacts/targets/LEE_normalization_stats.npz
   ```

2. **Tích hợp với quá trình tấn công**\
   Các scripts trong `scripts/attacks/` có thể đọc lại nhãn đã sinh hoặc import trực tiếp module để gọi `LocalOracleClient`. Điều này đảm bảo quá trình truy vấn target model hoạt động như một bước độc lập, không cần HTTP API hay server chạy nền.

## Giải Quyết Vấn Đề Không Tương Đồng Về Số Chiều Đặc Trưng

Dự án này xử lý vấn đề khi mô hình mục tiêu và kẻ tấn công sử dụng số đặc trưng khác nhau (ví dụ: mô hình mục tiêu dùng 50 đặc trưng, kẻ tấn công dùng 100 đặc trưng). Giải pháp được triển khai dựa trên hai chiến lược chính:

### 1. Nguyên Lý Cốt Lõi: Độc Lập Về Không Gian Đặc Trưng

Giải pháp coi mô hình mục tiêu (Target) là một **hộp đen**. Kẻ tấn công không cần (và thường là không thể) tái tạo chính xác các đặc trưng nội bộ mà mục tiêu đang sử dụng. Thay vào đó, kẻ tấn công xây dựng một không gian đặc trưng **riêng** để huấn luyện mô hình thay thế (Surrogate).

Mục tiêu của mô hình thay thế là học mối tương quan:
$$F_{surrogate}(attacker\_features) \approx Output_{target}$$

### 2. Cơ Chế Thực Thi Cụ Thể

#### Đối Với Mục Tiêu Là Antivirus (Black-box Hoàn Toàn)

- **Giao tiếp:** Điểm chung duy nhất là **tệp nhị phân gốc (raw binary file)**
- **Quy trình:** 
  - Kẻ tấn công gửi file gốc vào AV để lấy nhãn (Malware/Benign)
  - Đồng thời, kẻ tấn công trích xuất đặc trưng (theo định nghĩa của họ, ví dụ: Ember features) từ file đó
- **Giải quyết:** Sự không tương đồng bị xóa bỏ vì AV xử lý file theo cách của nó (với đặc trưng ẩn riêng), còn mô hình thay thế học trên đặc trưng của kẻ tấn công. Miễn là mô hình thay thế dự đoán đúng nhãn của AV, cuộc tấn công thành công mà không cần biết đặc trưng của AV là gì.

**Trong code:** `FileBasedTarget` xử lý AV targets - nhận index và trả về labels đã có sẵn, không cần vector đặc trưng.

#### Đối Với Mục Tiêu Là Mô Hình Máy Học (Gray-box/Input-aware)

- **Giao tiếp:** Đầu vào bắt buộc là một vector đặc trưng cụ thể
- **Quy trình:** 
  - Bài báo giả định kẻ tấn công biết định dạng đầu vào của mục tiêu (ví dụ: Ember v2 features)
  - Kẻ tấn công phải thực hiện **Tuân thủ giao diện (Interface Compliance)**
- **Giải quyết:** Nếu bộ dữ liệu tấn công có nhiều đặc trưng hơn mục tiêu yêu cầu (ví dụ: 100 vs 50), kẻ tấn công bắt buộc phải **cắt bỏ** các đặc trưng thừa để khớp với đầu vào của mục tiêu khi thực hiện truy vấn (query).

**Trong code:** Các target classes (`LGBTarget`, `TorchTarget`, `KerasCNNTarget`) tự động xử lý feature alignment:
- Tự động phát hiện số đặc trưng yêu cầu của target model
- Tự động cắt bỏ đặc trưng thừa nếu input có nhiều đặc trưng hơn
- Báo lỗi nếu input có ít đặc trưng hơn yêu cầu

### 3. Tổng Kết

Giải pháp cho sự bất đồng bộ này là:

1. **Với hệ thống thực tế (AV):** Sử dụng **file gốc** làm trung gian, cho phép mô hình tấn công và mô hình mục tiêu hoạt động trên hai không gian đặc trưng hoàn toàn khác biệt nhau.

2. **Với mô hình ML cụ thể:** Bắt buộc **đồng bộ hóa đầu vào** (feature alignment) dựa trên kiến thức về mô hình mục tiêu - tự động cắt bỏ đặc trưng thừa khi cần.

## Example
The following command will create a LightGBM surrogate model and it will store it in the output folder (`/tmp/logs`) along with a log file with the results for each iteration.

```bash
python scripts/attacks/model_extraction.py --data_dir /data/mari/sorel-data --dataset sorel --seed 42 --method medoids --type LGB --target_model sorel-FCNN --num_epochs 1 --num_queries 10 --log_dir "/tmp/logs/" --budget 2500 --fpr 0.006
```
If you use this code please cite:
```
@article{RIGAKI2023103192,
  title = {Stealing and evading malware classifiers and antivirus at low false positive conditions},
  journal = {Computers & Security},
  volume = {129},
  pages = {103192},
  year = {2023},
  issn = {0167-4048},
  doi = {https://doi.org/10.1016/j.cose.2023.103192},
  url = {https://www.sciencedirect.com/science/article/pii/S0167404823001025},
  author = {M. Rigaki and S. Garcia},
}
```
