# Hướng Dẫn Setup Môi Trường

## Yêu Cầu Hệ Thống

- Python 3.8 trở lên
- pip (thường đi kèm với Python)

## Cách 1: Sử dụng Script Tự Động (Khuyến nghị)

Chạy script setup:

```bash
./setup_venv.sh
```

Script này sẽ:
1. Kiểm tra Python version
2. Tạo virtual environment tại `venv/`
3. Cài đặt tất cả dependencies từ `requirements.txt`
4. Cài đặt package `ember` từ GitHub (vì không có trên PyPI)

## Cách 2: Setup Thủ Công

### Bước 1: Tạo Virtual Environment

```bash
python3 -m venv venv
```

### Bước 2: Kích Hoạt Virtual Environment

**Trên Linux/Mac:**
```bash
source venv/bin/activate
```

**Trên Windows:**
```bash
venv\Scripts\activate
```

### Bước 3: Cài Đặt Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt

# Cài đặt package ember từ GitHub (không có trên PyPI)
pip install git+https://github.com/endgameinc/ember.git
```

## Sử Dụng

Sau khi setup xong, mỗi lần làm việc với dự án:

1. Kích hoạt môi trường ảo:
   ```bash
   source venv/bin/activate
   ```

2. Chạy các script:
   ```bash
   python scripts/model_extraction.py --help
   ```

3. Tắt môi trường ảo khi xong:
   ```bash
   deactivate
   ```

## Cấu Trúc Thư Mục Mới

```
model_extraction_malware/
├── src/                    # Source code chính
│   ├── models/            # Model definitions (DNN, Sorel networks)
│   ├── attackers/         # Surrogate model attackers
│   ├── targets/           # Target model wrappers
│   ├── datasets/          # Dataset loaders
│   ├── utils/             # Utility functions
│   └── sampling/          # Active learning sampling strategies
├── scripts/               # Executable scripts
│   ├── model_extraction.py
│   ├── extract_final_model.py
│   ├── evaluate_surrogate_similarity.py
│   └── run_multiple_extractions.py
├── data/                  # Data files (nên thêm vào .gitignore)
├── output/                # Output files và results
├── logs/                  # Log files
├── config/                # Configuration files
├── venv/                  # Virtual environment (tự động tạo)
├── requirements.txt       # Python dependencies
└── setup_venv.sh         # Setup script
```

## Troubleshooting

### Lỗi khi import modules

Nếu gặp lỗi import, đảm bảo bạn đã kích hoạt virtual environment và đang ở thư mục gốc của project.

### Lỗi cài đặt TensorFlow

Nếu gặp vấn đề với TensorFlow, thử:
```bash
pip install tensorflow --upgrade
```

### Lỗi cài đặt PyTorch

PyTorch có thể cần cài đặt riêng tùy theo hệ điều hành và GPU. Xem: https://pytorch.org/get-started/locally/

### Lỗi cài đặt ember

Package `ember` không có trên PyPI, cần cài từ GitHub. Nếu gặp lỗi, thử:
```bash
pip install tqdm lief
pip install git+https://github.com/endgameinc/ember.git
```

