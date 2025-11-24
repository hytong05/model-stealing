#!/usr/bin/env python3
"""
Ví dụ CỰC KỲ ĐƠN GIẢN: Attacker chỉ cần tên model!

Đây là cách đơn giản nhất để query oracle:
1. Chỉ cần tên model (CEE, LEE, CSE, LSE)
2. Gửi raw features
3. Nhận prediction (0 hoặc 1)

Không cần biết:
- Model type (h5 hay lgb) - tự động detect
- Normalization stats - tự động tìm
- Feature alignment - tự động xử lý
- Preprocessing - tự động xử lý
"""

import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.targets.oracle_client import create_oracle_from_name


# ========================================
# CHỈ CẦN 2 DÒNG CODE!
# ========================================

# 1. Khởi tạo oracle - chỉ cần tên model
oracle = create_oracle_from_name("LEE")

# 2. Query với raw features
sample = np.random.randn(2381).astype(np.float32)
prediction = oracle.predict(sample)

print(f"Prediction: {prediction[0]} ({'Malware' if prediction[0] == 1 else 'Benign'})")

# ========================================
# XONG! Đơn giản vậy thôi!
# ========================================

