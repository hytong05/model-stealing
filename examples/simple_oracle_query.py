#!/usr/bin/env python3
"""
Ví dụ đơn giản: Cách attacker query oracle với raw features

Điểm quan trọng:
- Attacker chỉ cần gửi raw features (numpy array)
- Oracle client tự động xử lý:
  * Normalization (nếu có normalization stats)
  * Feature alignment
  * Trả về binary prediction (0 hoặc 1)
- Không cần phải lo về preprocessing, normalization, etc.
"""

import numpy as np
import sys
from pathlib import Path

# Thêm project root vào path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.targets.oracle_client import create_oracle_from_name


def main():
    # ========================================
    # 1. KHỞI TẠO ORACLE CLIENT - CHỈ CẦN TÊN MODEL!
    # ========================================
    # Chỉ cần cung cấp tên model (CEE, LEE, CSE, LSE)
    # Oracle sẽ tự động:
    # - Tìm model file (.h5 hoặc .lgb)
    # - Detect model type
    # - Tìm normalization stats
    # - Khởi tạo và sẵn sàng sử dụng
    
    model_name = "LEE"  # Chỉ cần tên model!
    
    oracle_client = create_oracle_from_name(
        model_name=model_name,
        threshold=0.5,
        feature_dim=2381,  # Mặc định: 2381 (có thể bỏ qua)
    )
    
    print("✅ Oracle client đã sẵn sàng!")
    print(f"   Model: {model_name}")
    print(f"   Model yêu cầu: {oracle_client.get_required_feature_dim()} features")
    
    # ========================================
    # 2. QUERY VỚI RAW FEATURES
    # ========================================
    # Attacker chỉ cần gửi raw features (numpy array)
    # Oracle sẽ tự động:
    # - Normalize (nếu có stats)
    # - Align feature dimensions
    # - Trả về binary prediction
    
    # Ví dụ 1: Query một sample
    sample = np.random.randn(2381).astype(np.float32)  # Raw features
    prediction = oracle_client.predict(sample)
    print(f"\n📊 Query một sample:")
    print(f"   Input shape: {sample.shape}")
    print(f"   Prediction: {prediction[0]} ({'Malware' if prediction[0] == 1 else 'Benign'})")
    
    # Ví dụ 2: Query nhiều samples (batch)
    batch = np.random.randn(10, 2381).astype(np.float32)  # 10 samples
    predictions = oracle_client.predict(batch)
    print(f"\n📊 Query batch (10 samples):")
    print(f"   Input shape: {batch.shape}")
    print(f"   Predictions: {predictions}")
    print(f"   Distribution: {dict(zip(*np.unique(predictions, return_counts=True)))}")
    
    # Ví dụ 3: Query với probabilities (nếu cần)
    if oracle_client.supports_probabilities():
        probs = oracle_client.predict_proba(batch)
        print(f"\n📊 Probabilities:")
        print(f"   Probabilities shape: {probs.shape}")
        print(f"   First 5 probabilities: {probs[:5]}")
    
    print("\n✅ Hoàn tất! Attacker chỉ cần gửi raw features, oracle tự động xử lý mọi thứ.")


if __name__ == "__main__":
    main()

