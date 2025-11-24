#!/usr/bin/env python3
"""
Phân tích ảnh hưởng của class imbalance trong Seed và Val
"""
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 80)
print("PHÂN TÍCH ẢNH HƯỞNG CỦA CLASS IMBALANCE TRONG SEED VÀ VAL")
print("=" * 80)

# Ví dụ với distribution thực tế
seed_dist = {0: 1944, 1: 56}
val_dist = {0: 971, 1: 29}

print("\n1. PHÂN TÍCH DISTRIBUTION:")
print(f"   Seed: {seed_dist}")
print(f"      Class 0: {seed_dist[0]}/{sum(seed_dist.values())} ({seed_dist[0]/sum(seed_dist.values())*100:.1f}%)")
print(f"      Class 1: {seed_dist[1]}/{sum(seed_dist.values())} ({seed_dist[1]/sum(seed_dist.values())*100:.1f}%)")
print(f"      Imbalance ratio: {seed_dist[0]/seed_dist[1]:.1f}:1")

print(f"\n   Val: {val_dist}")
print(f"      Class 0: {val_dist[0]}/{sum(val_dist.values())} ({val_dist[0]/sum(val_dist.values())*100:.1f}%)")
print(f"      Class 1: {val_dist[1]}/{sum(val_dist.values())} ({val_dist[1]/sum(val_dist.values())*100:.1f}%)")
print(f"      Imbalance ratio: {val_dist[0]/val_dist[1]:.1f}:1")

print("\n2. ẢNH HƯỞNG ĐẾN KẾT QUẢ TẤN CÔNG:")
print("""
   A. ROUND 0 TRAINING (Initial Model):
      - Model học từ seed data mất cân bằng (97% class 0, 3% class 1)
      - Model có xu hướng predict class 0 nhiều hơn
      - Model không học được pattern của class 1 tốt
      - Probabilities thấp cho class 1
      
   B. VALIDATION TRONG TRAINING:
      - Model được đánh giá trên val data mất cân bằng (97% class 0, 3% class 1)
      - Val accuracy cao nhưng không phản ánh đúng performance
      - Val loss thấp nhưng model bias về class 0
      - Early stopping có thể dừng sớm vì val loss giảm (do class imbalance)
      
   C. MODEL BIAS:
      - Model học được: "hầu hết samples là class 0"
      - Model output probabilities thấp cho class 1
      - Threshold phải thấp (0.1) để detect class 1
      - Agreement và accuracy thấp hơn
      
   D. PROPAGATION:
      - Model bias từ Round 0 ảnh hưởng đến Round 1, 2, ...
      - Queries được chọn dựa trên model bias
      - Vòng lặp: bias → queries bias → model bias hơn
""")

print("\n3. SO SÁNH VỚI CÂN BẰNG:")
balanced_seed = {0: 1000, 1: 1000}
balanced_val = {0: 500, 1: 500}

print(f"   Seed cân bằng: {balanced_seed}")
print(f"      Class 0: {balanced_seed[0]/sum(balanced_seed.values())*100:.1f}%")
print(f"      Class 1: {balanced_seed[1]/sum(balanced_seed.values())*100:.1f}%")

print(f"\n   Val cân bằng: {balanced_val}")
print(f"      Class 0: {balanced_val[0]/sum(balanced_val.values())*100:.1f}%")
print(f"      Class 1: {balanced_val[1]/sum(balanced_val.values())*100:.1f}%")

print("\n   Lợi ích của cân bằng:")
print("   ✅ Model học được cả 2 classes đều nhau")
print("   ✅ Probabilities cao hơn và calibrated tốt hơn")
print("   ✅ Threshold gần 0.5 (không cần thấp)")
print("   ✅ Val accuracy phản ánh đúng performance")
print("   ✅ Early stopping hoạt động tốt hơn")
print("   ✅ Agreement và accuracy cao hơn")

print("\n" + "=" * 80)
print("KẾT LUẬN")
print("=" * 80)
print("""
✅ Class imbalance trong Seed và Val CÓ ẢNH HƯỞNG NGHIÊM TRỌNG:

1. Model bias ngay từ Round 0
2. Validation không phản ánh đúng performance
3. Early stopping có thể dừng sớm
4. Probabilities thấp và threshold phải thấp
5. Agreement và accuracy thấp hơn

💡 GIẢI PHÁP:
   - Dùng stratified sampling cho Seed và Val
   - Đảm bảo 50/50 distribution (hoặc tỷ lệ gần nhất có thể)
   - Cải thiện model training và validation
""")

