#!/usr/bin/env python3
"""
Script để truy vấn các mô hình ML đã được huấn luyện.

Usage:
    python scripts/inference/predict_models.py \
        --input data/test_samples.csv \
        --output output/predictions/ \
        --model lightgbm
"""

import os
import sys
import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Import ML frameworks
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import tensorflow as tf
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
except ImportError:
    tf = None

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
except ImportError:
    TabNetClassifier = None

# ========================================
# CONFIGURATION
# ========================================

# Đường dẫn đến thư mục chứa models
ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "artifacts" / "targets"

# Mapping các mô hình
MODEL_CONFIGS = {
    "lightgbm": {
        "model_file": "LEE.lgb",
        "stats_file": "LEE_normalization_stats.npz",
    },
    "cnn": {
        "model_file": "CNN.h5",  # Sử dụng CNN.h5 thay vì CEE.h5
        "stats_file": "CNN_normalization_stats.npz",
    },
    "xgboost": {
        "model_file": "xgboost_ember.json",
        "stats_file": "xgboost_normalization_stats.npz",
    },
    "tabnet": {
        "model_file": "tabnet_ember.zip",
        "stats_file": "tabnet_normalization_stats.npz",
    },
    "dualffnn": {
        "model_file": "dualffnn_ember.pt",
        "stats_file": "dualffnn_normalization_stats.npz",
    }
}

CLASS_NAMES = ["Benign", "Malware"]


# ========================================
# MODEL LOADERS
# ========================================

def load_normalization_stats(stats_path):
    """Load normalization statistics từ file .npz"""
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Không tìm thấy file normalization stats: {stats_path}")
    
    data = np.load(stats_path, allow_pickle=True)
    feature_means = data['feature_means']
    feature_stds = data['feature_stds']
    
    # Xử lý trường hợp std = 0
    feature_stds = np.where(feature_stds == 0, 1.0, feature_stds)
    
    return feature_means.astype(np.float32), feature_stds.astype(np.float32)


def load_lightgbm_model(model_path):
    """Load LightGBM model từ file .lgb"""
    if lgb is None:
        raise ImportError("LightGBM chưa được cài đặt. Cài đặt bằng: pip install lightgbm")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy model: {model_path}")
    
    model = lgb.Booster(model_file=str(model_path))
    return model


def load_cnn_model(model_path):
    """Load CNN model từ file .h5 (TensorFlow/Keras)"""
    if tf is None:
        raise ImportError("TensorFlow chưa được cài đặt. Cài đặt bằng: pip install tensorflow")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy model: {model_path}")
    
    # Xử lý lỗi version incompatibility với batch_shape
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except (TypeError, ValueError) as e:
        error_str = str(e)
        if 'batch_shape' in error_str or 'Unrecognized keyword' in error_str:
            print("⚠️  Phát hiện lỗi tương thích batch_shape, đang thử load với custom_objects...")
            try:
                # Thử load với custom_objects để bỏ qua DTypePolicy
                from tensorflow.keras import mixed_precision
                custom_objects = {
                    'DTypePolicy': mixed_precision.Policy,
                }
                model = tf.keras.models.load_model(
                    model_path,
                    compile=False,
                    custom_objects=custom_objects
                )
                return model
            except Exception as e2:
                # Nếu vẫn lỗi, thử sửa file HDF5 tạm thời
                print("⚠️  Đang thử sửa file HDF5 để bỏ qua batch_shape...")
                try:
                    import h5py
                    import tempfile
                    import shutil
                    import json
                    
                    # Tạo file tạm
                    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
                        tmp_path = tmp_file.name
                    
                    # Copy file
                    shutil.copy2(model_path, tmp_path)
                    
                    # Đọc và sửa file HDF5 để loại bỏ batch_shape
                    with h5py.File(tmp_path, 'r+') as f:
                        # Sửa model_config (JSON string chứa config của model)
                        if 'model_config' in f.attrs:
                            try:
                                model_config_str = f.attrs['model_config']
                                if isinstance(model_config_str, bytes):
                                    model_config_str = model_config_str.decode('utf-8')
                                
                                model_config = json.loads(model_config_str)
                                
                                # Đệ quy xóa batch_shape trong config
                                def remove_batch_shape(obj):
                                    if isinstance(obj, dict):
                                        obj.pop('batch_shape', None)
                                        obj.pop('batch_input_shape', None)
                                        for key, value in obj.items():
                                            remove_batch_shape(value)
                                    elif isinstance(obj, list):
                                        for item in obj:
                                            remove_batch_shape(item)
                                
                                remove_batch_shape(model_config)
                                
                                # Cập nhật lại model_config
                                f.attrs['model_config'] = json.dumps(model_config).encode('utf-8')
                            except Exception as config_error:
                                print(f"⚠️  Không thể sửa model_config: {config_error}")
                    
                    # Load từ file đã sửa với custom_objects
                    from tensorflow.keras import mixed_precision
                    custom_objects = {
                        'DTypePolicy': mixed_precision.Policy,
                    }
                    model = tf.keras.models.load_model(tmp_path, compile=False, custom_objects=custom_objects)
                    os.unlink(tmp_path)  # Xóa file tạm
                    return model
                except Exception as e3:
                    raise RuntimeError(
                        f"Không thể load CNN model từ {model_path}.\n"
                        f"Lỗi gốc: {e}\n"
                        f"Lỗi khi load với custom_objects: {e2}\n"
                        f"Lỗi khi sửa file: {e3}\n"
                        f"Có thể do version incompatibility của TensorFlow."
                    )
        else:
            raise


def load_xgboost_model(model_path):
    """Load XGBoost model từ file .json"""
    if xgb is None:
        raise ImportError("XGBoost chưa được cài đặt. Cài đặt bằng: pip install xgboost")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy model: {model_path}")
    
    model = xgb.Booster()
    model.load_model(str(model_path))
    return model


def load_tabnet_model(model_path):
    """Load TabNet model từ file .zip"""
    if TabNetClassifier is None:
        raise ImportError("pytorch-tabnet chưa được cài đặt. Cài đặt bằng: pip install pytorch-tabnet")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy model: {model_path}")
    
    # TabNet load từ zip file
    model = TabNetClassifier()
    model.load_model(str(model_path))
    return model


def load_dualffnn_model(model_path, stats_path):
    """Load dualffnn model từ file .pt (PyTorch)"""
    if torch is None:
        raise ImportError("PyTorch chưa được cài đặt. Cài đặt bằng: pip install torch")
    
    # Load normalization stats để lấy thông tin về input dimensions
    stats_data = np.load(stats_path, allow_pickle=True)
    in_dim_1 = int(stats_data['in_dim_1'])
    in_dim_2 = int(stats_data['in_dim_2'])
    
    # Định nghĩa lại architecture (từ notebook) - CẦN ĐỊNH NGHĨA TRƯỚC KHI LOAD
    class DualFFNN(nn.Module):
        def __init__(self, in_dim_1, in_dim_2, hidden1_branch=256, hidden2_branch=128,
                     hidden_joint=256, out_dim=2, dropout_p=0.2):
            super().__init__()
            self.in_dim_1 = in_dim_1
            self.in_dim_2 = in_dim_2
            
            self.branch1 = nn.Sequential(
                nn.Linear(in_dim_1, hidden1_branch),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(hidden1_branch, hidden2_branch),
                nn.ReLU(),
            )
            
            self.branch2 = nn.Sequential(
                nn.Linear(in_dim_2, hidden1_branch),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(hidden1_branch, hidden2_branch),
                nn.ReLU(),
            )
            
            self.joint = nn.Sequential(
                nn.Linear(hidden2_branch * 2, hidden_joint),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(hidden_joint, out_dim),
            )
        
        def forward(self, x):
            x1 = x[:, :self.in_dim_1]
            x2 = x[:, self.in_dim_1:self.in_dim_1 + self.in_dim_2]
            z1 = self.branch1(x1)
            z2 = self.branch2(x2)
            z = torch.cat([z1, z2], dim=1)
            out = self.joint(z)
            return out
    
    # Kiểm tra xem có full model không (ưu tiên)
    full_model_path = str(model_path).replace('.pt', '_full.pt')
    if os.path.exists(full_model_path):
        # Thử load full model (dễ nhất)
        # PyTorch 2.6+ yêu cầu weights_only=False để load custom classes
        try:
            model = torch.load(full_model_path, map_location='cpu', weights_only=False)
            model.eval()
            return model
        except (AttributeError, RuntimeError) as e:
            # Nếu không load được full model (do class không match), load state_dict
            print(f"⚠️  Không thể load full model, sẽ load state_dict: {e}")
    
    # Load state_dict từ checkpoint
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy model: {model_path}")
    
    # Load model checkpoint (state_dict)
    # PyTorch 2.6+ yêu cầu weights_only=False để load custom classes
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Lấy hyperparameters từ checkpoint hoặc dùng defaults
    hidden1_branch = checkpoint.get('hidden1_branch', 256)
    hidden2_branch = checkpoint.get('hidden2_branch', 128)
    hidden_joint = checkpoint.get('hidden_joint', 256)
    dropout_p = checkpoint.get('dropout_p', 0.2)
    
    # Tạo model và load weights
    model = DualFFNN(
        in_dim_1=in_dim_1,
        in_dim_2=in_dim_2,
        hidden1_branch=hidden1_branch,
        hidden2_branch=hidden2_branch,
        hidden_joint=hidden_joint,
        dropout_p=dropout_p,
    )
    
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    return model


# ========================================
# DATA PROCESSING
# ========================================

def load_csv_data(csv_path):
    """Load dữ liệu từ CSV file"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Không tìm thấy file CSV: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    if df.empty:
        raise ValueError("File CSV rỗng!")
    
    return df


def normalize_features(X, means, stds):
    """Normalize features sử dụng means và stds"""
    X = (X - means) / stds
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X.astype(np.float32)


# ========================================
# PREDICTION FUNCTIONS
# ========================================

def predict_lightgbm(model, X):
    """Predict với LightGBM model"""
    # LightGBM có thể predict trực tiếp từ numpy array
    probs = model.predict(X, num_iteration=model.best_iteration if hasattr(model, 'best_iteration') else None)
    predictions = (probs >= 0.5).astype(int)
    return predictions, probs


def predict_cnn(model, X):
    """Predict với CNN model (TensorFlow/Keras)"""
    # CNN được train với data đã clip về [-10, 10] sau normalize
    # Cần clip data để đảm bảo consistency với training
    CLIP_VALUE = 10.0
    X_clipped = np.clip(X, -CLIP_VALUE, CLIP_VALUE)
    
    # CNN cần reshape thành (n_samples, n_features, 1)
    n_features = X_clipped.shape[1]
    X_reshaped = X_clipped.reshape((-1, n_features, 1))
    
    # Predict
    probs = model.predict(X_reshaped, verbose=0)
    # probs có shape (n_samples, 2) cho binary classification
    if probs.shape[1] == 2:
        probs = probs[:, 1]  # Lấy probability của class 1 (Malware)
    predictions = (probs >= 0.5).astype(int)
    return predictions, probs


def predict_xgboost(model, X):
    """Predict với XGBoost model"""
    # XGBoost cần DMatrix
    dmat = xgb.DMatrix(X)
    probs = model.predict(dmat)
    predictions = (probs >= 0.5).astype(int)
    return predictions, probs


def predict_tabnet(model, X):
    """Predict với TabNet model"""
    probs = model.predict_proba(X)[:, 1]  # Probability của class 1
    predictions = model.predict(X).astype(int)
    return predictions, probs


def predict_dualffnn(model, X, device='cpu'):
    """Predict với dualffnn model (PyTorch)"""
    if torch is None:
        raise ImportError("PyTorch chưa được cài đặt")
    
    model.to(device)
    model.eval()
    
    # Convert numpy to tensor
    X_tensor = torch.from_numpy(X).float().to(device)
    
    # Predict
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # Probability của class 1
        predictions = torch.argmax(logits, dim=1).cpu().numpy()
    
    return predictions, probs


# ========================================
# OUTPUT FUNCTIONS
# ========================================

def save_predictions_csv(df, predictions, probs, output_path):
    """Lưu predictions vào CSV file"""
    # Tạo bản sao của dataframe
    df_output = df.copy()
    
    # Thêm cột predictions
    df_output['prediction'] = predictions
    df_output['prediction_label'] = df_output['prediction'].map({0: 'Benign', 1: 'Malware'})
    df_output['prediction_prob'] = probs
    
    # Lưu file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_output.to_csv(output_path, index=False)
    
    print(f"✅ Đã lưu predictions vào: {output_path}")
    return df_output


def generate_report(predictions, probs, output_path, model_name, num_samples):
    """Tạo report markdown"""
    # Tính toán thống kê
    benign_count = int(np.sum(predictions == 0))
    malware_count = int(np.sum(predictions == 1))
    benign_pct = (benign_count / num_samples * 100) if num_samples > 0 else 0
    malware_pct = (malware_count / num_samples * 100) if num_samples > 0 else 0
    
    avg_prob = float(np.mean(probs))
    
    # Tạo nội dung report
    report = f"""# Model Prediction Report

## Thông tin mô hình
- **Mô hình**: {model_name.upper()}
- **Thời gian**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Thống kê predictions
- **Tổng số samples**: {num_samples:,}
- **Benign**: {benign_count:,} ({benign_pct:.2f}%)
- **Malware**: {malware_count:,} ({malware_pct:.2f}%)

## Thống kê probabilities
- **Trung bình probability**: {avg_prob:.4f}
- **Min probability**: {float(np.min(probs)):.4f}
- **Max probability**: {float(np.max(probs)):.4f}
- **Std probability**: {float(np.std(probs)):.4f}

## Phân bố predictions
```
Benign:  {'█' * (benign_count // max(1, num_samples // 50))}
Malware: {'█' * (malware_count // max(1, num_samples // 50))}
```
"""
    
    # Lưu file
    report_path = os.path.join(output_path, 'report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ Đã tạo report tại: {report_path}")
    return report_path


# ========================================
# MAIN FUNCTION
# ========================================

def main():
    parser = argparse.ArgumentParser(
        description='Truy vấn các mô hình ML đã được huấn luyện',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python scripts/inference/predict_models.py \\
      --input data/test_samples.csv \\
      --output output/predictions/ \\
      --model lightgbm

Các mô hình hỗ trợ: lightgbm, cnn, xgboost, tabnet, dualffnn
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Đường dẫn đến file CSV đầu vào (chỉ có features)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Đường dẫn đến folder chứa file CSV đầu ra và report markdown'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['lightgbm', 'cnn', 'xgboost', 'tabnet', 'dualffnn'],
        help='Tên mô hình cần sử dụng'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.input):
        print(f"❌ Lỗi: Không tìm thấy file CSV đầu vào: {args.input}")
        sys.exit(1)
    
    # Tạo output directory nếu chưa có
    os.makedirs(args.output, exist_ok=True)
    
    # Lấy config cho model
    model_config = MODEL_CONFIGS[args.model]
    model_path = ARTIFACTS_DIR / model_config['model_file']
    stats_path = ARTIFACTS_DIR / model_config['stats_file']
    
    print("=" * 60)
    print(f"🔍 Truy vấn mô hình: {args.model.upper()}")
    print("=" * 60)
    
    # Load normalization stats
    print("\n📊 Đang load normalization statistics...")
    try:
        feature_means, feature_stds = load_normalization_stats(str(stats_path))
        print(f"✅ Đã load normalization stats: {stats_path}")
    except Exception as e:
        print(f"❌ Lỗi khi load normalization stats: {e}")
        sys.exit(1)
    
    # Load model
    print(f"\n🤖 Đang load mô hình: {model_path}")
    try:
        if args.model == 'lightgbm':
            model = load_lightgbm_model(str(model_path))
        elif args.model == 'cnn':
            model = load_cnn_model(str(model_path))
        elif args.model == 'xgboost':
            model = load_xgboost_model(str(model_path))
        elif args.model == 'tabnet':
            model = load_tabnet_model(str(model_path))
        elif args.model == 'dualffnn':
            model = load_dualffnn_model(str(model_path), str(stats_path))
        print(f"✅ Đã load mô hình thành công")
    except Exception as e:
        print(f"❌ Lỗi khi load mô hình: {e}")
        sys.exit(1)
    
    # Load CSV data
    print(f"\n📂 Đang load dữ liệu từ CSV: {args.input}")
    try:
        df = load_csv_data(args.input)
        print(f"✅ Đã load {len(df):,} samples, {len(df.columns)} features")
    except Exception as e:
        print(f"❌ Lỗi khi load CSV: {e}")
        sys.exit(1)
    
    # Extract features (loại bỏ các cột không phải feature)
    # Loại bỏ các cột phổ biến không phải feature: label, filename, id, hash, etc.
    exclude_cols = ['Label', 'label', 'target', 'filename', 'file_name', 'id', 'ID', 
                    'hash', 'Hash', 'sha256', 'SHA256', 'md5', 'MD5', 'sha1', 'SHA1']
    
    # Lọc các cột có thể convert sang số
    feature_cols = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        # Thử convert một sample để kiểm tra xem có phải số không
        try:
            pd.to_numeric(df[col].iloc[0], errors='raise')
            feature_cols.append(col)
        except (ValueError, TypeError, IndexError):
            # Không phải số, bỏ qua
            print(f"⚠️  Bỏ qua cột không phải số: {col}")
            continue
    
    if len(feature_cols) == 0:
        raise ValueError("Không tìm thấy cột feature nào trong CSV! Vui lòng kiểm tra lại file CSV.")
    
    print(f"📊 Sử dụng {len(feature_cols)} cột feature (đã loại bỏ {len(df.columns) - len(feature_cols)} cột không phải số)")
    
    # Convert sang float, xử lý lỗi nếu có
    X = df[feature_cols].copy()
    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Fill NaN bằng 0 (nếu có)
    X = X.fillna(0)
    X = X.values.astype(np.float32)
    
    # Kiểm tra số lượng features
    expected_features = len(feature_means)
    actual_features = X.shape[1]
    
    if actual_features != expected_features:
        print(f"⚠️  Cảnh báo: Số lượng features không khớp!")
        print(f"   CSV có {actual_features} features, model mong đợi {expected_features} features")
        
        if actual_features < expected_features:
            print(f"❌ Không đủ features! Cần thêm {expected_features - actual_features} features.")
            sys.exit(1)
        else:
            print(f"⚠️  CSV có nhiều features hơn. Sẽ chỉ sử dụng {expected_features} features đầu tiên.")
            X = X[:, :expected_features]
    
    # Normalize features
    print("\n🔄 Đang normalize features...")
    X_normalized = normalize_features(X, feature_means, feature_stds)
    print("✅ Đã normalize xong")
    
    # Predict
    print(f"\n🔮 Đang thực hiện prediction...")
    try:
        if args.model == 'lightgbm':
            predictions, probs = predict_lightgbm(model, X_normalized)
        elif args.model == 'cnn':
            predictions, probs = predict_cnn(model, X_normalized)
        elif args.model == 'xgboost':
            predictions, probs = predict_xgboost(model, X_normalized)
        elif args.model == 'tabnet':
            predictions, probs = predict_tabnet(model, X_normalized)
        elif args.model == 'dualffnn':
            device = 'cuda' if torch and torch.cuda.is_available() else 'cpu'
            predictions, probs = predict_dualffnn(model, X_normalized, device)
        
        print(f"✅ Đã hoàn tất prediction cho {len(predictions):,} samples")
    except Exception as e:
        print(f"❌ Lỗi khi predict: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Lưu kết quả
    print(f"\n💾 Đang lưu kết quả...")
    output_csv_path = os.path.join(args.output, f'predictions_{args.model}.csv')
    df_output = save_predictions_csv(df, predictions, probs, output_csv_path)
    
    # Tạo report
    generate_report(predictions, probs, args.output, args.model, len(predictions))
    
    print("\n" + "=" * 60)
    print("✅ Hoàn tất!")
    print("=" * 60)
    print(f"📁 CSV output: {output_csv_path}")
    print(f"📄 Report: {os.path.join(args.output, 'report.md')}")


if __name__ == '__main__':
    main()

