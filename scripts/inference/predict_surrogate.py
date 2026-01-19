#!/usr/bin/env python3
"""
Script để truy vấn các surrogate model dualffnn đã được extract.

Usage:
    python scripts/inference/predict_surrogate.py \
        --model output/DUALFFNN-ember-dualDNN-5000/surrogate_model.h5 \
        --output output/predictions/surrogate_dualffnn_5000 \
        --input data/test_samples.csv
"""

import os
import sys
import argparse
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
    import tensorflow as tf
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
except ImportError:
    tf = None

try:
    from sklearn.preprocessing import RobustScaler
    import joblib
except ImportError:
    RobustScaler = None
    joblib = None

CLASS_NAMES = ["Benign", "Malware"]


# ========================================
# MODEL LOADERS
# ========================================

def load_surrogate_model(model_path):
    """Load surrogate model từ file .h5 (TensorFlow/Keras)"""
    if tf is None:
        raise ImportError("TensorFlow chưa được cài đặt. Cài đặt bằng: pip install tensorflow")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy surrogate model: {model_path}")
    
    # Xử lý lỗi version incompatibility với batch_shape
    # Thử nhiều cách load khác nhau
    import warnings
    
    # Cách 1: Load bình thường
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
                return model
            except TypeError:
                model = tf.keras.models.load_model(model_path, compile=False)
                return model
    except Exception as e:
        error_str = str(e)
        # Cách 2: Load với custom_objects rỗng và bỏ qua warnings
        if 'batch_shape' in error_str or 'Unrecognized keyword' in error_str or 'rms_scaling' in error_str:
            print("⚠️  Phát hiện lỗi tương thích, đang thử load với custom_objects rỗng...")
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        model = tf.keras.models.load_model(model_path, compile=False, custom_objects={}, safe_mode=False)
                        return model
                    except TypeError:
                        model = tf.keras.models.load_model(model_path, compile=False, custom_objects={})
                        return model
            except Exception as e2:
                # Cách 3: Thử load với custom_objects có DTypePolicy
                print("⚠️  Đang thử load với DTypePolicy custom_objects...")
                try:
                    from tensorflow.keras import mixed_precision
                    custom_objects = {
                        'DTypePolicy': mixed_precision.Policy,
                    }
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        try:
                            model = tf.keras.models.load_model(model_path, compile=False, custom_objects=custom_objects, safe_mode=False)
                            return model
                        except TypeError:
                            model = tf.keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)
                            return model
                except Exception as e3:
                    # Cách 4: Nếu vẫn lỗi, thử sửa file HDF5 tạm thời
                    print("⚠️  Đang thử sửa file HDF5 để bỏ qua các keyword không hợp lệ...")
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
                                    
                                    # Đệ quy xóa các keyword không hợp lệ trong config
                                    def remove_invalid_keywords(obj):
                                        if isinstance(obj, dict):
                                            # Loại bỏ các keyword không hợp lệ
                                            obj.pop('batch_shape', None)
                                            obj.pop('batch_input_shape', None)
                                            obj.pop('rms_scaling', None)
                                            # Xử lý các key khác
                                            for key, value in list(obj.items()):
                                                if isinstance(value, dict):
                                                    remove_invalid_keywords(value)
                                                elif isinstance(value, list):
                                                    for item in value:
                                                        remove_invalid_keywords(item)
                                        elif isinstance(obj, list):
                                            for item in obj:
                                                remove_invalid_keywords(item)
                                    
                                    remove_invalid_keywords(model_config)
                                    
                                    # Cập nhật lại model_config
                                    f.attrs['model_config'] = json.dumps(model_config).encode('utf-8')
                                except Exception as config_error:
                                    print(f"⚠️  Không thể sửa model_config: {config_error}")
                        
                        # Load từ file đã sửa với custom_objects
                        from tensorflow.keras import mixed_precision
                        custom_objects = {
                            'DTypePolicy': mixed_precision.Policy,
                        }
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            try:
                                model = tf.keras.models.load_model(tmp_path, compile=False, custom_objects=custom_objects, safe_mode=False)
                            except TypeError:
                                model = tf.keras.models.load_model(tmp_path, compile=False, custom_objects=custom_objects)
                        os.unlink(tmp_path)  # Xóa file tạm
                        return model
                    except Exception as e4:
                        # Nếu vẫn lỗi, raise error với tất cả các lỗi
                        raise RuntimeError(
                            f"Không thể load surrogate model từ {model_path}.\n"
                            f"Lỗi gốc: {e}\n"
                            f"Lỗi khi load với custom_objects rỗng: {e2}\n"
                            f"Lỗi khi load với DTypePolicy: {e3}\n"
                            f"Lỗi khi sửa file: {e4}\n"
                            f"Có thể do version incompatibility của TensorFlow.\n"
                            f"Thử cài đặt TensorFlow version tương thích hoặc rebuild model."
                        )
        else:
            raise


def load_robust_scaler(scaler_path):
    """Load RobustScaler từ file .joblib"""
    if joblib is None:
        raise ImportError("joblib chưa được cài đặt. Cài đặt bằng: pip install joblib")
    
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Không tìm thấy robust scaler: {scaler_path}")
    
    scaler = joblib.load(scaler_path)
    return scaler


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


def clip_scale(scaler, X):
    """Scale data với RobustScaler và clip về [-5, 5]"""
    if scaler is None:
        raise ValueError("Scaler không được khởi tạo")
    
    transformed = scaler.transform(X)
    return np.clip(transformed, -5, 5)


# ========================================
# PREDICTION FUNCTIONS
# ========================================

def predict_surrogate(model, X):
    """Predict với surrogate model (Keras)"""
    # Kiểm tra xem model có multiple inputs không
    input_shape = model.input_shape
    has_multiple_inputs = isinstance(input_shape, list) and len(input_shape) > 1
    
    if has_multiple_inputs:
        # Model có multiple inputs - cần tạo input thứ 2
        # Thường input thứ 2 là một giá trị constant hoặc metadata
        # Tạm thời tạo input thứ 2 với shape (n_samples, 1) filled với 0
        X_input2 = np.zeros((X.shape[0], 1), dtype=np.float32)
        # Predict với multiple inputs
        probs_raw = model.predict([X, X_input2], verbose=0)
    else:
        # Single input
        probs_raw = model.predict(X, verbose=0)
    
    # Xử lý output shape
    if len(probs_raw.shape) == 1:
        # Shape (n_samples,) - binary output
        probs = probs_raw
    elif probs_raw.shape[1] == 1:
        # Shape (n_samples, 1) - single output
        probs = probs_raw[:, 0]
    elif probs_raw.shape[1] == 2:
        # Shape (n_samples, 2) - two outputs, lấy class 1 (Malware)
        probs = probs_raw[:, 1]
    else:
        # Fallback: lấy output đầu tiên
        probs = probs_raw[:, 0]
    
    # Đảm bảo probs trong khoảng [0, 1]
    probs = np.clip(probs, 0, 1)
    
    # Predictions
    predictions = (probs >= 0.5).astype(int)
    
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
    report = f"""# Surrogate Model Prediction Report

## Thông tin mô hình
- **Mô hình**: {model_name}
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
        description='Truy vấn các surrogate model dualffnn đã được extract',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python scripts/inference/predict_surrogate.py \\
      --model output/DUALFFNN-ember-dualDNN-5000/surrogate_model.h5 \\
      --output output/predictions/surrogate_dualffnn_5000 \\
      --input data/test_samples.csv
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Đường dẫn đến surrogate model (.h5 file)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Đường dẫn đến folder chứa file CSV đầu ra và report markdown'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Đường dẫn đến file CSV đầu vào (chỉ có features)'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.model):
        print(f"❌ Lỗi: Không tìm thấy surrogate model: {args.model}")
        sys.exit(1)
    
    if not os.path.exists(args.input):
        print(f"❌ Lỗi: Không tìm thấy file CSV đầu vào: {args.input}")
        sys.exit(1)
    
    # Tạo output directory nếu chưa có
    os.makedirs(args.output, exist_ok=True)
    
    # Tìm robust scaler trong cùng folder với model
    model_dir = os.path.dirname(args.model)
    scaler_path = os.path.join(model_dir, 'robust_scaler.joblib')
    
    print("=" * 60)
    print(f"🔍 Truy vấn Surrogate Model")
    print("=" * 60)
    print(f"📁 Model: {args.model}")
    print(f"📁 Scaler: {scaler_path}")
    print(f"📁 Input: {args.input}")
    print(f"📁 Output: {args.output}")
    
    # Load robust scaler
    print(f"\n📊 Đang load robust scaler...")
    try:
        scaler = load_robust_scaler(scaler_path)
        print(f"✅ Đã load robust scaler: {scaler_path}")
    except Exception as e:
        print(f"❌ Lỗi khi load robust scaler: {e}")
        sys.exit(1)
    
    # Load surrogate model
    print(f"\n🤖 Đang load surrogate model: {args.model}")
    try:
        model = load_surrogate_model(args.model)
        print(f"✅ Đã load surrogate model thành công")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
    except Exception as e:
        print(f"❌ Lỗi khi load surrogate model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Load CSV data
    print(f"\n📂 Đang load dữ liệu từ CSV: {args.input}")
    try:
        df = load_csv_data(args.input)
        print(f"✅ Đã load {len(df):,} samples, {len(df.columns)} columns")
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
    
    # Kiểm tra số lượng features với model input
    # model.input_shape có thể là tuple, list, hoặc list of tuples (multiple inputs)
    input_shape = model.input_shape
    
    # Xử lý trường hợp multiple inputs
    if isinstance(input_shape, list) and len(input_shape) > 0:
        # Nếu là list, lấy input đầu tiên (thường là main input)
        first_input = input_shape[0]
        if isinstance(first_input, (list, tuple)):
            # Lấy số features từ input shape đầu tiên
            if len(first_input) > 1:
                expected_features = first_input[1] if first_input[1] is not None else first_input[-1]
            else:
                expected_features = first_input[0] if first_input[0] is not None else X.shape[1]
        else:
            # Nếu không phải tuple/list, thử lấy trực tiếp
            expected_features = first_input if isinstance(first_input, int) else X.shape[1]
    elif isinstance(input_shape, (list, tuple)) and len(input_shape) > 1:
        # Single input với shape là tuple/list
        expected_features = input_shape[1] if input_shape[1] is not None else input_shape[-1]
    elif isinstance(input_shape, (list, tuple)) and len(input_shape) == 1:
        expected_features = input_shape[0] if input_shape[0] is not None else X.shape[1]
    else:
        # Fallback: thử lấy từ model.input
        try:
            if hasattr(model, 'input') and model.input is not None:
                if isinstance(model.input, list):
                    # Multiple inputs
                    expected_features = model.input[0].shape[-1] if model.input[0].shape[-1] is not None else X.shape[1]
                else:
                    expected_features = model.input.shape[-1] if model.input.shape[-1] is not None else X.shape[1]
            else:
                expected_features = X.shape[1]  # Sử dụng số features hiện tại
        except:
            expected_features = X.shape[1]  # Sử dụng số features hiện tại
    
    # Đảm bảo expected_features là số nguyên
    if not isinstance(expected_features, (int, np.integer)):
        print(f"⚠️  Không thể xác định số features từ model input shape: {input_shape}")
        print(f"   Sẽ sử dụng số features từ CSV: {X.shape[1]}")
        expected_features = X.shape[1]
    
    actual_features = X.shape[1]
    
    if actual_features != expected_features:
        print(f"⚠️  Cảnh báo: Số lượng features không khớp!")
        print(f"   CSV có {actual_features} features, model mong đợi {expected_features} features")
        
        if actual_features < expected_features:
            # Pad với zeros
            print(f"⚠️  CSV có ít features hơn. Sẽ pad thêm {expected_features - actual_features} features bằng 0.")
            X_padded = np.zeros((X.shape[0], expected_features), dtype=np.float32)
            X_padded[:, :actual_features] = X
            X = X_padded
        else:
            print(f"⚠️  CSV có nhiều features hơn. Sẽ chỉ sử dụng {expected_features} features đầu tiên.")
            X = X[:, :expected_features]
    
    # Scale và clip data
    print("\n🔄 Đang scale và clip dữ liệu...")
    try:
        X_scaled = clip_scale(scaler, X)
        print("✅ Đã scale và clip xong")
    except Exception as e:
        print(f"❌ Lỗi khi scale data: {e}")
        sys.exit(1)
    
    # Predict
    print(f"\n🔮 Đang thực hiện prediction...")
    try:
        predictions, probs = predict_surrogate(model, X_scaled)
        print(f"✅ Đã hoàn tất prediction cho {len(predictions):,} samples")
    except Exception as e:
        print(f"❌ Lỗi khi predict: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Lưu kết quả
    print(f"\n💾 Đang lưu kết quả...")
    model_name = os.path.basename(os.path.dirname(args.model))
    output_csv_path = os.path.join(args.output, f'predictions_{model_name}.csv')
    df_output = save_predictions_csv(df, predictions, probs, output_csv_path)
    
    # Tạo report
    generate_report(predictions, probs, args.output, model_name, len(predictions))
    
    print("\n" + "=" * 60)
    print("✅ Hoàn tất!")
    print("=" * 60)
    print(f"📁 CSV output: {output_csv_path}")
    print(f"📄 Report: {os.path.join(args.output, 'report.md')}")


if __name__ == '__main__':
    main()

