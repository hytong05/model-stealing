import abc
import os

import lightgbm as lgb
import numpy as np

# Import flexible targets
from .flexible_target import FlexibleKerasTarget, FlexibleLGBTarget, FlexibleSKLearnTarget

class AbstractTarget(abc.ABC):
    """
    Abstract base class cho các target models.
    
    Giải quyết vấn đề "không tương đồng về số chiều đặc trưng":
    - Với AV (black-box): Sử dụng file gốc làm trung gian, không cần đồng bộ hóa đặc trưng
    - Với ML models (gray-box): Cần đồng bộ hóa đầu vào (cắt bỏ đặc trưng thừa nếu cần)
    """
    
    def __init__(self, model_path, thresh):
        self.model_endpoint = model_path
        self.model_threshold = thresh

    @abc.abstractmethod
    def __call__(self, X):
        raise NotImplementedError
    
    def get_required_feature_dim(self):
        """
        Trả về số đặc trưng mà target model yêu cầu.
        Returns None nếu không cần (như AV - dùng file gốc).
        """
        return None
    
    def _align_features(self, X):
        """
        Đồng bộ hóa số chiều đặc trưng của input với yêu cầu của target model.
        Nếu X có nhiều đặc trưng hơn, cắt bỏ các đặc trưng thừa ở cuối.
        Nếu X có ít đặc trưng hơn, raise ValueError.
        
        Args:
            X: Input features array (n_samples, n_features)
            
        Returns:
            X_aligned: Input đã được đồng bộ hóa
        """
        required_dim = self.get_required_feature_dim()
        if required_dim is None:
            # Không cần đồng bộ hóa (như AV)
            return X
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        actual_dim = X.shape[1]
        
        if actual_dim == required_dim:
            return X
        elif actual_dim > required_dim:
            # Cắt bỏ đặc trưng thừa ở cuối
            return X[:, :required_dim]
        else:
            # Không đủ đặc trưng - raise error
            raise ValueError(
                f"Input has {actual_dim} features, but target model requires {required_dim}. "
                f"Cannot pad features - please provide correct feature set."
            )


class LGBTarget(AbstractTarget):
    """"
    Class for Ember and Sorel-20M LightGBM models
    
    Xử lý feature dimension mismatch: Tự động cắt bỏ đặc trưng thừa nếu input 
    có nhiều đặc trưng hơn model yêu cầu (Interface Compliance).
    """
    def __init__(self, model_path, thresh, name):
        super().__init__(model_path, thresh)
        
        self.name = name
        self.model = lgb.Booster(model_file=self.model_endpoint)
        # Lấy số đặc trưng yêu cầu từ model
        self._required_feature_dim = self.model.num_feature()

    def get_required_feature_dim(self):
        """Trả về số đặc trưng mà LightGBM model yêu cầu"""
        return self._required_feature_dim

    def __call__(self, X):
        # Đồng bộ hóa số chiều đặc trưng trước khi predict
        X = self._align_features(X)
        scores = self.model.predict(X)
        # output = np.atleast_2d(scores)
        return np.array([int(score > self.model_threshold) for score in scores])


class TorchTarget(AbstractTarget):
    """"
    Class for Sorel-20M FCNN models
    
    Xử lý feature dimension mismatch: Tự động cắt bỏ đặc trưng thừa nếu input 
    có nhiều đặc trưng hơn model yêu cầu (Interface Compliance).
    """
    def __init__(self, model_path, thresh, name):
        super().__init__(model_path, thresh)

        self.name = name

        from ..models.sorel_nets import PENetwork  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel

        self._torch = torch
        # Sorel-20M FCNN model yêu cầu 2381 đặc trưng
        self._required_feature_dim = 2381
        self.model = PENetwork(
            use_malware=True, use_counts=False, use_tags=True, n_tags=11, feature_dimension=self._required_feature_dim
        )

        self.model.load_state_dict(self._torch.load(self.model_endpoint))
    
        # Set model to inference mode
        self.model.eval()
    
    def get_required_feature_dim(self):
        """Trả về số đặc trưng mà Sorel FCNN model yêu cầu"""
        return self._required_feature_dim

    """
    From sorel-20m code
    """
    def _features_postproc_func(self, x):
        x1 = np.copy(x)
        lz = x1 < 0
        gz = x1 > 0
        x1[lz] = - np.log(1 - x1[lz])
        x1[gz] = np.log(1 + x1[gz])
        return x1

    def __call__(self, X):
        # Đồng bộ hóa số chiều đặc trưng trước khi predict
        X = self._align_features(X)
        X = self._torch.from_numpy(self._features_postproc_func(X))
        predictions = self.model(X)
        scores = predictions["malware"].detach().numpy().ravel()
        return np.array([int(score > self.model_threshold) for score in scores])


class FileBasedTarget(AbstractTarget):
    """
    Class for targets that we have offline labels, such as AVs
    
    Với AV (black-box), giải pháp cho feature dimension mismatch:
    - Không cần đồng bộ hóa đặc trưng vì AV xử lý file gốc
    - Kẻ tấn công trích xuất đặc trưng riêng từ file, không cần biết đặc trưng của AV
    - Miễn là surrogate model dự đoán đúng nhãn của AV là thành công
    """
    def __init__(self, model_path, name, labels):

        self.name = name
        self.labels = labels
    
    def get_required_feature_dim(self):
        """
        AV không yêu cầu feature dimension cụ thể vì nó xử lý file gốc.
        Kẻ tấn công có thể dùng bất kỳ số đặc trưng nào cho surrogate model.
        """
        return None
        
    def __call__(self, idx):
        """
        AV target: Nhận index và trả về labels đã có sẵn.
        Không cần vector đặc trưng vì AV xử lý file gốc trực tiếp.
        """
        scores = self.labels[idx]
        self.labels = np.delete(self.labels, idx, axis=0)
        return scores


class KerasCNNTarget(AbstractTarget):
    """
    Oracle wrapper cho mô hình CNN Keras đã huấn luyện sẵn (final_model.h5)
    
    Xử lý feature dimension mismatch: Tự động cắt bỏ đặc trưng thừa nếu input 
    có nhiều đặc trưng hơn model yêu cầu (Interface Compliance).
    """

    def __init__(self, weights_path, feature_dim=2381, threshold=0.5, name="keras-cnn-target"):
        super().__init__(weights_path, threshold)
        self.name = name
        self.feature_dim = feature_dim
        self._required_feature_dim = feature_dim
        self._model = self._build_model()
        # TensorFlow yêu cầu biến môi trường này khi dùng legacy Keras API
        os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
        self._model.load_weights(self.model_endpoint)
    
    def get_required_feature_dim(self):
        """Trả về số đặc trưng mà Keras CNN model yêu cầu"""
        return self._required_feature_dim

    def _build_model(self):
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import (BatchNormalization, Conv1D, Dense,
                                             Dropout, Flatten, MaxPooling1D)
        from tensorflow.keras.regularizers import l2

        model = Sequential([
            Conv1D(
                64,
                5,
                strides=2,
                padding="same",
                activation="relu",
                input_shape=(self.feature_dim, 1),
            ),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),
            Conv1D(64, 3, padding="same", activation="relu"),
            BatchNormalization(),
            Conv1D(32, 3, padding="same", activation="relu"),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.4),
            Flatten(),
            Dense(256, activation="relu", kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.5),
            Dense(128, activation="relu", kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.5),
            Dense(2, activation="softmax", dtype="float32"),
        ])
        return model

    def _prepare_input(self, X):
        # Đồng bộ hóa số chiều đặc trưng trước khi chuẩn bị input
        X = self._align_features(X)
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.expand_dims(X, axis=-1)

    def predict_proba(self, X, batch_size=512):
        X = self._prepare_input(X)
        return self._model.predict(X, batch_size=batch_size, verbose=0)

    def __call__(self, X, batch_size=512):
        probs = self.predict_proba(X, batch_size=batch_size)
        return (probs[:, 1] >= self.model_threshold).astype(int)


class XGBoostTarget(AbstractTarget):
    """
    Class for XGBoost models with normalization support.
    
    Xử lý feature dimension mismatch: Tự động cắt bỏ đặc trưng thừa nếu input 
    có nhiều đặc trưng hơn model yêu cầu (Interface Compliance).
    """
    
    def __init__(self, model_path, normalization_stats_path=None, thresh=0.5, name="xgboost-target"):
        super().__init__(model_path, thresh)
        self.name = name
        
        # Load XGBoost model
        import xgboost as xgb
        self.model = xgb.Booster()
        self.model.load_model(self.model_endpoint)
        
        # Lấy số đặc trưng yêu cầu từ model
        # XGBoost sử dụng num_features() (method), không phải num_feature() như LightGBM
        self._required_feature_dim = self.model.num_features()
        
        # Load normalization stats (nếu có)
        self.feature_means = None
        self.feature_stds = None
        self.feature_cols = None
        self.use_normalization = False
        
        if normalization_stats_path is not None:
            self._load_normalization_stats(normalization_stats_path)
    
    def _load_normalization_stats(self, stats_path):
        """Load normalization statistics từ file .npz"""
        try:
            stats = np.load(stats_path, allow_pickle=True)
            
            if 'feature_means' in stats:
                self.feature_means = stats['feature_means']
            else:
                raise ValueError(f"File {stats_path} không chứa 'feature_means'")
            
            if 'feature_stds' in stats:
                self.feature_stds = stats['feature_stds']
            else:
                raise ValueError(f"File {stats_path} không chứa 'feature_stds'")
            
            if 'feature_cols' in stats:
                self.feature_cols = stats['feature_cols'].tolist() if hasattr(stats['feature_cols'], 'tolist') else stats['feature_cols']
            else:
                self.feature_cols = None
            
            self.use_normalization = True
            print(f"✅ Loaded normalization stats from {stats_path}")
            print(f"   Feature means shape: {self.feature_means.shape}")
            print(f"   Feature stds shape: {self.feature_stds.shape}")
        except Exception as e:
            print(f"⚠️  Warning: Cannot load normalization stats from {stats_path}: {type(e).__name__}: {str(e)}")
            print(f"   Will use features without normalization")
            self.use_normalization = False
    
    def get_required_feature_dim(self):
        """Trả về số đặc trưng mà XGBoost model yêu cầu"""
        return self._required_feature_dim
    
    def _normalize_features(self, X):
        """
        Normalize features nếu có stats.
        """
        if not self.use_normalization:
            return X
        
        # Align features trước khi normalize
        X_aligned = self._align_features(X)
        
        # Kiểm tra xem stats có khớp với model requirements không
        stats_feature_dim = self.feature_means.shape[0]
        
        if stats_feature_dim == self._required_feature_dim:
            # Stats khớp - normalize bình thường
            feature_means_used = self.feature_means
            feature_stds_used = self.feature_stds
        else:
            # Stats không khớp - chỉ dùng số features model cần
            if stats_feature_dim >= self._required_feature_dim:
                feature_means_used = self.feature_means[:self._required_feature_dim]
                feature_stds_used = self.feature_stds[:self._required_feature_dim]
            else:
                # Stats ít hơn model cần - không normalize
                print(f"⚠️  Stats có ít features hơn model cần - bỏ qua normalization")
                return X_aligned
        
        # Normalize
        features_normalized = (X_aligned - feature_means_used) / feature_stds_used
        features_normalized = np.nan_to_num(features_normalized, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features_normalized
    
    def predict_proba(self, X):
        """
        Predict probabilities.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            probabilities: Array of probabilities (n_samples,)
        """
        # Normalize và align features
        if self.use_normalization:
            features_array = self._normalize_features(X)
        else:
            features_array = self._align_features(X)
        
        # XGBoost cần DMatrix object, không phải numpy array
        import xgboost as xgb
        dmatrix = xgb.DMatrix(features_array)
        
        # Predict
        prediction_prob = self.model.predict(dmatrix)
        
        # Đảm bảo output là 1D array
        if prediction_prob.ndim > 1:
            prediction_prob = np.squeeze(prediction_prob)
        
        return prediction_prob
    
    def __call__(self, X):
        """
        Predict binary labels.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            labels: Binary labels (n_samples,)
        """
        probs = self.predict_proba(X)
        return (probs >= self.model_threshold).astype(int)


class DualFFNNTarget(AbstractTarget):
    """
    Class for dualFFNN PyTorch models with normalization support.
    
    dualFFNN là dual-branch feedforward network:
    - Input được chia thành 2 nhánh (in_dim_1 và in_dim_2)
    - Mỗi nhánh đi qua các layers riêng, sau đó concatenate và đi qua joint layers
    - Output là 2 classes (binary classification)
    
    Xử lý feature dimension mismatch: Tự động cắt bỏ đặc trưng thừa nếu input 
    có nhiều đặc trưng hơn model yêu cầu (Interface Compliance).
    """
    
    def __init__(self, model_path, normalization_stats_path=None, thresh=0.5, name="dualffnn-target", 
                 in_dim_1=None, in_dim_2=None, hidden1_branch=256, hidden2_branch=128, 
                 hidden_joint=256, dropout_p=0.2):
        super().__init__(model_path, thresh)
        self.name = name
        
        import torch
        
        self._torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model info từ file nếu có, hoặc dùng defaults
        model_info = self._load_model_info(model_path)
        
        # Xác định input dimensions
        if in_dim_1 is None or in_dim_2 is None:
            if model_info and 'in_dim_1' in model_info and 'in_dim_2' in model_info:
                self.in_dim_1 = model_info['in_dim_1']
                self.in_dim_2 = model_info['in_dim_2']
            else:
                # Default: chia đôi input (giống notebook)
                # Giả sử total input là 2381 (EMBER), chia thành 1190 và 1191
                total_dim = 2381
                self.in_dim_1 = total_dim // 2
                self.in_dim_2 = total_dim - self.in_dim_1
                print(f"⚠️  Using default input dimensions: in_dim_1={self.in_dim_1}, in_dim_2={self.in_dim_2}")
        else:
            self.in_dim_1 = in_dim_1
            self.in_dim_2 = in_dim_2
        
        # Total input dimension
        self._required_feature_dim = self.in_dim_1 + self.in_dim_2
        
        # Override với model info nếu có
        if model_info:
            hidden1_branch = model_info.get('hidden1_branch', hidden1_branch)
            hidden2_branch = model_info.get('hidden2_branch', hidden2_branch)
            hidden_joint = model_info.get('hidden_joint', hidden_joint)
            dropout_p = model_info.get('dropout_p', dropout_p)
        
        # Build model architecture
        self.model = self._build_model(hidden1_branch, hidden2_branch, hidden_joint, dropout_p)
        
        # Load model weights
        self._load_model_weights(model_path)
        
        # Set model to inference mode
        self.model.eval()
        
        # Load normalization stats (nếu có)
        self.feature_means = None
        self.feature_stds = None
        self.feature_cols = None
        self.use_normalization = False
        
        if normalization_stats_path is not None:
            self._load_normalization_stats(normalization_stats_path)
    
    def _load_model_info(self, model_path):
        """Thử load model info từ file .pt (có thể chứa metadata)"""
        try:
            import torch
            checkpoint = torch.load(model_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_info' in checkpoint:
                return checkpoint['model_info']
            elif isinstance(checkpoint, dict) and 'in_dim_1' in checkpoint:
                return checkpoint
        except Exception:
            pass
        return None
    
    def _build_model(self, hidden1_branch, hidden2_branch, hidden_joint, dropout_p):
        """Build dualFFNN architecture"""
        import torch
        import torch.nn as nn
        
        class DualFFNN(nn.Module):
            def __init__(self, in_dim_1, in_dim_2, hidden1_branch, hidden2_branch, 
                         hidden_joint, out_dim=2, dropout_p=0.2):
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
                # x: [batch, INPUT_DIM], tách thành 2 nhánh
                x1 = x[:, :self.in_dim_1]
                x2 = x[:, self.in_dim_1:self.in_dim_1 + self.in_dim_2]

                z1 = self.branch1(x1)
                z2 = self.branch2(x2)
                z = torch.cat([z1, z2], dim=1)
                out = self.joint(z)
                return out
        
        model = DualFFNN(
            self.in_dim_1, self.in_dim_2, hidden1_branch, hidden2_branch, 
            hidden_joint, out_dim=2, dropout_p=dropout_p
        ).to(self.device)
        
        return model
    
    def _load_model_weights(self, model_path):
        """Load model weights từ file .pt"""
        try:
            # PyTorch 2.6+ thay đổi default của weights_only thành True.
            # Để tránh phụ thuộc vào class gốc (__main__.DualFFNN), ta dùng
            # weights_only=True để chỉ load state_dict một cách an toàn.
            try:
                checkpoint = self._torch.load(
                    model_path, map_location=self.device, weights_only=True
                )
            except TypeError:
                # Older PyTorch: không có weights_only, load bình thường
                checkpoint = self._torch.load(model_path, map_location=self.device)
            
            # Nếu checkpoint là full nn.Module (ít khả năng với weights_only=True)
            if isinstance(checkpoint, self._torch.nn.Module):
                self.model.load_state_dict(checkpoint.state_dict())
                self.model.to(self.device).eval()
            # Nếu checkpoint là dict (state_dict hoặc wrapper)
            elif isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    # Thử dùng trực tiếp như state_dict
                    state_dict = checkpoint
                self.model.load_state_dict(state_dict)
                self.model.to(self.device).eval()
            else:
                raise ValueError(f"Unknown checkpoint format in {model_path}: {type(checkpoint)}")
            
            print(f"✅ Loaded dualFFNN model from {model_path}")
        except Exception as e:
            raise ValueError(f"Cannot load dualFFNN model from {model_path}: {type(e).__name__}: {str(e)}")
    
    def _load_normalization_stats(self, stats_path):
        """Load normalization statistics từ file .npz"""
        try:
            stats = np.load(stats_path, allow_pickle=True)
            
            if 'feature_means' in stats:
                self.feature_means = stats['feature_means']
            else:
                raise ValueError(f"File {stats_path} không chứa 'feature_means'")
            
            if 'feature_stds' in stats:
                self.feature_stds = stats['feature_stds']
            else:
                raise ValueError(f"File {stats_path} không chứa 'feature_stds'")
            
            if 'feature_cols' in stats:
                self.feature_cols = stats['feature_cols'].tolist() if hasattr(stats['feature_cols'], 'tolist') else stats['feature_cols']
            else:
                self.feature_cols = None
            
            self.use_normalization = True
            print(f"✅ Loaded normalization stats from {stats_path}")
            print(f"   Feature means shape: {self.feature_means.shape}")
            print(f"   Feature stds shape: {self.feature_stds.shape}")
        except Exception as e:
            print(f"⚠️  Warning: Cannot load normalization stats from {stats_path}: {type(e).__name__}: {str(e)}")
            print(f"   Will use features without normalization")
            self.use_normalization = False
    
    def get_required_feature_dim(self):
        """Trả về số đặc trưng mà dualFFNN model yêu cầu"""
        return self._required_feature_dim
    
    def _normalize_features(self, X):
        """
        Normalize features nếu có stats.
        """
        if not self.use_normalization:
            return X
        
        # Align features trước khi normalize
        X_aligned = self._align_features(X)
        
        # Kiểm tra xem stats có khớp với model requirements không
        stats_feature_dim = self.feature_means.shape[0]
        
        if stats_feature_dim == self._required_feature_dim:
            # Stats khớp - normalize bình thường
            feature_means_used = self.feature_means
            feature_stds_used = self.feature_stds
        else:
            # Stats không khớp - chỉ dùng số features model cần
            if stats_feature_dim >= self._required_feature_dim:
                feature_means_used = self.feature_means[:self._required_feature_dim]
                feature_stds_used = self.feature_stds[:self._required_feature_dim]
            else:
                # Stats ít hơn model cần - không normalize
                print(f"⚠️  Stats có ít features hơn model cần - bỏ qua normalization")
                return X_aligned
        
        # Normalize
        features_normalized = (X_aligned - feature_means_used) / feature_stds_used
        features_normalized = np.nan_to_num(features_normalized, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features_normalized
    
    def predict_proba(self, X, batch_size=512):
        """
        Predict probabilities.
        
        Args:
            X: Input features (n_samples, n_features)
            batch_size: Batch size for prediction
            
        Returns:
            probabilities: Array of probabilities for class 1 (n_samples,)
        """
        # Normalize và align features
        if self.use_normalization:
            features_array = self._normalize_features(X)
        else:
            features_array = self._align_features(X)
        
        features_array = np.asarray(features_array, dtype=np.float32)
        if features_array.ndim == 1:
            features_array = features_array.reshape(1, -1)
        
        # Predict in batches
        all_probs = []
        self.model.eval()
        
        with self._torch.no_grad():
            for i in range(0, len(features_array), batch_size):
                batch = features_array[i:i+batch_size]
                batch_tensor = self._torch.from_numpy(batch).to(self.device)
                
                logits = self.model(batch_tensor)
                # Apply softmax to get probabilities
                probs = self._torch.softmax(logits, dim=1)
                # Get probability of class 1 (malware)
                prob_class1 = probs[:, 1].cpu().numpy()
                all_probs.append(prob_class1)
        
        prediction_prob = np.concatenate(all_probs, axis=0)
        return prediction_prob
    
    def __call__(self, X, batch_size=512):
        """
        Predict binary labels.
        
        Args:
            X: Input features (n_samples, n_features)
            batch_size: Batch size for prediction
            
        Returns:
            labels: Binary labels (n_samples,)
        """
        probs = self.predict_proba(X, batch_size=batch_size)
        return (probs >= self.model_threshold).astype(int)


class TabNetTarget(AbstractTarget):
    """
    Class for TabNet models with normalization support.
    
    Xử lý feature dimension mismatch: Tự động cắt bỏ đặc trưng thừa nếu input 
    có nhiều đặc trưng hơn model yêu cầu (Interface Compliance).
    """
    
    def __init__(self, model_path, normalization_stats_path=None, thresh=0.5, name="tabnet-target", 
                 input_dim=None):
        super().__init__(model_path, thresh)
        self.name = name
        
        # Load TabNet model
        from pytorch_tabnet.tab_model import TabNetClassifier
        
        # Set default input_dim if not provided
        if input_dim is None:
            input_dim = 2381  # Default for EMBER
        
        # Initialize with default - will be updated after loading
        self._required_feature_dim = input_dim
        
        # Create a dummy model to load weights
        # TabNetClassifier needs to be initialized before loading
        self.model = TabNetClassifier(
            n_d=24, n_a=24, n_steps=5, gamma=1.5,
            n_independent=2, n_shared=2, epsilon=1e-15,
            seed=42, verbose=0
        )
        
        # Load model from file
        try:
            self.model.load_model(model_path)
            # Get input dimension from loaded model
            # TabNet stores input_dim in the model after loading
            if hasattr(self.model, 'input_dim') and self.model.input_dim is not None:
                self._required_feature_dim = self.model.input_dim
            elif hasattr(self.model, 'n_features') and self.model.n_features is not None:
                self._required_feature_dim = self.model.n_features
            print(f"✅ Loaded TabNet model from {model_path}")
            print(f"   Model input dimension: {self._required_feature_dim}")
        except Exception as e:
            print(f"⚠️  Warning: Error loading TabNet model: {e}")
            print(f"   Will use default input_dim: {input_dim}")
            self._required_feature_dim = input_dim
        
        # Load normalization stats (nếu có)
        self.feature_means = None
        self.feature_stds = None
        self.feature_cols = None
        self.use_normalization = False
        
        if normalization_stats_path is not None:
            self._load_normalization_stats(normalization_stats_path)
    
    def _load_normalization_stats(self, stats_path):
        """Load normalization statistics từ file .npz"""
        try:
            stats = np.load(stats_path, allow_pickle=True)
            
            if 'feature_means' in stats:
                self.feature_means = stats['feature_means']
            else:
                raise ValueError(f"File {stats_path} không chứa 'feature_means'")
            
            if 'feature_stds' in stats:
                self.feature_stds = stats['feature_stds']
            else:
                raise ValueError(f"File {stats_path} không chứa 'feature_stds'")
            
            if 'feature_cols' in stats:
                self.feature_cols = stats['feature_cols'].tolist() if hasattr(stats['feature_cols'], 'tolist') else stats['feature_cols']
            else:
                self.feature_cols = None
            
            self.use_normalization = True
            print(f"✅ Loaded normalization stats from {stats_path}")
            print(f"   Feature means shape: {self.feature_means.shape}")
            print(f"   Feature stds shape: {self.feature_stds.shape}")
        except Exception as e:
            print(f"⚠️  Warning: Cannot load normalization stats from {stats_path}: {type(e).__name__}: {str(e)}")
            print(f"   Will use features without normalization")
            self.use_normalization = False
    
    def get_required_feature_dim(self):
        """Trả về số đặc trưng mà TabNet model yêu cầu"""
        return self._required_feature_dim
    
    def _normalize_features(self, X):
        """
        Normalize features nếu có stats.
        """
        if not self.use_normalization:
            return X
        
        # Align features trước khi normalize
        X_aligned = self._align_features(X)
        
        # Kiểm tra xem stats có khớp với model requirements không
        stats_feature_dim = self.feature_means.shape[0]
        
        if stats_feature_dim == self._required_feature_dim:
            # Stats khớp - normalize bình thường
            feature_means_used = self.feature_means
            feature_stds_used = self.feature_stds
        else:
            # Stats không khớp - chỉ dùng số features model cần
            if stats_feature_dim >= self._required_feature_dim:
                feature_means_used = self.feature_means[:self._required_feature_dim]
                feature_stds_used = self.feature_stds[:self._required_feature_dim]
            else:
                # Stats ít hơn model cần - không normalize
                print(f"⚠️  Stats có ít features hơn model cần - bỏ qua normalization")
                return X_aligned
        
        # Normalize
        features_normalized = (X_aligned - feature_means_used) / feature_stds_used
        features_normalized = np.nan_to_num(features_normalized, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features_normalized
    
    def predict_proba(self, X):
        """
        Predict probabilities.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            probabilities: Array of probabilities for class 1 (n_samples,)
        """
        # Normalize và align features
        if self.use_normalization:
            features_array = self._normalize_features(X)
        else:
            features_array = self._align_features(X)
        
        features_array = np.asarray(features_array, dtype=np.float32)
        if features_array.ndim == 1:
            features_array = features_array.reshape(1, -1)
        
        # Predict
        prediction_prob = self.model.predict_proba(features_array)
        
        # TabNet returns (n_samples, n_classes), get class 1 probability
        if prediction_prob.ndim > 1:
            if prediction_prob.shape[1] == 2:
                prediction_prob = prediction_prob[:, 1]
            else:
                prediction_prob = prediction_prob[:, -1]
        
        return prediction_prob
    
    def __call__(self, X):
        """
        Predict binary labels.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            labels: Binary labels (n_samples,)
        """
        probs = self.predict_proba(X)
        return (probs >= self.model_threshold).astype(int)

