import abc
import os

import lightgbm as lgb
import numpy as np

# Import flexible targets
from .flexible_target import FlexibleKerasTarget, FlexibleLGBTarget

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

