import lightgbm as lgb
from ..models.dnn import create_dnn, create_dnn2, create_cnn
import tensorflow as tf
import joblib
import abc
import numpy as np

# This is optional and can also be called from the command line
try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except ImportError:
    patch_sklearn = None
from sklearn import svm

# XGBoost and TabNet imports
try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    import torch
except ImportError:
    TabNetClassifier = None
    torch = None

class AbstractAttacker(abc.ABC):
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def __call__(self, X):
        raise NotImplementedError

    @abc.abstractmethod
    def train_model(self, X, y, X_val, y_val):
        raise NotImplementedError
    
    @abc.abstractmethod
    def save_model(self):
        raise NotImplementedError


class LGBAttacker(AbstractAttacker):
    def __init__(self, seed=42):
        # Cập nhật hyperparameters để khớp với target model (LEE.lgb)
        # Target model sử dụng: num_leaves=15, learning_rate=0.01, max_depth=7, 
        # lambda_l1=0.1, lambda_l2=0.1, min_data_in_leaf=30
        self.lgb_params = {
            "boosting_type" : "gbdt",
            "objective" : "binary",
            "learning_rate" : 0.01,  # Giảm từ 0.05 xuống 0.01 để khớp target
            "num_leaves": 15,  # Giảm từ 2048 xuống 15 để tránh overfitting và khớp target
            "max_depth" : 7,  # Giảm từ 15 xuống 7 để khớp target
            "min_data_in_leaf": 30,  # Khớp với min_data_in_leaf của target
            "lambda_l1": 0.1,  # Thêm L1 regularization để khớp target
            "lambda_l2": 0.1,  # Thêm L2 regularization để khớp target
            "feature_fraction": 0.8,  # Thêm feature_fraction để khớp target
            "bagging_fraction": 0.8,  # Thêm bagging_fraction để khớp target
            "bagging_freq": 5,  # Thêm bagging_freq để khớp target
            "force_row_wise": True,  # Thêm force_row_wise để khớp target
            "verbose": -1,
            "seed": seed
        }
        self.model = None

    def train_model(self, X, y, X_val, y_val, boosting_rounds=2000, early_stopping=100):
        # Tính scale_pos_weight để xử lý class imbalance
        train_label_counts = np.bincount(y)
        num_negative = train_label_counts[0] if len(train_label_counts) > 0 else 0
        num_positive = train_label_counts[1] if len(train_label_counts) > 1 else 0
        
        if num_positive > 0 and num_negative > 0:
            scale_pos_weight = num_negative / num_positive
            self.lgb_params['scale_pos_weight'] = scale_pos_weight
            print(f"   📊 Class distribution: {num_negative} negative, {num_positive} positive")
            print(f"   📊 scale_pos_weight = {scale_pos_weight:.4f}")
        
        train_data = lgb.Dataset(X, label=y)
        self.val_data = lgb.Dataset(X_val, y_val)
        # LightGBM mới dùng callbacks cho early stopping và logging
        callbacks = [
            lgb.log_evaluation(period=0),  # period=0 để không log
            lgb.early_stopping(stopping_rounds=early_stopping)  # Early stopping
        ]
        self.model = lgb.train(
            self.lgb_params, 
            train_data,
            num_boost_round=boosting_rounds,
            valid_sets=[self.val_data],
            callbacks=callbacks
        )                
    
    def __call__(self, X):
        return self.model.predict(X)

    def save_model(self, path):
        self.model.save_model(path+".txt")


class KerasAttacker(AbstractAttacker):
    def __init__(self, early_stopping=30, seed=42, mc=False, input_shape=(2381,)):

        self.model = create_dnn(seed=seed, input_shape=input_shape, mc=mc)
        self.checkpoint_filepath = '/tmp/checkpoint.weights.h5'
        self.model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_filepath,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)

        self.early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping)

    def train_model(self, X, y, X_val, y_val, num_epochs):
        
        self.model.fit(X, y,
            batch_size=128, 
            epochs=num_epochs, 
            validation_data=(X_val, y_val),
            callbacks=[self.model_checkpoint_callback, self.early_stopping])   

        # Load the best weights after training
        self.model.load_weights(self.checkpoint_filepath)       
    
    def __call__(self, X):        
        return self.model.predict(X)

    def save_model(self, path):
        self.model.save(path+".h5")


class CNNAttacker(AbstractAttacker):
    def __init__(self, early_stopping=30, seed=42, mc=False, input_shape=(2381, 1)):
        """
        CNN Attacker sử dụng architecture CNN đơn giản.
        
        Args:
            early_stopping: Patience cho early stopping
            seed: Random seed
            mc: Monte Carlo dropout flag
            input_shape: Input shape (features, channels) - default: (2381, 1) cho EMBER
        """
        self.input_shape = input_shape
        self.model = create_cnn(seed=seed, input_shape=input_shape, mc=mc)
        self.checkpoint_filepath = '/tmp/checkpoint_cnn.weights.h5'
        self.model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_filepath,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)

        self.early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping)

    def train_model(self, X, y, X_val, y_val, num_epochs):
        """
        Train CNN model. Input X sẽ được reshape thành (n_samples, n_features, 1) nếu cần.
        """
        # Reshape X nếu cần (nếu X là 2D, cần reshape thành 3D cho Conv1D)
        if len(X.shape) == 2:
            # X có shape (n_samples, n_features) -> reshape thành (n_samples, n_features, 1)
            X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))
        else:
            X_reshaped = X
            
        if len(X_val.shape) == 2:
            X_val_reshaped = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
        else:
            X_val_reshaped = X_val
        
        self.model.fit(X_reshaped, y,
            batch_size=128, 
            epochs=num_epochs, 
            validation_data=(X_val_reshaped, y_val),
            callbacks=[self.model_checkpoint_callback, self.early_stopping])   

        # Load the best weights after training
        self.model.load_weights(self.checkpoint_filepath)       
    
    def __call__(self, X):
        """
        Predict với CNN model. Input X sẽ được reshape nếu cần.
        """
        # Reshape X nếu cần
        if len(X.shape) == 2:
            X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))
        else:
            X_reshaped = X
        return self.model.predict(X_reshaped)

    def save_model(self, path):
        self.model.save(path+".h5")

class KerasDualAttacker(AbstractAttacker):
    def __init__(self, early_stopping=30, seed=42, mc=False, input_shape=(2381,)):

        self.model = create_dnn2(seed=seed, mc=mc, input_shape=input_shape)

        # Keras yêu cầu filepath phải kết thúc bằng .weights.h5 khi save_weights_only=True
        self.checkpoint_filepath = '/tmp/checkpoint2.weights.h5'
        self.model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_filepath,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)

        self.early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping)

    def train_model(self, X, y, y_true, X_val, y_val, y_val_true, num_epochs):
        
        self.model.fit((X, y_true), y,
            batch_size=128, 
            epochs=num_epochs, 
            validation_data=((X_val, y_val_true), y_val),
            callbacks=[self.model_checkpoint_callback, self.early_stopping])  
        
        self.model.load_weights(self.checkpoint_filepath)          
    
    def __call__(self, X, y_true):
        return self.model.predict((X, y_true))

    def save_model(self, path):
        self.model.save(path+".h5")


class SVMAttacker(AbstractAttacker):
    def __init__(self, seed=42, max_iter=1000):
        self.model = svm.SVC(C=10., 
                    kernel='linear',
                    max_iter=max_iter, 
                    random_state=seed, 
                    probability=True)

    def train_model(self, X, y):
        self.model.fit(X, y)
        print(f"SVM was fitted properly: {self.model.fit_status_}")

    def __call__(self, X):
        return self.model.predict_proba(X)[:, 1]

    def save_model(self, path):
        joblib.dump(self.model, path+".joblib")


class KNNAttacker(AbstractAttacker):
    def __init__(self, seed=42, n_neighbors=5, weights='uniform', metric='euclidean'):
        """
        KNN Attacker sử dụng sklearn KNeighborsClassifier.
        
        Args:
            seed: Random seed (không được sử dụng trực tiếp trong KNN, nhưng giữ cho consistency)
            n_neighbors: Số neighbors (default: 5)
            weights: Cách tính trọng số - 'uniform' hoặc 'distance' (default: 'uniform')
            metric: Metric để tính khoảng cách - 'euclidean', 'manhattan', etc. (default: 'euclidean')
        """
        from sklearn.neighbors import KNeighborsClassifier
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            n_jobs=-1  # Sử dụng tất cả CPU cores
        )
        self.seed = seed  # Lưu lại cho reference

    def train_model(self, X, y, X_val=None, y_val=None):
        """
        Train KNN model. KNN không cần validation set cho training (không có training phase).
        Validation set có thể được dùng để chọn hyperparameters, nhưng ở đây đơn giản hóa.
        """
        self.model.fit(X, y)
        print(f"KNN was fitted with {len(X)} samples")

    def __call__(self, X):
        """
        Predict probabilities với KNN model.
        """
        # KNN trả về probabilities cho cả 2 classes, chỉ cần class 1 (malware)
        return self.model.predict_proba(X)[:, 1]

    def save_model(self, path):
        """
        Save KNN model bằng joblib.
        """
        joblib.dump(self.model, path+".pkl")


class XGBoostAttacker(AbstractAttacker):
    def __init__(self, seed=42):
        """
        XGBoost Attacker sử dụng XGBoost library.
        
        Args:
            seed: Random seed
        """
        if xgb is None:
            raise ImportError("xgboost package is required. Install it with: pip install xgboost")
        
        self.xgb_params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",  # GPU or CPU hist
            "max_depth": 8,
            "eta": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": seed,
            "verbosity": 0
        }
        self.model = None
        self.seed = seed

    def train_model(self, X, y, X_val, y_val, boosting_rounds=200, early_stopping=20):
        """
        Train XGBoost model.
        
        Args:
            X: Training features
            y: Training labels
            X_val: Validation features
            y_val: Validation labels
            boosting_rounds: Maximum number of boosting rounds
            early_stopping: Early stopping rounds
        """
        # Tính scale_pos_weight để xử lý class imbalance
        train_label_counts = np.bincount(y.astype(int))
        num_negative = train_label_counts[0] if len(train_label_counts) > 0 else 0
        num_positive = train_label_counts[1] if len(train_label_counts) > 1 else 0
        
        if num_positive > 0 and num_negative > 0:
            scale_pos_weight = num_negative / num_positive
            self.xgb_params['scale_pos_weight'] = scale_pos_weight
            print(f"   📊 Class distribution: {num_negative} negative, {num_positive} positive")
            print(f"   📊 scale_pos_weight = {scale_pos_weight:.4f}")
        
        # Tạo DMatrix cho train và validation
        dtrain = xgb.DMatrix(X, label=y)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        watchlist = [(dtrain, "train"), (dval, "valid")]
        
        self.model = xgb.train(
            params=self.xgb_params,
            dtrain=dtrain,
            num_boost_round=boosting_rounds,
            evals=watchlist,
            early_stopping_rounds=early_stopping,
            verbose_eval=False
        )
    
    def __call__(self, X):
        """
        Predict probabilities với XGBoost model.
        """
        dtest = xgb.DMatrix(X)
        # XGBoost predict trả về probability của class 1 (malware)
        return self.model.predict(dtest)

    def save_model(self, path):
        """
        Save XGBoost model dưới dạng .json (giống target model).
        """
        self.model.save_model(path + ".json")


class TabNetAttacker(AbstractAttacker):
    def __init__(self, seed=42, device_name=None):
        """
        TabNet Attacker sử dụng pytorch_tabnet library.
        
        Args:
            seed: Random seed
            device_name: Device name ('cuda' or 'cpu'), None để auto-detect
        """
        if TabNetClassifier is None:
            raise ImportError("pytorch_tabnet package is required. Install it with: pip install pytorch-tabnet")
        
        if device_name is None:
            if torch is not None and torch.cuda.is_available():
                device_name = "cuda"
            else:
                device_name = "cpu"
        
        self.device_name = device_name
        self.seed = seed
        
        # TabNet hyperparameters từ notebook
        self.tabnet_params = {
            "n_d": 24,
            "n_a": 24,
            "n_steps": 3,
            "gamma": 1.5,
            "n_independent": 1,
            "n_shared": 1,
            "momentum": 0.02,
        }
        
        self.model = None

    def train_model(self, X, y, X_val, y_val, max_epochs=100, patience=10, batch_size=1024):
        """
        Train TabNet model.
        
        Args:
            X: Training features
            y: Training labels
            X_val: Validation features
            y_val: Validation labels
            max_epochs: Maximum number of epochs
            patience: Early stopping patience
            batch_size: Batch size for training
        """
        # Khởi tạo TabNetClassifier
        self.model = TabNetClassifier(
            n_d=self.tabnet_params["n_d"],
            n_a=self.tabnet_params["n_a"],
            n_steps=self.tabnet_params["n_steps"],
            gamma=self.tabnet_params["gamma"],
            n_independent=self.tabnet_params["n_independent"],
            n_shared=self.tabnet_params["n_shared"],
            momentum=self.tabnet_params["momentum"],
            device_name=self.device_name,
        )
        
        # TabNet sử dụng numpy arrays trực tiếp
        # Fit model với early stopping.
        # LƯU Ý: Với một số trường hợp cực lệch class (như SOMLAP dưới oracle TABNET),
        # pytorch-tabnet + sklearn metrics (AUC/logloss) có thể lỗi nếu y_val chỉ có 1 class.
        # Để tránh crash:
        #  - Thử train với eval_set + eval_metric="logloss"
        #  - Nếu lỗi (single-class validation), fallback train KHÔNG eval_set/metric
        try:
            self.model.fit(
                X, y,
                eval_set=[(X_val, y_val)],
                eval_name=["valid"],
                eval_metric=["logloss"],
                max_epochs=max_epochs,
                patience=patience,
                batch_size=batch_size,
                virtual_batch_size=batch_size // 4,
                num_workers=0,
                drop_last=False,
            )
        except Exception as e:
            print(f"⚠️  Warning: TabNet fit with validation metrics failed: {type(e).__name__}: {e}")
            print("   💡 Fallback: train TabNet WITHOUT eval_set / eval_metric (no early stopping).")
            self.model.fit(
                X, y,
                max_epochs=max_epochs,
                patience=patience,
                batch_size=batch_size,
                virtual_batch_size=batch_size // 4,
                num_workers=0,
                drop_last=False,
            )
    
    def __call__(self, X):
        """
        Predict probabilities với TabNet model.
        """
        # TabNet predict_proba trả về probabilities cho cả 2 classes
        # Lấy probability của class 1 (malware)
        proba = self.model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        else:
            # Nếu chỉ có 1 class, trả về trực tiếp
            return proba.flatten()

    def save_model(self, path):
        """
        Save TabNet model bằng save_model() method (tạo .zip file).
        """
        self.model.save_model(path)
