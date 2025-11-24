import lightgbm as lgb
from ..models.dnn import create_dnn, create_dnn2
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
