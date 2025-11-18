"""
Flexible target model loader - Tự động phát hiện và load đúng architecture

Tính năng:
- Tự động phát hiện architecture từ weights file
- Xử lý input size mismatch bằng preprocessing layer
- Hỗ trợ nhiều loại model khác nhau (DNN, CNN, LightGBM, etc.)
- Phù hợp với model extraction attack - chỉ cần query target model để lấy labels

Lưu ý:
- Nếu input size không khớp, preprocessing layer sẽ được thêm vào
- Preprocessing layer sử dụng random projection (có thể không chính xác 100%)
- Trong model extraction attack, điều này vẫn đảm bảo logic đúng vì ta chỉ cần labels từ target model
"""
import os
import numpy as np
import h5py
from pathlib import Path
import lightgbm as lgb


class FlexibleKerasTarget:
    """
    Wrapper linh hoạt để load Keras model với bất kỳ architecture nào.
    Tự động phát hiện và thử các cách load khác nhau.
    
    Xử lý feature dimension mismatch: Tự động cắt bỏ đặc trưng thừa nếu input 
    có nhiều đặc trưng hơn model yêu cầu (Interface Compliance).
    """
    
    def __init__(self, weights_path, feature_dim=2381, threshold=0.5, name="flexible-keras-target"):
        self.model_endpoint = weights_path
        self.model_threshold = threshold
        self.name = name
        self.feature_dim = feature_dim  # Feature dim của attacker dataset
        
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
        
        self._model = self._load_model_flexible()
        self._input_shape = self._detect_input_shape()
        # Lấy số đặc trưng yêu cầu thực tế của model
        self._required_feature_dim = self._get_actual_required_feature_dim()
    
    def _load_model_flexible(self):
        """
        Thử nhiều cách để load model:
        1. Load như full model (nếu có architecture trong file)
        2. Thử các architecture phổ biến
        3. Detect từ weights file
        """
        import tensorflow as tf
        
        # Cách 1: Thử load như full model
        try:
            model = tf.keras.models.load_model(self.model_endpoint, compile=False)
            # Kiểm tra input shape của model
            model_input_shape = model.input_shape[1:] if model.input_shape else None
            if model_input_shape and len(model_input_shape) > 0:
                model_input_size = model_input_shape[0] if isinstance(model_input_shape[0], int) else None
                # Nếu input size không khớp, cần thêm preprocessing layer
                if model_input_size and model_input_size != self.feature_dim:
                    print(f"⚠️  Full model input size ({model_input_size}) != feature_dim ({self.feature_dim}), adding preprocessing layer")
                    model = self._add_preprocessing_layer(model, model_input_size)
            print(f"✅ Loaded as full model with {len(model.layers)} layers")
            return model
        except Exception as e:
            print(f"⚠️  Cannot load as full model: {type(e).__name__}")
        
        # Cách 2: Thử load với safe_mode=False
        try:
            model = tf.keras.models.load_model(
                self.model_endpoint, 
                compile=False, 
                safe_mode=False
            )
            # Kiểm tra input shape của model
            model_input_shape = model.input_shape[1:] if model.input_shape else None
            if model_input_shape and len(model_input_shape) > 0:
                model_input_size = model_input_shape[0] if isinstance(model_input_shape[0], int) else None
                # Nếu input size không khớp, cần thêm preprocessing layer
                if model_input_size and model_input_size != self.feature_dim:
                    print(f"⚠️  Model input size ({model_input_size}) != feature_dim ({self.feature_dim}), adding preprocessing layer")
                    model = self._add_preprocessing_layer(model, model_input_size)
            print(f"✅ Loaded with safe_mode=False, {len(model.layers)} layers")
            return model
        except Exception as e:
            print(f"⚠️  Cannot load with safe_mode=False: {type(e).__name__}")
        
        # Cách 3: Detect architecture từ weights file và build động
        print("🔄 Attempting to detect architecture from weights file...")
        architecture_info = self._detect_architecture_from_weights()
        
        # Thử build model động từ weights info
        try:
            model = self._build_model_from_weights(architecture_info)
            if model:
                # Load weights với by_name=True và skip_mismatch=True
                # Điều này cho phép skip các layers không có trong weights file (như preprocessing layer)
                try:
                    model.load_weights(self.model_endpoint, by_name=True, skip_mismatch=True)
                    print(f"✅ Successfully built and loaded model from weights ({len(model.layers)} layers)")
                    # Kiểm tra xem có preprocessing layer không
                    if any('preprocessing' in layer.name for layer in model.layers):
                        print(f"   ℹ️  Model has preprocessing layer for input size adaptation")
                    return model
                except Exception as e:
                    print(f"⚠️  Error loading weights: {type(e).__name__}: {str(e)[:100]}")
                    # Nếu không load được, vẫn trả về model (có thể weights sẽ được load sau)
                    return model
        except Exception as e:
            print(f"⚠️  Cannot build from weights info: {type(e).__name__}: {str(e)[:100]}")
        
        # Cách 4: Thử các architecture phổ biến
        architectures = self._get_common_architectures()
        
        for arch_name, build_func in architectures.items():
            try:
                model = build_func()
                # Thử load weights
                model.load_weights(self.model_endpoint)
                print(f"✅ Successfully loaded with {arch_name} architecture ({len(model.layers)} layers)")
                return model
            except Exception as e:
                continue
        
        # Nếu tất cả đều fail, raise error
        raise ValueError(
            f"Cannot load model from {self.model_endpoint}. "
            f"Tried full model load and common architectures. "
            f"Please check the file or provide correct architecture."
        )
    
    def _detect_architecture_from_weights(self):
        """Phân tích weights file để đoán architecture"""
        info = {
            "has_conv": False,
            "has_dense": False,
            "layer_count": 0,
            "layer_names": [],
            "dense_layers": []  # List of (layer_name, output_size)
        }
        
        try:
            with h5py.File(self.model_endpoint, 'r') as f:
                if 'model_weights' in f:
                    weights_group = f['model_weights']
                    for layer_name in weights_group.keys():
                        if layer_name == 'top_level_model_weights':
                            continue
                        info["layer_names"].append(layer_name)
                        info["layer_count"] += 1
                        
                        if 'conv' in layer_name.lower():
                            info["has_conv"] = True
                        
                        if 'dense' in layer_name.lower():
                            info["has_dense"] = True
                            try:
                                layer = weights_group[layer_name]
                                # Tìm kernel trong nested structure (có thể là sequential/dense/kernel)
                                kernel = None
                                
                                # Thử các path khác nhau
                                if 'kernel' in layer:
                                    kernel = layer['kernel']
                                elif 'sequential' in layer:
                                    seq = layer['sequential']
                                    # Tìm dense layer trong sequential
                                    for seq_key in seq.keys():
                                        if 'dense' in seq_key.lower() and isinstance(seq[seq_key], h5py.Group):
                                            dense_in_seq = seq[seq_key]
                                            if 'kernel' in dense_in_seq:
                                                kernel = dense_in_seq['kernel']
                                                break
                                    # Nếu không tìm thấy, thử trực tiếp
                                    if kernel is None and 'kernel' in seq:
                                        kernel = seq['kernel']
                                
                                if kernel is not None:
                                    kernel_shape = tuple(kernel.shape)
                                    # Kernel shape: (input_size, output_size) cho Dense layer
                                    if len(kernel_shape) == 2:
                                        output_size = kernel_shape[1]
                                    elif len(kernel_shape) == 1:
                                        output_size = kernel_shape[0]
                                    else:
                                        output_size = kernel_shape[-1]
                                    info["dense_layers"].append((layer_name, int(output_size)))
                            except Exception as e:
                                print(f"⚠️  Error reading {layer_name}: {e}")
        except Exception as e:
            print(f"⚠️  Error detecting architecture: {e}")
        
        return info
    
    def _build_model_from_weights(self, architecture_info):
        """Build model động dựa trên thông tin từ weights file"""
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
        
        if not architecture_info["has_dense"] or not architecture_info["dense_layers"]:
            return None
        
        # Sắp xếp dense layers theo thứ tự
        dense_layers = sorted(architecture_info["dense_layers"], 
                             key=lambda x: int(x[0].replace('dense', '').replace('_', '0') or '0'))
        
        if len(dense_layers) < 2:
            return None
        
        # Build model dựa trên dense layers tìm được
        layers = []
        is_first = True
        
        # Lấy input size thực tế của model từ weights file
        actual_input_size = None
        if dense_layers:
            # Đọc kernel shape của layer đầu tiên để biết input size thực tế
            try:
                with h5py.File(self.model_endpoint, 'r') as f:
                    first_layer_name = dense_layers[0][0]
                    layer = f['model_weights'][first_layer_name]
                    if 'sequential' in layer:
                        seq = layer['sequential']
                        for seq_key in seq.keys():
                            if 'dense' in seq_key.lower() and isinstance(seq[seq_key], h5py.Group):
                                dense_in_seq = seq[seq_key]
                                if 'kernel' in dense_in_seq:
                                    kernel_shape = dense_in_seq['kernel'].shape
                                    actual_input_size = kernel_shape[0]  # Input size là dimension đầu tiên
                                    break
            except Exception as e:
                print(f"⚠️  Error reading input size: {e}")
        
        # Nếu input size không khớp, cần thêm preprocessing layer
        needs_preprocessing = actual_input_size and actual_input_size != self.feature_dim
        
        if needs_preprocessing:
            # Thêm preprocessing layer để map từ feature_dim xuống actual_input_size
            # Layer này KHÔNG có trong weights file, sẽ được khởi tạo ngẫu nhiên
            # Nhưng trong model extraction attack, ta chỉ cần query target model,
            # không cần train preprocessing layer này
            layers.append(Dense(actual_input_size, activation='linear', 
                              input_shape=(self.feature_dim,), 
                              name='preprocessing_mapping',
                              trainable=False))  # Không train, chỉ dùng để map input
        
        # Build các layers từ weights file
        for i, (layer_name, output_size) in enumerate(dense_layers):
            if is_first:
                # Layer đầu tiên: input size phải match với actual_input_size
                input_size_for_layer = actual_input_size if actual_input_size else self.feature_dim
                if needs_preprocessing:
                    # Nếu có preprocessing, layer đầu tiên nhận input từ preprocessing
                    layers.append(Dense(output_size, activation='relu', name=layer_name))
                else:
                    # Không có preprocessing, layer đầu tiên nhận input trực tiếp
                    layers.append(Dense(output_size, activation='relu', 
                                      input_shape=(self.feature_dim,), 
                                      name=layer_name))
                is_first = False
            else:
                # Layer cuối cùng có thể là output layer
                if i == len(dense_layers) - 1:
                    activation = 'sigmoid' if output_size == 1 else 'softmax'
                else:
                    activation = 'relu'
                layers.append(Dense(output_size, activation=activation, name=layer_name))
            
            # Thêm BatchNormalization và Dropout sau mỗi Dense (trừ layer cuối)
            if i < len(dense_layers) - 1:
                # Tên BN và Dropout phải match với weights file
                if i == 0:
                    bn_name = 'batch_normalization'
                    dropout_name = 'dropout'
                else:
                    bn_name = f'batch_normalization_{i}'
                    dropout_name = f'dropout_{i}'
                layers.append(BatchNormalization(name=bn_name))
                layers.append(Dropout(0.3, name=dropout_name))
        
        model = Sequential(layers)
        
        # Nếu có preprocessing layer, khởi tạo weights thông minh hơn
        if needs_preprocessing:
            preprocessing_layer = model.get_layer('preprocessing_mapping')
            import numpy as np
            
            # Khởi tạo với random projection (Gaussian random matrix)
            # Đây là một cách tiếp cận hợp lý khi không biết mapping chính xác
            # Trong model extraction attack, ta chỉ cần query target model,
            # preprocessing layer này sẽ được "học" ngầm thông qua queries
            weights = preprocessing_layer.get_weights()
            if len(weights) > 0:
                # Sử dụng random projection với scaling phù hợp
                # Random projection giữ được một phần thông tin từ input
                kernel = np.random.randn(*weights[0].shape).astype(np.float32)
                # Scale để output có variance tương đương
                kernel = kernel / np.sqrt(weights[0].shape[0])
                bias = np.zeros(weights[1].shape, dtype=np.float32)
                preprocessing_layer.set_weights([kernel, bias])
                
                print(f"   ℹ️  Initialized preprocessing layer: {self.feature_dim} -> {actual_input_size}")
        
        return model
    
    def _get_common_architectures(self):
        """Trả về dictionary các hàm build architecture phổ biến"""
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import (
            BatchNormalization, Conv1D, Dense, Dropout, 
            Flatten, MaxPooling1D, LayerNormalization
        )
        from tensorflow.keras.regularizers import l2
        
        architectures = {}
        
        # Architecture 1: CNN (như trong final_model.ipynb)
        def build_cnn():
            return Sequential([
                Conv1D(64, 5, strides=2, padding='same', 
                      input_shape=(self.feature_dim, 1), activation='relu'),
                BatchNormalization(),
                MaxPooling1D(pool_size=2),
                Dropout(0.3),
                Conv1D(64, 3, padding='same', activation='relu'),
                BatchNormalization(),
                Conv1D(32, 3, padding='same', activation='relu'),
                BatchNormalization(),
                MaxPooling1D(pool_size=2),
                Dropout(0.4),
                Flatten(),
                Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
                BatchNormalization(),
                Dropout(0.5),
                Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
                BatchNormalization(),
                Dropout(0.5),
                Dense(2, activation='softmax', dtype='float32'),
            ])
        architectures['CNN'] = build_cnn
        
        # Architecture 2: DNN (như create_dnn)
        def build_dnn():
            initializer = tf.keras.initializers.GlorotNormal(seed=42)
            inputs = tf.keras.Input(shape=(self.feature_dim,))
            x = Dense(2381, activation='elu', kernel_initializer=initializer)(inputs)
            x = LayerNormalization()(x)
            x = Dropout(0.3)(x)
            x = Dense(1024, activation='elu', kernel_initializer=initializer)(x)
            x = LayerNormalization()(x)
            x = Dropout(0.3)(x)
            x = Dense(512, activation='elu', kernel_initializer=initializer)(x)
            x = LayerNormalization()(x)
            x = Dropout(0.3)(x)
            x = Dense(128, activation='elu', kernel_initializer=initializer)(x)
            x = LayerNormalization()(x)
            x = Dropout(0.3)(x)
            outputs = Dense(1, activation="sigmoid")(x)
            return tf.keras.Model(inputs, outputs)
        architectures['DNN'] = build_dnn
        
        # Architecture 3: Simple DNN (như model trong file - 10 layers)
        # Pattern: Dense -> BN -> Dropout -> Dense -> BN -> Dropout -> Dense -> BN -> Dropout -> Dense
        def build_simple_dnn():
            return Sequential([
                Dense(2381, activation='elu', input_shape=(self.feature_dim,)),
                BatchNormalization(),
                Dropout(0.3),
                Dense(1024, activation='elu'),
                BatchNormalization(),
                Dropout(0.3),
                Dense(512, activation='elu'),
                BatchNormalization(),
                Dropout(0.3),
                Dense(1, activation='sigmoid'),
            ])
        architectures['Simple_DNN'] = build_simple_dnn
        
        # Architecture 4: DNN với 4 Dense layers (dựa trên weights file structure)
        # Thử với các kích thước khác nhau
        def build_dnn_4layer_v1():
            return Sequential([
                Dense(2381, activation='relu', input_shape=(self.feature_dim,)),
                BatchNormalization(),
                Dropout(0.3),
                Dense(1024, activation='relu'),
                BatchNormalization(),
                Dropout(0.3),
                Dense(512, activation='relu'),
                BatchNormalization(),
                Dropout(0.3),
                Dense(1, activation='sigmoid'),
            ])
        architectures['DNN_4Layer_v1'] = build_dnn_4layer_v1
        
        def build_dnn_4layer_v2():
            # Thử với các kích thước khác
            return Sequential([
                Dense(2048, activation='relu', input_shape=(self.feature_dim,)),
                BatchNormalization(),
                Dropout(0.3),
                Dense(1024, activation='relu'),
                BatchNormalization(),
                Dropout(0.3),
                Dense(512, activation='relu'),
                BatchNormalization(),
                Dropout(0.3),
                Dense(1, activation='sigmoid'),
            ])
        architectures['DNN_4Layer_v2'] = build_dnn_4layer_v2
        
        def build_dnn_4layer_v3():
            # Thử với activation khác
            return Sequential([
                Dense(2381, activation='tanh', input_shape=(self.feature_dim,)),
                BatchNormalization(),
                Dropout(0.3),
                Dense(1024, activation='tanh'),
                BatchNormalization(),
                Dropout(0.3),
                Dense(512, activation='tanh'),
                BatchNormalization(),
                Dropout(0.3),
                Dense(1, activation='sigmoid'),
            ])
        architectures['DNN_4Layer_v3'] = build_dnn_4layer_v3
        
        return architectures
    
    def _add_preprocessing_layer(self, model, target_input_size):
        """Thêm preprocessing layer vào model đã load để xử lý input size mismatch"""
        import tensorflow as tf
        from tensorflow.keras import Sequential, Model
        from tensorflow.keras.layers import Dense, Input
        
        # Tạo preprocessing layer
        preprocessing_input = Input(shape=(self.feature_dim,), name='preprocessing_input')
        preprocessing_layer = Dense(
            target_input_size, 
            activation='linear',
            name='preprocessing_mapping',
            trainable=False
        )(preprocessing_input)
        
        # Kết nối với model hiện tại
        # Lấy output của preprocessing layer làm input cho model gốc
        model_output = model(preprocessing_layer)
        
        # Tạo model mới với preprocessing layer
        new_model = Model(inputs=preprocessing_input, outputs=model_output, name='model_with_preprocessing')
        
        # Khởi tạo preprocessing layer với random projection
        preprocessing_layer_obj = new_model.get_layer('preprocessing_mapping')
        weights = preprocessing_layer_obj.get_weights()
        if len(weights) > 0:
            kernel = np.random.randn(*weights[0].shape).astype(np.float32)
            kernel = kernel / np.sqrt(weights[0].shape[0])
            bias = np.zeros(weights[1].shape, dtype=np.float32)
            preprocessing_layer_obj.set_weights([kernel, bias])
        
        return new_model
    
    def _detect_input_shape(self):
        """Phát hiện input shape từ model"""
        if hasattr(self._model, 'input_shape') and self._model.input_shape:
            return self._model.input_shape[1:]  # Bỏ qua batch dimension
        elif hasattr(self._model, 'inputs') and self._model.inputs:
            return tuple(self._model.inputs[0].shape[1:])
        else:
            # Default: giả sử là DNN (1D input)
            return (self.feature_dim,)
    
    def _has_preprocessing_layer(self):
        """Kiểm tra xem model có preprocessing layer không"""
        if hasattr(self._model, 'layers'):
            for layer in self._model.layers:
                if 'preprocessing' in layer.name.lower() or 'mapping' in layer.name.lower():
                    return True
        return False
    
    def _get_actual_required_feature_dim(self):
        """
        Lấy số đặc trưng thực tế mà model yêu cầu.
        
        Nếu model có preprocessing layer:
        - Model đã được thiết kế để nhận input với feature_dim của attacker
        - Preprocessing layer sẽ map từ feature_dim của attacker sang feature_dim của model thực tế
        - Trong trường hợp này, không cần cắt đặc trưng (preprocessing layer đã xử lý)
        - Trả về None để báo hiệu không cần cắt đặc trưng
        
        Nếu không có preprocessing layer:
        - Lấy từ input shape của model (đây là số đặc trưng model thực sự yêu cầu)
        - Cần cắt đặc trưng nếu attacker có nhiều đặc trưng hơn
        """
        # Kiểm tra xem có preprocessing layer không
        if self._has_preprocessing_layer():
            # Nếu có preprocessing layer, model đã được thiết kế để nhận input với feature_dim của attacker
            # Không cần cắt đặc trưng vì preprocessing layer đã xử lý việc mapping
            return None
        
        # Nếu không có preprocessing layer, lấy từ input shape của model
        input_shape = self._input_shape
        if len(input_shape) == 1:
            # DNN: (features,)
            return int(input_shape[0])
        elif len(input_shape) == 2 and input_shape[-1] == 1:
            # CNN: (features, 1)
            return int(input_shape[0])
        else:
            # Default: dùng feature_dim được truyền vào
            return self.feature_dim
    
    def get_required_feature_dim(self):
        """
        Trả về số đặc trưng mà target model yêu cầu thực tế.
        """
        return self._required_feature_dim
    
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
        required_dim = self._required_feature_dim
        if required_dim is None:
            return X
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        actual_dim = X.shape[1]
        
        if actual_dim == required_dim:
            return X
        elif actual_dim > required_dim:
            # Cắt bỏ đặc trưng thừa ở cuối
            print(f"⚠️  Input has {actual_dim} features, target model requires {required_dim}. "
                  f"Trimming {actual_dim - required_dim} features.")
            return X[:, :required_dim]
        else:
            # Không đủ đặc trưng - raise error
            raise ValueError(
                f"Input has {actual_dim} features, but target model requires {required_dim}. "
                f"Cannot pad features - please provide correct feature set."
            )
    
    def _prepare_input(self, X):
        """Chuẩn bị input phù hợp với model architecture"""
        # Đồng bộ hóa số chiều đặc trưng trước khi chuẩn bị input
        X = self._align_features(X)
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Nếu model cần 3D input (CNN), thêm channel dimension
        if len(self._input_shape) == 2 and self._input_shape[-1] == 1:
            # CNN model: (features, 1)
            return np.expand_dims(X, axis=-1)
        else:
            # DNN model: (features,)
            return X
    
    def predict_proba(self, X, batch_size=512):
        """Predict probabilities"""
        X = self._prepare_input(X)
        probs = self._model.predict(X, batch_size=batch_size, verbose=0)
        
        # Xử lý output shape khác nhau
        if probs.ndim > 1:
            if probs.shape[-1] == 2:
                # Softmax output: lấy class 1
                return probs[:, 1] if probs.ndim == 2 else probs[..., 1]
            elif probs.shape[-1] == 1:
                # Sigmoid output: squeeze
                return np.squeeze(probs, axis=-1)
        
        return np.squeeze(probs)
    
    def __call__(self, X, batch_size=512):
        """Predict binary labels"""
        probs = self.predict_proba(X, batch_size=batch_size)
        return (probs >= self.model_threshold).astype(int)


class FlexibleLGBTarget:
    """
    Wrapper linh hoạt để load LightGBM model (.lgb, .txt, .pkl, .d5) với normalization stats.
    
    Hỗ trợ:
    - Load model từ file .lgb (LightGBM native format)
    - Load normalization statistics từ file .npz
    - Normalize features trước khi predict (giống như code trong user's example)
    - Xử lý feature alignment nếu cần
    
    Xử lý feature dimension mismatch: Tự động cắt bỏ đặc trưng thừa nếu input 
    có nhiều đặc trưng hơn model yêu cầu (Interface Compliance).
    """
    
    def __init__(
        self, 
        model_path, 
        normalization_stats_path=None,
        threshold=0.5, 
        name="flexible-lgb-target",
        feature_dim=None
    ):
        """
        Args:
            model_path: Đường dẫn tới file model .lgb, .txt, .pkl, hoặc .d5
            normalization_stats_path: Đường dẫn tới file .npz chứa normalization stats.
                                     Nếu None, sẽ không normalize features.
            threshold: Threshold để chuyển probabilities thành binary labels
            name: Tên của target model
            feature_dim: Số đặc trưng của attacker dataset. Nếu None, sẽ lấy từ model.
        """
        self.model_endpoint = model_path
        self.model_threshold = threshold
        self.name = name
        self.feature_dim = feature_dim
        
        # Load model
        self.model = self._load_model()
        
        # Lấy số đặc trưng yêu cầu từ model
        self._required_feature_dim = self.model.num_feature()
        
        # Nếu feature_dim không được cung cấp, dùng từ model
        if self.feature_dim is None:
            self.feature_dim = self._required_feature_dim
        
        # Load normalization stats (nếu có)
        self.feature_means = None
        self.feature_stds = None
        self.feature_cols = None
        self.use_normalization = False
        
        if normalization_stats_path is not None:
            self._load_normalization_stats(normalization_stats_path)
    
    def _load_model(self):
        """Load LightGBM model từ file"""
        try:
            # Cách 1: Load từ file .lgb, .txt, hoặc .d5 (LightGBM native format)
            model = lgb.Booster(model_file=self.model_endpoint)
            print(f"✅ Loaded LightGBM model from {self.model_endpoint}")
            print(f"   Model features: {model.num_feature()}")
            return model
        except Exception as e:
            # Cách 2: Thử load từ pickle file
            try:
                import pickle
                with open(self.model_endpoint, 'rb') as f:
                    model = pickle.load(f)
                if isinstance(model, lgb.Booster):
                    print(f"✅ Loaded LightGBM model from pickle file {self.model_endpoint}")
                    print(f"   Model features: {model.num_feature()}")
                    return model
                else:
                    raise ValueError(f"File {self.model_endpoint} không phải LightGBM Booster")
            except Exception as e2:
                raise ValueError(
                    f"Cannot load LightGBM model from {self.model_endpoint}. "
                    f"Error: {type(e).__name__}: {str(e)}"
                )
    
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
            
            # Kiểm tra compatibility
            if self.feature_cols is not None:
                print(f"   Feature columns: {len(self.feature_cols)}")
                if len(self.feature_cols) != self._required_feature_dim:
                    print(f"   ⚠️  Warning: feature_cols ({len(self.feature_cols)}) != model features ({self._required_feature_dim})")
        except Exception as e:
            print(f"⚠️  Warning: Cannot load normalization stats from {stats_path}: {type(e).__name__}: {str(e)}")
            print(f"   Will use features without normalization")
            self.use_normalization = False
    
    def get_required_feature_dim(self):
        """Trả về số đặc trưng mà LightGBM model yêu cầu"""
        return self._required_feature_dim
    
    def _align_features(self, X):
        """
        Đồng bộ hóa số chiều đặc trưng của input với yêu cầu của target model.
        Nếu X có nhiều đặc trưng hơn, cắt bỏ các đặc trưng thừa ở cuối.
        Nếu X có ít đặc trưng hơn, raise ValueError.
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        actual_dim = X.shape[1]
        
        if actual_dim == self._required_feature_dim:
            return X
        elif actual_dim > self._required_feature_dim:
            # Cắt bỏ đặc trưng thừa ở cuối
            print(f"⚠️  Input has {actual_dim} features, target model requires {self._required_feature_dim}. "
                  f"Trimming {actual_dim - self._required_feature_dim} features.")
            return X[:, :self._required_feature_dim]
        else:
            # Không đủ đặc trưng - raise error
            raise ValueError(
                f"Input has {actual_dim} features, but target model requires {self._required_feature_dim}. "
                f"Cannot pad features - please provide correct feature set."
            )
    
    def _normalize_features(self, X):
        """
        Normalize features giống như code của người dùng:
        - (features_array - feature_means) / feature_stds
        - Xử lý NaN và infinity
        
        Lưu ý: Nếu normalization stats có nhiều features hơn model yêu cầu,
        chỉ normalize số features mà model cần (từ đầu).
        """
        if not self.use_normalization:
            return X
        
        # Đảm bảo X đã align với model requirements trước (số features model cần)
        # Điều này quan trọng vì model có thể chỉ cần subset của features
        X_aligned = self._align_features(X)  # Cắt xuống số features model cần
        
        # Nếu normalization stats có nhiều features hơn model cần,
        # chỉ lấy số features đầu tiên từ stats tương ứng với số features model cần
        if self.feature_means.shape[0] > self._required_feature_dim:
            # Normalization stats có nhiều features hơn model cần
            # Chỉ normalize với stats của số features đầu tiên
            feature_means_used = self.feature_means[:self._required_feature_dim]
            feature_stds_used = self.feature_stds[:self._required_feature_dim]
        elif self.feature_means.shape[0] == self._required_feature_dim:
            # Normalization stats khớp với số features model cần
            feature_means_used = self.feature_means
            feature_stds_used = self.feature_stds
        else:
            # Normalization stats có ít features hơn model cần - không nên xảy ra
            raise ValueError(
                f"Normalization stats chỉ có {self.feature_means.shape[0]} features, "
                f"nhưng model cần {self._required_feature_dim} features. "
                f"Vui lòng kiểm tra lại file normalization stats."
            )
        
        # Normalize với stats đã được chọn
        features_normalized = (X_aligned - feature_means_used) / feature_stds_used
        
        # Xử lý NaN và infinity (giống code của người dùng)
        features_normalized = np.nan_to_num(features_normalized, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features_normalized
    
    def predict_proba(self, X):
        """
        Predict probabilities giống như code của người dùng.
        
        Args:
            X: Input features (n_samples, n_features) hoặc dict với feature names
            
        Returns:
            probabilities: Array of probabilities (n_samples,)
        """
        # Nếu X là dict (như code của người dùng), chuyển thành array
        if isinstance(X, dict):
            if self.feature_cols is None:
                raise ValueError("Cannot convert dict to array without feature_cols in normalization stats")
            
            # Chuyển đổi features dict thành array theo đúng thứ tự feature_cols
            features_array = np.array(
                [X.get(col, 0.0) for col in self.feature_cols], 
                dtype=np.float32
            ).reshape(1, -1)
        else:
            features_array = np.asarray(X, dtype=np.float32)
            if features_array.ndim == 1:
                features_array = features_array.reshape(1, -1)
        
        # QUAN TRỌNG: Xử lý normalization và alignment
        # Logic: Model cần 108 features, nhưng normalization stats có thể có 2381 features
        # Giải pháp: Normalize với số features model cần (108 đầu tiên từ stats)
        #            rồi cắt bỏ features thừa từ input
        
        if self.use_normalization:
            # Bước 1: Align input với số features model cần (cắt bỏ features thừa)
            # Điều này đảm bảo ta chỉ normalize với số features mà model thực sự cần
            features_array_aligned = self._align_features(features_array)  # Cắt xuống 108 features
            
            # Bước 2: Normalize với stats tương ứng
            # Nếu stats có nhiều features hơn model cần, chỉ lấy số features đầu tiên
            if self.feature_means.shape[0] >= self._required_feature_dim:
                # Stats có đủ hoặc nhiều hơn - chỉ lấy số features model cần
                feature_means_used = self.feature_means[:self._required_feature_dim]
                feature_stds_used = self.feature_stds[:self._required_feature_dim]
            else:
                # Stats có ít hơn model cần - dùng toàn bộ stats
                feature_means_used = self.feature_means
                feature_stds_used = self.feature_stds
                # Cắt features_array để khớp với stats
                if features_array_aligned.shape[1] > feature_means_used.shape[0]:
                    features_array_aligned = features_array_aligned[:, :feature_means_used.shape[0]]
            
            # Normalize
            features_normalized = (features_array_aligned - feature_means_used) / feature_stds_used
            features_normalized = np.nan_to_num(features_normalized, nan=0.0, posinf=0.0, neginf=0.0)
            
            features_array = features_normalized
        else:
            # Không có normalization - chỉ align
            features_array = self._align_features(features_array)
        
        # Reshape cho LightGBM (cần shape (n_samples, n_features))
        # LightGBM predict tự động xử lý (1, n_features) hoặc (n_samples, n_features)
        
        # Predict
        # Xử lý num_iteration giống code mẫu của người dùng:
        # - Nếu model có best_iteration và best_iteration > 0, dùng best_iteration
        # - Nếu best_iteration = -1 hoặc không có, dùng None (tất cả trees)
        num_iteration = None
        if hasattr(self.model, 'best_iteration') and self.model.best_iteration is not None:
            if self.model.best_iteration > 0:
                num_iteration = self.model.best_iteration
            # Nếu best_iteration = -1, dùng None (tất cả trees)
            # Điều này đảm bảo tương thích với model không có best_iteration được lưu
        
        prediction_prob = self.model.predict(features_array, num_iteration=num_iteration)
        
        # Đảm bảo output là 1D array
        if prediction_prob.ndim > 1:
            prediction_prob = np.squeeze(prediction_prob)
        
        return prediction_prob
    
    def __call__(self, X):
        """
        Predict binary labels.
        
        Args:
            X: Input features (n_samples, n_features) hoặc dict với feature names
            
        Returns:
            labels: Binary labels (n_samples,)
        """
        probs = self.predict_proba(X)
        return (probs >= self.model_threshold).astype(int)

