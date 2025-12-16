import tensorflow as tf
from tensorflow.python.keras.engine import training


def create_cnn(seed=42, input_shape=(2381, 1), mc=False):
    """
    Tạo CNN model đơn giản cho surrogate architecture.
    
    Args:
        seed: Random seed
        input_shape: Input shape (features, channels) - default: (2381, 1) cho EMBER
        mc: Monte Carlo dropout flag
    
    Returns:
        Compiled Keras CNN model
    """
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import (
        Conv1D, BatchNormalization, MaxPooling1D, 
        Dropout, Flatten, Dense
    )
    from tensorflow.keras.regularizers import l2
    
    initializer = tf.keras.initializers.GlorotNormal(seed=seed)
    
    model = Sequential([
        # Conv Block 1
        Conv1D(32, 5, strides=2, padding='same', 
               input_shape=input_shape, activation='relu',
               kernel_initializer=initializer),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3, seed=seed),
        
        # Conv Block 2
        Conv1D(64, 3, padding='same', activation='relu',
               kernel_initializer=initializer),
        BatchNormalization(),
        Conv1D(32, 3, padding='same', activation='relu',
               kernel_initializer=initializer),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.4, seed=seed),
        
        # Flatten và Dense layers
        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=l2(0.01),
              kernel_initializer=initializer),
        BatchNormalization(),
        Dropout(0.5, seed=seed),
        
        Dense(64, activation='relu', kernel_regularizer=l2(0.01),
              kernel_initializer=initializer),
        BatchNormalization(),
        Dropout(0.5, seed=seed),
        
        # Output layer
        Dense(1, activation='sigmoid')
    ])
    
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    optim = tf.keras.optimizers.Adam()
    
    model.compile(optimizer=optim, loss=bce, metrics=['accuracy'])
    
    return model


def create_dnn(seed=42, input_shape=(2381), mc=False):
    """
    This function compiles and returns a Keras model.
    """

    initializer = tf.keras.initializers.GlorotNormal(seed=seed)

    # model = tf.keras.models.Sequential()
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(2381, activation='elu', kernel_initializer=initializer)(inputs)
    x = tf.keras.layers.LayerNormalization() (x)
    x = tf.keras.layers.Dropout(0.3)(x, training=mc)

    x = tf.keras.layers.Dense(1024, activation='elu', kernel_initializer=initializer)(x)
    x = tf.keras.layers.LayerNormalization() (x)
    x = tf.keras.layers.Dropout(0.3)(x, training=mc)

    x = tf.keras.layers.Dense(512, activation='elu', kernel_initializer=initializer)(x)
    x = tf.keras.layers.LayerNormalization() (x)
    x = tf.keras.layers.Dropout(0.3)(x, training=mc)

    x = tf.keras.layers.Dense(128, activation='elu', kernel_initializer=initializer)(x)
    x = tf.keras.layers.LayerNormalization() (x)
    x = tf.keras.layers.Dropout(0.3)(x, training=mc)

    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)

    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    # optim = tf.keras.optimizers.RMSprop(lr=1e-3, momentum=0.9)
    optim = tf.keras.optimizers.Adam()

    model.compile(optimizer=optim, loss=bce, 
        metrics=['accuracy'])
#     metrics=['accuracy',
#             tf.keras.metrics.SensitivityAtSpecificity(0.99, name="TPR_01")])

    
    return model


def create_dnn2(seed=42, mc=False, input_shape=(2381,)):
    """
    This function compiles and returns a Keras model.
    
    Args:
        seed: Random seed
        mc: Monte Carlo dropout flag
        input_shape: Shape of input features (default: (2381,) for EMBER dataset)
    """

    initializer = tf.keras.initializers.GlorotNormal(seed=seed)

    # model = tf.keras.models.Sequential()
    input1 = tf.keras.Input(shape=input_shape)
    input2 = tf.keras.Input(shape=(1, ))

    x = tf.keras.layers.concatenate([input1, input2])
    # Tính toán số units dựa trên input_shape + 1 (cho y_true)
    # input1 có shape (input_shape[0],), input2 có shape (1,), concat sẽ có shape (input_shape[0] + 1,)
    feature_dim = input_shape[0] if isinstance(input_shape, tuple) else input_shape
    concat_dim = feature_dim + 1  # input1 + input2 = feature_dim + 1
    x = tf.keras.layers.Dense(concat_dim, activation='elu', kernel_initializer=initializer)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x, training=mc)

    x = tf.keras.layers.Dense(1024, activation='elu', kernel_initializer=initializer)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x, training=mc)

    x = tf.keras.layers.Dense(512, activation='elu', kernel_initializer=initializer)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x, training=mc)

    x = tf.keras.layers.Dense(128, activation='elu', kernel_initializer=initializer)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x, training=mc)

    outputs = tf.keras.layers.Dense(1)(x)
    # Sử dụng Add layer thay vì tf.add() để tương thích với KerasTensor
    out = tf.keras.layers.Add()([outputs, input2])

    # out = tf.clip_by_value(out, 0, 1)
    # Sử dụng Activation layer thay vì tf.sigmoid() để tương thích với KerasTensor
    out = tf.keras.layers.Activation('sigmoid')(out)
    model = tf.keras.Model((input1, input2), out)

    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    # mae = tf.keras.losses.MeanAbsoluteError()
    # mse = tf.keras.losses.MeanSquaredError()
    optim = tf.keras.optimizers.Adam()

    model.compile(optimizer=optim, loss=bce, 
        metrics=['accuracy'])
#     metrics=['accuracy',
#             tf.keras.metrics.SensitivityAtSpecificity(0.99, name="TPR_01")])
    return model


def create_dnn2_deeper(seed=42, mc=False, input_shape=(2381,)):
    """
    dualFFNN-1: Deeper Network variant of dualFFNN.
    Thêm các lớp trung gian nhỏ hơn để mô hình học các biểu diễn phức tạp hơn một cách từ từ.
    Kiến trúc: 2382 → 2382 → 1024 → 512 → 128 → 64 → 32 → 1
    
    Args:
        seed: Random seed
        mc: Monte Carlo dropout flag
        input_shape: Shape of input features (default: (2381,) for EMBER dataset)
    
    Returns:
        Compiled Keras model with dual inputs
    """
    initializer = tf.keras.initializers.GlorotNormal(seed=seed)

    input1 = tf.keras.Input(shape=input_shape)
    input2 = tf.keras.Input(shape=(1, ))

    x = tf.keras.layers.concatenate([input1, input2])
    feature_dim = input_shape[0] if isinstance(input_shape, tuple) else input_shape
    concat_dim = feature_dim + 1  # input1 + input2 = feature_dim + 1
    
    # Layer 1: 2382 → 2382
    x = tf.keras.layers.Dense(concat_dim, activation='elu', kernel_initializer=initializer)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x, training=mc)

    # Layer 2: 2382 → 1024
    x = tf.keras.layers.Dense(1024, activation='elu', kernel_initializer=initializer)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x, training=mc)

    # Layer 3: 1024 → 512
    x = tf.keras.layers.Dense(512, activation='elu', kernel_initializer=initializer)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x, training=mc)

    # Layer 4: 512 → 128
    x = tf.keras.layers.Dense(128, activation='elu', kernel_initializer=initializer)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x, training=mc)

    # Layer 5: 128 → 64 (bổ sung cho deeper network)
    x = tf.keras.layers.Dense(64, activation='elu', kernel_initializer=initializer)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x, training=mc)

    # Layer 6: 64 → 32 (bổ sung cho deeper network)
    x = tf.keras.layers.Dense(32, activation='elu', kernel_initializer=initializer)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x, training=mc)

    # Output: 32 → 1
    outputs = tf.keras.layers.Dense(1)(x)
    out = tf.keras.layers.Add()([outputs, input2])
    out = tf.keras.layers.Activation('sigmoid')(out)
    model = tf.keras.Model((input1, input2), out)

    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    optim = tf.keras.optimizers.Adam()

    model.compile(optimizer=optim, loss=bce, metrics=['accuracy'])
    return model


def create_dnn2_narrower(seed=42, mc=False, input_shape=(2381,)):
    """
    dualFFNN-2: Narrower Network variant of dualFFNN.
    Giảm số lượng tham số để kiểm tra xem mô hình có thể tổng quát hóa tốt hơn với ít nơ-ron hơn hay không (tránh overfitting).
    Kiến trúc: 2382 → 1024 → 512 → 256 → 64 → 1
    
    Args:
        seed: Random seed
        mc: Monte Carlo dropout flag
        input_shape: Shape of input features (default: (2381,) for EMBER dataset)
    
    Returns:
        Compiled Keras model with dual inputs
    """
    initializer = tf.keras.initializers.GlorotNormal(seed=seed)

    input1 = tf.keras.Input(shape=input_shape)
    input2 = tf.keras.Input(shape=(1, ))

    x = tf.keras.layers.concatenate([input1, input2])
    feature_dim = input_shape[0] if isinstance(input_shape, tuple) else input_shape
    concat_dim = feature_dim + 1  # input1 + input2 = feature_dim + 1
    
    # Layer 1: 2382 → 1024 (bỏ qua layer 2382 đầu tiên)
    x = tf.keras.layers.Dense(1024, activation='elu', kernel_initializer=initializer)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x, training=mc)

    # Layer 2: 1024 → 512
    x = tf.keras.layers.Dense(512, activation='elu', kernel_initializer=initializer)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x, training=mc)

    # Layer 3: 512 → 256 (thay vì 512 → 128)
    x = tf.keras.layers.Dense(256, activation='elu', kernel_initializer=initializer)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x, training=mc)

    # Layer 4: 256 → 64 (thay vì 128)
    x = tf.keras.layers.Dense(64, activation='elu', kernel_initializer=initializer)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x, training=mc)

    # Output: 64 → 1
    outputs = tf.keras.layers.Dense(1)(x)
    out = tf.keras.layers.Add()([outputs, input2])
    out = tf.keras.layers.Activation('sigmoid')(out)
    model = tf.keras.Model((input1, input2), out)

    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    optim = tf.keras.optimizers.Adam()

    model.compile(optimizer=optim, loss=bce, metrics=['accuracy'])
    return model
