import tensorflow as tf
from tensorflow.python.keras.engine import training


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
