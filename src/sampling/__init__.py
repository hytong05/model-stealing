"""
Sampling module - Các chiến lược sampling cho active learning.
"""

from typing import Callable, List, Tuple, Union

import lightgbm as lgb
import numpy as np
import tensorflow as tf
from scipy.stats import entropy
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model

from ..models.dnn import create_dnn, create_dnn2


def shuffled_argmax(values: np.ndarray, n_instances: int = 1) -> np.ndarray:
    """
    Trộn thứ tự rồi sort để phá vỡ tie khi cùng utility.
    """
    assert (
        n_instances <= values.shape[0]
    ), "n_instances must be less or equal than the size of utility"

    shuffled_idx = np.random.permutation(len(values))
    shuffled_values = values[shuffled_idx]

    sorted_query_idx = np.argsort(shuffled_values, kind="mergesort")[
        len(shuffled_values) - n_instances :
    ]

    query_idx = shuffled_idx[sorted_query_idx]
    return query_idx


def multi_argmax(values: np.ndarray, n_instances: int = 1) -> np.ndarray:
    """
    Chọn indices của n_instances giá trị lớn nhất.
    """
    assert (
        n_instances <= values.shape[0]
    ), "n_instances must be less or equal than the size of utility"

    max_idx = np.argpartition(-values, n_instances - 1, axis=0)[:n_instances]
    return max_idx


def classifier_entropy(
    classifier: Model,
    X: np.ndarray,
    y: np.ndarray,
    binary_labels: bool = True,
    dual: bool = False,
) -> np.ndarray:
    """
    Entropy của predictions cho tập mẫu X.
    """
    if dual:
        classwise_uncertainty = classifier(X, y).reshape(-1, 1)
        classwise_uncertainty = np.hstack(
            (1 - classwise_uncertainty, classwise_uncertainty)
        )
    else:
        classwise_uncertainty = classifier(X).reshape(-1, 1)
        classwise_uncertainty = np.hstack(
            (1 - classwise_uncertainty, classwise_uncertainty)
        )

    return np.transpose(entropy(np.transpose(classwise_uncertainty)))


def entropy_sampling(
    classifier: Model,
    X: np.ndarray,
    y: np.ndarray,
    binary_labels: bool = True,
    n_instances: int = 1,
    dual: bool = False,
    random_tie_break: bool = False,
) -> np.ndarray:
    """
    Chọn các mẫu với entropy cao nhất.
    """
    ent = classifier_entropy(classifier, X, y, binary_labels, dual)

    if not random_tie_break:
        query_idx = multi_argmax(ent, n_instances=n_instances)
    else:
        query_idx = shuffled_argmax(ent, n_instances=n_instances)

    return query_idx


def random_sampling(
    X: np.ndarray, rg: np.random.Generator, n_instances: int = 1
) -> np.ndarray:
    """
    Random sampling query strategy. Chọn random samples trong X.
    """
    query_idx = rg.integers(0, X.shape[0] - 1, size=n_instances)

    return query_idx


def mc_dropout(
    X_seed: np.ndarray,
    y_seed: np.ndarray,
    y_seed_true: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    y_val_true: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    n_models: int = 1,
    variance: bool = False,
    n_instances: int = 1,
    dual: bool = False,
) -> np.ndarray:
    """
    MC-dropout implementation.
    """
    checkpoint_filepath = "/tmp/checkpoint3"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=30
    )
    if dual:
        # Sử dụng input_shape từ X_seed
        input_shape = (X_seed.shape[1],) if len(X_seed.shape) > 1 else (X_seed.shape[0],)
        model = create_dnn2(mc=True, input_shape=input_shape)

        model.fit(
            (X_seed, y_seed_true),
            y_seed,
            batch_size=128,
            epochs=100,
            validation_data=((X_val, y_val_true), y_val),
            callbacks=[model_checkpoint_callback, early_stopping],
        )

    else:
        model = create_dnn(mc=True)

        model.fit(
            X_seed,
            y_seed,
            batch_size=128,
            epochs=100,
            validation_data=(X_val, y_val),
            callbacks=[model_checkpoint_callback, early_stopping],
        )

        model.load_weights(checkpoint_filepath)

    predictions = []
    for _ in range(n_models):
        if dual:
            y_pred = model.predict((X, y)).reshape(-1, 1)
        else:
            y_pred = model.predict(X).reshape(-1, 1)

        predictions.append(y_pred)

    print(np.array(predictions).shape)

    if variance:
        var = np.var(predictions, axis=0)
        query_idx = multi_argmax(var, n_instances=n_instances).squeeze()
        del var, predictions, model
    else:
        mean_pred = np.mean(predictions, axis=0)
        del predictions
        class_predictions = np.hstack((1 - mean_pred, mean_pred))
        ent = np.transpose(entropy(np.transpose(class_predictions)))
        query_idx = multi_argmax(ent, n_instances=n_instances)
        del ent, class_predictions, mean_pred, model

    return query_idx


def k_center(
    X_cluster: np.ndarray, X_med: np.ndarray, n_instances: int = 1
) -> Tuple[List, np.ndarray]:
    """
    Greedy K-center implementation.
    """
    query_idx: List[int] = []
    for _ in range(n_instances):
        dist = euclidean_distances(X_med, X_cluster)

        d_min = np.min(dist, axis=1)
        d_min_argmax = np.argmax(d_min)
        X_cluster = np.vstack([X_cluster, X_med[d_min_argmax]])
        query_idx.append(d_min_argmax)

    assert len(query_idx) == n_instances

    return query_idx, X_cluster


def ensemble(
    X_seed: np.ndarray,
    y_seed: np.ndarray,
    X: np.ndarray,
    num_models: int = 3,
    n_instances: int = 1,
) -> np.ndarray:
    """
    Ensemble LightGBM strategy dựa trên variance.
    """
    uncertainties = []
    for _ in range(num_models):
        X_train, X_test, y_train, y_test = train_test_split(
            X_seed,
            y_seed,
            test_size=0.1,
            random_state=np.random.randint(0, 1000),
        )
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_test, label=y_test)
        lgb_params = {
            "boosting_type": "gbdt",
            "objective": "binary",
            "learning_rate": 0.05,
            "num_leaves": 2048,
            "max_depth": 15,
            "min_child_samples": 30,
            "verbose": -1,
        }

        model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            verbose_eval=False,
            early_stopping_rounds=50,
        )

        classwise_uncertainty = model.predict(X).reshape(-1, 1)
        uncertainties.append(classwise_uncertainty)

    var = np.var(uncertainties, axis=0)
    query_idx = multi_argmax(var, n_instances=n_instances).squeeze()
    del var, uncertainties
    return query_idx


__all__ = [
    "shuffled_argmax",
    "multi_argmax",
    "classifier_entropy",
    "entropy_sampling",
    "random_sampling",
    "mc_dropout",
    "k_center",
    "ensemble",
]
