"""
Utils module - Các hàm tiện ích dùng chung.
"""

import hashlib
import os
from zipfile import ZipFile

import numpy as np
from mlflow import log_metric
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    hamming_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .query import Query


def rename_files_to_sha256(path):
    """
    Rename toàn bộ file trong thư mục sang tên SHA256 của nội dung.
    """
    files = os.listdir(path)

    for filename in files:
        with open(os.path.join(path, filename), "rb") as f:
            bytes_content = f.read()  # đọc toàn bộ file
            readable_hash = hashlib.sha256(bytes_content).hexdigest()
            print(readable_hash)
            os.rename(
                os.path.join(path, filename), os.path.join(path, readable_hash)
            )


def compress_files(file_list):
    """
    Nhận list các file path và nén thành một file zip.
    Trả về: test.zip
    """
    with ZipFile("test.zip", mode="w") as zf:
        for file_path in file_list:
            try:
                zf.write(file_path)
            except FileNotFoundError:
                print(f"{file_path} does not exist.")


def get_fpr(y_true, y_pred):
    """
    Tính False Positive Rate từ y_true và y_pred.
    """
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (tn + fp)
    return fpr


def get_tpr_at_fpr(y_true, y_pred, target_fpr):
    """
    Tính True Positive Rate tại một FPR cụ thể.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    return np.interp(target_fpr, fpr, tpr)


def find_threshold(y_true, y_pred, fpr_target):
    """
    Tìm threshold quyết định cho mức FPR mục tiêu.
    """
    fpr, _, thresh = roc_curve(y_true, y_pred)
    return np.interp(fpr_target, fpr, thresh)


def init_scores():
    """
    Khởi tạo dict chứa các metric trong quá trình training/evaluation.
    """
    scores = {
        "acc": [],
        "agg": [],
        "fpr": [],
        "rec": [],
        "pres": [],
        "auc": [],
        "confs": [],
        "threshold": [],
        "nums": [],
    }
    return scores


def log_and_score(y_proba, y_pred_target, y_test, scores, fpr_target, logging=True):
    """
    Ghi log metric (nếu bật) và cập nhật dict scores.
    """
    thresh = find_threshold(y_test, y_proba, fpr_target)
    y_pred = [int(i > thresh) for i in y_proba]

    test_score = accuracy_score(y_test, y_pred)
    agg_score = 1.0 - hamming_loss(y_pred_target, y_pred)
    fpr_score = get_fpr(y_test, y_pred)
    rec_score = recall_score(y_test, y_pred)
    pres_score = precision_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred)

    print(f"Threshold for target FPR {fpr_target}: {thresh}")
    print("Accuracy score:", test_score)
    print("Agreement:", agg_score)
    print("FPR:", fpr_score)
    print("Recall:", rec_score)
    print("Precision:", pres_score)
    print("AUC:", auc_score)
    print("Confusion matrix:", conf_mat)

    scores["acc"].append(test_score)
    scores["agg"].append(agg_score)
    scores["fpr"].append(fpr_score)
    scores["rec"].append(rec_score)
    scores["pres"].append(pres_score)
    scores["auc"].append(auc_score)
    scores["threshold"].append(thresh)
    scores["confs"].append(conf_mat)

    if logging:
        num_samples = scores["nums"][-1]
        log_metric("accuracy", test_score, step=num_samples)
        log_metric("agreement", agg_score, step=num_samples)
        log_metric("FPR", fpr_score, step=num_samples)
        log_metric("Recall", rec_score, step=num_samples)
        log_metric("Precision", pres_score, step=num_samples)
        log_metric("AUC", auc_score, step=num_samples)
        log_metric("Threshold", thresh, step=num_samples)

    return scores


__all__ = [
    "rename_files_to_sha256",
    "compress_files",
    "get_fpr",
    "get_tpr_at_fpr",
    "find_threshold",
    "init_scores",
    "log_and_score",
    "Query",
]
