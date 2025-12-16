from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .flexible_target import FlexibleKerasTarget, FlexibleLGBTarget, FlexibleSKLearnTarget
from . import XGBoostTarget, TabNetTarget, DualFFNNTarget

# Default paths
DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "targets"


class BaseOracleClient:
    """Common interface for querying target models."""

    def predict(self, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
        raise NotImplementedError

    def get_required_feature_dim(self) -> Optional[int]:
        return None

    def supports_probabilities(self) -> bool:
        return False

    def predict_proba(self, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
        raise NotImplementedError("Probability predictions not supported for this oracle.")


class LocalOracleClient(BaseOracleClient):
    """
    Loads target weights locally (without running an HTTP server) and returns binary labels.
    Data fed into `predict` must already be preprocessed (scaled/normalized) exactly as the
    target expects.
    """

    def __init__(
        self,
        model_type: str,
        model_path: str | Path,
        normalization_stats_path: str | Path | None = None,
        threshold: float = 0.5,
        feature_dim: Optional[int] = None,
    ):
        self.model_type = model_type.lower()
        self.model_path = str(Path(model_path).resolve())
        self.threshold = threshold
        self._required_feature_dim: Optional[int] = None

        if self.model_type == "lgb":
            if normalization_stats_path is None:
                raise ValueError("LightGBM oracle cần cung cấp normalization_stats_path.")
            norm_path = str(Path(normalization_stats_path).resolve())
            self._oracle = FlexibleLGBTarget(
                model_path=self.model_path,
                normalization_stats_path=norm_path,
                threshold=threshold,
                name=f"lgb-local-{Path(self.model_path).name}",
                feature_dim=feature_dim,
            )
        elif self.model_type == "sklearn":
            # Sklearn model (KNN, etc.) - normalization stats là optional
            norm_path = str(Path(normalization_stats_path).resolve()) if normalization_stats_path else None
            self._oracle = FlexibleSKLearnTarget(
                model_path=self.model_path,
                normalization_stats_path=norm_path,
                threshold=threshold,
                name=f"sklearn-local-{Path(self.model_path).name}",
                feature_dim=feature_dim,
            )
        elif self.model_type == "xgb":
            # XGBoost model (.json)
            norm_path = str(Path(normalization_stats_path).resolve()) if normalization_stats_path else None
            self._oracle = XGBoostTarget(
                model_path=self.model_path,
                normalization_stats_path=norm_path,
                thresh=threshold,
                name=f"xgb-local-{Path(self.model_path).name}",
            )
        elif self.model_type == "tabnet":
            # TabNet model (.zip)
            norm_path = str(Path(normalization_stats_path).resolve()) if normalization_stats_path else None
            self._oracle = TabNetTarget(
                model_path=self.model_path,
                normalization_stats_path=norm_path,
                thresh=threshold,
                name=f"tabnet-local-{Path(self.model_path).name}",
            )
        elif self.model_type == "pytorch":
            # dualFFNN PyTorch model (.pt)
            norm_path = str(Path(normalization_stats_path).resolve()) if normalization_stats_path else None
            self._oracle = DualFFNNTarget(
                model_path=self.model_path,
                normalization_stats_path=norm_path,
                thresh=threshold,
                name=f"dualffnn-local-{Path(self.model_path).name}",
            )
        else:
            # Keras model (h5)
            self._oracle = FlexibleKerasTarget(
                self.model_path,
                feature_dim=feature_dim if feature_dim is not None else 2381,
                threshold=threshold,
                name=f"keras-local-{Path(self.model_path).name}",
                normalization_stats_path=normalization_stats_path,
            )

        self._required_feature_dim = self._oracle.get_required_feature_dim()

    def _pad_or_truncate_features(self, X: np.ndarray, method: str = "zero") -> np.ndarray:
        """
        Pad hoặc truncate features để khớp với required_feature_dim của target model.
        
        Args:
            X: Input features array (n_samples, n_features) hoặc (n_features,) cho single sample
            method: Padding method - "zero" (zeros), "mean" (mean value), "random" (random values)
        
        Returns:
            X được pad/truncate để khớp với required_feature_dim
        """
        if self._required_feature_dim is None:
            # Không có yêu cầu cụ thể (có preprocessing layer) - không cần pad/truncate
            return X
        
        # Xử lý cả 1D và 2D arrays
        is_1d = len(X.shape) == 1
        if is_1d:
            X = X.reshape(1, -1)  # Reshape thành (1, n_features)
        
        current_dim = X.shape[1]
        
        if current_dim == self._required_feature_dim:
            # Không cần thay đổi
            return X[0] if is_1d else X
        elif current_dim > self._required_feature_dim:
            # Truncate: Cắt bỏ features thừa
            # Giữ N features đầu tiên (N = required_feature_dim)
            X_truncated = X[:, :self._required_feature_dim]
            return X_truncated[0] if is_1d else X_truncated
        else:
            # Padding: Thêm features thiếu
            n_samples = X.shape[0]
            n_pad = self._required_feature_dim - current_dim
            
            if method == "zero":
                pad_values = np.zeros((n_samples, n_pad), dtype=X.dtype)
            elif method == "mean":
                # Sử dụng mean của features hiện có (tính trên tất cả features và samples)
                mean_val = np.mean(X)
                pad_values = np.full((n_samples, n_pad), mean_val, dtype=X.dtype)
            elif method == "random":
                # Random values trong khoảng [-1, 1]
                pad_values = np.random.uniform(-1.0, 1.0, size=(n_samples, n_pad)).astype(X.dtype)
            else:
                # Default: zeros
                pad_values = np.zeros((n_samples, n_pad), dtype=X.dtype)
            
            X_padded = np.hstack([X, pad_values])
            return X_padded[0] if is_1d else X_padded
    
    def predict(self, X: np.ndarray, batch_size: int = 512, padding_method: str = "zero") -> np.ndarray:
        # Pad hoặc truncate features trước khi predict
        X_processed = self._pad_or_truncate_features(X, method=padding_method)
        
        if self.model_type in {"lgb", "sklearn", "xgb", "tabnet", "pytorch"}:
            return self._oracle(X_processed)
        else:
            return self._oracle(X_processed, batch_size=batch_size)

    def get_required_feature_dim(self) -> Optional[int]:
        return self._required_feature_dim

    def supports_probabilities(self) -> bool:
        return hasattr(self._oracle, "predict_proba")

    def predict_proba(self, X: np.ndarray, batch_size: int = 512, padding_method: str = "zero") -> np.ndarray:
        if not self.supports_probabilities():
            return super().predict_proba(X, batch_size=batch_size)
        
        # Pad hoặc truncate features trước khi predict_proba
        X_processed = self._pad_or_truncate_features(X, method=padding_method)
        
        if self.model_type in {"lgb", "sklearn", "xgb", "tabnet", "pytorch"}:
            return self._oracle.predict_proba(X_processed)
        else:
            return self._oracle.predict_proba(X_processed, batch_size=batch_size)

    def set_threshold(self, threshold: float) -> None:
        self.threshold = threshold
        if hasattr(self._oracle, "model_threshold"):
            self._oracle.model_threshold = threshold

    def get_threshold(self) -> float:
        if hasattr(self._oracle, "model_threshold"):
            return float(self._oracle.model_threshold)
        return self.threshold


def create_oracle_from_name(
    model_name: str,
    models_dir: Path | str | None = None,
    threshold: float = 0.5,
    feature_dim: Optional[int] = None,
    blackbox: bool = True,
) -> BaseOracleClient:
    """
    Tạo oracle client từ tên model - tự động detect mọi thứ.
    
    Args:
        model_name: Tên model (ví dụ: "CEE", "LEE", "CSE", "LSE")
        models_dir: Thư mục chứa models (mặc định: artifacts/targets/)
        threshold: Threshold cho binary classification (mặc định: 0.5)
        feature_dim: Số features trong dataset (mặc định: 2381)
    
    Returns:
        LocalOracleClient đã được khởi tạo và sẵn sàng sử dụng
    
    Ví dụ:
        >>> oracle = create_oracle_from_name("LEE")
        >>> prediction = oracle.predict(sample)
    """
    if models_dir is None:
        models_dir = DEFAULT_MODELS_DIR
    else:
        models_dir = Path(models_dir).expanduser().resolve()
    
    if not models_dir.exists():
        raise FileNotFoundError(f"❌ Không tìm thấy thư mục models: {models_dir}")
    
    model_name_upper = model_name.upper().strip()
    model_name_lower = model_name.lower().strip()
    
    # Tìm model file - thử cả .h5, .lgb, .pkl, .d5, .json, .zip, và .pt
    # Hỗ trợ cả uppercase và lowercase, và các suffix như _ember
    model_path = None
    model_type = None
    
    possible_extensions = [".h5", ".lgb", ".pkl", ".d5", ".json", ".zip", ".pt"]
    # Các pattern để tìm file: exact match, lowercase, và với các suffix phổ biến
    search_patterns = [
        model_name_upper,                # XGBOOST / DUALFFNN
        model_name_lower,                # xgboost / dualffnn
        f"{model_name_lower}_ember",     # xgboost_ember / dualffnn_ember
        f"{model_name_upper}_EMBER",     # XGBOOST_EMBER / DUALFFNN_EMBER
        f"{model_name_lower}_ember_full",  # dualffnn_ember_full (PyTorch dualFFNN)
        f"{model_name_upper}_EMBER_FULL",
    ]
    
    for pattern in search_patterns:
        for ext in possible_extensions:
            candidate = models_dir / f"{pattern}{ext}"
            if candidate.exists():
                model_path = candidate
                if ext == ".h5":
                    model_type = "h5"
                elif ext == ".pkl":
                    # .pkl có thể là LightGBM hoặc sklearn - cần detect
                    # Thử với LightGBM trước, nếu fail thì dùng sklearn
                    try:
                        import lightgbm as lgb
                        temp_model = lgb.Booster(model_file=str(candidate))
                        model_type = "lgb"
                        print(f"✅ Detected {pattern}.pkl as LightGBM model")
                    except Exception:
                        # Không phải LightGBM, có thể là sklearn
                        model_type = "sklearn"
                        print(f"✅ Detected {pattern}.pkl as sklearn model")
                elif ext in [".lgb", ".d5"]:
                    model_type = "lgb"
                elif ext == ".json":
                    # .json thường là XGBoost model
                    model_type = "xgb"
                    print(f"✅ Detected {pattern}.json as XGBoost model")
                elif ext == ".zip":
                    # .zip thường là TabNet model
                    model_type = "tabnet"
                    print(f"✅ Detected {pattern}.zip as TabNet model")
                elif ext == ".pt":
                    # .pt thường là dualFFNN PyTorch model
                    model_type = "pytorch"
                    print(f"✅ Detected {pattern}.pt as dualFFNN (PyTorch) model")
                break
        if model_path is not None:
            break
    
    if model_path is None or model_type is None:
        raise FileNotFoundError(
            f"❌ Không tìm thấy model '{model_name}' trong {models_dir}. "
            f"Đã thử các extension: {', '.join(possible_extensions)} "
            f"và các pattern: {', '.join(search_patterns)}"
        )
    
    # Tìm normalization stats - tự động tìm
    # Sử dụng tên file thực tế (có thể có suffix) để tìm stats
    model_file_stem = model_path.stem  # Tên file không có extension
    normalization_stats_path = None
    possible_stats_paths = [
        models_dir / f"{model_file_stem}_normalization_stats.npz",  # xgboost_ember_normalization_stats.npz
        models_dir / f"{model_name_upper}_normalization_stats.npz",  # XGBOOST_normalization_stats.npz
        models_dir / f"{model_name_lower}_normalization_stats.npz",  # xgboost_normalization_stats.npz
        models_dir / f"{model_name_upper}.npz",
        models_dir / f"{model_name_lower}.npz",
        models_dir / "normalization_stats.npz",
    ]
    
    for stats_path in possible_stats_paths:
        if stats_path.exists():
            normalization_stats_path = str(stats_path)
            break
    
    # Với LightGBM, normalization stats là bắt buộc
    # Với sklearn, normalization stats là optional
    if model_type == "lgb" and normalization_stats_path is None:
        raise FileNotFoundError(
            f"❌ LightGBM model '{model_name}' cần normalization stats nhưng không tìm thấy. "
            f"Đã tìm trong: {models_dir}"
        )
    
    # Auto-detect feature_dim từ model nếu không được cung cấp
    # Để tránh nhầm lẫn (ví dụ: LSE train trên SOMLAP nhưng default là EMBER)
    if feature_dim is None:
        # Thử load model tạm thời để detect số features
        try:
            if model_type == "lgb":
                import lightgbm as lgb
                temp_model = lgb.Booster(model_file=str(model_path))
                feature_dim = temp_model.num_feature()
                print(f"✅ Auto-detected feature_dim={feature_dim} từ model {model_name}")
            elif model_type == "xgb":
                import xgboost as xgb
                temp_model = xgb.Booster()
                temp_model.load_model(str(model_path))
                feature_dim = temp_model.num_features()
                print(f"✅ Auto-detected feature_dim={feature_dim} từ XGBoost model {model_name}")
            elif model_type == "sklearn":
                import joblib
                temp_model = joblib.load(str(model_path))
                try:
                    feature_dim = temp_model.n_features_in_
                    print(f"✅ Auto-detected feature_dim={feature_dim} từ sklearn model {model_name}")
                except AttributeError:
                    if hasattr(temp_model, '_fit_X'):
                        feature_dim = temp_model._fit_X.shape[1]
                        print(f"✅ Auto-detected feature_dim={feature_dim} từ sklearn model {model_name} (từ training data)")
                    else:
                        raise ValueError("Cannot detect feature_dim from sklearn model")
            else:
                # Với Keras, cần load model để detect - phức tạp hơn
                # Fallback về default cho Keras models
                feature_dim = 2381  # EMBER dataset default cho Keras models
                print(f"⚠️  Sử dụng default feature_dim={feature_dim} cho Keras model (có thể không chính xác)")
        except Exception as e:
            # Nếu không thể detect, dùng default
            print(f"⚠️  Không thể auto-detect feature_dim từ model: {e}")
            # Auto-detect dựa trên model_name
            if model_name in ["LSE"]:
                feature_dim = 108  # SOMLAP dataset
                print(f"   Sử dụng feature_dim={feature_dim} dựa trên model_name={model_name} (SOMLAP)")
            else:
                feature_dim = 2381  # EMBER dataset default
                print(f"   Sử dụng default feature_dim={feature_dim} (EMBER)")
    
    # Tạo oracle client
    local_oracle = LocalOracleClient(
        model_type=model_type,
        model_path=str(model_path),
        normalization_stats_path=normalization_stats_path,
        threshold=threshold,
        feature_dim=feature_dim,
    )
    
    # Nếu blackbox=True, wrap trong BlackBoxOracleClient để ẩn implementation details
    if blackbox:
        # Tạo BlackBoxOracleClient instance mà không gọi __init__
        blackbox_oracle = object.__new__(BlackBoxOracleClient)
        blackbox_oracle._oracle = local_oracle
        blackbox_oracle.model_name = model_name
        return blackbox_oracle
    else:
        return local_oracle


class BlackBoxOracleClient(BaseOracleClient):
    """
    Black Box Oracle Client - Ẩn hoàn toàn implementation details khỏi attacker.
    
    Trong black box attack, attacker CHỈ được biết:
    - Tên model (hoặc API endpoint)
    - Raw features (có thể query)
    - Predictions (0 hoặc 1, hoặc probabilities)
    
    Attacker KHÔNG được biết:
    - Model type (Keras vs LightGBM)
    - Normalization statistics
    - Model architecture
    - Preprocessing details
    
    Oracle client (của nhà cung cấp) tự động:
    - Detect model type
    - Load normalization stats
    - Xử lý preprocessing
    - Trả về predictions
    """
    
    def __init__(
        self,
        model_name: str,
        models_dir: Path | str | None = None,
        threshold: float = 0.5,
        feature_dim: Optional[int] = None,
    ):
        """
        Khởi tạo black box oracle client từ tên model.
        
        Args:
            model_name: Tên model (ví dụ: "CEE", "LEE", "CSE", "LSE")
            models_dir: Thư mục chứa models (mặc định: artifacts/targets/)
            threshold: Threshold cho binary classification (mặc định: 0.5)
            feature_dim: Số features trong dataset (mặc định: 2381)
        
        Tất cả implementation details (model type, normalization stats, etc.) 
        được tự động detect và ẩn khỏi attacker.
        """
        # Sử dụng create_oracle_from_name với blackbox=False để lấy LocalOracleClient
        # Sau đó wrap nó để ẩn implementation details
        local_oracle = create_oracle_from_name(
            model_name=model_name,
            models_dir=models_dir,
            threshold=threshold,
            feature_dim=feature_dim,
            blackbox=False,  # Lấy LocalOracleClient, không wrap
        )
        self._oracle = local_oracle
        self.model_name = model_name.upper().strip()
    
    def predict(self, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """
        Predict binary labels từ raw features.
        
        Args:
            X: Raw features (chưa normalize/preprocess)
            batch_size: Batch size cho prediction (chỉ dùng với Keras models)
        
        Returns:
            Binary predictions (0 hoặc 1)
        
        Oracle client tự động xử lý:
        - Normalization (nếu cần)
        - Preprocessing
        - Feature alignment
        """
        return self._oracle.predict(X, batch_size=batch_size)
    
    def predict_proba(self, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """
        Predict probabilities từ raw features.
        
        Args:
            X: Raw features (chưa normalize/preprocess)
            batch_size: Batch size cho prediction (chỉ dùng với Keras models)
        
        Returns:
            Probabilities (0.0 đến 1.0)
        
        Oracle client tự động xử lý preprocessing.
        """
        if not self.supports_probabilities():
            raise NotImplementedError("Oracle không hỗ trợ probability predictions")
        return self._oracle.predict_proba(X, batch_size=batch_size)
    
    def supports_probabilities(self) -> bool:
        """Kiểm tra xem oracle có hỗ trợ probability predictions không."""
        return self._oracle.supports_probabilities()
    
    def get_required_feature_dim(self) -> Optional[int]:
        """
        Lấy số features yêu cầu của model.
        
        Trong black box attack, đây có thể là thông tin hữu ích cho attacker
        để biết input size (thông qua API documentation hoặc trial-and-error).
        """
        return self._oracle.get_required_feature_dim()
    
    def set_threshold(self, threshold: float) -> None:
        """Đặt threshold cho binary classification."""
        self._oracle.set_threshold(threshold)
    
    def get_threshold(self) -> float:
        """Lấy threshold hiện tại."""
        return self._oracle.get_threshold()



