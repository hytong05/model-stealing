import json
from typing import Iterable, List, Optional
from urllib import error, request

import numpy as np


class HTTPOracle:
    """
    Minimal client that queries a remote black-box API endpoint.

    The endpoint must accept POST requests with payload:
        {"features": [...], "request_id": "..."}

    and respond with:
        {"prediction": 0/1}
    """

    def __init__(self, api_url: str, timeout: float = 15.0, extra_headers: Optional[dict] = None):
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        if extra_headers:
            self.headers.update(extra_headers)

    def _query_single(self, features: Iterable[float], request_id: Optional[str] = None) -> int:
        payload = {"features": list(map(float, features))}
        if request_id is not None:
            payload["request_id"] = request_id
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(self.api_url + "/predict", data=data, headers=self.headers, method="POST")
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                response = json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode('utf-8')}") from exc
        except error.URLError as exc:  # noqa: PERF203
            raise RuntimeError(f"Không thể kết nối tới oracle API: {exc}") from exc
        if "prediction" not in response:
            raise RuntimeError(f"Phản hồi không hợp lệ: {response}")
        return int(response["prediction"])

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        predictions: List[int] = []
        for sample in X:
            predictions.append(self._query_single(sample))
        return np.asarray(predictions, dtype=np.int32)

