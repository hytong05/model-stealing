"""
HTTP query helper cho target API.
"""

import requests


class Query:
    def __init__(self, base_url: str = "http://192.168.56.3:5000") -> None:
        self.base_url = base_url

    def do_get(self, path: str) -> int:
        response = requests.get(self.base_url + path, timeout=30)
        return response.status_code

    def do_post(self, path: str, filename: str) -> str:
        with open(filename, "rb") as file_obj:
            files = {"file": file_obj}
            response = requests.post(
                self.base_url + path, files=files, timeout=60
            )

        return response.text


__all__ = ["Query"]

