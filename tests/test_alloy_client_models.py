from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from alloyai_client import AlloyClient, Modality


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class AlloyClientModelsEndpointTests(unittest.TestCase):
    def test_models_uses_get_models_endpoint(self) -> None:
        payload = {
            "image": [
                {
                    "model_id": "qwen-image",
                    "active_requests": 1,
                    "is_supported": True,
                    "supports_concurrent_requests": True,
                    "capabilities": [
                        {"inputs": ["text"], "outputs": ["image"], "name": "text-to-image"}
                    ],
                    "allocation_status": "allocated",
                }
            ],
            "audio": [],
            "video": [],
            "text": [],
        }
        client = AlloyClient("http://node0:8000")
        with patch.object(client, "_get", return_value=_FakeResponse(payload)) as mocked_get:
            response = client.models()

        mocked_get.assert_called_once_with("/models", timeout_s=None)
        self.assertEqual(len(response.image), 1)
        self.assertEqual(response.image[0].model_id, "qwen-image")
        self.assertEqual(response.image[0].capabilities[0].outputs, {Modality.IMAGE})


if __name__ == "__main__":
    unittest.main()
