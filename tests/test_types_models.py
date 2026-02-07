from __future__ import annotations

import unittest

from alloyai_client import AllocationStatus, AlloyModelsResponse, Modality


class AlloyModelsTypesTests(unittest.TestCase):
    def test_alloy_models_defaults_supports_concurrent_requests(self) -> None:
        payload = {
            "image": [
                {
                    "model_id": "qwen-image",
                    "active_requests": 0,
                    "is_supported": True,
                    "capabilities": [
                        {
                            "inputs": ["text"],
                            "outputs": ["image"],
                            "name": "text-to-image",
                        }
                    ],
                    "allocation_status": "deallocated",
                }
            ],
            "audio": [],
            "video": [],
            "text": [],
        }

        parsed = AlloyModelsResponse.model_validate(payload)
        model = parsed.image[0]
        self.assertFalse(model.supports_concurrent_requests)
        self.assertEqual(model.allocation_status, AllocationStatus.DEALLOCATED)
        self.assertEqual(model.capabilities[0].outputs, {Modality.IMAGE})

    def test_alloy_models_parses_supports_concurrent_requests(self) -> None:
        payload = {
            "image": [],
            "audio": [],
            "video": [],
            "text": [
                {
                    "model_id": "qwen3-medium",
                    "active_requests": 2,
                    "is_supported": True,
                    "supports_concurrent_requests": True,
                    "capabilities": [
                        {
                            "inputs": ["text", "image"],
                            "outputs": ["text"],
                            "name": "vlm-chat",
                        }
                    ],
                    "allocation_status": "allocated",
                }
            ],
        }

        parsed = AlloyModelsResponse.model_validate(payload)
        model = parsed.text[0]
        self.assertTrue(model.supports_concurrent_requests)
        self.assertEqual(model.allocation_status, AllocationStatus.ALLOCATED)
        self.assertEqual(model.capabilities[0].inputs, {Modality.TEXT, Modality.IMAGE})


if __name__ == "__main__":
    unittest.main()
