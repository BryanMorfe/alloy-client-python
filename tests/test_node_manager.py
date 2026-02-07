from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from unittest.mock import patch

from alloyai_client import (
    AlloyModelsResponse,
    AlloyNodeManager,
    NodeConfig,
    NodeQueryMode,
)


def _model_payload(
    *,
    model_id: str,
    active_requests: int,
    is_supported: bool = True,
    supports_concurrent_requests: bool = True,
    allocation_status: str = "deallocated",
    output: str = "image",
) -> dict:
    return {
        "model_id": model_id,
        "active_requests": active_requests,
        "is_supported": is_supported,
        "supports_concurrent_requests": supports_concurrent_requests,
        "capabilities": [
            {
                "inputs": ["text"],
                "outputs": [output],
                "name": f"text-to-{output}",
            }
        ],
        "allocation_status": allocation_status,
    }


def _models_response(*, image: list[dict] | None = None, text: list[dict] | None = None) -> dict:
    return {
        "image": image or [],
        "audio": [],
        "video": [],
        "text": text or [],
    }


@dataclass
class _BackendState:
    models_payload: dict
    models_calls: int = 0
    image_calls: int = 0
    image_stream_calls: int = 0


class _FakeAlloyClient:
    backends: dict[str, _BackendState] = {}

    def __init__(self, base_url: str, *, timeout_s: float = 300.0) -> None:
        self.base_url = base_url
        self.timeout_s = timeout_s

    def models(self, *, timeout_s=None) -> AlloyModelsResponse:
        backend = self.backends[self.base_url]
        backend.models_calls += 1
        return AlloyModelsResponse.model_validate(backend.models_payload)

    def image(self, model_id: str, prompt, *, stream=False, **kwargs):
        backend = self.backends[self.base_url]
        if stream:
            backend.image_stream_calls += 1

            def _generator():
                yield {"event": "received", "payload": {"model_id": model_id}}
                yield {"event": "done", "payload": {}}

            return _generator()
        backend.image_calls += 1
        return {"node": self.base_url, "model_id": model_id}

    def chat(self, *args, **kwargs):
        return {"message": {"content": "ok"}}

    def audio(self, *args, **kwargs):
        return {"outputs": [], "sample_rate": 24000}


class AlloyNodeManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeAlloyClient.backends = {}

    def _new_manager(self, mode: NodeQueryMode = NodeQueryMode.LOCAL_ONLY, max_nodes_to_query: int = 2):
        nodes = [
            NodeConfig(base_url="http://node0:8000", name="node0"),
            NodeConfig(base_url="http://node1:8000", name="node1"),
        ]
        return AlloyNodeManager(nodes=nodes, mode=mode, max_nodes_to_query=max_nodes_to_query)

    @patch("alloyai_client.node_manager.AlloyClient", new=_FakeAlloyClient)
    def test_selects_node_with_lower_active_requests(self) -> None:
        _FakeAlloyClient.backends = {
            "http://node0:8000": _BackendState(
                models_payload=_models_response(
                    image=[_model_payload(model_id="qwen-image", active_requests=4, allocation_status="allocated")]
                )
            ),
            "http://node1:8000": _BackendState(
                models_payload=_models_response(
                    image=[_model_payload(model_id="qwen-image", active_requests=1, allocation_status="allocated")]
                )
            ),
        }
        manager = self._new_manager()

        response = manager.image(model_id="qwen-image", prompt="fox")
        self.assertEqual(response["node"], "http://node1:8000")

    @patch("alloyai_client.node_manager.AlloyClient", new=_FakeAlloyClient)
    def test_non_concurrent_penalty_can_outweigh_lower_queue_depth(self) -> None:
        _FakeAlloyClient.backends = {
            "http://node0:8000": _BackendState(
                models_payload=_models_response(
                    image=[
                        _model_payload(
                            model_id="qwen-image",
                            active_requests=1,
                            supports_concurrent_requests=False,
                            allocation_status="allocated",
                        )
                    ]
                )
            ),
            "http://node1:8000": _BackendState(
                models_payload=_models_response(
                    image=[
                        _model_payload(
                            model_id="qwen-image",
                            active_requests=2,
                            supports_concurrent_requests=True,
                            allocation_status="allocated",
                        )
                    ]
                )
            ),
        }
        manager = self._new_manager()

        response = manager.image(model_id="qwen-image", prompt="fox")
        self.assertEqual(response["node"], "http://node1:8000")

    @patch("alloyai_client.node_manager.AlloyClient", new=_FakeAlloyClient)
    def test_query_everytime_refreshes_all_candidates(self) -> None:
        _FakeAlloyClient.backends = {
            "http://node0:8000": _BackendState(
                models_payload=_models_response(
                    image=[_model_payload(model_id="qwen-image", active_requests=0, allocation_status="allocated")]
                )
            ),
            "http://node1:8000": _BackendState(
                models_payload=_models_response(
                    image=[_model_payload(model_id="qwen-image", active_requests=0, allocation_status="deallocated")]
                )
            ),
        }
        manager = self._new_manager(mode=NodeQueryMode.QUERY_EVERYTIME)
        initial_calls = sum(backend.models_calls for backend in _FakeAlloyClient.backends.values())
        manager.image(model_id="qwen-image", prompt="fox")
        post_calls = sum(backend.models_calls for backend in _FakeAlloyClient.backends.values())

        self.assertEqual(post_calls - initial_calls, 2)

    @patch("alloyai_client.node_manager.AlloyClient", new=_FakeAlloyClient)
    def test_controlled_querying_limits_refreshes(self) -> None:
        _FakeAlloyClient.backends = {
            "http://node0:8000": _BackendState(
                models_payload=_models_response(
                    image=[_model_payload(model_id="qwen-image", active_requests=1, allocation_status="allocated")]
                )
            ),
            "http://node1:8000": _BackendState(
                models_payload=_models_response(
                    image=[_model_payload(model_id="qwen-image", active_requests=2, allocation_status="deallocated")]
                )
            ),
        }
        manager = self._new_manager(mode=NodeQueryMode.CONTROLLED_QUERYING, max_nodes_to_query=1)
        initial_calls = sum(backend.models_calls for backend in _FakeAlloyClient.backends.values())
        manager.image(model_id="qwen-image", prompt="fox")
        post_calls = sum(backend.models_calls for backend in _FakeAlloyClient.backends.values())

        self.assertEqual(post_calls - initial_calls, 1)

    @patch("alloyai_client.node_manager.AlloyClient", new=_FakeAlloyClient)
    def test_stream_request_releases_inflight_counters(self) -> None:
        _FakeAlloyClient.backends = {
            "http://node0:8000": _BackendState(
                models_payload=_models_response(
                    image=[_model_payload(model_id="qwen-image", active_requests=0, allocation_status="allocated")]
                )
            ),
            "http://node1:8000": _BackendState(
                models_payload=_models_response(
                    image=[_model_payload(model_id="qwen-image", active_requests=4, allocation_status="allocated")]
                )
            ),
        }
        manager = self._new_manager(mode=NodeQueryMode.LOCAL_ONLY)
        events = list(manager.image(model_id="qwen-image", prompt="fox", stream=True))
        self.assertTrue(events)
        self.assertEqual(events[-1]["event"], "done")

        selected_node = next(node for node in manager._nodes if node.name == "node0")
        self.assertEqual(selected_node.local_inflight_total, 0)
        self.assertNotIn("qwen-image", selected_node.local_inflight_by_model)

    @patch("alloyai_client.node_manager.AlloyClient", new=_FakeAlloyClient)
    def test_models_aggregates_across_nodes(self) -> None:
        _FakeAlloyClient.backends = {
            "http://node0:8000": _BackendState(
                models_payload=_models_response(
                    image=[
                        _model_payload(
                            model_id="qwen-image",
                            active_requests=1,
                            allocation_status="allocated",
                            supports_concurrent_requests=False,
                        )
                    ]
                )
            ),
            "http://node1:8000": _BackendState(
                models_payload=_models_response(
                    image=[
                        _model_payload(
                            model_id="qwen-image",
                            active_requests=2,
                            allocation_status="deallocated",
                            supports_concurrent_requests=True,
                        )
                    ]
                )
            ),
        }
        manager = self._new_manager()
        combined = manager.models()
        self.assertEqual(len(combined.image), 1)
        model = combined.image[0]
        self.assertEqual(model.model_id, "qwen-image")
        self.assertEqual(model.active_requests, 3)
        self.assertEqual(model.allocation_status.value, "allocated")
        self.assertTrue(model.supports_concurrent_requests)


if __name__ == "__main__":
    unittest.main()
