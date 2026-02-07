from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence, Union

from .alloyai_client import AlloyClient
from .client_protocol import AlloyClientProtocol
from .types import (
    AllocationStatus,
    AlloyModel,
    AlloyModelsResponse,
    ChatResponse,
    JsonSchemaValue,
    Message,
    ModelCapability,
    Modality,
    Tool,
)


class NodeQueryMode(str, Enum):
    LOCAL_ONLY = "local_only"
    QUERY_EVERYTIME = "query_everytime"
    CONTROLLED_QUERYING = "controlled_querying"


@dataclass(frozen=True)
class NodeConfig:
    base_url: str
    name: Optional[str] = None
    weight: float = 1.0


@dataclass
class _NodeState:
    name: str
    client: AlloyClient
    weight: float
    models: Dict[str, AlloyModel] = field(default_factory=dict)
    categories_by_model_id: Dict[str, set[Modality]] = field(default_factory=dict)
    supported_model_count: int = 0
    remote_active_total: int = 0
    local_inflight_total: int = 0
    local_inflight_by_model: Dict[str, int] = field(default_factory=dict)
    last_refresh_ts: float = 0.0
    last_refresh_error: Optional[str] = None


class AlloyNodeManager(AlloyClientProtocol):
    def __init__(
        self,
        nodes: Sequence[Union[str, NodeConfig]],
        *,
        timeout_s: float = 300.0,
        mode: NodeQueryMode | str = NodeQueryMode.CONTROLLED_QUERYING,
        max_nodes_to_query: int = 2,
        strict_init: bool = False,
    ) -> None:
        if not nodes:
            raise ValueError("nodes must include at least one node")
        if max_nodes_to_query <= 0:
            raise ValueError("max_nodes_to_query must be positive")

        self._timeout_s = timeout_s
        self._mode = NodeQueryMode(mode)
        self._max_nodes_to_query = int(max_nodes_to_query)
        self._lock = threading.Lock()
        self._nodes: List[_NodeState] = []

        for index, node in enumerate(nodes):
            if isinstance(node, str):
                base_url = node
                node_name = f"node-{index}"
                node_weight = 1.0
            else:
                base_url = node.base_url
                node_name = node.name or f"node-{index}"
                node_weight = float(node.weight)
            self._nodes.append(
                _NodeState(
                    name=node_name,
                    client=AlloyClient(base_url=base_url, timeout_s=timeout_s),
                    weight=node_weight,
                )
            )

        errors = self.refresh_nodes()
        if strict_init and errors:
            error_lines = [f"{node}: {message}" for node, message in errors.items()]
            raise RuntimeError("failed to initialize node manager: " + "; ".join(error_lines))
        if all(not node.models for node in self._nodes):
            raise RuntimeError("No nodes provided a valid /models response")

    def image(
        self,
        model_id: str,
        prompt: Any,
        *,
        stream: bool = False,
        decode_images: bool = True,
        timeout_s: Optional[float] = None,
        **params: Any,
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        return self._dispatch(
            model_id=model_id,
            stream=stream,
            invoke=lambda client: client.image(
                model_id=model_id,
                prompt=prompt,
                stream=stream,
                decode_images=decode_images,
                timeout_s=timeout_s,
                **params,
            ),
        )

    def chat(
        self,
        model: str,
        messages: Optional[Sequence[Union[Mapping[str, Any], Message]]],
        *,
        think: bool | Literal["low", "medium", "high"] | None = None,
        tools: Optional[Sequence[Union[Mapping[str, Any], Tool, Callable]]] = None,
        options: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        format: Optional[Union[JsonSchemaValue, Literal["", "json"]]] = None,
        keep_alive: float | str | None = None,
    ) -> Union[Iterator[ChatResponse], ChatResponse]:
        return self._dispatch(
            model_id=model,
            stream=stream,
            invoke=lambda client: client.chat(
                model=model,
                messages=messages,
                think=think,
                tools=tools,
                options=options,
                stream=stream,
                format=format,
                keep_alive=keep_alive,
            ),
        )

    def audio(
        self,
        model_id: str,
        text: Any,
        *,
        language: Any = None,
        speaker: Any = None,
        instruct: Any = None,
        ref_audio: Any = None,
        ref_text: Any = None,
        stream: bool = False,
        keep_alive: float | str | None = None,
        timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        result = self._dispatch(
            model_id=model_id,
            stream=stream,
            invoke=lambda client: client.audio(
                model_id=model_id,
                text=text,
                language=language,
                speaker=speaker,
                instruct=instruct,
                ref_audio=ref_audio,
                ref_text=ref_text,
                stream=stream,
                keep_alive=keep_alive,
                timeout_s=timeout_s,
            ),
        )
        return result  # type: ignore[return-value]

    def models(
        self,
        *,
        timeout_s: Optional[float] = None,
    ) -> AlloyModelsResponse:
        self.refresh_nodes(timeout_s=timeout_s)
        return self._combined_models_response()

    def refresh_nodes(
        self,
        *,
        timeout_s: Optional[float] = None,
        node_names: Optional[Sequence[str]] = None,
    ) -> Dict[str, str]:
        timeout = self._timeout_s if timeout_s is None else timeout_s
        with self._lock:
            node_map = {node.name: node for node in self._nodes}
        selected_names = set(node_names) if node_names else set(node_map.keys())
        selected_nodes = [node for name, node in node_map.items() if name in selected_names]
        if not selected_nodes:
            return {}

        errors: Dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=max(1, min(16, len(selected_nodes)))) as executor:
            futures = {
                executor.submit(node.client.models, timeout_s=timeout): node for node in selected_nodes
            }
            for future in as_completed(futures):
                node = futures[future]
                try:
                    response = future.result()
                except Exception as exc:
                    with self._lock:
                        node.last_refresh_error = str(exc)
                    errors[node.name] = str(exc)
                    continue
                models, categories = _index_models(response)
                with self._lock:
                    node.models = models
                    node.categories_by_model_id = categories
                    node.supported_model_count = sum(
                        1 for model in node.models.values() if model.is_supported
                    )
                    node.remote_active_total = sum(
                        int(model.active_requests) for model in node.models.values()
                    )
                    node.last_refresh_ts = time.time()
                    node.last_refresh_error = None
        return errors

    def _dispatch(
        self,
        *,
        model_id: str,
        stream: bool,
        invoke: Callable[[AlloyClient], Any],
    ) -> Any:
        node = self._select_node_for_model(model_id)
        self._increment_inflight(node, model_id)
        try:
            result = invoke(node.client)
        except Exception:
            self._decrement_inflight(node, model_id)
            raise

        if stream and isinstance(result, Iterator):
            return self._wrap_stream_result(result, node, model_id)

        self._decrement_inflight(node, model_id)
        return result

    def _wrap_stream_result(
        self,
        result: Iterator[Any],
        node: _NodeState,
        model_id: str,
    ) -> Iterator[Any]:
        try:
            for item in result:
                yield item
        finally:
            self._decrement_inflight(node, model_id)

    def _select_node_for_model(self, model_id: str) -> _NodeState:
        with self._lock:
            candidates = [node for node in self._nodes if _is_model_supported(node, model_id)]

        if not candidates:
            self.refresh_nodes()
            with self._lock:
                candidates = [node for node in self._nodes if _is_model_supported(node, model_id)]
            if not candidates:
                raise ValueError(f"No candidate node supports model '{model_id}'")

        if self._mode == NodeQueryMode.QUERY_EVERYTIME:
            self.refresh_nodes(node_names=[node.name for node in candidates])
        elif self._mode == NodeQueryMode.CONTROLLED_QUERYING:
            ranked = sorted(candidates, key=lambda node: self._node_score(node, model_id))
            to_query = ranked[: min(self._max_nodes_to_query, len(ranked))]
            self.refresh_nodes(node_names=[node.name for node in to_query])

        with self._lock:
            candidates = [node for node in self._nodes if _is_model_supported(node, model_id)]
            if not candidates:
                raise ValueError(f"No candidate node supports model '{model_id}'")
            chosen = min(candidates, key=lambda node: self._node_score(node, model_id))
            return chosen

    def _node_score(self, node: _NodeState, model_id: str) -> float:
        model = node.models.get(model_id)
        if model is None or not model.is_supported:
            return float("inf")

        remote_active = int(model.active_requests)
        local_active_model = int(node.local_inflight_by_model.get(model_id, 0))
        active_requests = remote_active + local_active_model

        # Active load is the highest-priority factor.
        load_score = float(active_requests)
        if not model.supports_concurrent_requests:
            load_score *= 10.0

        status_penalty = {
            AllocationStatus.ALLOCATED: 0.0,
            AllocationStatus.DEALLOCATED: 1.0,
            AllocationStatus.QUEUE: 4.0,
        }.get(model.allocation_status, 1.5)

        # Nodes that support fewer models get a slight boost to avoid under-utilization.
        scarcity_bias = max(node.supported_model_count, 1) * 0.01
        weight_bias = -max(node.weight, 0.0) * 0.25
        node_load_bias = float(node.local_inflight_total) * 0.1
        remote_load_bias = float(node.remote_active_total) * 0.01

        return load_score + status_penalty + scarcity_bias + node_load_bias + remote_load_bias + weight_bias

    def _increment_inflight(self, node: _NodeState, model_id: str) -> None:
        with self._lock:
            node.local_inflight_total += 1
            node.local_inflight_by_model[model_id] = (
                node.local_inflight_by_model.get(model_id, 0) + 1
            )

    def _decrement_inflight(self, node: _NodeState, model_id: str) -> None:
        with self._lock:
            if node.local_inflight_total > 0:
                node.local_inflight_total -= 1
            active = node.local_inflight_by_model.get(model_id, 0)
            if active <= 1:
                node.local_inflight_by_model.pop(model_id, None)
            else:
                node.local_inflight_by_model[model_id] = active - 1

    def _combined_models_response(self) -> AlloyModelsResponse:
        with self._lock:
            model_summary: Dict[str, AlloyModel] = {}
            categories: Dict[str, set[Modality]] = {}

            for node in self._nodes:
                for model_id, model in node.models.items():
                    categories.setdefault(model_id, set()).update(
                        node.categories_by_model_id.get(model_id, set())
                    )
                    if model_id not in model_summary:
                        model_summary[model_id] = AlloyModel.model_validate(model.model_dump())
                        continue

                    existing = model_summary[model_id]
                    existing.active_requests += int(model.active_requests)
                    existing.is_supported = existing.is_supported or model.is_supported
                    existing.supports_concurrent_requests = (
                        existing.supports_concurrent_requests
                        or model.supports_concurrent_requests
                    )
                    if existing.allocation_status != AllocationStatus.ALLOCATED:
                        if model.allocation_status == AllocationStatus.ALLOCATED:
                            existing.allocation_status = AllocationStatus.ALLOCATED
                        elif (
                            existing.allocation_status == AllocationStatus.DEALLOCATED
                            and model.allocation_status == AllocationStatus.QUEUE
                        ):
                            existing.allocation_status = AllocationStatus.QUEUE
                    if not existing.capabilities and model.capabilities:
                        existing.capabilities = model.capabilities

            grouped: Dict[Modality, List[AlloyModel]] = {
                Modality.IMAGE: [],
                Modality.AUDIO: [],
                Modality.VIDEO: [],
                Modality.TEXT: [],
            }
            for model_id, model in model_summary.items():
                model_categories = categories.get(model_id)
                if not model_categories and model.capabilities:
                    model_categories = {
                        modality
                        for capability in model.capabilities
                        for modality in capability.outputs
                    }
                for modality in model_categories or set():
                    if modality in grouped:
                        grouped[modality].append(model)

        for models in grouped.values():
            models.sort(key=lambda item: item.model_id)
        return AlloyModelsResponse(
            image=grouped[Modality.IMAGE],
            audio=grouped[Modality.AUDIO],
            video=grouped[Modality.VIDEO],
            text=grouped[Modality.TEXT],
        )


def _is_model_supported(node: _NodeState, model_id: str) -> bool:
    model = node.models.get(model_id)
    return model is not None and bool(model.is_supported)


def _index_models(
    response: AlloyModelsResponse,
) -> tuple[Dict[str, AlloyModel], Dict[str, set[Modality]]]:
    models: Dict[str, AlloyModel] = {}
    categories: Dict[str, set[Modality]] = {}

    grouped_lists = {
        Modality.IMAGE: response.image,
        Modality.AUDIO: response.audio,
        Modality.VIDEO: response.video,
        Modality.TEXT: response.text,
    }
    for modality, model_list in grouped_lists.items():
        for model in model_list:
            if model.model_id not in models:
                models[model.model_id] = AlloyModel.model_validate(model.model_dump())
            categories.setdefault(model.model_id, set()).add(modality)
            if not models[model.model_id].capabilities and model.capabilities:
                models[model.model_id].capabilities = [
                    ModelCapability.model_validate(capability.model_dump())
                    for capability in model.capabilities
                ]

    return models, categories
