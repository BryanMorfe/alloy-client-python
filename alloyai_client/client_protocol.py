from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Union,
    runtime_checkable,
)

from .types import AlloyModelsResponse, ChatResponse, JsonSchemaValue, Message, Tool


@runtime_checkable
class AlloyClientProtocol(Protocol):
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
        ...

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
        ...

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
        ...

    def models(
        self,
        *,
        timeout_s: Optional[float] = None,
    ) -> AlloyModelsResponse:
        ...
