from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Union,
)

from .types import AlloyModelsResponse, ChatResponse, JsonSchemaValue, Message, Tool


class AlloyClientError(RuntimeError):
    def __init__(self, status_code: int, message: str, body: Optional[str] = None) -> None:
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code
        self.body = body


class AlloyClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        *,
        timeout_s: float = 300.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s

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
        payload: Dict[str, Any] = {
            "model_id": model_id,
            "prompt": prompt,
            "stream": stream,
        }
        payload.update(params)

        if stream:
            response = self._post("/image", payload, stream=True, timeout_s=timeout_s)
            return self._stream_events(response, decode_images)

        response = self._post("/image", payload, stream=False, timeout_s=timeout_s)
        with response:
            data = self._read_json(response)
        return self._maybe_decode_images(data, decode_images)

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
        if stream:
            raise ValueError("Streaming chat is not supported yet")

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        if think is not None:
            payload["think"] = think
        if tools is not None:
            payload["tools"] = tools
        if options is not None:
            payload["options"] = options
        if format is not None:
            payload["format"] = format
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive

        response = self._post("/chat", payload, stream=False, timeout_s=None)
        with response:
            return self._read_json(response)

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
        if stream:
            raise ValueError("Streaming audio is not supported yet")

        payload: Dict[str, Any] = {
            "model_id": model_id,
            "text": text,
            "stream": stream,
        }
        if language is not None:
            payload["language"] = language
        if speaker is not None:
            payload["speaker"] = speaker
        if instruct is not None:
            payload["instruct"] = instruct
        if ref_audio is not None:
            payload["ref_audio"] = ref_audio
        if ref_text is not None:
            payload["ref_text"] = ref_text
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive

        response = self._post("/audio", payload, stream=False, timeout_s=timeout_s)
        with response:
            return self._read_json(response)

    def models(
        self,
        *,
        timeout_s: Optional[float] = None,
    ) -> AlloyModelsResponse:
        response = self._get("/models", timeout_s=timeout_s)
        with response:
            payload = self._read_json(response)
        return AlloyModelsResponse.model_validate(payload)

    def _post(
        self,
        path: str,
        payload: Dict[str, Any],
        *,
        stream: bool,
        timeout_s: Optional[float],
    ):
        url = f"{self._base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
        }
        request = urllib.request.Request(url, data=data, method="POST", headers=headers)
        timeout = self._timeout_s if timeout_s is None else timeout_s
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "ignore")
            message = body or exc.reason
            raise AlloyClientError(exc.code, message, body=body) from None

    def _get(
        self,
        path: str,
        *,
        timeout_s: Optional[float],
    ):
        url = f"{self._base_url}{path}"
        request = urllib.request.Request(url, method="GET")
        timeout = self._timeout_s if timeout_s is None else timeout_s
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "ignore")
            message = body or exc.reason
            raise AlloyClientError(exc.code, message, body=body) from None

    def _read_json(self, response) -> Dict[str, Any]:
        raw = response.read()
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _maybe_decode_images(self, data: Dict[str, Any], decode_images: bool) -> Dict[str, Any]:
        if not decode_images:
            return data
        images = data.get("images")
        if isinstance(images, list):
            data = dict(data)
            data["images"] = [base64.b64decode(item) for item in images]
        return data

    def _stream_events(
        self,
        response,
        decode_images: bool,
    ) -> Iterator[Dict[str, Any]]:
        try:
            for event in self._iter_sse(response):
                payload = event.get("payload")
                if decode_images and isinstance(payload, dict):
                    images = payload.get("images")
                    if isinstance(images, list):
                        payload = dict(payload)
                        payload["images"] = [base64.b64decode(item) for item in images]
                        event = dict(event)
                        event["payload"] = payload
                yield event
        finally:
            response.close()

    def _iter_sse(self, response) -> Iterable[Dict[str, Any]]:
        event: Dict[str, Any] = {}
        for raw_line in response:
            line = raw_line.decode("utf-8", "ignore").strip()
            if not line:
                if event:
                    yield event
                    event = {}
                continue
            if line.startswith("data:"):
                data = line[len("data:") :].strip()
                if not data:
                    continue
                event["payload"] = json.loads(data)
            elif line.startswith("event:"):
                event["event"] = line[len("event:") :].strip()
        if event:
            yield event
