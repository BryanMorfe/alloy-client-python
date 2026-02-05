import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from ..alloyai_client import AlloyClient

DEFAULT_MODELS = [
    "gpt-oss-20b",
    "qwen3-medium",
    "qwen3-lite",
    "gpt-oss-120b",
]

DEFAULT_MESSAGES = [
    {"role": "user", "content": "Give me a one-sentence fun fact about space."},
]


def _load_json_env(name: str) -> Optional[Any]:
    raw = os.getenv(name)
    if not raw:
        return None
    return json.loads(raw)


def _models_from_env() -> List[str]:
    raw = os.getenv("ALLOY_MODELS")
    if raw:
        return [item.strip() for item in raw.split(",") if item.strip()]
    return list(DEFAULT_MODELS)


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _save_response(output_dir: str, request_tag: str, response: Dict[str, Any]) -> None:
    path = os.path.join(output_dir, f"{request_tag}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(response, handle, indent=2, ensure_ascii=True)
    print(f"saved {path}")


def _run_request(
    client: AlloyClient,
    model_id: str,
    request_index: int,
    messages: List[Dict[str, Any]],
    options: Optional[Dict[str, Any]],
    keep_alive: Optional[str],
    output_dir: str,
) -> None:
    request_tag = f"{model_id}_{request_index}_{uuid.uuid4().hex[:8]}"
    response = client.chat(
        model=model_id,
        messages=messages,
        options=options,
        keep_alive=keep_alive,
    )
    if not isinstance(response, dict):
        raise RuntimeError(f"{request_tag} returned unexpected response type")
    _save_response(output_dir, request_tag, response)


def main() -> int:
    base_url = os.getenv("ALLOY_BASE_URL", "http://127.0.0.1:8000")
    output_dir = os.getenv("ALLOY_OUTPUT_DIR", "client_chat_outputs")
    os.makedirs(output_dir, exist_ok=True)

    models = _models_from_env()
    messages = _load_json_env("ALLOY_CHAT_MESSAGES_JSON") or DEFAULT_MESSAGES
    options = _load_json_env("ALLOY_CHAT_OPTIONS_JSON")
    keep_alive = os.getenv("ALLOY_KEEP_ALIVE")

    concurrency = _int_env("ALLOY_CONCURRENCY", 4)
    per_model = _int_env("ALLOY_REQUESTS_PER_MODEL", 2)

    client = AlloyClient(base_url)
    start = time.monotonic()

    futures = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        for model_id in models:
            for request_index in range(per_model):
                futures.append(
                    executor.submit(
                        _run_request,
                        client,
                        model_id,
                        request_index,
                        messages,
                        options,
                        keep_alive,
                        output_dir,
                    )
                )

        for future in as_completed(futures):
            future.result()

    elapsed = time.monotonic() - start
    print(f"completed {len(futures)} requests in {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
