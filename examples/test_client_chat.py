import json
import os
from typing import Any, Dict, List, Optional

from ..alloyai_client import AlloyClient

DEFAULT_MODELS = [
    "gpt-oss-20b",
    "qwen3-medium",
    "qwen3-lite",
    "gpt-oss-120b",
]

DEFAULT_MESSAGES = [
    {"role": "user", "content": "Say hello and tell me the current weekday."},
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


def _save_response(output_dir: str, model_id: str, response: Dict[str, Any]) -> None:
    path = os.path.join(output_dir, f"{model_id}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(response, handle, indent=2, ensure_ascii=True)
    print(f"saved {path}")


def main() -> int:
    base_url = os.getenv("ALLOY_BASE_URL", "http://127.0.0.1:8000")
    output_dir = os.getenv("ALLOY_OUTPUT_DIR", "client_chat_outputs")
    os.makedirs(output_dir, exist_ok=True)

    models = _models_from_env()
    messages = _load_json_env("ALLOY_CHAT_MESSAGES_JSON") or DEFAULT_MESSAGES
    options = _load_json_env("ALLOY_CHAT_OPTIONS_JSON")
    keep_alive = os.getenv("ALLOY_KEEP_ALIVE")

    client = AlloyClient(base_url)

    for model_id in models:
        response = client.chat(
            model=model_id,
            messages=messages,
            options=options,
            keep_alive=keep_alive,
        )
        if not isinstance(response, dict):
            raise RuntimeError(f"{model_id} returned unexpected response type")
        _save_response(output_dir, model_id, response)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
