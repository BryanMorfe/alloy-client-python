import os
import sys
from typing import Any, Dict, Iterable, List

from ..alloyai_client import AlloyClient

PROMPT = """
A 24-year-old adult East Asian girl with delicate, charming features and large, bright eyes—expressive and lively, \
with a cheerful or subtly smiling expression. Her naturally wavy long hair is either loose or tied in \
twin ponytails behind her, reaching her shoulders. She has fair skin and light makeup accentuating her youthful freshness. She is wearing casual, \
comfortable clothing suitable for lounging at home. She stands indoors in her bedroom, an anime posters-filled room. \
She is lifting her hands up in the air with a joyful expression, as if stretching after waking up from a restful sleep in the morning \
sunlight streaming through the window.
""".strip()

# Modify to test different models
MODELS = ["z-image-turbo", "z-image"]


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _save_images(images: Iterable[bytes], output_dir: str, prefix: str) -> None:
    for idx, image in enumerate(images):
        path = os.path.join(output_dir, f"{prefix}_{idx}.png")
        with open(path, "wb") as handle:
            handle.write(image)
        print(f"saved {path}")


def _run_non_stream(client: AlloyClient, model_id: str, params: Dict[str, Any], output_dir: str) -> None:
    response = client.image(model_id=model_id, stream=False, **params)
    images = response.get("images")
    if not images:
        raise RuntimeError(f"{model_id} returned no images")
    _save_images(images, output_dir, model_id)


def _run_stream(client: AlloyClient, model_id: str, params: Dict[str, Any], output_dir: str) -> None:
    for event in client.image(model_id=model_id, stream=True, **params):
        event_type = event.get("event")
        payload = event.get("payload") or {}
        if event_type == "progress":
            step = payload.get("step")
            total = payload.get("total_steps")
            progress = payload.get("progress")
            if total is not None and step is not None:
                pct = int(float(progress or 0) * 100)
                print(f"{model_id} progress: step {int(step) + 1}/{int(total)} ({pct}%)")
            elif step is not None:
                print(f"{model_id} progress: step {int(step) + 1}")
            continue
        if event_type in {"received", "queued", "allocated"}:
            print(f"{model_id} event: {event_type} {payload}")
        if event_type == "completed":
            images = payload.get("images")
            if not images:
                raise RuntimeError(f"{model_id} completed without images")
            _save_images(images, output_dir, model_id)
            return
        if event_type == "error":
            payload = event.get("payload") or {}
            message = payload.get("message") or "unknown error"
            raise RuntimeError(f"{model_id} failed: {message}")
    raise RuntimeError(f"{model_id} stream ended without completion")


def main() -> int:
    base_url = os.getenv("ALLOY_BASE_URL", "http://127.0.0.1:8000")
    output_dir = os.getenv("ALLOY_OUTPUT_DIR", "client_outputs")
    os.makedirs(output_dir, exist_ok=True)

    params_zturbo: Dict[str, Any] = {
        "prompt": PROMPT,
        "width": 1024,
        "height": 1536,
        "num_inference_steps": 9,
        "num_images_per_prompt": 2,
        "keep_alive": 60,
    }

    params_zimage: Dict[str, Any] = {
        "prompt": PROMPT,
        "width": 1024,
        "height": 1536,
        "num_inference_steps": 50,
        "num_images_per_prompt": 2,
        "keep_alive": 60,
    }

    params_qwen: Dict[str, Any] = {
        "prompt": PROMPT,
        "width": 1024,
        "height": 1536,
        "num_inference_steps": 50,
        "num_images_per_prompt": 2,
        "keep_alive": 60,
    }

    params_flux2: Dict[str, Any] = {
        "prompt": PROMPT,
        "width": 1024,
        "height": 1536,
        "num_inference_steps": 50,
        "num_images_per_prompt": 2,
        "keep_alive": 60,
    }

    params_flux2_turbo: Dict[str, Any] = {
        "prompt": PROMPT,
        "width": 1024,
        "height": 1536,
        "num_inference_steps": 9,
        "num_images_per_prompt": 2,
        "keep_alive": 60,
    }

    client = AlloyClient(base_url)
    use_stream = _bool_env("ALLOY_STREAM", default=_bool_env("ALLOW_STREAM", default=False))

    for model_id in MODELS:
        if model_id == "z-image-turbo":
            params = params_zturbo
        elif model_id == "z-image":
            params = params_zimage
        elif model_id == "qwen-image":
            params = params_qwen
        elif model_id == "flux2-dev":
            params = params_flux2
        elif model_id == "flux2-dev-turbo":
            params = params_flux2_turbo
        else:
            raise RuntimeError(f"Unknown model_id: {model_id}")

        if use_stream:
            _run_stream(client, model_id, params, output_dir)
        else:
            _run_non_stream(client, model_id, params, output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
