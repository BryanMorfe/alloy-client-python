import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
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
MODEL_PARAMS: Dict[str, Dict[str, Any]] = {
    "z-image-turbo": {
        "prompt": PROMPT,
        "width": 1024,
        "height": 1536,
        "num_inference_steps": 9,
        "keep_alive": 120,
    },
    "flux2-dev-turbo": {
        "prompt": PROMPT,
        "width": 1024,
        "height": 1536,
        "num_inference_steps": 9,
        "keep_alive": 120,
    },
    "qwen-image": {
        "prompt": PROMPT,
        "width": 1024,
        "height": 1536,
        "num_inference_steps": 50,
        "true_cfg_scale": 4.0,
        "keep_alive": 120,
    },
    "flux2-dev": {
        "prompt": PROMPT,
        "width": 1024,
        "height": 1536,
        "num_inference_steps": 50,
        "keep_alive": 120,
    },
}


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _save_images(images: Iterable[bytes], output_dir: str, prefix: str) -> None:
    for idx, image in enumerate(images):
        path = os.path.join(output_dir, f"{prefix}_{idx}.png")
        with open(path, "wb") as handle:
            handle.write(image)
        print(f"saved {path}")


def _run_non_stream(
    client: AlloyClient,
    model_id: str,
    request_tag: str,
    params: Dict[str, Any],
    output_dir: str,
) -> None:
    response = client.image(model_id=model_id, stream=False, **params)
    images = response.get("images")
    if not images:
        raise RuntimeError(f"{request_tag} returned no images")
    _save_images(images, output_dir, request_tag)


def _run_stream(
    client: AlloyClient,
    model_id: str,
    request_tag: str,
    params: Dict[str, Any],
    output_dir: str,
) -> None:
    for event in client.image(model_id=model_id, stream=True, **params):
        event_type = event.get("event")
        payload = event.get("payload") or {}
        if event_type == "progress":
            step = payload.get("step")
            total = payload.get("total_steps")
            progress = payload.get("progress")
            if total is not None and step is not None:
                pct = int(float(progress or 0) * 100)
                print(f"{request_tag} progress: step {int(step) + 1}/{int(total)} ({pct}%)")
            elif step is not None:
                print(f"{request_tag} progress: step {int(step) + 1}")
            continue
        if event_type in {"received", "queued", "allocated"}:
            print(f"{request_tag} event: {event_type} {payload}")
        if event_type == "completed":
            images = payload.get("images")
            if not images:
                raise RuntimeError(f"{request_tag} completed without images")
            _save_images(images, output_dir, request_tag)
            return
        if event_type == "error":
            message = payload.get("message") or "unknown error"
            raise RuntimeError(f"{request_tag} failed: {message}")
    raise RuntimeError(f"{request_tag} stream ended without completion")


def _run_request(
    client: AlloyClient,
    model_id: str,
    request_index: int,
    use_stream: bool,
    output_dir: str,
) -> None:
    params = dict(MODEL_PARAMS[model_id])
    request_tag = f"{model_id}_{request_index}_{uuid.uuid4().hex[:8]}"
    if use_stream:
        _run_stream(client, model_id, request_tag, params, output_dir)
    else:
        _run_non_stream(client, model_id, request_tag, params, output_dir)


def main() -> int:
    base_url = os.getenv("ALLOY_BASE_URL", "http://127.0.0.1:8000")
    output_dir = os.getenv("ALLOY_OUTPUT_DIR", "client_outputs")
    os.makedirs(output_dir, exist_ok=True)

    use_stream = _bool_env("ALLOY_STREAM", default=_bool_env("ALLOW_STREAM", default=False))
    concurrency = _int_env("ALLOY_CONCURRENCY", 4)
    per_model = _int_env("ALLOY_REQUESTS_PER_MODEL", 2)

    models = os.getenv("ALLOY_MODELS")
    if models:
        model_ids = [item.strip() for item in models.split(",") if item.strip()]
    else:
        model_ids = list(MODEL_PARAMS.keys())

    for model_id in model_ids:
        if model_id not in MODEL_PARAMS:
            raise ValueError(f"Unknown model_id '{model_id}'")

    client = AlloyClient(base_url)
    start = time.monotonic()

    futures = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        for model_id in model_ids:
            for request_index in range(per_model):
                futures.append(
                    executor.submit(
                        _run_request,
                        client,
                        model_id,
                        request_index,
                        use_stream,
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
