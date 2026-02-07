# alloy-client

A lightweight Python client for the Alloy server.

## Install

```bash
pip install alloyai-client
```

For local development:

```bash
pip install -e .
```

## Quick start

```python
from alloyai_client import AlloyClient

client = AlloyClient("http://127.0.0.1:8000")

# Images
result = client.image(model_id="qwen-image", prompt="a cinematic portrait")
with open("output.png", "wb") as f:
    f.write(result["images"][0])

# Chat (non-streaming)
response = client.chat(
    model="qwen3-medium",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response["message"]["content"])

# Audio (non-streaming)
audio = client.audio(
    model_id="qwen3-tts-base",
    text="Hello from Alloy",
)
print(audio["sample_rate"], len(audio["outputs"]))

# Node metadata
models = client.models()
print([model.model_id for model in models.image])
```

## Distributed node manager

```python
from alloyai_client import AlloyNodeManager, NodeConfig, NodeQueryMode

manager = AlloyNodeManager(
    nodes=[
        NodeConfig(base_url="http://node0:8000", name="node0", weight=1.0),
        NodeConfig(base_url="http://node1:8000", name="node1", weight=1.0),
    ],
    mode=NodeQueryMode.CONTROLLED_QUERYING,
    max_nodes_to_query=2,
)

result = manager.image(model_id="qwen-image", prompt="a cinematic portrait")
```

## Types

The client re-exports minimal Ollama-style types so you can annotate inputs without
pulling in the `ollama` dependency:

```python
from alloyai_client import Message, JsonSchemaValue, AlloyModelsResponse
```

## Notes

- Streaming is supported for `/image` only.
- `/chat` and `/audio` are non-streaming for now.
