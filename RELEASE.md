# Release Guide

This package uses `pyproject.toml` (no `setup.py` needed).

## Build

```bash
python -m build
```

Artifacts will appear in `dist/` as a wheel and sdist.

## Publish to PyPI

```bash
python -m twine upload dist/*
```

## Publish to TestPyPI (optional)

```bash
python -m twine upload --repository testpypi dist/*
```

## Notes

- You need a PyPI account and an API token.
- Set `TWINE_USERNAME=__token__` and `TWINE_PASSWORD=<your-token>` or use a `.pypirc`.
