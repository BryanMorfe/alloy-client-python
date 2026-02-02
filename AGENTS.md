# Agent Guide

This repo contains the `alloy_client` package for talking to an Alloy server.

## Quick commands
- Install (client-only): `pip install -e .`
- Run type-checking/tests: project-specific (none by default)

## Project notes
- Keep dependencies minimal (no `ollama`, `diffusers`, etc.).
- Types live in `alloy_client/types.py` and mirror common Ollama response fields.
- Streaming is supported only for `/image`.

## Changes
- Keep diffs focused and update README examples when behavior changes.
