"""Explicit, read-only discovery using the server's existing OpenAI credential."""

import json
import re

import httpx


class ModelCatalogError(ValueError):
    pass


async def list_openai_models(api_key: str) -> list[str]:
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=False) as client:
            async with client.stream(
                "GET",
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            ) as response:
                response.raise_for_status()
                raw = bytearray()
                async for chunk in response.aiter_bytes():
                    raw.extend(chunk)
                    if len(raw) > 1024 * 1024:
                        raise ValueError("Model list exceeds the size limit")
        data = json.loads(raw)
        if not isinstance(data, dict) or not isinstance(data.get("data"), list):
            raise ValueError("Invalid model list")
        models = set()
        for item in data["data"]:
            identity = item.get("id") if isinstance(item, dict) else None
            if (
                isinstance(identity, str)
                and re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", identity)
                and api_key not in identity
            ):
                models.add(identity)
        return sorted(models)
    except httpx.HTTPError, ValueError, TypeError:
        raise ModelCatalogError(
            "OpenAI models could not be loaded. Check the saved API key and retry, "
            "or enter a model ID manually."
        ) from None
