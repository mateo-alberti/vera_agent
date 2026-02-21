from __future__ import annotations

from typing import Any, Iterable, Mapping

def build_langsmith_config(
    *,
    tags: Iterable[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if tags:
        clean_tags = [tag for tag in tags if tag]
        if clean_tags:
            config["tags"] = clean_tags
    if metadata:
        config["metadata"] = dict(metadata)
    return config
