"""Dataset-specific adapters and protocols."""

from importlib import import_module


MODULES = {
    "ebnerd": "ebnerd",
    "kuairand-1k": "kuairand",
    "music-crs-2026": "music_crs_2026",
    "synerise-2025": "synerise_2025",
}


def get_dataset(name):
    try:
        return import_module(f"{__name__}.{MODULES[name]}")
    except KeyError as exc:
        raise ValueError(f"unknown dataset: {name}") from exc
