"""Model registry and .env loading (stdlib only)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RUNS_DIR = RESULTS / "runs"
RAW_DIR = RESULTS / "raw"


def load_env(path: Path = ROOT / ".env") -> None:
    """Load KEY=VALUE lines from .env into os.environ without overriding variables already set."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip().removeprefix("export ").strip(), v.strip().strip('"').strip("'")
        if k and v and k not in os.environ:
            os.environ[k] = v


@dataclass
class ModelSpec:
    id: str
    label: str
    provider: str            # anthropic | openai | gemini
    family: str              # Claude | GPT | Gemini
    max_tokens: int = 4096
    extra: dict = field(default_factory=dict)

    @property
    def slug(self) -> str:
        return self.id.replace("/", "_").replace(":", "_")


def load_models(path: Path = Path(__file__).with_name("models.json")) -> list[ModelSpec]:
    data = json.loads(path.read_text())
    return [ModelSpec(**m) for m in data["models"]]


def find_model(name: str, models: list[ModelSpec] | None = None) -> ModelSpec:
    """Look up a model by id or label; an unknown id is accepted if it starts with a known prefix."""
    models = models if models is not None else load_models()
    for m in models:
        if name in (m.id, m.label):
            return m
    for prefix, prov, fam in (("claude", "anthropic", "Claude"), ("gpt", "openai", "GPT"),
                              ("o", "openai", "GPT"), ("gemini", "gemini", "Gemini")):
        if name.startswith(prefix):
            return ModelSpec(id=name, label=name, provider=prov, family=fam)
    raise KeyError(f"unknown model {name!r}; add it to llm_bench/models.json")
