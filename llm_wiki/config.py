"""Configuration for an LLM-Wiki instance.

Hyperparameter defaults follow the paper (arXiv 2605.25480):
SelectPages k=5, agent tool budget Tmax=15, search patience P=3,
LLM periodic repair every N=10 articles, 3 finalization repair rounds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

DEFAULT_MODEL = "claude-opus-5"

CONFIG_FILENAME = "wiki.json"


@dataclass
class WikiConfig:
    root: Path

    provider: str = "anthropic"  # "anthropic" | "ollama"
    compiler_model: str = DEFAULT_MODEL
    agent_model: str = DEFAULT_MODEL
    judge_model: str = DEFAULT_MODEL
    ollama_base_url: str = "http://localhost:11434"
    ollama_num_ctx: int = 16384

    select_pages_k: int = 5
    agent_max_tool_calls: int = 15
    agent_patience: int = 3
    repair_every_n_articles: int = 10
    finalization_rounds: int = 3
    contradiction_sample_pairs: int = 20
    # Delete facts the judge model deems unsupported. Keep off for weak local
    # judges — a wrong judgment destroys good facts; findings are always
    # recorded in the error book either way.
    judge_autoremove: bool = True
    shed_cooldown_seconds: float = 120.0

    max_output_tokens: int = 8192

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    # --- layout -------------------------------------------------------------

    @property
    def pages_dir(self) -> Path:
        return self.root / "pages"

    @property
    def sources_dir(self) -> Path:
        return self.root / "sources"

    @property
    def digests_dir(self) -> Path:
        return self.sources_dir / "digests"

    @property
    def articles_dir(self) -> Path:
        return self.sources_dir / "articles"

    @property
    def error_book_path(self) -> Path:
        return self.root / "error_book.yaml"

    @property
    def config_path(self) -> Path:
        return self.root / CONFIG_FILENAME

    # --- persistence --------------------------------------------------------

    def save(self) -> None:
        data = asdict(self)
        data["root"] = str(self.root)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(json.dumps(data, indent=2) + "\n")

    @classmethod
    def load(cls, root: Path | str) -> "WikiConfig":
        root = Path(root)
        path = root / CONFIG_FILENAME
        if path.exists():
            data = json.loads(path.read_text())
            data["root"] = root
            return cls(**data)
        return cls(root=root)
