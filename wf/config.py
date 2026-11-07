"""Config loading: built-in defaults, overridden by config.toml (or $WORDFORGE_CONFIG)."""
from __future__ import annotations

import copy
import os
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

ROOT = Path(__file__).resolve().parent.parent

POS_LIST = ["noun", "adjective", "verb", "adverb", "pronoun",
            "article", "preposition", "conjunction", "interjection"]

DEFAULTS = {
    "paths": {"model_dir": "model", "data_dir": "data"},
    "server": {"host": "0.0.0.0", "port": 80, "domain": "", "rate_limit_per_minute": 60, "log_level": "info"},
    "https": {"enabled": False, "port": 443, "certfile": "", "keyfile": "", "keyfile_password": "",
              "redirect_http": True, "hsts": True},
    "hardware": {"device": "auto", "device_index": 0, "low_vram_gib": 6.5, "min_compute_capability": 5.0},
    "data": {
        "max_corpus_gb": 4.0, "keep_raw": False, "rebuild": False, "max_def_chars": 220,
        "min_def_chars": 6, "skip_inflections": True,
        "sources": [
            {"name": "wiktionary-en", "type": "kaikki_raw",
             "url": "https://kaikki.org/dictionary/raw-wiktextract-data.jsonl.gz"},
            {"name": "wordnet", "type": "wordnet"},
        ],
        # Merged later by "train more" (see README). Same verified kaikki.org dump as the first source.
        "extra_in_first_build": False,
        "extra_sources": [
            {"name": "wiktionary-latin", "type": "kaikki_raw", "lang_codes": ["la"], "latin_only": True,
             "url": "https://kaikki.org/dictionary/raw-wiktextract-data.jsonl.gz", "enabled": True},
            {"name": "wiktionary-european", "type": "kaikki_raw",
             "lang_codes": ["fr", "de", "es", "it", "pt", "nl"], "latin_only": True,
             "url": "https://kaikki.org/dictionary/raw-wiktextract-data.jsonl.gz", "enabled": False},
        ],
    },
    "train": {
        "preset": "auto", "max_minutes": 90, "epochs": 0, "batch_size": 0, "learning_rate": 0.0,
        "max_seq_len": 96, "vocab_size": 8192, "prompt_weight": 0.15, "params_per_token": 4.0,
        "max_model_gb": 4.5, "compile": "auto", "val_entries": 2000, "seed": 1337,
        "ask_to_train_more": True, "ask_timeout_seconds": 20,
        "resume_epochs": 2, "resume_lr_factor": 0.3,
    },
    "generate": {"candidates": 16, "temperature": 0.85, "top_k": 40, "top_p": 0.95,
                 "max_word_tokens": 20, "allow_existing_words": False},
}


def _merge(base: dict, over: dict) -> dict:
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def config_path(path=None) -> Path:
    return Path(path or os.environ.get("WORDFORGE_CONFIG") or ROOT / "config.toml").expanduser().resolve()


def load(path=None) -> dict:
    p = config_path(path)
    cfg = copy.deepcopy(DEFAULTS)
    if p.exists():
        with open(p, "rb") as f:
            _merge(cfg, tomllib.load(f))
    else:
        print(f"[config] {p} not found, using built-in defaults")
    cfg["_config_file"] = str(p)
    for key in ("model_dir", "data_dir"):
        d = Path(cfg["paths"][key]).expanduser()
        cfg["paths"][key] = str(d if d.is_absolute() else (ROOT / d).resolve())
    return cfg
