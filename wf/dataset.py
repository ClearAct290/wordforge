"""Dictionary building: download dictionaries, keep word / part of speech / first definition, merge, dedupe.

Output: <data_dir>/dictionary.txt, one entry per line:  word <TAB> part-of-speech <TAB> first definition
State:  <data_dir>/dictionary_state.json remembers which sources are already merged, so "train more" only
        has to fetch the extra dictionaries, never the big first download again.
"""
from __future__ import annotations

import gzip
import io
import json
import time
from pathlib import Path

import requests
from tqdm import tqdm

from .config import POS_LIST
from .textutil import normalize_def, valid_word

# kaikki / wiktextract part-of-speech codes -> WordForge's nine
POS_MAP = {"noun": "noun", "verb": "verb", "adj": "adjective", "adv": "adverb", "pron": "pronoun",
           "article": "article", "prep": "preposition", "conj": "conjunction", "intj": "interjection"}
WN_POS = {"n": "noun", "v": "verb", "a": "adjective", "s": "adjective", "r": "adverb"}
JSON_POS = {**{p: p for p in POS_LIST}, "n": "noun", "v": "verb", "a": "adjective", "adj": "adjective",
            "adv": "adverb", "pron": "pronoun", "prep": "preposition", "conj": "conjunction",
            "interj": "interjection", "intj": "interjection"}

SKIP_TAGS = {"form-of", "alt-of", "misspelling", "abbreviation", "initialism", "acronym", "alternative"}
SKIP_PREFIX = ("alternative ", "obsolete spelling", "obsolete form", "archaic form", "archaic spelling",
               "misspelling", "common misspelling", "eye dialect", "nonstandard spelling", "synonym of",
               "abbreviation of", "initialism of", "acronym of", "clipping of", "plural of", "past tense of")


def is_latin(word: str) -> bool:
    return all(ord(c) < 0x250 for c in word)       # Basic Latin .. Latin Extended-B


# --------------------------------------------------------------------------- downloading
def download(url: str, dest: Path, retries: int = 4) -> Path:
    """Resumable download with a progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            have = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={have}-"} if have else {}
            with requests.get(url, stream=True, headers=headers, timeout=60) as r:
                if r.status_code == 416:  # already complete
                    break
                r.raise_for_status()
                if have and r.status_code != 206:
                    have = 0  # server ignored Range
                total = int(r.headers.get("content-length", 0)) + have
                with open(part, "ab" if have else "wb") as f, tqdm(
                        total=total or None, initial=have, unit="B", unit_scale=True,
                        desc=f"download {dest.name}") as bar:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                        bar.update(len(chunk))
            break
        except (requests.RequestException, OSError) as e:
            print(f"[data] download failed ({e}); retry {attempt}/{retries}")
            time.sleep(3 * attempt)
    else:
        raise RuntimeError(f"could not download {url}")
    part.replace(dest)
    return dest


# --------------------------------------------------------------------------- source readers
# Each reader yields (word, pos, definition) in "first wins" order.
def iter_kaikki(path: Path, skip_inflections: bool, lang_codes=("en",), latin_only: bool = False):
    codes = set(lang_codes)
    pats = [p for c in codes for p in (f'"lang_code": "{c}"'.encode(), f'"lang_code":"{c}"'.encode())]
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as raw:
        for line in io.BufferedReader(raw, 1 << 22):
            # cheap prefilter: the raw dump holds every language
            if not any(p in line for p in pats):
                continue
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("lang_code") not in codes:
                continue
            pos = POS_MAP.get(e.get("pos", ""))
            word = e.get("word", "")
            if not pos or not valid_word(word) or (latin_only and not is_latin(word)):
                continue
            for s in e.get("senses", ()):
                g = s.get("glosses")
                if not g or len(g) != 1:        # top-level senses only
                    continue
                if skip_inflections and (s.get("form_of") or s.get("alt_of")
                                         or SKIP_TAGS.intersection(s.get("tags", ()))):
                    continue
                yield word, pos, g[0]
                break


def iter_wordnet():
    import nltk
    from nltk.corpus import wordnet as wn
    try:
        wn.ensure_loaded()
    except LookupError:
        nltk.download("wordnet", quiet=True)
        wn.ensure_loaded()
    for lemma in wn.all_lemma_names():
        if "_" in lemma:
            continue
        syn = wn.synsets(lemma)
        if syn and syn[0].pos() in WN_POS:
            yield lemma, WN_POS[syn[0].pos()], syn[0].definition()


def iter_tsv(path: Path):
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 2)
            if len(parts) == 3:
                yield parts[0].strip().lower(), parts[1].strip().lower(), parts[2]


def iter_json_words(path: Path):
    """{"word": [{"partOfSpeech": "noun", "text": "definition"}, ...]} - the OPTED-to-JSON / Webster 1913 layout.
    `path` can be one .json file or a folder of them."""
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        for word, entries in data.items():
            for ent in entries:
                pos = JSON_POS.get(str(ent.get("partOfSpeech", "")).strip().lower().rstrip("."))
                text = ent.get("text") or ent.get("definition")
                if pos and text:
                    yield word.strip().lower(), pos, text
                    break


def _fetch(url: str, data_dir: Path, raw_files: list) -> Path:
    path = download(url, data_dir / "raw" / Path(url.split("?")[0]).name)
    raw_files.append(path)
    return path


def _open_source(src: dict, d: dict, data_dir: Path, raw_files: list):
    kind = src["type"]
    if kind == "kaikki_raw":
        path = _fetch(src["url"], data_dir, raw_files)
        return iter_kaikki(path, d["skip_inflections"], tuple(src.get("lang_codes", ["en"])),
                           bool(src.get("latin_only", src.get("lang_codes", ["en"]) != ["en"])))
    if kind == "wordnet":
        return iter_wordnet()
    if kind in ("tsv", "json_words"):
        if kind == "json_words" and src.get("urls"):
            for u in src["urls"]:
                _fetch(u, data_dir, raw_files)
            path = data_dir / "raw"
        elif src.get("url"):
            path = _fetch(src["url"], data_dir, raw_files)
        else:
            path = Path(src["path"]).expanduser()
        return iter_tsv(path) if kind == "tsv" else iter_json_words(path)
    raise ValueError(f"unknown source type '{kind}'")


# --------------------------------------------------------------------------- state + builder
def _state_path(cfg: dict) -> Path:
    return Path(cfg["paths"]["data_dir"]) / "dictionary_state.json"


def _read_state(cfg: dict) -> dict:
    p = _state_path(cfg)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            pass
    # dictionary.txt without a state file: assume the main sources went in
    return {"sources": [s.get("name", s["type"]) for s in cfg["data"]["sources"]]}


def _extras(cfg: dict) -> list[dict]:
    return [s for s in cfg["data"].get("extra_sources", []) if s.get("enabled", True)]


def pending_sources(cfg: dict) -> list[str]:
    """Names of enabled extra dictionaries that are not in dictionary.txt yet."""
    done = set(_read_state(cfg)["sources"])
    return [s.get("name", s["type"]) for s in _extras(cfg) if s.get("name", s["type"]) not in done]


def _ingest(cfg, sources, seen, fout, written, cap, raw_files):
    d, data_dir = cfg["data"], Path(cfg["paths"]["data_dir"])
    used = []
    for src in sources:
        if written >= cap:
            print(f"[data] corpus cap reached ({written / 1e9:.2f} GB), skipping {src.get('name')}")
            break
        name = src.get("name", src["type"])
        print(f"[data] source '{name}' ({src['type']})")
        try:
            it = _open_source(src, d, data_dir, raw_files)
            added = 0
            for word, pos, definition in it:
                if word in seen or pos not in POS_LIST:
                    continue
                text = normalize_def(definition, d["max_def_chars"])
                if len(text) < d["min_def_chars"]:
                    continue
                if d["skip_inflections"] and text.startswith(SKIP_PREFIX):
                    continue
                line = f"{word}\t{pos}\t{text.replace(chr(9), ' ')}\n"
                seen.add(word)
                fout.write(line)
                written += len(line.encode("utf-8"))
                added += 1
                if added % 200_000 == 0:
                    print(f"[data]   {name}: {added:,} entries ({written / 1e6:.0f} MB)")
                if written >= cap:
                    break
            print(f"[data] '{name}' added {added:,} new words (total {len(seen):,})")
            used.append(name)
        except Exception as e:  # noqa: BLE001 - one bad source must not kill the build
            print(f"[data] source '{name}' failed: {type(e).__name__}: {e}")
    return written, used


def _cleanup(cfg, raw_files, keep_names=frozenset()):
    """Delete the big raw downloads, except ones that a not-yet-merged extra source will read again."""
    if not cfg["data"]["keep_raw"]:
        for p in raw_files:
            if Path(p).name not in keep_names:
                Path(p).unlink(missing_ok=True)


def _needed_later(cfg) -> set:
    pending = set(pending_sources(cfg))
    return {Path(s["url"].split("?")[0]).name for s in _extras(cfg)
            if s.get("url") and s.get("name", s["type"]) in pending}


def build_dictionary(cfg: dict) -> Path:
    d = cfg["data"]
    data_dir = Path(cfg["paths"]["data_dir"])
    out = data_dir / "dictionary.txt"
    if out.exists() and not d["rebuild"]:
        print(f"[data] using existing {out} ({out.stat().st_size / 1e6:.0f} MB)")
        return out
    data_dir.mkdir(parents=True, exist_ok=True)
    sources = list(d["sources"]) + (_extras(cfg) if d.get("extra_in_first_build") else [])
    seen: set[str] = set()
    raw_files: list = []
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fout:
        written, used = _ingest(cfg, sources, seen, fout, 0, int(float(d["max_corpus_gb"]) * 1e9), raw_files)
    if not seen:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("no dictionary entries could be built; check your network and [data.sources]")
    tmp.replace(out)
    _state_path(cfg).write_text(json.dumps({"sources": used, "entries": len(seen)}), encoding="utf-8")
    print(f"[data] dictionary.txt: {len(seen):,} words, {written / 1e6:.1f} MB")
    _cleanup(cfg, raw_files, _needed_later(cfg))
    return out


def extend_dictionary(cfg: dict):
    """Append the enabled extra sources that are not merged yet. Returns (path, [names added])."""
    out = build_dictionary(cfg)
    pending = set(pending_sources(cfg))
    todo = [s for s in _extras(cfg) if s.get("name", s["type"]) in pending]
    if not todo:
        print("[data] no new dictionaries to add")
        return out, []
    with open(out, encoding="utf-8") as f:
        seen = {line.split("\t", 1)[0] for line in f}
    before = len(seen)
    raw_files: list = []
    with open(out, "a", encoding="utf-8", newline="\n") as fout:
        written, used = _ingest(cfg, todo, seen, fout, out.stat().st_size,
                                int(float(cfg["data"]["max_corpus_gb"]) * 1e9), raw_files)
    state = _read_state(cfg)
    state["sources"] = list(dict.fromkeys(state["sources"] + used))
    state["entries"] = len(seen)
    _state_path(cfg).write_text(json.dumps(state), encoding="utf-8")
    print(f"[data] added {len(seen) - before:,} words from {len(used)} extra dictionaries; "
          f"dictionary.txt is now {written / 1e6:.1f} MB")
    _cleanup(cfg, raw_files)
    return out, used


def load_dictionary(path: Path):
    words, pos, defs = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 2)
            if len(parts) == 3 and parts[1] in POS_LIST:
                words.append(parts[0]); pos.append(parts[1]); defs.append(parts[2])
    return words, pos, defs
