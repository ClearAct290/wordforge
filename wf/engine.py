"""Inference engine: loads the trained model from model/ and coins a word for (definition, part of speech)."""
from __future__ import annotations

import json
import random
import threading
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .hw import Hardware, detect
from .textutil import normalize_def, valid_word


def _load_model(model_dir: Path, dtype):
    import transformers
    from transformers import AutoModelForCausalLM
    ver = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    kw = {"dtype": dtype} if ver >= (4, 56) else {"torch_dtype": dtype}
    return AutoModelForCausalLM.from_pretrained(str(model_dir), **kw)


class Engine:
    def __init__(self, cfg: dict, model_dir: str | Path | None = None, hw: Hardware | None = None):
        self.g = cfg["generate"]
        md = Path(model_dir or cfg["paths"]["model_dir"])
        self.meta = json.loads((md / "wordforge_meta.json").read_text(encoding="utf-8"))
        self.hw = hw or detect(cfg)
        self.tok = Tokenizer.from_file(str(md / "tokenizer.json"))
        dtype = torch.float16 if self.hw.device == "cuda" else torch.float32
        self.model = _load_model(md, dtype).to(self.hw.torch_device).eval()

        tid = self.tok.token_to_id
        self.PAD, self.BOS, self.EOS = tid("<pad>"), tid("<bos>"), tid("<eos>")
        self.DEF, self.WORD = tid("<|def|>"), tid("<|word|>")
        self.n_special = len(self.meta["special_tokens"])
        self.pos_ids = {p: tid(f"<|{p}|>") for p in self.meta["pos"]}
        kw = md / "known_words.txt"
        self.known = set(kw.read_text(encoding="utf-8").split("\n")) if kw.exists() else set()
        self.n = int(self.g["candidates"])
        if self.hw.low_mode:                       # low-gpu mode: fewer parallel candidates
            self.n = min(self.n, 8)
        self.lock = threading.Lock()
        self.params = sum(p.numel() for p in self.model.parameters())
        try:
            self.generate("a test of the engine", "noun")   # warm-up
        except Exception:  # noqa: BLE001
            pass

    # ---------------------------------------------------------------- public API
    def generate(self, definition: str, pos: str) -> dict:
        pos = (pos or "").strip().lower()
        if pos not in self.pos_ids:
            raise ValueError(f"unknown part of speech '{pos}'")
        text = normalize_def(definition, int(self.meta["max_def_chars"]))
        if len(text) < 2:
            raise ValueError("the definition is empty")
        max_def = int(self.meta["max_seq_len"]) - 5 - int(self.g["max_word_tokens"])
        d_ids = self.tok.encode(text, add_special_tokens=False).ids[:max_def]
        prompt = [self.BOS, self.pos_ids[pos], self.DEF, *d_ids, self.WORD]

        allow_existing = bool(self.g["allow_existing_words"])
        temp, fallback = float(self.g["temperature"]), None
        for _ in range(3):
            cands = self._sample(prompt, temp)
            novel = [c for c in cands if allow_existing or c[0] not in self.known]
            if novel:
                word = random.choice(novel[:3])[0]      # pick among the three most likely
                return {"word": word, "pos": pos, "novel": word not in self.known}
            if cands and fallback is None:
                fallback = cands[0][0]
            temp = min(temp + 0.15, 1.3)                # every candidate was an existing word: be bolder
        if fallback:
            return {"word": fallback, "pos": pos, "novel": False}
        raise RuntimeError("the model did not produce a usable word, try rephrasing the definition")

    # ---------------------------------------------------------------- sampling
    @torch.inference_mode()
    def _sample(self, prompt: list[int], temperature: float) -> list[tuple[str, float]]:
        g = self.g
        dev = self.hw.torch_device
        inp = torch.tensor([prompt] * self.n, device=dev)
        with self.lock:
            out = self.model.generate(
                input_ids=inp, attention_mask=torch.ones_like(inp), do_sample=True,
                temperature=temperature, top_k=int(g["top_k"]), top_p=float(g["top_p"]),
                max_new_tokens=int(g["max_word_tokens"]) + 1,
                pad_token_id=self.PAD, eos_token_id=self.EOS,
                return_dict_in_generate=True, output_scores=True)
            seqs = out.sequences[:, inp.shape[1]:]
            logp = torch.stack(out.scores, 1).float().log_softmax(-1)
            tok_lp = logp.gather(2, seqs[:, : logp.shape[1]].unsqueeze(-1)).squeeze(-1)
            seqs, tok_lp = seqs.cpu().tolist(), tok_lp.cpu().tolist()

        best: dict[str, float] = {}
        for ids, lp in zip(seqs, tok_lp):
            if self.EOS not in ids:
                continue
            e = ids.index(self.EOS)
            if e == 0 or any(i < self.n_special for i in ids[:e]):
                continue
            word = self.tok.decode(ids[:e], skip_special_tokens=True).strip()
            if not valid_word(word):
                continue
            score = sum(lp[: e + 1]) / (e + 1)          # mean log-probability of the word
            if score > best.get(word, -1e9):
                best[word] = score
        return sorted(best.items(), key=lambda kv: kv[1], reverse=True)
