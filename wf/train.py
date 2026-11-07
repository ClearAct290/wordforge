"""WordForge trainer.

Builds the dictionary, trains a byte-level BPE tokenizer and a Llama-architecture causal LM from scratch
(real tensors, real weights) on CUDA (CPU fallback), and writes a fp16 safetensors model to model/.

Training example:   <bos> <|pos|> <|def|> definition tokens <|word|> word tokens <eos>
Generation prompt:  <bos> <|pos|> <|def|> definition tokens <|word|>   ->  the model writes the word.

Run directly:  python -m wf.train [--config config.toml]
"""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from . import config as C
from .config import POS_LIST
from .dataset import build_dictionary, extend_dictionary, load_dictionary
from .hw import detect
from .textutil import normalize_def  # noqa: F401  (documented contract: same normalisation as the dataset)

SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>", "<|def|>", "<|word|>"] + [f"<|{p}|>" for p in POS_LIST]
PAD, BOS, EOS, UNK, DEF_ID, WORD_ID = range(6)
POS_ID = {p: 6 + i for i, p in enumerate(POS_LIST)}
N_SPECIAL = len(SPECIALS)

# d = hidden size, L = layers, ff = MLP size. Head size is always 64. Embeddings are tied.
PRESETS = {
    "nano":  dict(d=256,  L=4,  ff=704,  bs=128,  lr=3e-3,   epochs=4),   # CPU only
    "tiny":  dict(d=512,  L=8,  ff=1408, bs=256,  lr=2e-3,   epochs=6),   # low-gpu mode (<= 6 GB VRAM)
    "small": dict(d=768,  L=12, ff=2048, bs=512,  lr=1.5e-3, epochs=5),
    "base":  dict(d=1024, L=16, ff=2816, bs=768,  lr=1e-3,   epochs=4),
    "large": dict(d=1536, L=20, ff=4096, bs=1024, lr=6e-4,   epochs=3),
    "xl":    dict(d=2048, L=28, ff=5632, bs=1024, lr=4e-4,   epochs=3),
    "max":   dict(d=2304, L=32, ff=6144, bs=1024, lr=3e-4,   epochs=3),   # ~4.1 GB in fp16
}
GPU_ORDER = ["tiny", "small", "base", "large", "xl", "max"]


def param_count(p: dict, vocab: int) -> int:
    d, L, ff = p["d"], p["L"], p["ff"]
    return vocab * d + L * (4 * d * d + 3 * d * ff + 2 * d) + d


def pick_preset(t: dict, hw, n_tokens: int, vocab: int) -> str:
    name = str(t["preset"]).lower()
    max_gb = float(t["max_model_gb"])
    if name != "auto":
        if name not in PRESETS:
            raise SystemExit(f"unknown preset '{name}', choose from auto, {', '.join(PRESETS)}")
        gb = param_count(PRESETS[name], vocab) * 2 / 1e9
        if gb > max_gb:
            raise SystemExit(f"preset '{name}' would be {gb:.2f} GB in fp16, above max_model_gb={max_gb}")
        return name
    if hw.device == "cpu":
        return "nano"
    if hw.low_mode:
        return "tiny"
    cap_params = float(t["params_per_token"]) * n_tokens
    best = "tiny"
    for name in GPU_ORDER:
        p = param_count(PRESETS[name], vocab)
        if p * 2 / 1e9 > max_gb or p * 16 > 0.6 * hw.vram_gib * 1024 ** 3:
            break
        if name != "tiny" and p > cap_params:
            break
        best = name
    return best


# ------------------------------------------------------------------------------------------ tokenizer
def train_tokenizer(words, defs, vocab_size, rng, max_samples=700_000):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIALS, min_frequency=2,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=True)
    idx = rng.permutation(len(words))[:max_samples]

    def it():
        for i in idx:
            yield defs[i]
            yield words[i]

    tk.train_from_iterator(it(), trainer=trainer, length=2 * len(idx))
    for i, s in enumerate(SPECIALS):
        assert tk.token_to_id(s) == i, f"special token {s} has id {tk.token_to_id(s)}, expected {i}"
    return tk


def encode_all(tk, texts, chunk=50_000):
    flats, lens = [], []
    for i in range(0, len(texts), chunk):
        ids = [e.ids for e in tk.encode_batch(texts[i:i + chunk], add_special_tokens=False)]
        n = np.fromiter((len(x) for x in ids), dtype=np.int32, count=len(ids))
        flats.append(np.fromiter(itertools.chain.from_iterable(ids), dtype=np.uint16, count=int(n.sum())))
        lens.append(n)
    return np.concatenate(flats), np.concatenate(lens)


# ------------------------------------------------------------------------------------------ data
class Packed:
    """Flat token storage; examples are assembled per batch (fast, ~2 bytes per token)."""

    def __init__(self, dflat, dlen, wflat, wlen, pos_ids):
        self.dflat, self.dlen, self.wflat, self.wlen, self.pos = dflat, dlen, wflat, wlen, pos_ids
        self.doff = np.concatenate(([0], np.cumsum(dlen[:-1], dtype=np.int64)))
        self.woff = np.concatenate(([0], np.cumsum(wlen[:-1], dtype=np.int64)))
        self.total = 5 + dlen + wlen          # bos pos <def> ... <word> ... eos

    def batch(self, idx, prompt_weight, pad_to=16):
        B = len(idx)
        dl, wl, L = self.dlen[idx], self.wlen[idx], self.total[idx]
        T = int(-(-int(L.max()) // pad_to) * pad_to)
        ids = np.zeros((B, T), dtype=np.int64)
        ids[:, 0], ids[:, 1], ids[:, 2] = BOS, self.pos[idx], DEF_ID
        for r, i in enumerate(idx):
            n, m = int(dl[r]), int(wl[r])
            a, b = self.doff[i], self.woff[i]
            ids[r, 3:3 + n] = self.dflat[a:a + n]
            ids[r, 3 + n] = WORD_ID
            ids[r, 4 + n:4 + n + m] = self.wflat[b:b + m]
            ids[r, 4 + n + m] = EOS
        col = np.arange(T)[None, :]
        nd = dl[:, None]
        w = np.where((col >= 3) & (col < 3 + nd), prompt_weight, 0.0)       # definition tokens
        w = np.where((col >= 4 + nd) & (col < L[:, None]), 1.0, w)           # word tokens + <eos>
        return ids, w.astype(np.float32)


def epoch_batches(ds: Packed, indices, bs, seed):
    """Shuffle, then sort by length inside large chunks so batches have little padding."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(indices)
    out = []
    mega = bs * 64
    for i in range(0, len(perm), mega):
        chunk = perm[i:i + mega]
        chunk = chunk[np.argsort(ds.total[chunk], kind="stable")]
        out.extend(chunk[j:j + bs] for j in range(0, len(chunk) - bs + 1, bs))
    return [out[i] for i in rng.permutation(len(out))]


def prefetch(batches, ds, pw, pin, depth=6):
    q: queue.Queue = queue.Queue(depth)

    def work():
        for idx in batches:
            ids, w = ds.batch(idx, pw)
            ids, w = torch.from_numpy(ids), torch.from_numpy(w)
            if pin:
                ids, w = ids.pin_memory(), w.pin_memory()
            q.put((ids, w))
        q.put(None)

    threading.Thread(target=work, daemon=True).start()
    while (item := q.get()) is not None:
        yield item


# ------------------------------------------------------------------------------------------ model
def build_model(p: dict, vocab: int, max_len: int):
    from transformers import LlamaConfig, LlamaForCausalLM
    heads = p["d"] // 64
    cfg = LlamaConfig(
        vocab_size=vocab, hidden_size=p["d"], intermediate_size=p["ff"], num_hidden_layers=p["L"],
        num_attention_heads=heads, num_key_value_heads=heads, max_position_embeddings=max(256, max_len),
        rms_norm_eps=1e-5, initializer_range=0.02, tie_word_embeddings=True,
        pad_token_id=PAD, bos_token_id=BOS, eos_token_id=EOS)
    return LlamaForCausalLM(cfg)


def make_optimizer(model, lr, device):
    decay = [p for p in model.parameters() if p.ndim >= 2]
    nodecay = [p for p in model.parameters() if p.ndim < 2]
    kw = {"fused": True} if device.type == "cuda" else {}
    return torch.optim.AdamW([{"params": decay, "weight_decay": 0.05},
                              {"params": nodecay, "weight_decay": 0.0}],
                             lr=lr, betas=(0.9, 0.95), eps=1e-8, **kw)


def weighted_ce(logits, y, w):
    ce = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none")
    w = w.reshape(-1)
    return (ce * w).sum() / w.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate(model, ds, idx, bs, hw):
    model.eval()
    tot, cnt = 0.0, 0.0
    for i in range(0, len(idx), bs):
        ids, w = ds.batch(idx[i:i + bs], 0.0)
        ids, w = torch.from_numpy(ids).to(hw.torch_device), torch.from_numpy(w).to(hw.torch_device)
        with hw.autocast():
            logits = model(input_ids=ids[:, :-1], use_cache=False).logits
        ce = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1), reduction="none")
        tot += float((ce * w[:, 1:].reshape(-1)).sum())
        cnt += float(w[:, 1:].sum())
    model.train()
    return tot / max(cnt, 1.0)


def lr_at(step, warm, prog, peak, floor=0.1):
    if step < warm:
        return peak * (step + 1) / warm
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))


# ------------------------------------------------------------------------------------------ saving
def save_model(model, tk, out_dir: Path, meta: dict, known_words):
    out_dir.mkdir(parents=True, exist_ok=True)
    for pat in ("*.safetensors", "*.bin", "model.safetensors.index.json"):
        for f in out_dir.glob(pat):
            f.unlink()
    model = model.to(torch.float16).eval()
    bad = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
    if bad:
        raise RuntimeError(f"non-finite weights after fp16 cast: {bad[:3]}")
    model.save_pretrained(out_dir, safe_serialization=True, max_shard_size="5GB")
    tk.save(str(out_dir / "tokenizer.json"))
    try:  # also write the standard HF tokenizer files so transformers / ModelScope can load it
        from transformers import PreTrainedTokenizerFast
        PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<bos>", eos_token="<eos>",
                                pad_token="<pad>", unk_token="<unk>").save_pretrained(out_dir)
    except Exception as e:  # noqa: BLE001
        print(f"[save] HF tokenizer wrapper skipped: {e}")
    (out_dir / "known_words.txt").write_text("\n".join(sorted(set(known_words))), encoding="utf-8")
    (out_dir / "wordforge_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    size = sum(f.stat().st_size for f in out_dir.glob("*.safetensors"))
    print(f"[save] {out_dir}  weights: {size / 1e6:.1f} MB (fp16 safetensors)")
    return size


# ------------------------------------------------------------------------------------------ main
def train(cfg: dict, resume: bool = False, extend: bool = False):
    t = cfg["train"]
    torch.manual_seed(int(t["seed"]))
    rng = np.random.default_rng(int(t["seed"]))
    hw = detect(cfg)
    dev = hw.torch_device
    print(f"[hw] {hw.describe()}")
    if hw.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    # ---- data
    dict_path = build_dictionary(cfg)
    if extend or resume:
        dict_path, _ = extend_dictionary(cfg)
    words, pos, defs = load_dictionary(dict_path)
    print(f"[data] {len(words):,} dictionary entries")
    for p in POS_LIST:
        print(f"[data]   {p:<12}{pos.count(p):>9,}")
    model_dir = Path(cfg["paths"]["model_dir"])
    meta0 = {}
    if resume:
        from tokenizers import Tokenizer
        meta0 = json.loads((model_dir / "wordforge_meta.json").read_text(encoding="utf-8"))
        tk = Tokenizer.from_file(str(model_dir / "tokenizer.json"))   # the model's embeddings are tied to it
        max_len = int(meta0["max_seq_len"])
        print(f"[resume] continuing the '{meta0['preset']}' model in {model_dir}")
    else:
        max_len = min(int(t["max_seq_len"]), 64) if hw.low_mode else int(t["max_seq_len"])
        print("[tok] training tokenizer")
        tk = train_tokenizer(words, defs, int(t["vocab_size"]), rng)
    vocab = tk.get_vocab_size()
    print("[tok] encoding")
    dflat, dlen = encode_all(tk, defs)
    wflat, wlen = encode_all(tk, words)
    ds = Packed(dflat, dlen, wflat, wlen, np.array([POS_ID[p] for p in pos], dtype=np.int64))
    valid = np.nonzero(ds.total <= max_len)[0]
    print(f"[data] {len(valid):,} entries fit in {max_len} tokens ({len(ds.total) - len(valid):,} dropped)")
    perm = rng.permutation(valid)
    n_val = min(int(t["val_entries"]), max(1, len(perm) // 50))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    n_tokens = int(ds.total[train_idx].sum())

    # ---- model + schedule
    if resume:
        from .engine import _load_model
        name = meta0["preset"]
        P = PRESETS[name]
        model = _load_model(model_dir, torch.float32)          # fp16 weights -> fp32 master weights
        if model.config.vocab_size != vocab:
            raise SystemExit("tokenizer and model do not match; run a fresh training instead")
        n_params = sum(p.numel() for p in model.parameters())
        bs = int(t["batch_size"]) or P["bs"]
        peak_lr = (float(t["learning_rate"]) or P["lr"]) * float(t["resume_lr_factor"])
        epochs = int(t["resume_epochs"])
    else:
        name = pick_preset(t, hw, n_tokens, vocab)
        P = PRESETS[name]
        n_params = param_count(P, vocab)
        bs = int(t["batch_size"]) or P["bs"]
        peak_lr = float(t["learning_rate"]) or P["lr"]
        epochs = int(t["epochs"]) or P["epochs"]
        model = build_model(P, vocab, max_len)
    print(f"[model] preset '{name}': {n_params / 1e6:.1f}M params, {n_params * 2 / 1e6:.0f} MB in fp16, "
          f"{n_tokens / 1e6:.1f}M corpus tokens")
    model = model.to(dev)
    model.train()
    opt = make_optimizer(model, peak_lr, dev)
    use_scaler = hw.amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    autocast = hw.autocast
    pw = min(float(t["prompt_weight"]), 0.9)

    # optional torch.compile (CUDA, Volta+, not in low-gpu mode); falls back to eager on any failure
    fwd, compiled = model, False
    want = str(t["compile"]).lower()
    if hw.device == "cuda" and hw.capability[0] >= 7 and (want == "true" or (want == "auto" and not hw.low_mode)):
        try:
            import triton  # noqa: F401
            torch._dynamo.config.cache_size_limit = 32
            fwd, compiled = torch.compile(model, dynamic=True), True
            print("[train] torch.compile enabled")
        except Exception as e:  # noqa: BLE001
            print(f"[train] torch.compile unavailable ({type(e).__name__}); using eager mode")

    def run_fwd(x):
        nonlocal fwd, compiled
        try:
            with autocast():
                return fwd(input_ids=x, use_cache=False).logits
        except Exception as e:  # noqa: BLE001
            if not compiled or isinstance(e, torch.cuda.OutOfMemoryError):
                raise
            print(f"[train] compiled path failed ({type(e).__name__}); switching to eager")
            fwd, compiled = model, False
            with autocast():
                return fwd(input_ids=x, use_cache=False).logits

    # ---- batch-size probe: largest batch that fits at the longest sequence length
    if hw.device == "cuda":
        while bs >= 16:
            try:
                x = torch.randint(N_SPECIAL, vocab, (bs, max_len - 1), device=dev)
                loss = weighted_ce(run_fwd(x), x, torch.ones_like(x, dtype=torch.float32))
                loss.backward()
                opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                break
            except torch.cuda.OutOfMemoryError:
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                bs //= 2
                print(f"[train] out of memory, trying batch size {bs}")
        else:
            raise SystemExit("not enough GPU memory even for batch size 16; set [hardware] device = \"cpu\"")
        torch.cuda.empty_cache()
    peak_lr *= math.sqrt(bs / P["bs"]) if bs < P["bs"] else 1.0

    steps_per_epoch = max(1, len(train_idx) // bs - 1)
    total_steps = epochs * steps_per_epoch
    warm = max(20, int(0.03 * total_steps))
    budget = float(t["max_minutes"]) * 60.0
    print(f"[train] batch {bs}, {epochs} epochs x {steps_per_epoch:,} steps, peak lr {peak_lr:.2e}, "
          f"time budget {t['max_minutes']} min" if budget else "[train] no time budget")

    # ---- training loop
    step, t0, run_loss, run_n = 0, None, torch.zeros((), device=dev), 0
    bar = tqdm(total=total_steps, desc="train", unit="step", smoothing=0.05)
    done = False
    try:
        for epoch in range(epochs):
            batches = epoch_batches(ds, train_idx, bs, int(t["seed"]) + epoch)
            for ids, w in prefetch(batches, ds, pw, hw.device == "cuda"):
                prog = step / total_steps
                if budget and t0 is not None:
                    prog = max(prog, (time.time() - t0) / budget)
                if prog >= 1.0:
                    done = True
                    break
                lr = lr_at(step, warm, prog, peak_lr)
                for g in opt.param_groups:
                    g["lr"] = lr
                ids, w = ids.to(dev, non_blocking=True), w.to(dev, non_blocking=True)
                loss = weighted_ce(run_fwd(ids[:, :-1]), ids[:, 1:], w[:, 1:])
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step == 1:
                    t0 = time.time()      # the budget clock starts after compile/warm-up
                run_loss += loss.detach()
                run_n += 1
                bar.update(1)
                if step % 50 == 0:
                    bar.set_postfix(loss=f"{float(run_loss) / run_n:.3f}", lr=f"{lr:.1e}")
                    run_loss.zero_(); run_n = 0
            if done:
                print("\n[train] time budget reached")
                break
            vl = evaluate(model, ds, val_idx, bs, hw)
            print(f"\n[val] epoch {epoch + 1}/{epochs}: word loss {vl:.3f}  (perplexity {math.exp(min(vl, 20)):.1f})")
    except KeyboardInterrupt:
        print("\n[train] interrupted - saving the model as it is now")
    bar.close()
    final_loss = evaluate(model, ds, val_idx, bs, hw)
    print(f"[val] final word loss {final_loss:.3f}")

    # ---- save
    meta = {
        **meta0,
        "name": "WordForge", "version": 1, "pos": POS_LIST, "special_tokens": SPECIALS,
        "max_seq_len": max_len, "max_def_chars": int(cfg["data"]["max_def_chars"]),
        "entries": int(len(valid)), "params": int(n_params), "preset": name,
        "val_word_loss": round(final_loss, 4), "trained_steps": int(meta0.get("trained_steps", 0)) + step,
        "device": hw.name, "resumed": int(meta0.get("resumed", 0)) + (1 if resume else 0),
    }
    size = save_model(model, tk, model_dir, meta, words)
    if size / 1e9 > float(t["max_model_gb"]):
        print(f"[warn] model is {size / 1e9:.2f} GB, above max_model_gb")

    # ---- end-to-end sanity check through the same code the server uses
    del opt, fwd, model
    gc.collect()
    if hw.device == "cuda":
        torch.cuda.empty_cache()
    try:
        from .engine import Engine
        eng = Engine(cfg, hw=hw)
        print("[check] sample generations (held-out dictionary entries):")
        for i in val_idx[:6]:
            r = eng.generate(defs[i], pos[i])
            print(f"  [{pos[i]}] {defs[i][:60]!r}\n      true: {words[i]:<16} generated: {r['word']}")
    except Exception as e:  # noqa: BLE001
        print(f"[check] skipped: {type(e).__name__}: {e}")
    print("[done] model ready in", cfg["paths"]["model_dir"])


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train the WordForge model")
    ap.add_argument("--config", default=None)
    ap.add_argument("--resume", action="store_true", help="continue training the model in model/")
    ap.add_argument("--extend", action="store_true", help="merge the extra dictionaries into dictionary.txt first")
    args = ap.parse_args(argv)
    train(C.load(args.config), resume=args.resume, extend=args.extend)


if __name__ == "__main__":
    main()
