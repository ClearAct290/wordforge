"""WordForge - Gradio frontend.

Runs from this folder, either locally (`python app.py`) or as a Hugging Face Space. It is self-contained and does
not need the rest of the WordForge project: it loads the model files that sit next to it.

If the model files are not next to this file (for example a Space that only holds the app), set the
WORDFORGE_MODEL_REPO variable to a Hugging Face model repo id such as "your-name/wordforge-model" and the model
is downloaded on start. For a private repo also add an HF_TOKEN secret.

Optional environment variables:
    WORDFORGE_CANDIDATES       words sampled per request                (default 16; 8 on small GPUs/CPU)
    WORDFORGE_TEMPERATURE      sampling temperature                     (default 0.85)
    WORDFORGE_TOP_K / TOP_P    sampling filters                         (default 40 / 0.95)
    WORDFORGE_ALLOW_EXISTING   1 = may return words that already exist  (default 0)
    PORT                       local port                               (default 7860)
"""
import html
import json
import os
import random
import re
import threading
from pathlib import Path

import gradio as gr
import torch
import transformers
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM

HERE = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------------------------- model files
MODEL_DIR = HERE
if not (HERE / "wordforge_meta.json").exists():
    repo = os.environ.get("WORDFORGE_MODEL_REPO", "").strip()
    if not repo:
        raise SystemExit("No model files next to app.py. Copy the trained model folder here, or set "
                         "WORDFORGE_MODEL_REPO to a Hugging Face model repo id.")
    from huggingface_hub import snapshot_download
    MODEL_DIR = Path(snapshot_download(repo_id=repo, token=os.environ.get("HF_TOKEN") or None))

META = json.loads((MODEL_DIR / "wordforge_meta.json").read_text(encoding="utf-8"))
POS = META["pos"]
TOK = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
_ver = tuple(int(x) for x in transformers.__version__.split(".")[:2])
MODEL = AutoModelForCausalLM.from_pretrained(
    str(MODEL_DIR), **({"dtype": DTYPE} if _ver >= (4, 56) else {"torch_dtype": DTYPE})).to(DEVICE).eval()

_kw = MODEL_DIR / "known_words.txt"
KNOWN = set(_kw.read_text(encoding="utf-8").split("\n")) if _kw.exists() else set()

_tid = TOK.token_to_id
PAD, BOS, EOS, DEF, WORD = _tid("<pad>"), _tid("<bos>"), _tid("<eos>"), _tid("<|def|>"), _tid("<|word|>")
POS_IDS = {p: _tid(f"<|{p}|>") for p in POS}
N_SPECIAL = len(META["special_tokens"])
MAX_DEF_CHARS = int(META.get("max_def_chars", 220))
MAX_WORD_TOKENS = int(os.environ.get("WORDFORGE_MAX_WORD_TOKENS", "20"))

LOW = DEVICE == "cpu"
N_CAND = int(os.environ.get("WORDFORGE_CANDIDATES", "8" if LOW else "16"))
TEMP = float(os.environ.get("WORDFORGE_TEMPERATURE", "0.85"))
TOP_K = int(os.environ.get("WORDFORGE_TOP_K", "40"))
TOP_P = float(os.environ.get("WORDFORGE_TOP_P", "0.95"))
ALLOW_EXISTING = os.environ.get("WORDFORGE_ALLOW_EXISTING", "0") == "1"
LOCK = threading.Lock()

WORD_RE = re.compile(r"^[^\W\d_]+(?:['\-][^\W\d_]+)*$")


def normalize_def(text: str) -> str:
    t = re.sub(r"\s+", " ", str(text)).strip().lower().rstrip(" .;:,")
    if len(t) > MAX_DEF_CHARS:
        t = t[:MAX_DEF_CHARS].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return t


def valid_word(w: str) -> bool:
    return 2 <= len(w) <= 32 and w == w.lower() and bool(WORD_RE.match(w))


# ----------------------------------------------------------------------------------------------- generation
@torch.inference_mode()
def sample(prompt, temperature):
    inp = torch.tensor([prompt] * N_CAND, device=DEVICE)
    with LOCK:
        out = MODEL.generate(
            input_ids=inp, attention_mask=torch.ones_like(inp), do_sample=True, temperature=temperature,
            top_k=TOP_K, top_p=TOP_P, max_new_tokens=MAX_WORD_TOKENS + 1, pad_token_id=PAD, eos_token_id=EOS,
            return_dict_in_generate=True, output_scores=True)
        seqs = out.sequences[:, inp.shape[1]:]
        logp = torch.stack(out.scores, 1).float().log_softmax(-1)
        tok_lp = logp.gather(2, seqs[:, : logp.shape[1]].unsqueeze(-1)).squeeze(-1)
        seqs, tok_lp = seqs.cpu().tolist(), tok_lp.cpu().tolist()
    best = {}
    for ids, lp in zip(seqs, tok_lp):
        if EOS not in ids:
            continue
        e = ids.index(EOS)
        if e == 0 or any(i < N_SPECIAL for i in ids[:e]):
            continue
        word = TOK.decode(ids[:e], skip_special_tokens=True).strip()
        if valid_word(word):
            score = sum(lp[: e + 1]) / (e + 1)
            if score > best.get(word, -1e9):
                best[word] = score
    return sorted(best.items(), key=lambda kv: kv[1], reverse=True)


def coin_word(definition: str, pos: str) -> str:
    text = normalize_def(definition)
    if len(text) < 2:
        raise ValueError("Enter a definition first.")
    max_def = int(META["max_seq_len"]) - 5 - MAX_WORD_TOKENS
    prompt = [BOS, POS_IDS[pos], DEF, *TOK.encode(text, add_special_tokens=False).ids[:max_def], WORD]
    temp, fallback = TEMP, None
    for _ in range(3):
        cands = sample(prompt, temp)
        novel = [c for c in cands if ALLOW_EXISTING or c[0] not in KNOWN]
        if novel:
            return random.choice(novel[:3])[0]
        if cands and fallback is None:
            fallback = cands[0][0]
        temp = min(temp + 0.15, 1.3)
    if fallback:
        return fallback
    raise ValueError("The model did not produce a usable word. Try rephrasing the definition.")


# ----------------------------------------------------------------------------------------------- interface
STYLE = """
<style>
.gradio-container { max-width: 40rem !important; margin: 0 auto !important;
  --button-primary-background-fill: #c25a00; --button-primary-background-fill-hover: #e8710a;
  --button-primary-text-color: #ffffff; --color-accent: #e8710a; }
.wf-top { border-top: 3px solid #e8710a; margin: 0 -1rem 1.25rem; }
.wf-title { font-size: 2rem; font-weight: 700; margin: .25rem 0 .4rem; }
.wf-lede { color: #62625f; margin: 0 0 1rem; }
.wf-entry { border: 1px solid #dcdcd8; border-radius: 6px; padding: 1.1rem 1.4rem 1.25rem; }
.wf-head { display: flex; align-items: baseline; gap: .75rem; flex-wrap: wrap;
  padding-bottom: .6rem; margin-bottom: .7rem; border-bottom: 1px solid #dcdcd8; }
.wf-word { font: 700 2rem/1.2 Georgia, "Times New Roman", serif; word-break: break-word; }
.wf-pos { font: italic 1.05rem Georgia, "Times New Roman", serif; color: #c25a00; }
.wf-sense { font: 1.1rem/1.6 Georgia, "Times New Roman", serif; display: flex; gap: .6rem; }
.wf-sense span:first-child { color: #62625f; }
.wf-foot { text-align: center; color: #62625f; font-size: .85rem; margin-top: 1rem; }
.wf-eula { font-size: .93rem; }
.wf-eula h1 { font-size: 1.3rem; } .wf-eula h2 { font-size: 1rem; margin: 1.2rem 0 .3rem; }
.wf-eula .caps { font-size: .85rem; } .wf-eula .meta { color: #62625f; }
</style>
"""


def render(word: str, pos: str, definition: str) -> str:
    d = definition.strip()
    d = d[:1].upper() + d[1:]
    if not re.search(r"[.!?]$", d):
        d += "."
    return (f'<div class="wf-entry"><div class="wf-head"><span class="wf-word">{html.escape(word)}</span>'
            f'<span class="wf-pos">{html.escape(pos)}</span></div>'
            f'<p class="wf-sense"><span>1.</span><span>{html.escape(d)}</span></p></div>')


def forge(definition: str, pos: str) -> str:
    if not (definition or "").strip():
        raise gr.Error("Enter a definition first.")
    if len(definition) > 300:
        raise gr.Error("Keep the definition under 300 characters.")
    try:
        return render(coin_word(definition, pos), pos, definition)
    except ValueError as e:
        raise gr.Error(str(e))


def eula_html() -> str:
    f = HERE / "EULA.html"
    if f.exists():
        m = re.search(r'<article class="doc">(.*?)</article>', f.read_text(encoding="utf-8"), re.S)
        if m:
            body = re.sub(r'<a class="back".*?</a>', "", m.group(1), flags=re.S)
            return f'<div class="wf-eula">{body}</div>'
    return '<div class="wf-eula"><p>Copyright 2026 WFDev. Front-end under GPL 2.0.</p></div>'


def build_ui():
    try:
        demo = gr.Blocks(title="WordForge", theme=gr.themes.Soft(primary_hue="orange"))
    except TypeError:                                  # newer Gradio versions moved the theme argument
        demo = gr.Blocks(title="WordForge")
    with demo:
        gr.HTML(STYLE + '<div class="wf-top"></div><div class="wf-title">WordForge</div>'
                '<p class="wf-lede">Describe a meaning and pick a part of speech. WordForge coins a new English word '
                'for it, using the patterns it learned from a full dictionary.</p>')
        definition = gr.Textbox(label="Definition", lines=4, max_length=300,
                                placeholder="a person who collects and studies old coins")
        pos = gr.Dropdown(choices=POS, value="noun", label="Part of speech")
        go = gr.Button("Generate", variant="primary")
        result = gr.HTML()
        go.click(forge, [definition, pos], result)
        gr.HTML('<div class="wf-foot">GPL 2.0 2026 - WordForge - EULA License</div>')
        with gr.Accordion("EULA License", open=False):
            gr.HTML(eula_html())
    return demo


demo = build_ui()

if __name__ == "__main__":
    on_space = bool(os.environ.get("SPACE_ID"))
    demo.queue().launch(server_name="0.0.0.0" if on_space else "127.0.0.1",
                        server_port=int(os.environ.get("PORT", "7860")))
