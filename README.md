# WordForge

Type a definition, pick a part of speech, get a newly coined English word.

WordForge trains a language model **from scratch** on a dictionary of `word / part of speech / first definition`
entries (PyTorch on an NVIDIA GPU, CPU fallback), then serves it through a small web page. A Gradio frontend and a
Hugging Face upload guide are included in `model/`.

(This model is lightweight enough to use fans, me, the dev, doesn't want mass water consumption)

Runs on **Windows, Linux and macOS**.

- Trains a real model: real tensors, fp16 safetensors output, standard `LlamaForCausalLM` architecture
- Builds its own dictionary: English Wiktionary and WordNet, plus optional extra open dictionaries, duplicates removed
- Picks the model size from your GPU: 6 GB of VRAM or less runs in low-gpu (tiny) mode; no CUDA GPU falls back to CPU
- **Train it more** later: at boot it offers to keep training when the model is smaller than your size limit
- Plain HTTP on port 80 by default, HTTPS by config (domain, PEM files, redirect)
- Gradio app and Hugging Face guide in `model/`

---

## Quick start

```bash
git clone https://github.com/YOUR-NAME/wordforge.git
cd wordforge
python -m venv .venv
# activate the environment (see below), install PyTorch (see below), then:
pip install -r requirements.txt
python wordforge.py
```

The first run downloads the dictionary (about 2.5 GB), trains, saves the model to `model/`, closes the trainer, and
starts the server at `http://localhost/`. Later runs go straight to the server.

## Requirements

| | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.11 or 3.12 |
| GPU | none (CPU fallback, smoke tests only) | NVIDIA, compute capability 7.0+ (RTX 20-series or newer) |
| VRAM | 6 GB (low-gpu mode) | 8 GB or more |
| RAM | 8 GB | 16 GB |
| Disk | 6 GB free | 10 GB free |
| Network | needed on first run for the dictionary download | |

## Install

### 1. Virtual environment

| | Create | Activate |
|---|---|---|
| Linux, macOS | `python3 -m venv .venv` | `source .venv/bin/activate` |
| Windows (PowerShell) | `py -m venv .venv` | `.venv\Scripts\Activate.ps1` |
| Windows (cmd) | `py -m venv .venv` | `.venv\Scripts\activate.bat` |

If PowerShell blocks the activate script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

### 2. PyTorch

Install PyTorch **before** the other requirements. The selector at <https://pytorch.org/get-started/locally/> gives the
exact command for your system.

| System | Command |
|---|---|
| Windows or Linux with an NVIDIA GPU | `pip install torch --index-url https://download.pytorch.org/whl/cu126` (use the CUDA version the selector shows) |
| Linux without a GPU | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| macOS (Intel or Apple Silicon) | `pip install torch` |

Plain `pip install torch` on Windows gives a **CPU-only** build, so use the `--index-url` form there.
You need a recent NVIDIA driver for GPU use; check it with `nvidia-smi`. Verify PyTorch sees the GPU:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### 3. Everything else

```bash
pip install -r requirements.txt
```

## Run

```bash
python wordforge.py              # train if there is no model; offer to train more if it is small; then serve
python wordforge.py train        # train a new model from scratch and exit
python wordforge.py serve        # serve only
python wordforge.py --more       # continue training the existing model on the extra dictionaries, then serve
python wordforge.py --retrain    # train from scratch first, then serve
python wordforge.py --no-prompt  # never ask questions at boot (for unattended starts)
python wordforge.py --config path/to/other.toml
```

On Windows use `py wordforge.py ...` if `python` is not on your PATH.

**Ports.** The default is plain HTTP on `0.0.0.0:80`. Ports below 1024 need extra rights on Linux and macOS: run with
`sudo`, allow Python to bind them (`sudo setcap cap_net_bind_service=+ep "$(readlink -f .venv/bin/python)"` on Linux), or set
`[server] port = 8080`. On Windows, port 80 can already be taken by IIS, Skype or another web server; change the port if
you get an "address already in use" message.

Running it as a background service (systemd, Windows, launchd) is covered in [setup.md](setup.md).

## What happens on first run

1. **Hardware check.** The log shows your GPU and mode: `full-gpu`, `low-gpu (tiny)` for 6 GB of VRAM or less, or `CPU`.
2. **Dictionary build.** Downloads the English Wiktionary dump (resumable), reads WordNet, merges them and writes
   `data/dictionary.txt`. Takes tens of minutes depending on your connection and CPU. The large raw download is deleted
   afterwards unless `keep_raw = true` or an extra dictionary that has not been merged yet needs the same file.
3. **Training.** Trains the tokenizer and the model, prints validation loss after each epoch, stops at the epoch limit or
   `max_minutes`. Press `Ctrl+C` once to stop early; the model is saved as it stands.
4. **Save.** Writes the fp16 model to `model/` and prints sample generations as a self-check.
5. **Server.** The trainer process exits (releasing GPU memory) and the web server starts.

## Train it more

If a model already exists and it is **smaller than `max_model_gb`** (default 4.5), a normal start asks:

```
The model in model/ is 0.18 GB; your size limit is 4.5 GB.
Do you want to train it more?
  [1] Continue training this model (keeps its weights; 1 extra dictionaries not merged yet (wiktionary-latin))
  [2] Train a new model from scratch on all dictionaries (slower; may pick a larger model)
  [3] Not now, start the server  (default)
  [4] Do not ask again
```

- **[1] Continue** merges the enabled extra dictionaries into `dictionary.txt` (the first Wiktionary download is reused when it was kept; otherwise it is fetched again), then keeps
  training the saved model on the full dictionary for `resume_epochs` epochs at a lower learning rate.
- **[2] From scratch** merges the extras and trains a new model. The `auto` preset may now choose a larger model
  because there is more data.
- **[3]** is the default and is also chosen if nobody answers within `ask_timeout_seconds` (20), so unattended starts never
  block. The question only appears when the program is run from an interactive terminal.
- **[4]** stores the choice in `data/launcher_state.json`. Turn the question off for good with `ask_to_train_more = false`.

Skip the question and do it directly with `python wordforge.py --more`. If continued training fails (for example out of
memory), the existing model is kept and the server starts with it.

**Extra dictionaries** live under `[[data.extra_sources]]` in `config.toml`. They are open-licence sources:

| Name | What | Default |
|---|---|---|
| `wiktionary-latin` | Latin lemmas with English definitions, from the same kaikki.org dump | on |
| `wiktionary-european` | French, German, Spanish, Italian, Portuguese, Dutch | off |
| `webster-1913` (example in `config.toml`) | Public-domain Webster 1913 / OPTED as JSON that you download into `data/opted/` | off, needs your files |
| your own | a TSV `word<TAB>pos<TAB>definition` file or URL | add it |

A word already in the dictionary is never added twice. The non-English sources are Latin-script only and are meant to
add word roots; they can pull generated words away from English, which is why only Latin is on by default.

## How it works

- **Dictionary** (`wf/dataset.py`): one line per word, `word<TAB>pos<TAB>definition`. Only the word's first entry and its first
  non-inflection sense are kept, so the part of speech always belongs to the first definition. The nine parts of speech are
  noun, adjective, verb, adverb, pronoun, article, preposition, conjunction and interjection. Building stops at
  `max_corpus_gb` (default 4.0).
- **Model** (`wf/train.py`): a Llama-architecture decoder (tied embeddings, RMSNorm, RoPE, SwiGLU) with a byte-level BPE
  tokenizer. Format: `<bos> <|noun|> <|def|> definition <|word|> word <eos>`. The loss is weighted toward the word tokens.
  Mixed precision (bf16 or fp16) with fp32 master weights, fused AdamW, SDPA attention, length-bucketed batches, optional
  `torch.compile`, automatic batch-size probing for your VRAM, and a wall-clock budget (`max_minutes`) that the learning-rate
  schedule respects.
- **Output** in `model/`: `model.safetensors` (**fp16**), `config.json`, `generation_config.json`, `tokenizer.json` and the
  HF tokenizer files, `known_words.txt`, `wordforge_meta.json`. It loads with `transformers`
  (`AutoModelForCausalLM`) and with runtimes and hubs that support the `LlamaForCausalLM` architecture.
- **Generation** (`wf/engine.py`): samples several candidates in one batch, drops any that already exist in the dictionary
  (`allow_existing_words = true` allows them), and returns one of the three most likely.
- **Server** (`wf/server.py`): FastAPI and uvicorn. `GET /`, `GET /license`, `POST /api/generate`, `GET /api/health`,
  per-IP rate limit, security headers.

## Hardware modes

| Detected | Mode | Model |
|---|---|---|
| NVIDIA GPU, 6.5 GiB VRAM or less (6 GB cards report ~5.8-6.0 GiB) | **low-gpu / tiny** | `tiny`, ~30M parameters, 64-token examples, 8 candidates per request |
| NVIDIA GPU, more VRAM | full-gpu | largest preset that fits your VRAM and the amount of data (`auto`) |
| No CUDA GPU, GPU too old, or failed CUDA self-test | CPU | `nano`, ~5M parameters, fp32 |
| `[hardware] device = "mps"` on Apple Silicon (opt-in, experimental) | MPS | `tiny`, fp32 |

Presets: `nano 5M`, `tiny 30M`, `small 91M`, `base 214M`, `large 579M`, `xl 1.46B`, `max 2.06B (4.1 GB fp16)`.
Force one with `[train] preset`. `max_model_gb` (default 4.5) is a hard ceiling. On macOS the default is the CPU; the
Apple GPU is available as an experiment through `device = "mps"`.

## Configuration

Everything is in `config.toml`; every key is optional. The ones you will touch most:

| Setting | Default | Meaning |
|---|---|---|
| `[server] host`, `port` | `0.0.0.0`, `80` | Where the web server listens (plain HTTP) |
| `[server] domain` | empty | Your public domain (HTTPS redirects, startup message) |
| `[server] rate_limit_per_minute` | `60` | Per-IP limit on `/api/generate`, `0` = off |
| `[https] enabled` | `false` | Serve HTTPS (see below) |
| `[hardware] device` | `auto` | `auto`, `cuda`, `cpu`, `mps` |
| `[hardware] low_vram_gib` | `6.5` | GPUs at or below this use low-gpu mode |
| `[data] max_corpus_gb` | `4.0` | Stop adding dictionary entries at this size |
| `[data] extra_sources` | Latin on | Dictionaries merged by "train more" |
| `[train] preset` | `auto` | `auto`, `nano`, `tiny`, `small`, `base`, `large`, `xl`, `max` |
| `[train] max_minutes` | `90` | Training time budget, `0` = none |
| `[train] max_model_gb` | `4.5` | Size ceiling; also the threshold for the "train more" question |
| `[train] ask_to_train_more` | `true` | Ask at boot when the model is below the ceiling |
| `[generate] allow_existing_words` | `false` | `true` lets it return words that already exist |

Use another config file with `--config` or the `WORDFORGE_CONFIG` environment variable.

## HTTPS

```toml
[server]
domain = "wordforge.example.com"
port   = 80                      # HTTP port; redirects to HTTPS while [https] is on

[https]
enabled  = true
port     = 443
certfile = "/etc/letsencrypt/live/wordforge.example.com/fullchain.pem"
keyfile  = "/etc/letsencrypt/live/wordforge.example.com/privkey.pem"
redirect_http = true
hsts = true
```

1. Point your domain's DNS record at the server and open ports 80 and 443 in the firewall.
2. Get a certificate (any PEM certificate and key work). With Let's Encrypt, stop WordForge first because certbot needs port 80:
   `sudo certbot certonly --standalone -d wordforge.example.com`. On Windows, use a client such as win-acme.
3. The key files under `/etc/letsencrypt` are root-only; run WordForge as root or copy the two files where your service
   user can read them. If the key has a password, set `keyfile_password`.
4. The certificate is read at start-up, so restart WordForge after each renewal.

## Hugging Face and Gradio

The `model/` folder holds the trained model and a ready-made Gradio frontend:

| File | Purpose |
|---|---|
| `model/app.py` | Gradio UI (orange theme, definition box, part-of-speech dropdown, dictionary-style result, EULA) |
| `model/HUGGINGFACE.md` | Step-by-step: upload as a Space, or as a model repo plus a small Space |
| `model/README.md` | Hugging Face Space and model card (YAML settings included) |
| `model/requirements.txt`, `model/.gitattributes`, `model/EULA.html` | Dependencies, Git LFS rules, license text for the app |

Try it locally with `pip install -r model/requirements.txt && python model/app.py` (port 7860).
The trainer writes the model files into the same folder, so after training the folder is ready to upload.

## Project layout

```
wordforge.py          launcher (train, prompt, serve)
config.toml           all settings
requirements.txt
wf/                   backend: config, hardware, dataset, trainer, engine, server
web/                  frontend: index.html, license.html, static/style.css, static/app.js
model/                trained model + Gradio app + Hugging Face files (output of the trainer)
data/                 dictionary downloads and dictionary.txt (created on first run)
setup.md              running as a service, firewall, reverse proxy
LICENSE               GNU General Public License, version 2
```

## Things worth knowing

- **Size.** A word + part of speech + first-definition dictionary is small: English Wiktionary reduces to roughly 150-250 MB
  of text (about 1.4M entries). The 4 GB corpus setting is a ceiling, not something the open data can fill, and the model is
  not padded to 4.5 GB: with about 50M training tokens a 2B-parameter model mostly memorises, so `auto` picks about 90M
  parameters on a typical GPU. With a large GPU (80 GB class) you can force `preset = "max"`.
- **CPU training** exists as a fallback and is only good for a smoke test. Train on a GPU.
- **Quality.** Output is plausible, dictionary-shaped words (fitting endings and roots), not lookups.
- **Testing status.** The dictionary builder, batching, config, boot prompt and HTTPS redirect were tested offline. The GPU
  training, model loading, generation, the web server and the Gradio app have not been run end to end by the author of this
  package; read the trainer log on first run (`[hw]`, `[model]`, `[val]`, `[check]` lines).

## License

- `LICENSE` is the GNU General Public License, version 2.
- The WordForge EULA (`web/license.html`, shown at `/license` and in the Gradio app) is by WFDev, 2026. It licenses the
  front-end under GPL 2.0 and describes the back-end (`wf/`, `wordforge.py`) as closed source with the restrictions listed
  there. **These two statements conflict if the back-end source is published in a repository whose `LICENSE` is the GPL:**
  GPL section 6 forbids adding restrictions for people who receive GPL code. To keep the back-end closed, publish only the
  front-end (`web/`, `model/app.py`) and docs on GitHub and ship the back-end compiled (for example with Nuitka); to
  publish everything, license the back-end under the GPL too and change section 3 and 4 of the EULA.
- Dictionary data: English Wiktionary text is CC BY-SA, Princeton WordNet has its own license, Webster 1913 is public
  domain. Check what that means for distributing a model trained on them.
- The EULA is a template, not legal advice.

## Troubleshooting

**The log says "using CPU" but I have an NVIDIA GPU.** Run the check in the install section. If `torch.cuda.is_available()`
is `False`, reinstall PyTorch with the CUDA `--index-url` and update your driver. `CUDA self-test failed` means your PyTorch
build does not support your GPU architecture; try another CUDA build of PyTorch.

**`out of memory, trying batch size N`.** Normal; the trainer halves the batch size until it fits. If it still fails at 16, set
`[train] preset = "tiny"` or close other programs that use the GPU.

**`no permission to bind port 80` / `address already in use`.** See "Ports" under Run.

**`https.enabled = true but certfile/keyfile are missing`.** The paths in `[https]` are empty, wrong, or unreadable by the
user running WordForge.

**The dictionary download fails or stops.** Run the command again; the download resumes. If WordNet cannot download its data,
run `python -m nltk.downloader wordnet`. A failing source is skipped and the build goes on with the others.

**`No trained model in model/`.** Run `python wordforge.py train` and let it finish.

**Generated words look random.** Check that `[val] word loss` falls over the epochs. If `max_minutes` cut training short, raise
it or set `0`, or run `python wordforge.py --more`. Lowering `[generate] temperature` (for example `0.7`) also helps.

**Generation is slow.** On CPU, lower `[generate] candidates` (for example `4`). On a GPU, confirm the server log names your
GPU rather than CPU.

**Windows: strange characters in the log.** Run `chcp 65001` first, or use Windows Terminal.
