# Putting WordForge on Hugging Face

This folder (`model/`) is ready to upload as a Hugging Face **Space**, which runs the Gradio app in `app.py`. You can also
publish the trained model as a **model repository** and let the Space download it. Both ways are below.

You need a free account at <https://huggingface.co> and a trained model: `python wordforge.py train` must have
finished, so that this folder contains `model.safetensors`, `config.json`, `tokenizer.json`, `known_words.txt` and
`wordforge_meta.json` next to `app.py`.

## 0. Try the Gradio app locally first

```bash
pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:7860>. If this works, the Space will work.

## 1. Install the Hugging Face CLI and log in

```bash
pip install -U huggingface_hub
hf auth login
```

Paste an access token with **write** permission from <https://huggingface.co/settings/tokens>.
(Older versions of `huggingface_hub` call the command `huggingface-cli` instead of `hf`; the arguments are the same.)

## 2A. Everything in one Space (simplest)

Best for small models (the default `tiny`, `small` and `base` presets are well under 1 GB).

1. Create the Space at <https://huggingface.co/new-space>: pick a name, choose **Gradio** as the SDK, hardware
   **CPU basic**, and visibility **Public** or **Private**.
2. Upload this whole folder. From the project root:

   ```bash
   hf upload YOUR-NAME/wordforge ./model . --repo-type=space
   ```

   Or use the website: open the Space, **Files** tab, **Add file**, **Upload files**, and drag in everything in
   `model/` (including `README.md`, which holds the Space settings, and `.gitattributes`).
3. The Space builds from `requirements.txt` and starts `app.py`. The first start takes a few minutes. When the log says
   *Running*, the app is live at `https://huggingface.co/spaces/YOUR-NAME/wordforge`.

## 2B. Model repository plus a small Space (for large models)

Best when the weights are big or you want people to download the model itself.

1. Publish the model:

   ```bash
   hf repo create wordforge-model --repo-type model
   hf upload YOUR-NAME/wordforge-model ./model . --repo-type model
   ```

2. Create a Gradio Space as in step 2A.1, but upload only the app files:

   ```bash
   hf upload YOUR-NAME/wordforge ./model/app.py app.py --repo-type=space
   hf upload YOUR-NAME/wordforge ./model/README.md README.md --repo-type=space
   hf upload YOUR-NAME/wordforge ./model/requirements.txt requirements.txt --repo-type=space
   hf upload YOUR-NAME/wordforge ./model/EULA.html EULA.html --repo-type=space
   ```

3. In the Space: **Settings**, **Variables and secrets**, add a **Variable** named `WORDFORGE_MODEL_REPO` with the
   value `YOUR-NAME/wordforge-model`. If the model repo is private, also add a **Secret** named `HF_TOKEN` holding a
   read token. The app downloads the model when it starts.

## 3. Hardware

- **CPU basic** (free) is fine for the smaller presets. The app uses fp32 on CPU and samples 8 candidates per request.
- A bigger model (`large` and above) needs upgraded hardware (a GPU Space). On a GPU the app switches to fp16 by itself.
- Spaces on free hardware go to sleep when idle; the first request after a sleep reloads the model.

## 4. Settings you can change (Space variables)

| Variable | Default | Meaning |
|---|---|---|
| `WORDFORGE_MODEL_REPO` | empty | Model repo to download if no model files sit next to `app.py` |
| `WORDFORGE_CANDIDATES` | 16 (8 on CPU) | Words sampled per request |
| `WORDFORGE_TEMPERATURE` | 0.85 | Higher = more adventurous words |
| `WORDFORGE_TOP_K`, `WORDFORGE_TOP_P` | 40, 0.95 | Sampling filters |
| `WORDFORGE_ALLOW_EXISTING` | 0 | `1` lets it return words that already exist in the dictionary |

## 5. After you retrain the model

Run the upload command again; Hugging Face replaces the changed files and the Space restarts.

```bash
hf upload YOUR-NAME/wordforge ./model . --repo-type=space
```

## Notes

- Files over 10 MB (the weights and `known_words.txt`) are stored with Git LFS. The `.gitattributes` file in this folder
  already sets that up; `hf upload` and the website handle it for you.
- `README.md` starts with a YAML block (title, SDK, license). Edit the `license:` line if you publish the model under
  different terms. Keep `sdk: gradio` and `app_file: app.py`.
- `EULA.html` is a copy of `web/license.html` from the main project. If you edit one, copy it to the other.
- Hugging Face Spaces are public by default and run your code as is. Do not put secrets in the folder; use Space
  secrets for tokens.
