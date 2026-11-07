---
title: WordForge
emoji: 📖
colorFrom: yellow
colorTo: red
sdk: gradio
app_file: app.py
pinned: false
license: gpl-2.0
tags:
  - text-generation
  - word-generation
  - dictionary
---

# WordForge

Describe a meaning, pick a part of speech, and WordForge coins a new English word for it.

This folder is the trained WordForge model plus a Gradio frontend. It is laid out so the whole folder can be uploaded
to Hugging Face as a **Space** (runs `app.py`) or as a **model repository**. See `HUGGINGFACE.md` for the steps.

## Files

| File | What it is |
|---|---|
| `model.safetensors`, `config.json`, `generation_config.json` | The model: a `LlamaForCausalLM`, fp16 weights (written by the trainer) |
| `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json` | Byte-level BPE tokenizer (written by the trainer) |
| `known_words.txt` | Every word in the training dictionary; used to avoid returning words that already exist |
| `wordforge_meta.json` | Part-of-speech list, special tokens, sequence length (written by the trainer) |
| `app.py` | Gradio frontend |
| `requirements.txt` | Python dependencies for `app.py` |
| `EULA.html` | WordForge EULA, shown in the app (copy of `web/license.html`) |
| `HUGGINGFACE.md` | How to put this on Hugging Face |
| `.gitattributes` | Git LFS rules for the large files |

## Run it locally

```bash
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:7860.

## Prompt format

```
<bos> <|noun|> <|def|> a person who collects old coins <|word|>      ->  the model writes the word, then <eos>
```

The parts of speech are noun, adjective, verb, adverb, pronoun, article, preposition, conjunction and interjection.

Copyright 2026 WFDev. Front-end under GPL 2.0; see `EULA.html`.
