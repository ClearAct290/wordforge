#!/usr/bin/env python3
"""WordForge launcher (Windows, Linux, macOS).

    python wordforge.py              train if there is no model yet; otherwise offer to train it more when it is
                                     smaller than the size limit in config.toml; then start the web server
    python wordforge.py train        train a new model from scratch and exit
    python wordforge.py serve        start the web server only

    --more        continue training the existing model on the extra dictionaries, then serve (no question asked)
    --retrain     train a fresh model from scratch first
    --extend      with --retrain / train: also merge the extra dictionaries
    --no-prompt   never ask at boot (for unattended starts)
    --config PATH use another config file

The trainer runs in its own process; when it exits, all GPU memory is released before the server starts.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from wf import config as C  # noqa: E402


# ------------------------------------------------------------------------------------------ helpers
def model_ready(cfg) -> bool:
    d = Path(cfg["paths"]["model_dir"])
    return (d / "wordforge_meta.json").exists() and (d / "config.json").exists() and any(d.glob("*.safetensors"))


def model_size_gb(cfg) -> float:
    d = Path(cfg["paths"]["model_dir"])
    return sum(f.stat().st_size for f in d.glob("*.safetensors")) / 1e9


def _state_file(cfg) -> Path:
    return Path(cfg["paths"]["data_dir"]) / "launcher_state.json"


def _read_state(cfg) -> dict:
    try:
        return json.loads(_state_file(cfg).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_state(cfg, state: dict):
    f = _state_file(cfg)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(state), encoding="utf-8")


def ask_with_timeout(prompt: str, timeout: float):
    """input() with a timeout that also works on Windows. Returns None when nobody answered."""
    answer: list = []

    def reader():
        try:
            answer.append(input(prompt))
        except (EOFError, KeyboardInterrupt):
            answer.append("")

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    th.join(timeout)
    return answer[0].strip().lower() if answer else None


def interactive() -> bool:
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def train_more_menu(cfg, size_gb: float):
    """Returns 'continue', 'fresh' or None."""
    from wf.dataset import pending_sources
    limit = float(cfg["train"]["max_model_gb"])
    pending = pending_sources(cfg)
    timeout = float(cfg["train"]["ask_timeout_seconds"])
    print()
    print(f"The model in {cfg['paths']['model_dir']} is {size_gb:.2f} GB; your size limit is {limit:g} GB.")
    print("Do you want to train it more?")
    extra = f"{len(pending)} extra dictionaries not merged yet ({', '.join(pending)})" if pending \
        else "all configured extra dictionaries are already merged"
    print(f"  [1] Continue training this model (keeps its weights; {extra})")
    print("  [2] Train a new model from scratch on all dictionaries (slower; may pick a larger model)")
    print("  [3] Not now, start the server  (default)")
    print("  [4] Do not ask again")
    ans = ask_with_timeout(f"Choose 1-4 [3] (auto-continues to the server in {int(timeout)}s): ", timeout)
    if ans is None:
        print("\nNo answer, starting the server.")
        return None
    if ans == "1":
        return "continue"
    if ans == "2":
        return "fresh"
    if ans == "4":
        st = _read_state(cfg)
        st["ask_more"] = False
        _write_state(cfg, st)
        print("OK, I will not ask again. Use `python wordforge.py --more` to train more any time.")
    return None


def run_trainer(cfg, *flags) -> int:
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTORCH_ENABLE_MPS_FALLBACK="1")
    cmd = [sys.executable, "-m", "wf.train", "--config", cfg["_config_file"], *flags]
    return subprocess.call(cmd, cwd=ROOT, env=env)


# ------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="WordForge", formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("command", nargs="?", default="run", choices=["run", "train", "serve"])
    ap.add_argument("--config", default=None, help="path to config.toml")
    ap.add_argument("--retrain", action="store_true", help="train a fresh model first")
    ap.add_argument("--more", action="store_true", help="continue training the existing model, then serve")
    ap.add_argument("--extend", action="store_true", help="also merge the extra dictionaries (fresh training)")
    ap.add_argument("--no-prompt", action="store_true", help="never ask whether to train more")
    args = ap.parse_args()
    cfg = C.load(args.config)
    ready = model_ready(cfg)

    if args.command != "serve":
        flags = None                                   # None = do not train
        if args.more:
            if not ready:
                print("There is no model to continue yet; training a new one.")
                flags = ["--extend"] if args.extend else []
            else:
                flags = ["--resume", "--extend"]
        elif args.command == "train" or args.retrain or not ready:
            flags = ["--extend"] if args.extend else []
        elif (not args.no_prompt and cfg["train"]["ask_to_train_more"] and _read_state(cfg).get("ask_more", True)
              and model_size_gb(cfg) < float(cfg["train"]["max_model_gb"]) and interactive()):
            choice = train_more_menu(cfg, model_size_gb(cfg))
            if choice == "continue":
                flags = ["--resume", "--extend"]
            elif choice == "fresh":
                flags = ["--extend"]

        if flags is not None:
            print("=== WordForge trainer ===")
            rc = run_trainer(cfg, *flags)
            if rc != 0:
                if ready and args.command == "run":
                    print(f"Training failed (exit code {rc}); starting the server with the existing model.")
                else:
                    sys.exit(f"trainer exited with code {rc}")
            if args.command == "train":
                return

    if not model_ready(cfg):
        sys.exit(f"No trained model in {cfg['paths']['model_dir']}. Run: python wordforge.py train")
    print("=== WordForge server ===")
    from wf.server import serve
    serve(cfg)


if __name__ == "__main__":
    main()
