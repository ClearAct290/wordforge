"""Hardware detection: CUDA vs CPU, VRAM tier (low-gpu mode), mixed-precision dtype."""
from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch


@dataclass
class Hardware:
    device: str            # "cuda" | "cpu"
    name: str
    vram_gib: float
    capability: tuple
    low_mode: bool         # tiny / low-gpu mode (<= low_vram_gib VRAM, or CPU)
    amp_dtype: object      # torch.bfloat16 / torch.float16 / None (fp32)
    index: int = 0

    @property
    def torch_device(self) -> torch.device:
        if self.device == "cuda":
            return torch.device(f"cuda:{self.index}")
        return torch.device(self.device)          # "mps" or "cpu"

    def autocast(self):
        """Mixed-precision context, or a no-op when running in fp32."""
        if self.amp_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(self.device, dtype=self.amp_dtype)

    def describe(self) -> str:
        if self.device == "cpu":
            return f"CPU ({self.name}) - tiny mode"
        if self.device == "mps":
            return "Apple GPU (MPS, experimental) - tiny mode"
        mode = "low-gpu (tiny) mode" if self.low_mode else "full-gpu mode"
        amp = {torch.bfloat16: "bf16", torch.float16: "fp16"}.get(self.amp_dtype, "fp32")
        return (f"{self.name} - {self.vram_gib:.1f} GiB VRAM, "
                f"sm_{self.capability[0]}{self.capability[1]}, {amp}, {mode}")


def _cpu(reason: str) -> Hardware:
    print(f"[hw] using CPU ({reason})")
    return Hardware("cpu", "cpu", 0.0, (0, 0), True, None)


def detect(cfg: dict) -> Hardware:
    hw = cfg["hardware"]
    want = str(hw.get("device", "auto")).lower()
    if want == "cpu":
        return _cpu("forced by config")
    if want == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return Hardware("mps", "Apple GPU", 0.0, (0, 0), True, None)
        return _cpu("device = mps requested but MPS is not available")
    if not torch.cuda.is_available():
        return _cpu("no CUDA-capable PyTorch/GPU found")
    idx = int(hw.get("device_index", 0))
    try:
        props = torch.cuda.get_device_properties(idx)
        cap = (props.major, props.minor)
        if cap[0] + cap[1] / 10 < float(hw.get("min_compute_capability", 5.0)):
            return _cpu(f"GPU compute capability {cap[0]}.{cap[1]} is too old")
        # self-test: catches "no kernel image" errors from unsupported architectures
        x = torch.randn(64, 64, device=f"cuda:{idx}", dtype=torch.float16)
        float((x @ x).float().sum().item())
    except Exception as e:  # noqa: BLE001
        return _cpu(f"CUDA self-test failed: {type(e).__name__}: {str(e)[:120]}")

    vram = props.total_memory / 1024 ** 3
    low = vram <= float(hw.get("low_vram_gib", 6.5))
    amp = None
    if cap[0] >= 8 and torch.cuda.is_bf16_supported():
        amp = torch.bfloat16
    elif cap[0] >= 7:
        amp = torch.float16
    return Hardware("cuda", props.name, vram, cap, low, amp, idx)
