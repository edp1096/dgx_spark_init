"""Conditional SM121 Sol-Attn integration for the LTX-2.5 API."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


log = logging.getLogger("ltx-api.sol")


def stage2_tokens(width: int, height: int, frames: int) -> int:
    temporal = (frames - 1) // 8 + 1
    return temporal * (width // 32) * (height // 32)


def normalize_mode(value: str) -> str:
    mode = str(value or "auto").strip().lower()
    aliases = {
        "on": "sol",
        "force": "sol",
        "enabled": "sol",
        "off": "dense",
        "disabled": "dense",
        "false": "dense",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"auto", "dense", "sol"}:
        raise ValueError("acceleration must be auto, dense, or sol")
    return mode


@dataclass
class AccelerationPlan:
    requested_mode: str
    tokens: int
    enabled: bool
    backend: str
    exact_adaln: bool = False
    label: str = "dense"
    reason: str = "below token threshold"


class _OptimizedStage:
    def __init__(self, stage: Any, attention: Any, exact_adaln: Any | None) -> None:
        self.stage = stage
        self.attention = attention
        self.exact_adaln = exact_adaln
        self.call_index = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        from ltx_pipelines.utils.samplers import euler_denoising_loop

        self.call_index += 1
        stage_index = self.call_index
        original_loop = kwargs.get("loop") or euler_denoising_loop

        if self.exact_adaln is not None and stage_index == 2:
            def accelerated_loop(*loop_args: Any, **loop_kwargs: Any) -> Any:
                transformer = loop_kwargs["transformer"]
                self.exact_adaln.install(transformer)
                try:
                    return original_loop(*loop_args, **loop_kwargs)
                finally:
                    self.exact_adaln.uninstall(transformer)

            kwargs["loop"] = accelerated_loop

        with self.attention.stage2(stage_index == 2):
            return self.stage(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.stage, name)


class SolRuntime:
    def __init__(self, model_dir: Path, transformer: Path) -> None:
        self.model_dir = model_dir
        self.transformer = transformer
        self.default_mode = normalize_mode(os.environ.get("LTX_SOL_MODE", "auto"))
        self.min_tokens = max(1, int(os.environ.get("LTX_SOL_MIN_TOKENS", "32000")))
        self.exact_enabled = os.environ.get("LTX_SOL_EXACT_ADALN", "1").strip().lower() not in {
            "0", "false", "off", "no",
        }
        self._exact_tables: dict[int, Any] = {}
        self._lock = threading.Lock()
        self._last: dict[str, Any] = {}
        self._error = ""

    def backend(self) -> str:
        try:
            from sol_attn import get_sol_attn_backend

            return str(get_sol_attn_backend())
        except Exception as exc:
            self._error = str(exc)
            return "unavailable"

    def plan(
        self,
        width: int,
        height: int,
        frames: int,
        requested_mode: str = "",
        *,
        lora_active: bool = False,
    ) -> AccelerationPlan:
        mode = normalize_mode(requested_mode or self.default_mode)
        tokens = stage2_tokens(width, height, frames)
        enabled = mode == "sol" or (mode == "auto" and tokens >= self.min_tokens)
        backend = self.backend() if enabled else "dense"
        if mode == "dense":
            reason = "disabled by request"
        elif not enabled:
            reason = f"{tokens} tokens below {self.min_tokens}"
        elif backend == "unavailable":
            enabled = False
            reason = "Sol-Attn backend unavailable"
        else:
            reason = "automatic high-token acceleration" if mode == "auto" else "forced"
        exact = enabled and self.exact_enabled and not lora_active
        label = backend if enabled else "dense"
        if exact:
            label += "+exact-adaln"
        return AccelerationPlan(mode, tokens, enabled, backend, exact, label, reason)

    def _table_path(self, tokens: int) -> Path:
        return self.model_dir / "sol-cache" / f"exact-adaln-{tokens}t.pt"

    def _exact_adaln(self, tokens: int) -> Any:
        from models.ltx25.RTX5090.exact_adaln import LTX25ExactAdaLN

        with self._lock:
            cached = self._exact_tables.get(tokens)
            if cached is not None:
                return cached
            table = self._table_path(tokens)
            try:
                exact = LTX25ExactAdaLN(table, self.transformer)
            except (FileNotFoundError, ValueError, RuntimeError):
                table.parent.mkdir(parents=True, exist_ok=True)
                temporary = table.with_suffix(".tmp.pt")
                temporary.unlink(missing_ok=True)
                try:
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "models.ltx25.RTX5090.build_exact_adaln",
                            "--checkpoint",
                            str(self.transformer),
                            "--tokens",
                            str(tokens),
                            "--output",
                            str(temporary),
                        ],
                        check=True,
                    )
                    os.replace(temporary, table)
                finally:
                    temporary.unlink(missing_ok=True)
                exact = LTX25ExactAdaLN(table, self.transformer)
            self._exact_tables[tokens] = exact
            return exact

    @contextmanager
    def activate(
        self,
        pipeline: Any,
        width: int,
        height: int,
        frames: int,
        requested_mode: str = "",
        *,
        lora_active: bool = False,
    ) -> Iterator[AccelerationPlan]:
        plan = self.plan(
            width,
            height,
            frames,
            requested_mode,
            lora_active=lora_active,
        )
        if not plan.enabled:
            self._last = vars(plan).copy()
            yield plan
            return

        try:
            from models.ltx25.RTX5090.attention import LTX25Stage2SolAttention

            attention = LTX25Stage2SolAttention()
        except Exception as exc:
            plan.enabled = False
            plan.backend = "dense"
            plan.exact_adaln = False
            plan.label = "dense"
            plan.reason = "Sol-Attn setup failed; dense fallback"
            self._error = str(exc)
            self._last = vars(plan).copy()
            log.exception("Sol-Attn setup failed; continuing with dense attention")
            yield plan
            return
        exact = None
        if plan.exact_adaln:
            try:
                exact = self._exact_adaln(plan.tokens)
            except Exception as exc:
                plan.exact_adaln = False
                plan.label = plan.backend
                self._error = f"Exact AdaLN disabled: {exc}"
                log.exception("Exact AdaLN table preparation failed; continuing with Sol-Attn")

        original_stage = pipeline.stage
        pipeline.stage = _OptimizedStage(
            original_stage.with_attention(attention),
            attention,
            exact,
        )
        log.info(
            "LTX acceleration=%s tokens=%d lora=%s",
            plan.label,
            plan.tokens,
            lora_active,
        )
        try:
            yield plan
        finally:
            pipeline.stage = original_stage
            stats = attention.stats()
            self._last = {**vars(plan), "stats": stats}

    def status(self) -> dict[str, Any]:
        return {
            "mode": self.default_mode,
            "backend": self.backend(),
            "min_tokens": self.min_tokens,
            "exact_adaln": self.exact_enabled,
            "last": self._last or None,
            "error": self._error,
        }
