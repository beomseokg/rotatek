"""Opt-in W5 Video-MME progress and per-request latency profiling.

This module is deliberately inert unless ``W5_LATENCY_PROFILE=1``.  The two
Qwen2.5-VL wrappers use it to emit one JSONL record per completed question and
human-readable running accuracy without changing their normal evaluation path.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import torch


def enabled() -> bool:
    return os.environ.get("W5_LATENCY_PROFILE", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }


class _EventTimer:
    def __init__(self) -> None:
        self.pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self.start_event: torch.cuda.Event | None = None

    def start(self) -> None:
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()

    def end(self) -> None:
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        if self.start_event is not None:
            self.pairs.append((self.start_event, end))
            self.start_event = None

    def total_ms(self) -> float:
        return sum(start.elapsed_time(end) for start, end in self.pairs)


def profiled_generate(
    model: Any,
    generate: Callable[[], torch.Tensor],
    preprocessing_ms: float,
) -> tuple[torch.Tensor, dict[str, float | int] | None]:
    """Run generate once and return an opt-in CUDA-event stage breakdown."""
    if not enabled():
        return generate(), None

    vit = _EventTimer()
    prefill = _EventTimer()
    decode = _EventTimer()
    lm_head = _EventTimer()
    state: dict[str, Any] = {"calls": 0, "active": None}

    handles = [
        model.visual.register_forward_pre_hook(lambda _m, _a: vit.start()),
        model.visual.register_forward_hook(lambda _m, _a, _o: vit.end()),
        model.lm_head.register_forward_pre_hook(lambda _m, _a: lm_head.start()),
        model.lm_head.register_forward_hook(lambda _m, _a, _o: lm_head.end()),
    ]

    def backbone_pre(_module: Any, _args: tuple[Any, ...]) -> None:
        timer = prefill if state["calls"] == 0 else decode
        state["active"] = timer
        timer.start()

    def backbone_post(_module: Any, _args: tuple[Any, ...], _output: Any) -> None:
        if state["active"] is not None:
            state["active"].end()
            state["active"] = None
        state["calls"] += 1

    handles.extend([
        model.model.register_forward_pre_hook(backbone_pre),
        model.model.register_forward_hook(backbone_post),
    ])

    try:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = perf_counter()
        output = generate()
        torch.cuda.synchronize()
        wall_ms = (perf_counter() - start) * 1000.0
        component_sum = (
            vit.total_ms() + prefill.total_ms() + decode.total_ms()
            + lm_head.total_ms()
        )
        timing: dict[str, float | int] = {
            "preprocessing_ms": round(float(preprocessing_ms), 3),
            "generate_wall_ms": round(wall_ms, 3),
            "vit_ms": round(vit.total_ms(), 3),
            "llm_prefill_ms": round(prefill.total_ms(), 3),
            "llm_decode_ms": round(decode.total_ms(), 3),
            "lm_head_ms": round(lm_head.total_ms(), 3),
            "unattributed_generate_ms": round(wall_ms - component_sum, 3),
            "decode_steps": max(0, int(state["calls"]) - 1),
            "peak_allocated_mib": round(
                torch.cuda.max_memory_allocated() / (1024.0 * 1024.0), 3
            ),
        }
        return output, timing
    finally:
        for handle in handles:
            handle.remove()


def _extract_choice(text: str) -> str:
    # Match a standalone answer letter, not the leading "A" in "Answer".
    # Prefer the last occurrence so CoT-like accidental prefixes cannot mask
    # the final fixed-format answer.
    matches = re.findall(r"\b([ABCD])\b", text.upper())
    return matches[-1] if matches else ""


class ProgressLogger:
    """Append sample records and print shard-local running accuracy."""

    def __init__(self, total: int) -> None:
        self.total = int(total)
        self.completed = 0
        self.correct = 0
        self.row = os.environ.get("W5_ROW", "unknown")
        self.shard = os.environ.get("SHARD_RANK", "0")
        self.every = max(1, int(os.environ.get("W5_PROGRESS_EVERY", "10")))
        path = os.environ.get("W5_LATENCY_JSONL", "").strip()
        self.path = Path(path) if path else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        *,
        doc_id: int,
        doc: dict[str, Any],
        prediction: str,
        raw_response: str,
        timing: dict[str, float | int] | None,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        pred = _extract_choice(prediction)
        target = str(doc.get("answer", "")).strip().upper()
        is_correct = pred == target
        self.completed += 1
        self.correct += int(is_correct)

        record: dict[str, Any] = {
            "row": self.row,
            "shard": int(self.shard),
            "doc_id": int(doc_id),
            "question_id": doc.get("question_id"),
            "video_id": doc.get("videoID"),
            "prediction": pred,
            "target": target,
            "correct": is_correct,
            "raw_response": raw_response,
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
        }
        if timing is not None:
            record.update(timing)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        if self.completed == 1 or self.completed % self.every == 0 or self.completed == self.total:
            accuracy = 100.0 * self.correct / self.completed
            latency = ""
            if timing is not None:
                latency = (
                    f" pre={timing['preprocessing_ms']:.1f}ms"
                    f" vit={timing['vit_ms']:.1f}ms"
                    f" prefill={timing['llm_prefill_ms']:.1f}ms"
                    f" decode={timing['llm_decode_ms']:.1f}ms"
                )
            print(
                f"[W5-PROGRESS] row={self.row} shard={self.shard} "
                f"completed={self.completed}/{self.total} correct={self.correct} "
                f"accuracy={accuracy:.2f}%{latency}",
                flush=True,
            )
