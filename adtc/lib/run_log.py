"""Timestamped run logs for download / normalize / train stages.

Each run writes three artifacts under ``adtc/logs/<stage>/``:

* ``<utc>_<run_id>.log`` — human-readable progress (tee-friendly)
* ``<utc>_<run_id>.jsonl`` — one JSON event per line (incremental; survives Ctrl+C)
* ``<utc>_<run_id>.summary.json`` — final or partial rollup (ok / error / skipped)

Stages (convention):
  download_train | normalize_cpt | normalize_sft | mix_sft
  | download_models | train_sft | train_cpt | merge_lora

Usage::

    from lib.run_log import RunLogger

    log = RunLogger("download_train", meta={"profile": "first_experiment"})
    log.info("starting", n_sources=10)
    try:
        log.item_start("walia", hf_id="...")
        # ... work ...
        log.item_ok("walia", n_rows=1234)
    except Exception as e:
        log.item_error("walia", e)
    finally:
        log.finish()  # always write summary, even on KeyboardInterrupt
"""
from __future__ import annotations

import json
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ADTC_ROOT = Path(__file__).resolve().parents[1]
LOGS_ROOT = ADTC_ROOT / "logs"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        try:
            return str(obj.relative_to(ADTC_ROOT))
        except ValueError:
            return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, BaseException):
        return f"{type(obj).__name__}: {obj}"
    return str(obj)


class RunLogger:
    """Incremental success/failure logger for pipeline runs."""

    def __init__(
        self,
        stage: str,
        *,
        meta: dict[str, Any] | None = None,
        logs_root: Path | None = None,
        echo: bool = True,
    ) -> None:
        self.stage = stage
        self.meta = dict(meta or {})
        self.echo = echo
        self.run_id = uuid.uuid4().hex[:8]
        self.started_utc = _iso_now()
        self.finished_utc: str | None = None
        self.status = "running"  # running | ok | error | interrupted | partial

        root = Path(logs_root) if logs_root else LOGS_ROOT
        self.dir = root / stage
        self.dir.mkdir(parents=True, exist_ok=True)
        stamp = _utc_stamp()
        base = f"{stamp}_{self.run_id}"
        self.log_path = self.dir / f"{base}.log"
        self.jsonl_path = self.dir / f"{base}.jsonl"
        self.summary_path = self.dir / f"{base}.summary.json"
        self.latest_summary = self.dir / "latest.summary.json"

        self.items: list[dict[str, Any]] = []
        self._open_item: dict[str, Any] | None = None

        self._log_fp = self.log_path.open("w", encoding="utf-8")
        self._jsonl_fp = self.jsonl_path.open("w", encoding="utf-8")
        self._emit(
            "run_start",
            message=f"stage={stage} run_id={self.run_id}",
            meta=self.meta,
        )

    def info(self, message: str, **fields: Any) -> None:
        self._emit("info", message=message, **fields)

    def warn(self, message: str, **fields: Any) -> None:
        self._emit("warn", message=message, **fields)

    def item_start(self, key: str, **fields: Any) -> None:
        if self._open_item is not None:
            # previous item never closed — mark interrupted mid-item
            prev = self._open_item
            prev["status"] = "interrupted"
            prev["finished_utc"] = _iso_now()
            self.items.append(prev)
            self._write_event("item_interrupted", key=prev["key"], **{k: v for k, v in prev.items() if k != "key"})
            self._open_item = None
        entry = {
            "key": key,
            "status": "running",
            "started_utc": _iso_now(),
            **_jsonable(fields),
        }
        self._open_item = entry
        self._emit("item_start", key=key, **fields)

    def item_ok(self, key: str, **fields: Any) -> None:
        self._close_item(key, "ok", fields)

    def item_error(self, key: str, error: Any, **fields: Any) -> None:
        err = _jsonable(error)
        fields = {**fields, "error": err}
        if isinstance(error, BaseException) and error.__traceback__ is not None:
            fields["traceback"] = "".join(traceback.format_exception(type(error), error, error.__traceback__))[
                -4000:
            ]
        self._close_item(key, "error", fields)

    def item_skip(self, key: str, reason: str, **fields: Any) -> None:
        self._close_item(key, "skipped", {**fields, "reason": reason})

    def _close_item(self, key: str, status: str, fields: dict[str, Any]) -> None:
        now = _iso_now()
        if self._open_item and self._open_item.get("key") == key:
            entry = self._open_item
            entry.update(_jsonable(fields))
            entry["status"] = status
            entry["finished_utc"] = now
            self._open_item = None
        else:
            entry = {
                "key": key,
                "status": status,
                "started_utc": now,
                "finished_utc": now,
                **_jsonable(fields),
            }
        self.items.append(entry)
        event = {
            "ok": "item_ok",
            "skipped": "item_skip",
            "interrupted": "item_interrupted",
        }.get(status, "item_error")
        self._emit(event, key=key, status=status, **fields)
        # flush after every item so Ctrl+C still leaves a trail
        self._flush()
        self._write_summary(partial=True)

    def finish(self, *, status: str | None = None, message: str | None = None) -> Path:
        """Close the run and write the final summary. Safe to call more than once."""
        if self.finished_utc is not None:
            return self.summary_path

        if self._open_item is not None:
            open_key = self._open_item["key"]
            self._close_item(
                open_key,
                "interrupted",
                {"error": "interrupted before completion", "interrupted": True},
            )

        counts = self.counts()
        if status is not None:
            self.status = status
        elif counts["error"] and counts["ok"]:
            self.status = "partial"
        elif counts["error"] and not counts["ok"]:
            self.status = "error"
        elif counts.get("interrupted"):
            self.status = "interrupted"
        else:
            self.status = "ok"

        self.finished_utc = _iso_now()
        self._emit(
            "run_finish",
            message=message or f"status={self.status}",
            status=self.status,
            **counts,
        )
        path = self._write_summary(partial=False)
        self._flush()
        self._log_fp.close()
        self._jsonl_fp.close()
        return path

    def counts(self) -> dict[str, int]:
        c = {"ok": 0, "error": 0, "skipped": 0, "interrupted": 0, "running": 0, "total": len(self.items)}
        for it in self.items:
            st = it.get("status", "running")
            c[st] = c.get(st, 0) + 1
        return c

    def summary_dict(self, *, partial: bool) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "run_id": self.run_id,
            "status": "running" if partial else self.status,
            "partial": partial,
            "started_utc": self.started_utc,
            "finished_utc": None if partial else self.finished_utc,
            "meta": self.meta,
            "counts": self.counts(),
            "items": self.items,
            "paths": {
                "log": str(self.log_path.relative_to(ADTC_ROOT)),
                "jsonl": str(self.jsonl_path.relative_to(ADTC_ROOT)),
                "summary": str(self.summary_path.relative_to(ADTC_ROOT)),
            },
        }

    def _write_summary(self, *, partial: bool) -> Path:
        payload = self.summary_dict(partial=partial)
        text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        self.summary_path.write_text(text, encoding="utf-8")
        self.latest_summary.write_text(text, encoding="utf-8")
        return self.summary_path

    def _emit(self, event: str, message: str | None = None, **fields: Any) -> None:
        self._write_event(event, message=message, **fields)
        line = self._format_line(event, message=message, **fields)
        self._log_fp.write(line + "\n")
        if self.echo:
            print(line, flush=True)

    def _write_event(self, event: str, message: str | None = None, **fields: Any) -> None:
        rec = {
            "ts": _iso_now(),
            "stage": self.stage,
            "run_id": self.run_id,
            "event": event,
        }
        if message is not None:
            rec["message"] = message
        rec.update(_jsonable(fields))
        self._jsonl_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _format_line(self, event: str, message: str | None = None, **fields: Any) -> str:
        key = fields.get("key")
        status = fields.get("status")
        parts = [f"[{self.stage}]"]
        if event.startswith("item_"):
            label = {
                "item_start": "START",
                "item_ok": "OK",
                "item_error": "FAIL",
                "item_skip": "SKIP",
                "item_interrupted": "INTERRUPTED",
            }.get(event, event.upper())
            parts.append(label)
            if key:
                parts.append(str(key))
        elif event == "run_start":
            parts.append("RUN")
        elif event == "run_finish":
            parts.append("DONE")
            if status:
                parts.append(str(status))
        else:
            parts.append(event.upper())
        if message:
            parts.append("—")
            parts.append(message)
        # compact extras (skip noisy / large fields)
        skip = {"key", "status", "traceback", "meta"}
        extras = []
        for k, v in fields.items():
            if k in skip or v is None:
                continue
            if k == "error":
                extras.append(f"error={v}")
            elif isinstance(v, (str, int, float, bool)):
                extras.append(f"{k}={v}")
        if extras:
            parts.append("(" + ", ".join(extras) + ")")
        return " ".join(parts)

    def _flush(self) -> None:
        self._log_fp.flush()
        self._jsonl_fp.flush()
        sys.stdout.flush()


def import_run_log_path() -> None:
    """Ensure ``adtc/`` is on sys.path so ``from lib.run_log import RunLogger`` works."""
    root = str(ADTC_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
