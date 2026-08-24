#!/usr/bin/env python3
"""Fill ADTC-shaped scores and write perf leaderboard from per-model JSONs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PERF_DIR = ROOT / "docs" / "artifacts" / "perf"

S_ACC_KEYS = ("afrimgsm_amh", "afrimgsm_eng", "afrimmlu_amh", "en_stem_holdout")


def empty_record(model_key: str, gguf_path: str) -> dict:
    return {
        "schema_version": "1.0",
        "model_key": model_key,
        "gguf_path": gguf_path,
        "measured_on": "jubail_hpc",
        "systems": {
            "tps": None,
            "ttft_ms": None,
            "peak_rss_mb": None,
            "steady_rss_mb": None,
            "throttled": None,
            "core_temp_c_peak": None,
            "profiler_artifact": None,
        },
        "profiler_accuracy": {
            "task": "arc_easy",
            "limit": 50,
            "items": [],
            "mean": None,
        },
        "frozen_suites": {
            "afrimgsm_amh": {"n": 0, "correct": 0, "acc": None},
            "afrimgsm_eng": {"n": 0, "correct": 0, "acc": None},
            "afrimmlu_amh": {"n": 0, "correct": 0, "acc": None},
            "en_stem_holdout": {"n": 0, "correct": 0, "acc": None},
            "custom_tutoring": {"n": 0, "correct": 0, "acc": None},
        },
        "translate_test": {"direct_acc": None, "translate_acc": None, "gap": None, "n": 50},
        "sustained": {"ok": None, "notes": ""},
        "adtc_shaped": {
            "S_acc": None,
            "S_tps": None,
            "S_mem": None,
            "total": None,
            "formula": "0.5*S_acc + 0.3*S_tps + 0.2*S_mem",
        },
        "human_amharic_review": {"status": "pending", "n_samples": 0, "notes": ""},
    }


def merge_profiler(rec: dict, profiler_path: Path) -> None:
    data = json.loads(profiler_path.read_text(encoding="utf-8"))
    thr = data.get("throughput") or {}
    mem = data.get("memory") or {}
    therm = data.get("cpu_thermal") or {}
    rec["systems"]["tps"] = thr.get("tokens_per_second_generation")
    rec["systems"]["ttft_ms"] = thr.get("first_token_latency_ms")
    rec["systems"]["peak_rss_mb"] = mem.get("peak_rss_mb")
    rec["systems"]["steady_rss_mb"] = mem.get("steady_state_rss_mb")
    rec["systems"]["throttled"] = therm.get("throttled")
    rec["systems"]["core_temp_c_peak"] = therm.get("core_temp_c_peak")
    try:
        rel = str(profiler_path.relative_to(ROOT))
    except ValueError:
        rel = str(profiler_path)
    rec["systems"]["profiler_artifact"] = rel

    acc = data.get("accuracy") or []
    rec["profiler_accuracy"]["items"] = acc
    if acc:
        scores = []
        for item in acc:
            if isinstance(item, dict):
                for k in ("acc", "accuracy", "score", "exact_match"):
                    if k in item and item[k] is not None:
                        scores.append(float(item[k]))
                        break
        rec["profiler_accuracy"]["mean"] = (sum(scores) / len(scores)) if scores else None

    throttled = bool(therm.get("throttled"))
    peak_t = therm.get("core_temp_c_peak")
    ok = not throttled
    notes = []
    if throttled:
        notes.append("throttled=true")
    if peak_t is not None and float(peak_t) > 85:
        ok = False
        notes.append(f"core_temp_c_peak={peak_t}>85")
    rec["sustained"] = {"ok": ok, "notes": "; ".join(notes)}


def merge_frozen(rec: dict, frozen_path: Path) -> None:
    data = json.loads(frozen_path.read_text(encoding="utf-8"))
    suites = data.get("frozen_suites") or data.get("suites") or {}
    for k, v in suites.items():
        rec["frozen_suites"][k] = v


def merge_translate(rec: dict, tt_path: Path) -> None:
    data = json.loads(tt_path.read_text(encoding="utf-8"))
    rec["translate_test"] = {
        "direct_acc": data.get("direct_acc"),
        "translate_acc": data.get("translate_acc"),
        "gap": data.get("gap"),
        "n": data.get("n"),
    }


def fill_adtc_shaped(rec: dict) -> None:
    suites = rec.get("frozen_suites") or {}
    accs = []
    for k in S_ACC_KEYS:
        a = (suites.get(k) or {}).get("acc")
        if a is not None:
            accs.append(float(a))
    s_acc = 100.0 * (sum(accs) / len(accs)) if accs else None

    tps = rec.get("systems", {}).get("tps")
    s_tps = 100.0 * min(1.0, float(tps) / 15.0) if tps is not None else None

    peak_mb = rec.get("systems", {}).get("peak_rss_mb")
    if peak_mb is not None:
        peak_gb = float(peak_mb) / 1024.0
        s_mem = 100.0 * (7.0 - peak_gb) / 7.0
        s_mem = max(0.0, min(100.0, s_mem))
    else:
        s_mem = None

    total = None
    if s_acc is not None and s_tps is not None and s_mem is not None:
        total = 0.5 * s_acc + 0.3 * s_tps + 0.2 * s_mem

    rec["adtc_shaped"] = {
        "S_acc": s_acc,
        "S_tps": s_tps,
        "S_mem": s_mem,
        "total": total,
        "formula": "0.5*S_acc + 0.3*S_tps + 0.2*S_mem",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--perf-dir",
        type=Path,
        default=PERF_DIR,
        help="Directory with <model_key>_v0.json files",
    )
    ap.add_argument(
        "--keys",
        nargs="+",
        default=[
            "qwen3_1_7b_merged_v0-Q4_K_M",
            "gemma3_4b_merged_v0-Q4_K_M",
            "qwen3_4b_merged_v0-Q4_K_M",
        ],
    )
    args = ap.parse_args()
    perf_dir: Path = args.perf_dir
    perf_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for key in args.keys:
        path = perf_dir / f"{key}_v0.json"
        if not path.exists():
            print(f"[warn] missing {path}")
            continue
        rec = json.loads(path.read_text(encoding="utf-8"))
        fill_adtc_shaped(rec)
        path.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        records.append(rec)
        print(f"updated {path} total={rec['adtc_shaped']['total']}")

    records.sort(
        key=lambda r: (r.get("adtc_shaped") or {}).get("total") is not None,
        reverse=True,
    )
    records.sort(
        key=lambda r: (r.get("adtc_shaped") or {}).get("total") or -1.0,
        reverse=True,
    )

    board = {
        "schema_version": "1.0",
        "measured_on": "jubail_hpc",
        "formula": "0.5*S_acc + 0.3*S_tps + 0.2*S_mem",
        "models": [
            {
                "rank": i + 1,
                "model_key": r["model_key"],
                "gguf_path": r["gguf_path"],
                "adtc_shaped": r["adtc_shaped"],
                "systems": {
                    "tps": r["systems"].get("tps"),
                    "peak_rss_mb": r["systems"].get("peak_rss_mb"),
                },
                "frozen_suite_accs": {
                    k: (r["frozen_suites"].get(k) or {}).get("acc") for k in S_ACC_KEYS
                },
                "perf_json": f"docs/artifacts/perf/{r['model_key']}_v0.json",
            }
            for i, r in enumerate(records)
        ],
        "recommended_submission": records[0]["gguf_path"] if records else None,
    }
    (perf_dir / "leaderboard_v0.json").write_text(
        json.dumps(board, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    md = [
        "# GGUF performance leaderboard (v0)",
        "",
        f"Formula: `{board['formula']}`",
        "",
        "| rank | model | total | S_acc | S_tps | S_mem | TPS | peak RSS MB |",
        "|------|-------|-------|-------|-------|-------|-----|-------------|",
    ]
    for m in board["models"]:
        a = m["adtc_shaped"]
        md.append(
            f"| {m['rank']} | `{m['model_key']}` | {a.get('total')} | {a.get('S_acc')} | "
            f"{a.get('S_tps')} | {a.get('S_mem')} | {m['systems'].get('tps')} | "
            f"{m['systems'].get('peak_rss_mb')} |"
        )
    md.append("")
    if board["recommended_submission"]:
        md.append(f"**Recommended submission GGUF:** `{board['recommended_submission']}`")
        md.append("")
    (perf_dir / "leaderboard_v0.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {perf_dir / 'leaderboard_v0.json'}")
    print(f"recommended={board['recommended_submission']}")


if __name__ == "__main__":
    main()
