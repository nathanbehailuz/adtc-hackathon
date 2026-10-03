# TebebAI v8 data foundation

Phase A provides the canonical schema, deterministic validation, stable identities,
SQLite uniqueness registry, fail-closed source manifest, and no-generation supply
census.

Run from the `adtc/` project directory:

```bash
python3 -m pip install -r data/v8/requirements.txt
python3 -m unittest data.v8.test_phase_a -v
python3 -m data.v8.cli gate-a
```

Useful commands:

```bash
python3 -m data.v8.cli export-schema
python3 -m data.v8.cli init-registry
python3 -m data.v8.cli census
python3 -m data.v8.cli validate data/canonical/v8/train.jsonl
python3 -m data.v8.cli freeze-eval
python3 -m data.v8.cli gate-b
```

The census counts only metadata inventory rows from pinned, approved sources.
An unpinned source contributes zero. `gate_a.may_start_bulk_model_calls=false`
is a hard stop: do not begin Jev or generative-model bulk calls.

`gate-b` freezes exact and template hashes for every local evaluation prompt.
Future inventory rows must pass that firewall before the census counts them.

Inventory JSONL records use this metadata-only shape:

```json
{
  "source_id": "stem",
  "family_id": "sha256:...",
  "original_id": "row-123",
  "canonical_prompt": "A 2 kg object has a net force of 10 N...",
  "source_split": "train",
  "subject": "physics",
  "eligible_behaviors": ["solve"],
  "grade_band": "secondary_9_12",
  "grounding_available": false,
  "license_class": "permissive",
  "hard_gate_pass": true,
  "rejection_reasons": [],
  "template_fingerprint": "sha256:..."
}
```
