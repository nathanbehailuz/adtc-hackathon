"""Shared model loading helpers for ADTC training (Gemma / Qwen3 / Qwen3.5).

Qwen3.5 Afrique checkpoints use ``Qwen3_5ForConditionalGeneration`` (multimodal).
For text-only CPT/SFT we still load that class when needed, but LoRA targets must
exclude vision modules.
"""
from __future__ import annotations

from typing import Any


VISION_NAME_HINTS = (
    "vision",
    "visual",
    "mm_projector",
    "multi_modal",
    "merger",
    "patch_embed",
)


def resolve_model_class(cfg: dict[str, Any]) -> str:
    """Return ``causal`` | ``qwen3_5`` | ``auto``."""
    return str(cfg.get("model_class", "auto")).strip().lower()


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_pretrained_model(
    model_id: str,
    *,
    model_class: str = "auto",
    quantization_config=None,
    torch_dtype=None,
    device_map: str | dict | None = "auto",
):
    """Load HF model; prefer CausalLM, fall back to Qwen3.5 conditional gen."""
    from transformers import AutoConfig, AutoModelForCausalLM

    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": device_map,
    }
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype

    mc = (model_class or "auto").lower()
    if mc == "causal":
        return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    if mc == "qwen3_5":
        return _load_qwen35(model_id, **kwargs)

    # auto: try causal first, then qwen3.5 / image-text
    try:
        return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except Exception as causal_err:  # noqa: BLE001
        try:
            return _load_qwen35(model_id, **kwargs)
        except Exception as qwen_err:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to load {model_id} as CausalLM ({causal_err}) "
                f"and as Qwen3.5 ({qwen_err})"
            ) from qwen_err


def _load_qwen35(model_id: str, **kwargs):
    try:
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    except Exception:  # noqa: BLE001
        from transformers import AutoModel

        return AutoModel.from_pretrained(model_id, **kwargs)


def list_linear_module_names(model) -> list[str]:
    import torch.nn as nn

    names: list[str] = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            names.append(name)
    return names


def is_vision_module(name: str) -> bool:
    low = name.lower()
    return any(h in low for h in VISION_NAME_HINTS)


def suggest_lora_targets(model, *, text_only: bool = True) -> list[str]:
    """Return unique short leaf names suitable for PEFT target_modules."""
    leaves: set[str] = set()
    preferred = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    for name in list_linear_module_names(model):
        if text_only and is_vision_module(name):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf in preferred:
            leaves.add(leaf)
    if leaves:
        return sorted(leaves)
    # fallback: all non-vision linear leaf names
    for name in list_linear_module_names(model):
        if text_only and is_vision_module(name):
            continue
        leaves.add(name.rsplit(".", 1)[-1])
    return sorted(leaves)


def inspect_config(model_id: str) -> dict[str, Any]:
    from transformers import AutoConfig, AutoTokenizer

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    raw = cfg.to_dict() if hasattr(cfg, "to_dict") else {}
    text_cfg = getattr(cfg, "text_config", None)
    vision = getattr(cfg, "vision_config", None) is not None or "vision_config" in raw
    return {
        "hf_id": model_id,
        "model_type": getattr(cfg, "model_type", None),
        "architectures": getattr(cfg, "architectures", None),
        "has_vision_config": bool(vision),
        "vocab_size": getattr(tok, "vocab_size", None) or getattr(cfg, "vocab_size", None),
        "chat_template_present": bool(getattr(tok, "chat_template", None)),
        "max_position_embeddings": getattr(
            text_cfg or cfg,
            "max_position_embeddings",
            getattr(cfg, "max_position_embeddings", None),
        ),
        "hidden_size": getattr(text_cfg or cfg, "hidden_size", getattr(cfg, "hidden_size", None)),
        "num_hidden_layers": getattr(
            text_cfg or cfg, "num_hidden_layers", getattr(cfg, "num_hidden_layers", None)
        ),
    }
