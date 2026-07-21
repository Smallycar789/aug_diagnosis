# -*- coding: utf-8 -*-
"""Shared helpers for platform algorithm service scripts."""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

PLATFORM_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PLATFORM_ROOT.parent
RUNTIME_ROOT = PLATFORM_ROOT / "runtime"

_IMAGE_FIELD_KEYS = frozenset(
    {
        "signal_comparison",
        "frequency_comparison",
        "feature_plots",
        "loss_curves",
        "training_curves",
        "tsne_visualization",
        "target_confusion_matrix",
        "vae_comprehensive_results",
    }
)
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".svg", ".gif", ".webp")


def _is_image_path(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    lowered = value.lower()
    return lowered.endswith(_IMAGE_SUFFIXES) or "/png" in lowered


def sanitize_metrics_for_platform(metrics: Any) -> Any:
    """Remove image file references from metrics while keeping numeric indicators."""
    if isinstance(metrics, dict):
        cleaned: Dict[str, Any] = {}
        for key, value in metrics.items():
            if key in _IMAGE_FIELD_KEYS:
                continue
            if _is_image_path(value):
                continue
            cleaned[key] = sanitize_metrics_for_platform(value)
        return cleaned
    if isinstance(metrics, list):
        return [sanitize_metrics_for_platform(item) for item in metrics]
    return metrics


def public_platform_result(result: Dict[str, Any], allowed_image_fields: Optional[List[str]] = None) -> Dict[str, Any]:
    """Drop image paths from platform API output except explicitly allowed fields."""
    allowed = set(allowed_image_fields or ())
    public: Dict[str, Any] = {}
    for key, value in result.items():
        if key in allowed:
            public[key] = value
            continue
        if key in _IMAGE_FIELD_KEYS or _is_image_path(value):
            continue
        if key == "metrics" and isinstance(value, dict):
            public[key] = sanitize_metrics_for_platform(value)
            continue
        public[key] = value
    return public


def ensure_project_on_path() -> None:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def load_input(config_path: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """Load flat parameter dict from JSON file or kwargs."""
    params: Dict[str, Any] = {}
    if config_path:
        path = Path(config_path)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        if isinstance(payload, dict) and payload.get("InterfaceType") == "input":
            for item in payload.get("InterfaceParam", []):
                if isinstance(item, dict) and item.get("english_name"):
                    params[item["english_name"]] = item.get("param_value")
        elif isinstance(payload, dict):
            params.update(payload)
    for key, value in kwargs.items():
        if value is not None:
            params[key] = value
    return params


def resolve_param(params: Dict[str, Any], english_name: str, default: Any = None) -> Any:
    if english_name in params and params[english_name] not in (None, ""):
        return params[english_name]
    return default


def resolve_path(path_value: str, base: Optional[Path] = None) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        base = base or Path.cwd()
        path = (base / path).resolve()
    return path.resolve()


def build_output(status: str, **fields: Any) -> Dict[str, Any]:
    return {"status": status, **fields}


def build_interface_output(result: Dict[str, Any], field_specs: List[Dict[str, str]]) -> Dict[str, Any]:
    """Convert flat result dict to InterfaceType=output JSON."""
    interface_params: List[Dict[str, Any]] = []
    for spec in field_specs:
        english_name = spec["english_name"]
        interface_params.append(
            {
                "chinese_name": spec.get("chinese_name", english_name),
                "english_name": english_name,
                "param_type": spec.get("param_type", "str"),
                "desc": spec.get("desc", ""),
                "param_value": result.get(english_name),
            }
        )
    return {"InterfaceType": "output", "InterfaceParam": interface_params}


def save_platform_json(path_value: str, result: Dict[str, Any], field_specs: List[Dict[str, str]]) -> Path:
    out_path = resolve_path(path_value)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_interface_output(result, field_specs)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return out_path


def default_output_fields() -> List[Dict[str, str]]:
    return [
        {"chinese_name": "执行状态", "english_name": "status", "param_type": "str", "desc": "success/failed"},
        {"chinese_name": "输出目录", "english_name": "output_dir", "param_type": "str", "desc": "算法结果目录"},
        {"chinese_name": "算法名称", "english_name": "algorithm", "param_type": "str", "desc": "算法标识"},
        {"chinese_name": "数据集配置", "english_name": "dataset_profile", "param_type": "str", "desc": "数据集 profile"},
        {"chinese_name": "错误信息", "english_name": "error_message", "param_type": "str", "desc": "失败时的错误描述"},
    ]


def run_service(main_func, config_path: Optional[str] = None, output_field_specs: Optional[List[Dict[str, str]]] = None, **kwargs):
    """Execute service and optionally persist platform output JSON."""
    ensure_project_on_path()
    specs = output_field_specs or default_output_fields()
    try:
        result = main_func(config_path=config_path, **kwargs)
    except Exception as exc:
        result = build_output("failed", error_message=str(exc), traceback=traceback.format_exc())
    platform_json = resolve_param(result if isinstance(result, dict) else {}, "platform_output_json_file", None)
    if not platform_json:
        platform_json = resolve_param(load_input(config_path, **kwargs), "platform_output_json_file", None)
    if platform_json:
        save_platform_json(platform_json, result, specs)
    return result
