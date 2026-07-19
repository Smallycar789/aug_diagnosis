# -*- coding: utf-8 -*-
"""Platform service: fault diagnosis testing."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

PLATFORM_DIR = Path(__file__).resolve().parent
if str(PLATFORM_DIR) not in sys.path:
    sys.path.insert(0, str(PLATFORM_DIR))

from platform_common import build_output, default_output_fields, ensure_project_on_path, load_input, resolve_param, resolve_path, save_platform_json

OUTPUT_SPECS = default_output_fields() + [
    {"chinese_name": "诊断准确率", "english_name": "accuracy", "param_type": "float", "desc": "测试集准确率"},
    {"chinese_name": "平均F1", "english_name": "f1_macro", "param_type": "float", "desc": "宏平均 F1"},
    {"chinese_name": "指标文件", "english_name": "metrics_json", "param_type": "str", "desc": "test_metrics.json 路径"},
    {"chinese_name": "混淆矩阵图", "english_name": "confusion_matrix_png", "param_type": "str", "desc": "混淆矩阵图片路径"},
]


def _pick_metric(metrics, *keys):
    for key in keys:
        if key in metrics and metrics[key] is not None:
            return metrics[key]
    return None


def fault_diagnosis_test(config_path=None, **kwargs):
    """故障诊断测试服务入口。"""
    ensure_project_on_path()
    params = load_input(config_path, **kwargs)
    run_dir = resolve_param(params, "run_dir", None) or resolve_param(params, "checkpoint_dir", None)
    split = resolve_param(params, "split", "test")
    platform_output_json = resolve_param(params, "platform_output_json_file", None)

    if not run_dir:
        raise ValueError("run_dir or checkpoint_dir is required")

    run_path = resolve_path(run_dir)
    resolved_cfg = run_path / "config_resolved.yaml"
    if not resolved_cfg.exists():
        raise FileNotFoundError("config_resolved.yaml not found in run_dir: {}".format(run_path))

    from diagnosis.test import main as test_main

    test_main(str(resolved_cfg), output_dir=str(run_path), split=split)

    metrics_path = run_path / "test_metrics.json"
    if not metrics_path.exists():
        metrics_path = run_path / "test_metrics_{}.json".format(split)

    metrics = {}
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as fh:
            metrics = json.load(fh)

    cm_path = run_path / "confusion_matrix_{}.png".format(split)
    if not cm_path.exists():
        cm_path = run_path / "confusion_matrix.png"

    result = build_output(
        "success",
        output_dir=str(run_path),
        algorithm=metrics.get("model", ""),
        dataset_profile=metrics.get("dataset", ""),
        accuracy=_pick_metric(metrics, "诊断准确率", "accuracy", "accuracy_percent"),
        f1_macro=_pick_metric(metrics, "平均F1分数", "f1_macro"),
        metrics_json=str(metrics_path) if metrics_path.exists() else "",
        confusion_matrix_png=str(cm_path) if cm_path.exists() else "",
        error_message="",
        platform_output_json_file=platform_output_json or "",
    )
    if platform_output_json:
        save_platform_json(platform_output_json, result, OUTPUT_SPECS)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Platform fault diagnosis test")
    parser.add_argument("--config", default=None, help="InterfaceType=input JSON path")
    args = parser.parse_args()
    try:
        out = fault_diagnosis_test(config_path=args.config)
        print(json.dumps(out, ensure_ascii=False, indent=2))
    except Exception as exc:
        fail = build_output("failed", error_message=str(exc), traceback=traceback.format_exc())
        print(json.dumps(fail, ensure_ascii=False, indent=2))
        raise SystemExit(1)
