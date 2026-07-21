# -*- coding: utf-8 -*-
"""Platform service: data augmentation sample generation."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

PLATFORM_DIR = Path(__file__).resolve().parent
if str(PLATFORM_DIR) not in sys.path:
    sys.path.insert(0, str(PLATFORM_DIR))

from platform_common import (
    build_output,
    default_output_fields,
    ensure_project_on_path,
    load_input,
    public_platform_result,
    resolve_param,
    resolve_path,
    sanitize_metrics_for_platform,
    save_platform_json,
)

OUTPUT_SPECS = default_output_fields() + [
    {"chinese_name": "生成样本数", "english_name": "num_generated", "param_type": "int", "desc": "生成样本总数"},
    {"chinese_name": "生成文件列表", "english_name": "generated_files", "param_type": "list", "desc": "生成 npy 文件路径"},
    {"chinese_name": "指标详情", "english_name": "metrics", "param_type": "dict", "desc": "评估指标（不含图片路径）"},
]


def _collect_generated_files(run_path):
    files = []
    for pattern in ("generated_*.npy", "generated_samples*.npy"):
        files.extend(sorted(str(p) for p in run_path.glob(pattern)))
    return files


def data_augmentation_generate(config_path=None, **kwargs):
    """数据增强生成服务入口。"""
    ensure_project_on_path()
    params = load_input(config_path, **kwargs)
    run_dir = resolve_param(params, "run_dir", None)
    algorithm = resolve_param(params, "algorithm", None)
    platform_output_json = resolve_param(params, "platform_output_json_file", None)

    if not run_dir:
        raise ValueError("run_dir is required")

    run_path = resolve_path(run_dir)
    resolved_cfg = run_path / "config_resolved.yaml"
    if not resolved_cfg.exists():
        raise FileNotFoundError("config_resolved.yaml not found in run_dir: {}".format(run_path))

    from data_aug.train import main as train_main

    train_main(str(resolved_cfg), stage="generate", output_dir=str(run_path))

    generated_files = _collect_generated_files(run_path)
    metrics_path = run_path / "metrics.json"
    num_generated = len(generated_files)
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as fh:
            metrics = sanitize_metrics_for_platform(json.load(fh))
        num_generated = int(metrics.get("num_generated", num_generated))

    result = public_platform_result(
        build_output(
            "success",
            output_dir=str(run_path),
            algorithm=algorithm or metrics.get("model", ""),
            dataset_profile=metrics.get("dataset", ""),
            num_generated=num_generated,
            generated_files=generated_files,
            metrics=metrics,
            error_message="",
            platform_output_json_file=platform_output_json or "",
        )
    )
    if platform_output_json:
        save_platform_json(platform_output_json, result, OUTPUT_SPECS)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Platform data augmentation generate")
    parser.add_argument("--config", default=None, help="InterfaceType=input JSON path")
    args = parser.parse_args()
    try:
        out = data_augmentation_generate(config_path=args.config)
        print(json.dumps(out, ensure_ascii=False, indent=2))
    except Exception as exc:
        fail = build_output("failed", error_message=str(exc), traceback=traceback.format_exc())
        print(json.dumps(fail, ensure_ascii=False, indent=2))
        raise SystemExit(1)
