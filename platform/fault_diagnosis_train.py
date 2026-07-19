# -*- coding: utf-8 -*-
"""Platform service: fault diagnosis training."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

PLATFORM_DIR = Path(__file__).resolve().parent
if str(PLATFORM_DIR) not in sys.path:
    sys.path.insert(0, str(PLATFORM_DIR))

from config import DIAGNOSIS_ALGORITHMS, resolve_data_dir, write_diagnosis_config
from platform_common import build_output, default_output_fields, ensure_project_on_path, load_input, resolve_param, save_platform_json

OUTPUT_SPECS = default_output_fields() + [
    {"chinese_name": "配置文件", "english_name": "config_path", "param_type": "str", "desc": "生成的 yaml 路径"},
    {"chinese_name": "检查点", "english_name": "checkpoint_path", "param_type": "str", "desc": "最佳模型路径"},
]


def fault_diagnosis_train(config_path=None, **kwargs):
    """故障诊断训练服务入口。"""
    ensure_project_on_path()
    params = load_input(config_path, **kwargs)
    data_dir = resolve_param(params, "data_dir", None)
    profile = resolve_param(params, "dataset_profile", "sensitivity")
    algorithm = resolve_param(params, "algorithm", "cnn_bigru")
    device = resolve_param(params, "device", "cpu")
    smoke = bool(resolve_param(params, "smoke", True))
    platform_output_json = resolve_param(params, "platform_output_json_file", None)

    if algorithm not in DIAGNOSIS_ALGORITHMS:
        raise ValueError("algorithm must be one of: {}".format(", ".join(DIAGNOSIS_ALGORITHMS)))

    resolved_data = resolve_data_dir(data_dir, profile)
    if not resolved_data.exists():
        raise FileNotFoundError("data_dir does not exist: {}".format(resolved_data))

    yaml_path = write_diagnosis_config(profile, algorithm, resolved_data, device=device, smoke=smoke)
    from diagnosis.train import main as train_main

    run_dir = train_main(str(yaml_path))
    checkpoint = Path(run_dir) / "checkpoint_best.pth"

    result = build_output(
        "success",
        output_dir=str(run_dir),
        algorithm=algorithm,
        dataset_profile=profile,
        config_path=str(yaml_path),
        checkpoint_path=str(checkpoint) if checkpoint.exists() else "",
        error_message="",
        platform_output_json_file=platform_output_json or "",
    )
    if platform_output_json:
        save_platform_json(platform_output_json, result, OUTPUT_SPECS)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Platform fault diagnosis train")
    parser.add_argument("--config", default=None, help="InterfaceType=input JSON path")
    args = parser.parse_args()
    try:
        out = fault_diagnosis_train(config_path=args.config)
        print(json.dumps(out, ensure_ascii=False, indent=2))
    except Exception as exc:
        fail = build_output("failed", error_message=str(exc), traceback=traceback.format_exc())
        print(json.dumps(fail, ensure_ascii=False, indent=2))
        raise SystemExit(1)
