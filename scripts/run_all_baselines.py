from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chem_synonymizer.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lexical and semantic baselines.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    parser.add_argument("--models-config", default=str(ROOT / "configs" / "models.yaml"))
    parser.add_argument("--csv-path", default=None)
    parser.add_argument("--eval-frac", type=float, default=None)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--semantic-models", nargs="*", default=None)
    parser.add_argument("--skip-lexical", action="store_true")
    parser.add_argument("--skip-semantic", action="store_true")
    parser.add_argument("--make-plots", action="store_true")
    parser.add_argument("--set", action="append", default=None)
    return parser.parse_args()


def add_common_args(cmd: list[str], args: argparse.Namespace) -> list[str]:
    cmd += ["--config", args.config, "--models-config", args.models_config]
    if args.csv_path:
        cmd += ["--csv-path", args.csv_path]
    if args.eval_frac is not None:
        cmd += ["--eval-frac", str(args.eval_frac)]
    if args.limit_rows is not None:
        cmd += ["--limit-rows", str(args.limit_rows)]
    if args.device and "run_semantic_baseline.py" in cmd[1]:
        cmd += ["--device", args.device]
    for override in args.set or []:
        cmd += ["--set", override]
    return cmd


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.models_config, args.set)
    semantic_models = args.semantic_models or cfg.get("experiments", {}).get("semantic_models", [])

    if not args.skip_lexical:
        cmd = [sys.executable, str(ROOT / "scripts" / "run_lexical_baseline.py"), "--run-name", "lexical"]
        run(add_common_args(cmd, args))

    if not args.skip_semantic:
        for model_key in semantic_models:
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "run_semantic_baseline.py"),
                "--model-key",
                model_key,
                "--run-name",
                model_key,
            ]
            run(add_common_args(cmd, args))

    if args.make_plots:
        cmd = [sys.executable, str(ROOT / "scripts" / "make_plots.py")]
        run(cmd)


if __name__ == "__main__":
    main()
