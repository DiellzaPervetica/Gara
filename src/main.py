from __future__ import annotations

import argparse
import json
from pathlib import Path

from .solver import solve_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input")
    ap.add_argument("--output")
    ap.add_argument("--input-dir")
    ap.add_argument("--output-dir", default="outputs")
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()

    metrics = []
    if args.input:
        if not args.output:
            raise SystemExit("--output is required with --input")
        metrics.append(solve_file(args.input, args.output, args.time_limit, args.seeds))
    elif args.input_dir:
        indir = Path(args.input_dir)
        outdir = Path(args.output_dir)
        for path in sorted(indir.iterdir()):
            if path.suffix.lower() not in {".txt", ".in", ".docx"}:
                continue
            try:
                metrics.append(solve_file(path, outdir / f"{path.stem}.out", args.time_limit, args.seeds))
            except Exception as exc:
                item = {"instance": path.stem, "valid": False, "reason": str(exc)}
                metrics.append(item)
                print(json.dumps(item, indent=2))
    else:
        raise SystemExit("provide --input or --input-dir")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
