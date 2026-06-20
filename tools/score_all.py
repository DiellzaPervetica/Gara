"""Validate every outputs/best/*.out against its instance with the official
node-list semantics and print a comparison vs the colleague's baseline."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from official_validator import parse_instance, validate

ROOT = Path(__file__).resolve().parent.parent
INST = ROOT / "instances"
OUT = ROOT / "outputs" / "best"

# colleague's validated baseline (their outputs/best/summary.json)
BASELINE = {"train_a": 0.700943, "train_b": 0.967738, "train_n": 0.947249, "m": 0.742008}

rows = []
total = 0.0
for name in ["train_a", "train_b", "train_n", "m"]:
    inst_path = INST / f"{name}.txt"
    out_path = OUT / f"{name}.out"
    if not inst_path.exists() or not out_path.exists():
        rows.append((name, "MISSING", None, None)); continue
    inst = parse_instance(str(inst_path))
    valid, errors, sc = validate(inst, str(out_path))
    score = sc["score"] if valid else 0.0
    total += score
    base = BASELINE.get(name)
    delta = score - base if base is not None else None
    rows.append((name, "VALID" if valid else f"INVALID {errors[:1]}", score, delta))

print(f"{'instance':10} {'status':10} {'score':>10} {'colleague':>10} {'delta':>10}")
for name, status, score, delta in rows:
    base = BASELINE.get(name)
    s = f"{score:.6f}" if isinstance(score, float) else str(score)
    b = f"{base:.6f}" if base is not None else "-"
    d = f"{delta:+.6f}" if delta is not None else "-"
    print(f"{name:10} {status:10} {s:>10} {b:>10} {d:>10}")
base_total = sum(BASELINE.values())
print(f"\nTOTAL (a+b+n+m): {total:.6f}   colleague baseline: {base_total:.6f}   delta: {total-base_total:+.6f}")
print("train_d: infeasible (0) for everyone")
