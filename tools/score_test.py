"""Validate the scored test outputs (outputs/test/*.out) against the official
node-list semantics and print a summary. These are the competition instances."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from official_validator import parse_instance, validate

ROOT = Path(__file__).resolve().parent.parent
INST = ROOT / "instances" / "test"
OUT = ROOT / "outputs" / "test"

names = sorted(p.stem for p in INST.glob("*.txt"))
total = 0.0
print(f"{'instance':10} {'status':12} {'alpha':>6} {'score':>10} {'coverage':>9} {'efficiency':>10} {'mand':>8}")
for name in names:
    inst_path = INST / f"{name}.txt"
    out_path = OUT / f"{name}.out"
    inst = parse_instance(str(inst_path))
    if not out_path.exists():
        print(f"{name:10} {'NO OUTPUT':12} {inst['alpha']:6.2f}")
        continue
    valid, errors, sc = validate(inst, str(out_path))
    score = sc["score"] if valid else 0.0
    total += score
    status = "VALID" if valid else "INVALID"
    md = f"{sc['mandatory_cleaned']}/{sc['mandatory_total']}"
    print(f"{name:10} {status:12} {inst['alpha']:6.2f} {score:10.6f} {sc['coverage']:9.4f} {sc['efficiency']:10.4f} {md:>8}")
    if not valid:
        print("   errors:", errors[:3])
print(f"\nTEST TOTAL: {total:.6f}")
