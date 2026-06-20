"""Large-Neighbourhood-Search booster: ruin-and-recreate over OPTIONAL streets.

GLS/Tabu converge to a fixed optional selection. This operator repeatedly
removes a random subset of currently-cleaned optional tasks (freeing route
time) and a random subset of routes' optional tail, then re-inserts optionals
greedily and runs the cheap exchange/reorder operators. It keeps any validated
improvement. This explores a different part of the search space than the CP
metaheuristics and can lift coverage on time-saturated large instances.
"""
from __future__ import annotations
import argparse, random, time
from pathlib import Path
from . import clipp_solver as cs


def ruin_recreate(inst, paths, sol, rng, ruin_frac):
    work = sol.copy()
    # remove a random fraction of optional tasks
    for route in work.routes:
        kept = []
        for task in route.tasks:
            st = inst.streets[task.street_id]
            if st.category == "O" and rng.random() < ruin_frac:
                continue
            kept.append(task)
        route.tasks = kept
    # recreate
    cs.insert_optionals_large(work, inst, paths, rng)
    cs.improve_optional_exchanges_large(work, inst, paths, rounds=20)
    for route in work.routes:
        cs.optimize_route_order_exact(route, inst, paths, max_tasks=12)
    cs.insert_optionals_large(work, inst, paths, rng)
    return work


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed-output", required=True)
    ap.add_argument("--seconds", type=float, default=300.0)
    args = ap.parse_args()
    inst = cs.parse_instance(args.input)
    paths = cs.ShortestPaths(inst)
    best = cs.read_submission(Path(args.seed_output), inst)
    _v, _e, bscore = cs.validate_solution(best, inst, paths)
    print(f"seed {bscore.value:.6f} cov {bscore.coverage:.4f} eff {bscore.efficiency:.4f}", flush=True)
    rng = random.Random(12345)
    started = time.monotonic()
    it = 0
    while time.monotonic() - started < args.seconds:
        it += 1
        frac = rng.choice([0.1, 0.15, 0.2, 0.3, 0.4])
        cand = ruin_recreate(inst, paths, best, rng, frac)
        v, e, sc = cs.validate_solution(cand, inst, paths)
        if v and sc.value > bscore.value + cs.EPS:
            best, bscore = cand, sc
            print(f"  it{it} frac{frac}: IMPROVED -> {sc.value:.6f} cov {sc.coverage:.4f} eff {sc.efficiency:.4f}", flush=True)
    cs.write_submission(args.output, best, inst, paths)
    print(f"FINAL {inst.name}: {bscore.value:.6f} ({it} iterations) -> {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
