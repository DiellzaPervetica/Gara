"""Iterated, re-seeded OR-Tools boosting on top of the CLIPP engine.

Strategy to reach the maximum score on an instance:
  1. Collect candidate seed solutions (existing .out files + fresh multistarts).
  2. Keep the best valid candidate.
  3. Iterate OR-Tools polish, alternating Guided Local Search and Tabu Search,
     re-seeding each round from the current best. The polish only ever returns
     an improvement, so the score is monotonically non-decreasing.
  4. Stop when a full GLS+Tabu round yields no gain, or the wall-clock budget
     for the instance is exhausted.

Everything is validated with the in-module validator (which matches the
official netlify validator to 1e-6) before being accepted or written.
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

from . import clipp_solver as cs


def _fresh_candidates(instance, paths, seeds, time_budget):
    """Diversified fresh constructions + local improvement (no OR-Tools)."""
    out = []
    started = time.monotonic()
    mandatory_count = sum(s.category == "M" for s in instance.streets)
    big = mandatory_count > 100 or len(instance.streets) > 300
    # Robust: never rely on a single constructor. The regret heuristic can fail
    # on instances with many heavy-mandatory streets and few Large vehicles,
    # while the heavy-first "large" constructor succeeds (and vice-versa).
    constructors = (
        [cs.construct_mandatory_large, cs.construct_mandatory, cs.construct_mandatory_randomized]
        if big else
        [cs.construct_mandatory, cs.construct_mandatory_randomized, cs.construct_mandatory_large]
    )
    for seed in range(seeds):
        if seed and time.monotonic() - started > time_budget:
            break
        rng = random.Random(20_000 + seed * 2_654_435_761 % 1_000_003)
        cand = None
        for make in constructors:
            cand = make(instance, paths, rng)
            if cand is not None:
                break
        if cand is None:
            continue
        if big:
            cs.insert_optionals_large(cand, instance, paths, rng)
            cs.improve_optional_exchanges_large(cand, instance, paths, rounds=16)
        else:
            cs.insert_optionals(cand, instance, paths, rng)
            cs.local_search(cand, instance, paths, rng)
        valid, _e, sc = cs.validate_solution(cand, instance, paths)
        if valid:
            out.append((sc.value, cand))
    return out


def boost(instance_path: Path, seed_outputs, out_path: Path, total_seconds: float,
          fresh_seeds: int, round_seconds: int):
    inst = cs.parse_instance(instance_path)
    paths = cs.ShortestPaths(inst)
    from ortools.constraint_solver import routing_enums_pb2 as E

    started = time.monotonic()
    best = None
    best_score = None

    def consider(sol, tag):
        nonlocal best, best_score
        if sol is None:
            return False
        valid, _e, sc = cs.validate_solution(sol, inst, paths)
        if valid and (best_score is None or sc.value > best_score.value + cs.EPS):
            best, best_score = sol, sc
            print(f"    [{tag}] -> {sc.value:.6f} (cov {sc.coverage:.4f} eff {sc.efficiency:.4f})")
            return True
        return False

    # 1. existing seed outputs
    for sp in seed_outputs:
        if Path(sp).exists():
            try:
                consider(cs.read_submission(Path(sp), inst), f"seed:{Path(sp).name}")
            except Exception as exc:  # noqa: BLE001
                print(f"    seed {sp} unreadable: {exc}")

    # 2. fresh multistarts
    for val, cand in _fresh_candidates(inst, paths, fresh_seeds, total_seconds * 0.35):
        consider(cand, "fresh")

    if best is None:
        print("    no feasible seed found")
        return None

    # 2b. fresh CP construction from scratch can land in a different, higher
    #     basin than any insertion seed (especially for pure-coverage alpha=1).
    if time.monotonic() - started < total_seconds:
        scratch = cs.ortools_polish(inst, best, max(round_seconds, 2 * round_seconds),
                                    start_from_seed=False)
        consider(scratch, "cp-scratch")

    # 3. iterated re-seeded polish, alternating metaheuristics
    metas = [
        E.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
        E.LocalSearchMetaheuristic.TABU_SEARCH,
        E.LocalSearchMetaheuristic.GENERIC_TABU_SEARCH,
        E.LocalSearchMetaheuristic.SIMULATED_ANNEALING,
    ]
    stagnation = 0
    while time.monotonic() - started < total_seconds and stagnation < len(metas):
        improved_this_round = False
        for meta in metas:
            if time.monotonic() - started >= total_seconds:
                break
            polished = cs.ortools_polish(inst, best, round_seconds, metaheuristic=meta)
            if consider(polished, f"polish:{meta}"):
                improved_this_round = True
        stagnation = 0 if improved_this_round else stagnation + 1

    cs.write_submission(out_path, best, inst, paths)
    print(f"  FINAL {inst.name}: {best_score.value:.6f}  -> {out_path}")
    return best_score.value


def main() -> int:
    ap = argparse.ArgumentParser(description="Iterated OR-Tools boosting for CLIPP")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed-output", action="append", default=[],
                    help="existing .out files to use as starting seeds (repeatable)")
    ap.add_argument("--total-seconds", type=float, default=180.0)
    ap.add_argument("--round-seconds", type=int, default=30)
    ap.add_argument("--fresh-seeds", type=int, default=12)
    args = ap.parse_args()
    boost(args.input, args.seed_output, args.output, args.total_seconds,
          args.fresh_seeds, args.round_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
