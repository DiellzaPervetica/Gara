from __future__ import annotations

import json
import time
from pathlib import Path

from .alns import improve_alns
from .construction import (
    construct_mandatory,
    exact_group_construct_mandatory,
    exact_phase_construct_mandatory,
    harvest_free_cleaning,
    insert_optional,
    regret_construct_mandatory,
)
from .graph_utils import Graph
from .local_search import improve
from .models import Instance, Solution
from .parser import parse_instance
from .scorer import bounds
from .shortest_paths import ShortestPaths
from .validator import validate_solution


def solve_instance(instance: Instance, time_limit: float = 30.0, seeds: int = 20) -> Solution:
    graph = Graph(instance)
    sp = ShortestPaths(graph)
    start = time.monotonic()
    best: Solution | None = None
    attempts = max(1, seeds)
    per_seed_alns = max(0.0, time_limit * 0.35 / attempts)

    for seed in range(attempts):
        if time.monotonic() - start > time_limit:
            break
        base = exact_group_construct_mandatory(instance, graph, sp, seed)
        if base is None:
            base = exact_phase_construct_mandatory(instance, graph, sp, seed)
        if base is None:
            base = regret_construct_mandatory(instance, graph, sp, seed)
        if base is None:
            base = construct_mandatory(instance, graph, sp, seed)
        if base is None:
            continue
        base = harvest_free_cleaning(base, graph, sp, include_optional=False)
        if instance.m > 500:
            cand = harvest_free_cleaning(base, graph, sp, include_optional=True)
        else:
            cand = insert_optional(base, graph, sp, seed, passes=1)
            cand = improve(cand, graph, sp)
            cand = harvest_free_cleaning(cand, graph, sp, include_optional=True)
        cand = validate_solution(cand, graph, sp)
        if not cand.valid:
            continue
        if instance.m <= 500:
            cand = improve_alns(cand, graph, sp, per_seed_alns, seed)
        cand = validate_solution(cand, graph, sp)
        if cand.valid and (best is None or cand.score > best.score):
            best = cand

    if best is None:
        raise RuntimeError(f"failed to build valid solution for {instance.name}")
    return best


def solution_metrics(solution: Solution) -> dict:
    inst = solution.instance
    lmax, wmax = bounds(inst)
    counts = {cat: sum(1 for s in inst.streets if s.category == cat) for cat in ["M", "O", "C"]}
    return {
        "instance": inst.name,
        "N": inst.n,
        "M": inst.m,
        "T": inst.time_limit,
        "C": len(inst.vehicles),
        "depot": inst.depot,
        "alpha": inst.alpha,
        "counts": counts,
        "vehicle_counts": {k: sum(1 for v in inst.vehicles if v.kind == k) for k in ["S", "M", "L"]},
        "Lmax": lmax,
        "Wmax": wmax,
        "mandatory_cleaned": len(inst.mandatory_ids & set(solution.cleaned_ids())),
        "mandatory_total": len(inst.mandatory_ids),
        "optional_cleaned": len(inst.optional_ids & set(solution.cleaned_ids())),
        "optional_total": len(inst.optional_ids),
        "coverage": solution.coverage,
        "efficiency": solution.efficiency,
        "total_waste": solution.total_waste,
        "score": solution.score,
        "route_times": [r.time for r in solution.routes],
        "valid": solution.valid,
        "reason": solution.reason,
    }


def solve_file(input_path: str | Path, output_path: str | Path, time_limit: float, seeds: int) -> dict:
    instance = parse_instance(input_path)
    solution = solve_instance(instance, time_limit=time_limit, seeds=seeds)
    write_submission(solution, output_path)
    metrics = solution_metrics(solution)
    print(json.dumps(metrics, indent=2))
    return metrics


def write_submission(solution: Solution, output_path: str | Path) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(len(solution.instance.vehicles))]
    for route in solution.routes:
        nodes = route.nodes if route.nodes else [solution.instance.depot]
        if not route.tasks and nodes == [solution.instance.depot]:
            lines.append("0")
            lines.append(str(solution.instance.depot))
            lines.append("")
            continue
        # The public statement says n is the number of junctions, but the
        # official example and validator use n as the number of traversed
        # streets/steps, so the node line contains n + 1 junction indices.
        lines.append(str(max(0, len(nodes) - 1)))
        lines.append(" ".join(map(str, nodes)))
        lines.append(" ".join(map(str, route.cleaned_ids())))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
