from __future__ import annotations

import random
import time
from copy import deepcopy

from .construction import harvest_free_cleaning, insert_optional
from .graph_utils import Graph
from .local_search import improve
from .models import Solution
from .scorer import score_solution
from .shortest_paths import ShortestPaths
from .validator import validate_solution


def _remove_some_optionals(solution: Solution, rng: random.Random) -> Solution:
    candidate = deepcopy(solution)
    optionals = []
    for ri, route in enumerate(candidate.routes):
        for ti, task in enumerate(route.tasks):
            if task.category == "O":
                optionals.append((ri, ti))
    rng.shuffle(optionals)
    remove_count = max(1, len(optionals) // 5) if optionals else 0
    for ri, ti in sorted(optionals[:remove_count], reverse=True):
        del candidate.routes[ri].tasks[ti]
    return candidate


def improve_alns(
    initial: Solution,
    graph: Graph,
    sp: ShortestPaths,
    time_budget: float,
    seed: int,
) -> Solution:
    rng = random.Random(seed)
    best = improve(deepcopy(initial), graph, sp)
    best = harvest_free_cleaning(best, graph, sp, include_optional=True)
    best = validate_solution(best, graph, sp)
    if not best.valid:
        return initial
    current = deepcopy(best)
    end = time.monotonic() + max(0.0, time_budget)
    temp = 0.01
    while time.monotonic() < end:
        cand = _remove_some_optionals(current, rng)
        cand = insert_optional(cand, graph, sp, rng.randint(0, 10**9), passes=1)
        cand = improve(cand, graph, sp)
        cand = harvest_free_cleaning(cand, graph, sp, include_optional=True)
        cand = validate_solution(cand, graph, sp)
        if not cand.valid:
            continue
        delta = cand.score - current.score
        if delta >= 0 or rng.random() < pow(2.718281828, delta / max(temp, 1e-9)):
            current = cand
        if cand.score > best.score:
            best = deepcopy(cand)
        temp *= 0.995
    return best
