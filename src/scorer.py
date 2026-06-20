from __future__ import annotations

from collections import Counter

from .models import CAPACITY, Instance, Solution


def bounds(instance: Instance) -> tuple[int, float]:
    lmax = sum(s.length for s in instance.streets if s.cleanable)
    wmax = sum((30 - s.requirement) * s.length / 1000 for s in instance.streets if s.cleanable)
    return lmax, wmax


def edge_waste(capacity: int, requirement: int, length: int) -> float:
    return (capacity - requirement) * length / 1000


def optional_net_gain(instance: Instance, vehicle_capacity: int, street_id: int) -> float:
    street = instance.streets[street_id]
    lmax, wmax = bounds(instance)
    coverage = instance.alpha * street.length / lmax if lmax else 0.0
    if wmax == 0:
        penalty = 0.0
    else:
        penalty = (1 - instance.alpha) * edge_waste(vehicle_capacity, street.requirement, street.length) / wmax
    return coverage - penalty


def score_solution(solution: Solution) -> Solution:
    instance = solution.instance
    lmax, wmax = bounds(instance)
    cleanable = {s.id for s in instance.streets if s.cleanable}
    by_id = {s.id: s for s in instance.streets}
    unique: set[int] = set()
    waste = 0.0
    for route in solution.routes:
        cap = route.vehicle.capacity
        for sid in route.cleaned_ids():
            if sid not in cleanable:
                continue
            street = by_id[sid]
            waste += edge_waste(cap, street.requirement, street.length)
            unique.add(sid)
    cleaned_len = sum(by_id[sid].length for sid in unique)
    coverage = cleaned_len / lmax if lmax else 1.0
    if wmax == 0:
        efficiency = 1.0 if waste == 0 else 0.0
    else:
        efficiency = 1 - waste / wmax
    solution.coverage = coverage
    solution.efficiency = efficiency
    solution.total_waste = waste
    solution.score = instance.alpha * coverage + (1 - instance.alpha) * efficiency
    return solution


def duplicate_cleaned(solution: Solution) -> dict[int, int]:
    c = Counter(solution.cleaned_ids())
    return {sid: n for sid, n in c.items() if n > 1}
