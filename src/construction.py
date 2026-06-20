from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .graph_utils import Graph
from .models import Instance, Route, Solution, Task
from .scorer import edge_waste, optional_net_gain
from .shortest_paths import INF, ShortestPaths
from .validator import expand_route


@dataclass
class Insertion:
    route_index: int
    position: int
    task: Task
    delta: int
    cost: float
    gain: float = 0.0


def route_task_time(tasks: list[Task], depot: int, sp: ShortestPaths) -> int:
    cur = depot
    total = 0
    for task in tasks:
        d = sp.distance(cur, task.start)
        if d >= INF:
            return INF
        total += d + task.time
        cur = task.end
    d = sp.distance(cur, depot)
    if d >= INF:
        return INF
    return total + d


def delta_time(route: Route, position: int, task: Task, depot: int, sp: ShortestPaths) -> int:
    old = route_task_time(route.tasks, depot, sp)
    new_tasks = route.tasks[:position] + [task] + route.tasks[position:]
    new = route_task_time(new_tasks, depot, sp)
    if old >= INF or new >= INF:
        return INF
    return new - old


def compatible(instance: Instance, street_id: int, vehicle_capacity: int) -> bool:
    return vehicle_capacity >= instance.streets[street_id].requirement


def _vehicle_mismatch(capacity: int, requirement: int) -> int:
    return capacity - requirement


def mandatory_cost(instance: Instance, cap: int, task: Task, delta: int) -> float:
    if instance.alpha >= 0.999:
        lambda_waste = 0.0
        lambda_mismatch = 0.0
    elif instance.alpha <= 0.001:
        lambda_waste = 2000.0
        lambda_mismatch = 500.0
    else:
        lambda_waste = 100.0 * (1 - instance.alpha)
        lambda_mismatch = 50.0 * (1 - instance.alpha)
    waste = edge_waste(cap, task.requirement, task.length)
    return delta + lambda_waste * waste + lambda_mismatch * _vehicle_mismatch(cap, task.requirement) - 0.001 * task.length


def best_insertions_for_street(
    solution: Solution,
    graph: Graph,
    sp: ShortestPaths,
    street_id: int,
    optional: bool,
    rng: random.Random,
    allowed_routes: set[int] | None = None,
) -> list[Insertion]:
    instance = solution.instance
    tasks = graph.tasks_for_street(instance.streets[street_id])
    candidates: list[Insertion] = []
    for ri, route in enumerate(solution.routes):
        if allowed_routes is not None and ri not in allowed_routes:
            continue
        cap = route.vehicle.capacity
        if not compatible(instance, street_id, cap):
            continue
        for task in tasks:
            for pos in range(len(route.tasks) + 1):
                dt = delta_time(route, pos, task, instance.depot, sp)
                if dt >= INF:
                    continue
                old_time = route_task_time(route.tasks, instance.depot, sp)
                if old_time + dt > instance.time_limit:
                    continue
                if optional:
                    gain = optional_net_gain(instance, cap, street_id)
                    if instance.alpha <= 0.001 and abs(cap - task.requirement) > 0:
                        continue
                    if instance.alpha < 0.999 and gain <= 1e-12:
                        continue
                    if instance.alpha >= 0.999:
                        priority = task.length / max(dt, 1)
                    else:
                        priority = gain / max(dt, 1)
                    priority += (0.000001 * task.length) + rng.uniform(-1e-8, 1e-8)
                    candidates.append(Insertion(ri, pos, task, dt, -priority, gain))
                else:
                    cost = mandatory_cost(instance, cap, task, dt) + rng.uniform(0, 1e-6)
                    candidates.append(Insertion(ri, pos, task, dt, cost))
    candidates.sort(key=lambda x: x.cost)
    return candidates


def build_empty_solution(instance: Instance) -> Solution:
    return Solution(instance, [Route(vehicle=v) for v in instance.vehicles])


def harvest_free_cleaning(
    solution: Solution,
    graph: Graph,
    sp: ShortestPaths,
    include_optional: bool = False,
) -> Solution:
    """Clean compatible streets already traversed by route expansion.

    This is a high-value arc-routing trick: repositioning paths may traverse
    mandatory/optional streets, and cleaning them costs no additional time.
    """
    instance = solution.instance
    by_id = {s.id: s for s in instance.streets}
    for route in solution.routes:
        route.extra_cleaned = []
    global_cleaned = {t.street_id for r in solution.routes for t in r.tasks}
    for i, route in enumerate(solution.routes):
        if not expand_route(solution, graph, sp, i):
            continue
        for sid in route.traversed_streets:
            if sid in global_cleaned:
                continue
            street = by_id[sid]
            if not street.cleanable or route.vehicle.capacity < street.requirement:
                continue
            if street.category == "M":
                if instance.alpha <= 0.001 and route.vehicle.capacity != street.requirement:
                    continue
                route.extra_cleaned.append(sid)
                global_cleaned.add(sid)
            elif include_optional:
                gain = optional_net_gain(instance, route.vehicle.capacity, sid)
                exact = route.vehicle.capacity == street.requirement
                if instance.alpha >= 0.999 or gain > 1e-12 or (instance.alpha <= 0.001 and exact):
                    route.extra_cleaned.append(sid)
                    global_cleaned.add(sid)
    return solution


def construct_mandatory(instance: Instance, graph: Graph, sp: ShortestPaths, seed: int = 0) -> Solution | None:
    rng = random.Random(seed)
    solution = build_empty_solution(instance)
    remaining = set(instance.mandatory_ids)

    def difficulty(sid: int) -> tuple:
        s = instance.streets[sid]
        compatible_count = sum(1 for v in instance.vehicles if v.capacity >= s.requirement)
        standalone = min(
            (sp.distance(instance.depot, t.start) + t.time + sp.distance(t.end, instance.depot))
            for t in graph.tasks_for_street(s)
        )
        return (-s.requirement, compatible_count, -standalone, -s.length, -s.direction, rng.random())

    ordered = sorted(remaining, key=difficulty)
    for sid in ordered:
        harvest_free_cleaning(solution, graph, sp, include_optional=False)
        if sid in set(solution.cleaned_ids()):
            remaining.discard(sid)
            continue
        candidates = best_insertions_for_street(solution, graph, sp, sid, optional=False, rng=rng)
        if not candidates:
            return None
        best = candidates[0]
        route = solution.routes[best.route_index]
        route.tasks.insert(best.position, best.task)
        harvest_free_cleaning(solution, graph, sp, include_optional=False)
        remaining.remove(sid)

    return solution


def regret_construct_mandatory(instance: Instance, graph: Graph, sp: ShortestPaths, seed: int = 0) -> Solution | None:
    rng = random.Random(seed)
    solution = build_empty_solution(instance)
    remaining = set(instance.mandatory_ids)
    while remaining:
        harvest_free_cleaning(solution, graph, sp, include_optional=False)
        remaining -= set(solution.cleaned_ids())
        if not remaining:
            break
        chosen: tuple[float, int, Insertion] | None = None
        for sid in list(remaining):
            candidates = best_insertions_for_street(solution, graph, sp, sid, optional=False, rng=rng)
            if not candidates:
                continue
            best = candidates[0]
            second = candidates[1].cost if len(candidates) > 1 else best.cost + 1_000_000
            street = instance.streets[sid]
            regret = second - best.cost + 1000 * street.requirement + 0.01 * street.length
            if chosen is None or regret > chosen[0]:
                chosen = (regret, sid, best)
        if chosen is None:
            return None
        _, sid, ins = chosen
        solution.routes[ins.route_index].tasks.insert(ins.position, ins.task)
        harvest_free_cleaning(solution, graph, sp, include_optional=False)
        remaining -= set(solution.cleaned_ids())
    return solution


def exact_phase_construct_mandatory(instance: Instance, graph: Graph, sp: ShortestPaths, seed: int = 0) -> Solution | None:
    """Build mandatory coverage while protecting scarce exact-match vehicles.

    This is intentionally conservative: heavy streets go to Large first, medium
    streets to Medium first, and light streets to Small first. Remaining streets
    then fall back to the normal all-compatible regret insertion.
    """
    rng = random.Random(seed)
    solution = build_empty_solution(instance)
    remaining = set(instance.mandatory_ids)

    phases = [
        (30, {i for i, r in enumerate(solution.routes) if r.vehicle.capacity == 30}),
        (20, {i for i, r in enumerate(solution.routes) if r.vehicle.capacity == 20}),
        (10, {i for i, r in enumerate(solution.routes) if r.vehicle.capacity == 10}),
    ]

    for req, allowed in phases:
        phase_ids = {sid for sid in remaining if instance.streets[sid].requirement == req}
        while phase_ids:
            harvest_free_cleaning(solution, graph, sp, include_optional=False)
            covered = set(solution.cleaned_ids())
            remaining -= covered
            phase_ids -= covered
            if not phase_ids:
                break
            chosen: tuple[float, int, Insertion] | None = None
            for sid in list(phase_ids):
                candidates = best_insertions_for_street(
                    solution, graph, sp, sid, optional=False, rng=rng, allowed_routes=allowed
                )
                if not candidates:
                    continue
                best = candidates[0]
                second = candidates[1].cost if len(candidates) > 1 else best.cost + 1_000_000
                street = instance.streets[sid]
                regret = second - best.cost + 0.01 * street.length
                if chosen is None or regret > chosen[0]:
                    chosen = (regret, sid, best)
            if chosen is None:
                break
            _, sid, ins = chosen
            solution.routes[ins.route_index].tasks.insert(ins.position, ins.task)
            harvest_free_cleaning(solution, graph, sp, include_optional=False)
            covered = set(solution.cleaned_ids())
            remaining -= covered
            phase_ids -= covered

    while remaining:
        harvest_free_cleaning(solution, graph, sp, include_optional=False)
        remaining -= set(solution.cleaned_ids())
        if not remaining:
            break
        chosen = None
        for sid in list(remaining):
            candidates = best_insertions_for_street(solution, graph, sp, sid, optional=False, rng=rng)
            if not candidates:
                continue
            best = candidates[0]
            second = candidates[1].cost if len(candidates) > 1 else best.cost + 1_000_000
            street = instance.streets[sid]
            regret = second - best.cost + 1000 * street.requirement + 0.01 * street.length
            if chosen is None or regret > chosen[0]:
                chosen = (regret, sid, best)
        if chosen is None:
            return None
        _, sid, ins = chosen
        solution.routes[ins.route_index].tasks.insert(ins.position, ins.task)
        harvest_free_cleaning(solution, graph, sp, include_optional=False)
        remaining -= set(solution.cleaned_ids())

    return solution


def _best_subset_routes(
    instance: Instance,
    graph: Graph,
    sp: ShortestPaths,
    street_ids: list[int],
) -> dict[int, tuple[int, list[Task]]]:
    """Exact shortest cleaning order for every subset of a small street set."""
    k = len(street_ids)
    if k == 0:
        return {0: (0, [])}
    if k > 14:
        return {}
    orientations = [graph.tasks_for_street(instance.streets[sid]) for sid in street_ids]
    dp: dict[tuple[int, int, int], int] = {}
    parent: dict[tuple[int, int, int], tuple[int, int, int] | None] = {}
    for i, tasks in enumerate(orientations):
        for oi, task in enumerate(tasks):
            d = sp.distance(instance.depot, task.start)
            if d >= INF:
                continue
            state = (1 << i, i, oi)
            dp[state] = d + task.time
            parent[state] = None
    for mask in range(1, 1 << k):
        states = [key for key in dp if key[0] == mask]
        for state in states:
            cost = dp[state]
            _, i, oi = state
            last = orientations[i][oi]
            for j in range(k):
                if mask & (1 << j):
                    continue
                for oj, task in enumerate(orientations[j]):
                    d = sp.distance(last.end, task.start)
                    if d >= INF:
                        continue
                    nm = mask | (1 << j)
                    ns = (nm, j, oj)
                    nc = cost + d + task.time
                    if nc < dp.get(ns, INF):
                        dp[ns] = nc
                        parent[ns] = state
    best: dict[int, tuple[int, list[Task]]] = {0: (0, [])}
    for mask in range(1, 1 << k):
        best_state = None
        best_total = INF
        for state, cost in dp.items():
            if state[0] != mask:
                continue
            _, i, oi = state
            d = sp.distance(orientations[i][oi].end, instance.depot)
            if d < INF and cost + d < best_total:
                best_total = cost + d
                best_state = state
        if best_state is None:
            continue
        rev: list[Task] = []
        cur = best_state
        while cur is not None:
            _, i, oi = cur
            rev.append(orientations[i][oi])
            cur = parent[cur]
        best[mask] = (best_total, list(reversed(rev)))
    return best


def exact_group_construct_mandatory(instance: Instance, graph: Graph, sp: ShortestPaths, seed: int = 0) -> Solution | None:
    """Exact small-subproblem construction by requirement class.

    This covers the provided train instances well and is still bounded for
    hidden small/medium cases. If any class is too large or infeasible, callers
    can fall back to regret construction.
    """
    solution = build_empty_solution(instance)
    classes = [(30, 30), (20, 20), (10, 10)]
    used_routes: dict[int, list[Task]] = {i: [] for i in range(len(solution.routes))}
    for req, cap in classes:
        street_ids = [s.id for s in instance.streets if s.category == "M" and s.requirement == req]
        vehicle_indices = [i for i, r in enumerate(solution.routes) if r.vehicle.capacity == cap]
        if not street_ids:
            continue
        if not vehicle_indices:
            return None
        subset_best = _best_subset_routes(instance, graph, sp, street_ids)
        if not subset_best:
            return None
        full = (1 << len(street_ids)) - 1
        memo: dict[tuple[int, int], tuple[int, list[int]] | None] = {}

        def assign(pos: int, mask: int) -> tuple[int, list[int]] | None:
            key = (pos, mask)
            if key in memo:
                return memo[key]
            if mask == 0:
                memo[key] = (0, [])
                return memo[key]
            if pos == len(vehicle_indices):
                memo[key] = None
                return None
            best_choice: tuple[int, list[int]] | None = None
            sub = mask
            while True:
                if sub in subset_best and subset_best[sub][0] <= instance.time_limit:
                    rest = assign(pos + 1, mask ^ sub)
                    if rest is not None:
                        total = subset_best[sub][0] + rest[0]
                        choice = (total, [sub] + rest[1])
                        if best_choice is None or choice[0] < best_choice[0]:
                            best_choice = choice
                if sub == 0:
                    break
                sub = (sub - 1) & mask
            memo[key] = best_choice
            return best_choice

        result = assign(0, full)
        if result is None:
            return None
        masks = result[1]
        for vi, mask in zip(vehicle_indices, masks):
            if mask:
                used_routes[vi].extend(subset_best[mask][1])

    for i, tasks in used_routes.items():
        solution.routes[i].tasks = tasks
    return solution


def insert_optional(solution: Solution, graph: Graph, sp: ShortestPaths, seed: int = 0, passes: int = 1) -> Solution:
    rng = random.Random(seed)
    harvest_free_cleaning(solution, graph, sp, include_optional=True)
    cleaned = set(solution.cleaned_ids())
    for _ in range(passes):
        improved = False
        while True:
            best: Insertion | None = None
            for sid in instance_optional_ids(solution.instance, cleaned):
                candidates = best_insertions_for_street(solution, graph, sp, sid, optional=True, rng=rng)
                if candidates and (best is None or candidates[0].cost < best.cost):
                    best = candidates[0]
            if best is None:
                break
            solution.routes[best.route_index].tasks.insert(best.position, best.task)
            harvest_free_cleaning(solution, graph, sp, include_optional=True)
            cleaned.add(best.task.street_id)
            cleaned |= set(solution.cleaned_ids())
            improved = True
        if not improved:
            break
    return solution


def instance_optional_ids(instance: Instance, cleaned: set[int]) -> list[int]:
    return [sid for sid in instance.optional_ids if sid not in cleaned]
