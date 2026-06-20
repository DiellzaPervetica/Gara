from __future__ import annotations

from collections import Counter

from .graph_utils import Graph
from .models import Route, Solution
from .scorer import bounds, duplicate_cleaned, edge_waste, score_solution
from .shortest_paths import ShortestPaths


def expand_route(solution: Solution, graph: Graph, sp: ShortestPaths, route_index: int) -> bool:
    route = solution.routes[route_index]
    depot = solution.instance.depot
    nodes = [depot]
    streets: list[int] = []
    cur = depot
    total = 0
    for task in route.tasks:
        path = sp.path(cur, task.start)
        if path is None:
            solution.reason = f"no path from {cur} to {task.start}"
            return False
        nodes.extend(path.nodes[1:])
        streets.extend(path.streets)
        total += path.time
        if not graph.street_allows(task.street_id, task.start, task.end):
            solution.reason = f"task orientation invalid for street {task.street_id}"
            return False
        nodes.append(task.end)
        streets.append(task.street_id)
        total += task.time
        cur = task.end
    path = sp.path(cur, depot)
    if path is None:
        solution.reason = f"no path from {cur} to depot"
        return False
    nodes.extend(path.nodes[1:])
    streets.extend(path.streets)
    total += path.time
    route.nodes = nodes
    route.traversed_streets = streets
    route.time = total
    return True


def validate_solution(solution: Solution, graph: Graph, sp: ShortestPaths) -> Solution:
    instance = solution.instance
    by_id = {s.id: s for s in instance.streets}

    if len(solution.routes) != len(instance.vehicles):
        solution.valid = False
        solution.reason = "route count does not match vehicle count"
        return solution

    for i, route in enumerate(solution.routes):
        if not expand_route(solution, graph, sp, i):
            solution.valid = False
            return solution
        if not route.nodes or route.nodes[0] != instance.depot or route.nodes[-1] != instance.depot:
            solution.valid = False
            solution.reason = f"vehicle {i} does not start/end at depot"
            return solution
        if route.time > instance.time_limit:
            solution.valid = False
            solution.reason = f"vehicle {i} exceeds time limit: {route.time}>{instance.time_limit}"
            return solution
        traversed_counts = Counter(route.traversed_streets)
        for sid in route.cleaned_ids():
            street = by_id.get(sid)
            if street is None:
                solution.valid = False
                solution.reason = f"unknown cleaned street {sid}"
                return solution
            if street.category == "C":
                solution.valid = False
                solution.reason = f"connector cleaned {sid}"
                return solution
            if route.vehicle.capacity < street.requirement:
                solution.valid = False
                solution.reason = f"vehicle {i} cannot clean street {sid}"
                return solution
            if traversed_counts[sid] <= 0:
                solution.valid = False
                solution.reason = f"vehicle {i} claims untraversed street {sid}"
                return solution

    cleaned = set(solution.cleaned_ids())
    missing = instance.mandatory_ids - cleaned
    if missing:
        solution.valid = False
        solution.reason = f"missing mandatory streets: {sorted(missing)[:10]}"
        return solution
    dup = duplicate_cleaned(solution)
    if dup:
        solution.valid = False
        solution.reason = f"duplicate cleaned streets: {dup}"
        return solution

    solution.valid = True
    solution.reason = "ok"
    return score_solution(solution)


def validate_submission_lines(lines: list[str], solution: Solution, graph: Graph) -> tuple[bool, str]:
    instance = solution.instance
    if not lines or lines[0].strip() != str(len(instance.vehicles)):
        return False, "bad first line"
    if len(lines) != 1 + 3 * len(instance.vehicles):
        return False, "bad line count"
    return True, "ok"


def validate_submission_text(instance, graph: Graph, text: str) -> Solution:
    """Validate official node-list submission format directly.

    This path is stricter and closer to the web validator than validating the
    internal task-only representation, because a route may traverse a street as
    an anchor without claiming to clean it.
    """
    lines = text.splitlines()
    solution = Solution(instance, [Route(v) for v in instance.vehicles])
    if not lines or lines[0].strip() != str(len(instance.vehicles)):
        solution.valid = False
        solution.reason = "bad first line"
        return solution
    if len(lines) != 1 + 3 * len(instance.vehicles):
        solution.valid = False
        solution.reason = "bad line count"
        return solution

    seen: set[int] = set()
    cleaned_len_ids: set[int] = set()
    total_waste = 0.0
    for vi, vehicle in enumerate(instance.vehicles):
        base = 1 + 3 * vi
        try:
            n = int(lines[base].strip())
            nodes = [int(x) for x in lines[base + 1].split()]
            cleaned = [int(x) for x in lines[base + 2].split()] if lines[base + 2].strip() else []
        except ValueError:
            solution.valid = False
            solution.reason = f"parse error vehicle {vi}"
            return solution
        if len(nodes) != n + 1:
            solution.valid = False
            solution.reason = f"route node count mismatch for vehicle {vi}: expected {n + 1}, got {len(nodes)}"
            return solution
        if not nodes or nodes[0] != instance.depot or nodes[-1] != instance.depot:
            solution.valid = False
            solution.reason = f"vehicle {vi} does not start/end at depot"
            return solution
        traversed: list[int] = []
        route_time = 0
        for u, v in zip(nodes, nodes[1:]):
            arc = graph.best_arc.get((u, v))
            if arc is None:
                solution.valid = False
                solution.reason = f"invalid traversal {u}->{v} by vehicle {vi}"
                return solution
            traversed.append(arc.street_id)
            route_time += arc.time
        if route_time > instance.time_limit:
            solution.valid = False
            solution.reason = f"vehicle {vi} exceeds time limit: {route_time}>{instance.time_limit}"
            return solution
        traversed_set = set(traversed)
        for sid in cleaned:
            if sid in seen:
                solution.valid = False
                solution.reason = f"duplicate cleaned street {sid}"
                return solution
            if sid < 0 or sid >= len(instance.streets):
                solution.valid = False
                solution.reason = f"unknown cleaned street {sid}"
                return solution
            street = instance.streets[sid]
            if not street.cleanable:
                solution.valid = False
                solution.reason = f"connector cleaned {sid}"
                return solution
            if vehicle.capacity < street.requirement:
                solution.valid = False
                solution.reason = f"vehicle {vi} cannot clean street {sid}"
                return solution
            if sid not in traversed_set:
                solution.valid = False
                solution.reason = f"vehicle {vi} claims untraversed street {sid}"
                return solution
            seen.add(sid)
            cleaned_len_ids.add(sid)
            total_waste += edge_waste(vehicle.capacity, street.requirement, street.length)
        solution.routes[vi].nodes = nodes
        solution.routes[vi].traversed_streets = traversed
        solution.routes[vi].time = route_time

    missing = instance.mandatory_ids - seen
    if missing:
        solution.valid = False
        solution.reason = f"missing mandatory streets: {sorted(missing)[:10]}"
        return solution
    lmax, wmax = bounds(instance)
    solution.coverage = sum(instance.streets[sid].length for sid in cleaned_len_ids) / lmax if lmax else 1.0
    solution.efficiency = 1.0 if wmax == 0 and total_waste == 0 else (0.0 if wmax == 0 else 1 - total_waste / wmax)
    solution.total_waste = total_waste
    solution.score = instance.alpha * solution.coverage + (1 - instance.alpha) * solution.efficiency
    solution.valid = True
    solution.reason = "ok"
    return solution
