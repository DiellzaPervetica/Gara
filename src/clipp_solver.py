from __future__ import annotations

"""A dependency-free solver for the CLIPP Street Cleaning challenge.

The solver builds valid depot-to-depot routes, services every mandatory street,
then uses alpha-aware insertion and route local search for optional streets.
It intentionally keeps the submission writer and validator in the same module:
every output is checked before it is written.
"""

import argparse
import heapq
import json
import math
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


CAPACITY = {"S": 10, "M": 20, "L": 30}
EPS = 1e-9


@dataclass(frozen=True)
class Street:
    id: int
    a: int
    b: int
    direction: int
    travel_time: int
    length: int
    category: str
    requirement: int

    def orientations(self) -> tuple[tuple[int, int], ...]:
        return ((self.a, self.b),) if self.direction == 1 else ((self.a, self.b), (self.b, self.a))


@dataclass(frozen=True)
class Traversal:
    to: int
    street_id: int
    travel_time: int


@dataclass
class Instance:
    name: str
    nodes: int
    time_limit: int
    depot: int
    alpha: float
    streets: list[Street]
    vehicle_types: list[str]
    graph: list[list[Traversal]] = field(init=False)
    arc_street: dict[tuple[int, int], int] = field(init=False)
    cleanable_length: int = field(init=False)
    max_waste: float = field(init=False)

    def __post_init__(self) -> None:
        self.graph = [[] for _ in range(self.nodes)]
        self.arc_street = {}
        for street in self.streets:
            for u, v in street.orientations():
                self.graph[u].append(Traversal(v, street.id, street.travel_time))
                # The official data guarantees at most one street per node pair.
                self.arc_street.setdefault((u, v), street.id)
        cleanable = [s for s in self.streets if s.category != "C"]
        self.cleanable_length = sum(s.length for s in cleanable)
        self.max_waste = sum((30 - s.requirement) * s.length / 1000 for s in cleanable)

    @property
    def capacities(self) -> list[int]:
        return [CAPACITY[t] for t in self.vehicle_types]


@dataclass(frozen=True)
class Task:
    street_id: int
    start: int
    end: int


@dataclass
class Route:
    vehicle_id: int
    capacity: int
    tasks: list[Task] = field(default_factory=list)

    def copy(self) -> "Route":
        return Route(self.vehicle_id, self.capacity, list(self.tasks))


@dataclass
class Solution:
    routes: list[Route]

    def copy(self) -> "Solution":
        return Solution([route.copy() for route in self.routes])


@dataclass
class Score:
    value: float
    coverage: float
    efficiency: float
    cleaned_length: int
    waste: float
    duplicates: int


class ShortestPaths:
    """Dijkstra cache, retaining parents so paths can be expanded exactly."""

    def __init__(self, instance: Instance) -> None:
        self.instance = instance
        self.cache: dict[int, tuple[list[float], list[tuple[int, int] | None]]] = {}

    def _run(self, source: int) -> tuple[list[float], list[tuple[int, int] | None]]:
        n = self.instance.nodes
        dist = [math.inf] * n
        parent: list[tuple[int, int] | None] = [None] * n
        dist[source] = 0
        heap = [(0, source)]
        while heap:
            current, u = heapq.heappop(heap)
            if current != dist[u]:
                continue
            for arc in self.instance.graph[u]:
                candidate = current + arc.travel_time
                if candidate < dist[arc.to]:
                    dist[arc.to] = candidate
                    parent[arc.to] = (u, arc.street_id)
                    heapq.heappush(heap, (candidate, arc.to))
        return dist, parent

    def distance(self, source: int, target: int) -> float:
        if source not in self.cache:
            self.cache[source] = self._run(source)
        return self.cache[source][0][target]

    def path(self, source: int, target: int) -> tuple[list[int], list[int], int] | None:
        if source not in self.cache:
            self.cache[source] = self._run(source)
        dist, parent = self.cache[source]
        if math.isinf(dist[target]):
            return None
        nodes = [target]
        street_ids: list[int] = []
        cursor = target
        while cursor != source:
            previous = parent[cursor]
            if previous is None:
                return None
            cursor, street_id = previous
            nodes.append(cursor)
            street_ids.append(street_id)
        nodes.reverse()
        street_ids.reverse()
        return nodes, street_ids, int(dist[target])


def parse_instance(path: Path) -> Instance:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if not lines:
        raise ValueError(f"{path} is empty")
    header = lines[0].split()
    if len(header) != 6:
        raise ValueError(f"{path}: header must have six values")
    nodes, streets_count, time_limit, vehicles, depot, alpha = header
    n, m, c = int(nodes), int(streets_count), int(vehicles)

    # The published specification mentions coordinates, while its sample and
    # the supplied training files omit them. Support both encodings.
    offset = 1
    if len(lines) >= 1 + n + m + 1 and all(len(line.split()) == 2 for line in lines[1 : 1 + n]):
        offset += n
    if len(lines) < offset + m + 1:
        raise ValueError(f"{path}: incomplete street or fleet data")

    streets: list[Street] = []
    for street_id, line in enumerate(lines[offset : offset + m]):
        values = line.split()
        if len(values) != 7:
            raise ValueError(f"{path}: street {street_id} must have seven values")
        a, b, direction, travel, length, category, requirement = values
        street = Street(street_id, int(a), int(b), int(direction), int(travel), int(length), category, int(requirement))
        if not (0 <= street.a < n and 0 <= street.b < n) or street.a == street.b:
            raise ValueError(f"{path}: invalid endpoints for street {street_id}")
        if street.direction not in (1, 2) or street.category not in {"M", "O", "C"}:
            raise ValueError(f"{path}: invalid definition for street {street_id}")
        streets.append(street)
    vehicle_types = lines[offset + m].split()
    if len(vehicle_types) != c or any(v not in CAPACITY for v in vehicle_types):
        raise ValueError(f"{path}: invalid vehicle list")
    return Instance(path.stem, n, int(time_limit), int(depot), float(alpha), streets, vehicle_types)


def route_time(route: Route, instance: Instance, paths: ShortestPaths) -> float:
    cursor = instance.depot
    total = 0.0
    for task in route.tasks:
        connection = paths.distance(cursor, task.start)
        if math.isinf(connection):
            return math.inf
        street = instance.streets[task.street_id]
        total += connection + street.travel_time
        cursor = task.end
    finish = paths.distance(cursor, instance.depot)
    return total + finish


def expand_route(route: Route, instance: Instance, paths: ShortestPaths) -> tuple[list[int], list[int], int] | None:
    """Returns junctions, intentionally cleaned street ids, and total time."""
    nodes = [instance.depot]
    cleaned: list[int] = []
    total = 0
    cursor = instance.depot
    for task in route.tasks:
        connector = paths.path(cursor, task.start)
        if connector is None:
            return None
        connector_nodes, _connector_streets, connector_time = connector
        nodes.extend(connector_nodes[1:])
        total += connector_time
        street = instance.streets[task.street_id]
        if (task.start, task.end) not in instance.arc_street:
            return None
        nodes.append(task.end)
        total += street.travel_time
        cleaned.append(task.street_id)
        cursor = task.end
    connector = paths.path(cursor, instance.depot)
    if connector is None:
        return None
    connector_nodes, _connector_streets, connector_time = connector
    nodes.extend(connector_nodes[1:])
    total += connector_time
    return nodes, cleaned, total


def score_solution(solution: Solution, instance: Instance) -> Score:
    unique: set[int] = set()
    waste = 0.0
    action_count = 0
    for route in solution.routes:
        for task in route.tasks:
            action_count += 1
            street = instance.streets[task.street_id]
            unique.add(street.id)
            waste += (route.capacity - street.requirement) * street.length / 1000
    cleaned_length = sum(instance.streets[street_id].length for street_id in unique)
    coverage = cleaned_length / instance.cleanable_length if instance.cleanable_length else 1.0
    efficiency = 1.0 if instance.max_waste <= EPS else 1.0 - waste / instance.max_waste
    value = instance.alpha * coverage + (1.0 - instance.alpha) * efficiency
    return Score(value, coverage, efficiency, cleaned_length, waste, action_count - len(unique))


def validate_solution(solution: Solution, instance: Instance, paths: ShortestPaths | None = None) -> tuple[bool, list[str], Score]:
    errors: list[str] = []
    if len(solution.routes) != len(instance.vehicle_types):
        errors.append("wrong number of routes")
    cleaned: list[int] = []
    for index, route in enumerate(solution.routes):
        if index >= len(instance.vehicle_types):
            break
        if route.vehicle_id != index or route.capacity != CAPACITY[instance.vehicle_types[index]]:
            errors.append(f"route {index}: vehicle metadata does not match the fleet")
        seen_on_route: set[int] = set()
        for task in route.tasks:
            if not 0 <= task.street_id < len(instance.streets):
                errors.append(f"route {index}: unknown street {task.street_id}")
                continue
            street = instance.streets[task.street_id]
            if street.category == "C":
                errors.append(f"route {index}: connector {street.id} is marked as cleaned")
            if route.capacity < street.requirement:
                errors.append(f"route {index}: insufficient capacity for street {street.id}")
            if (task.start, task.end) not in street.orientations():
                errors.append(f"route {index}: invalid direction for street {street.id}")
            if street.id in seen_on_route:
                errors.append(f"route {index}: duplicate cleaning of street {street.id}")
            seen_on_route.add(street.id)
            cleaned.append(street.id)
        if paths is not None:
            expanded = expand_route(route, instance, paths)
            if expanded is None:
                errors.append(f"route {index}: cannot return to depot")
            elif expanded[2] > instance.time_limit:
                errors.append(f"route {index}: {expanded[2]}s exceeds {instance.time_limit}s")
    missing = [s.id for s in instance.streets if s.category == "M" and s.id not in cleaned]
    if missing:
        errors.append(f"missing mandatory streets: {missing}")
    return not errors, errors, score_solution(solution, instance)


def _task_options(street: Street) -> Iterable[Task]:
    for start, end in street.orientations():
        yield Task(street.id, start, end)


def _insertion_candidates(
    route: Route,
    street: Street,
    instance: Instance,
    paths: ShortestPaths,
    old_time: float | None = None,
) -> list[tuple[float, int, Task]]:
    """All feasible insertions using an O(1) exact delta per position."""
    if old_time is None:
        old_time = route_time(route, instance, paths)
    candidates: list[tuple[float, int, Task]] = []
    for position in range(len(route.tasks) + 1):
        previous = instance.depot if position == 0 else route.tasks[position - 1].end
        following = instance.depot if position == len(route.tasks) else route.tasks[position].start
        replaced_connection = paths.distance(previous, following)
        if math.isinf(replaced_connection):
            continue
        for task in _task_options(street):
            before = paths.distance(previous, task.start)
            after = paths.distance(task.end, following)
            if math.isinf(before) or math.isinf(after):
                continue
            new_time = old_time + before + street.travel_time + after - replaced_connection
            if new_time <= instance.time_limit:
                candidates.append((new_time - old_time, position, task))
    return candidates


def _insert(route: Route, position: int, task: Task) -> None:
    route.tasks.insert(position, task)


def _mandatory_insertion_cost(extra_time: float, route: Route, street: Street, instance: Instance) -> float:
    # At low alpha an exact capacity match is much more important than shaving a
    # few seconds of travel. At alpha=1 time is the only tie-breaker.
    waste = (route.capacity - street.requirement) * street.length / 1000
    waste_weight = 0.0 if instance.alpha >= 1.0 - EPS else (1.0 - instance.alpha) * 10000.0
    return extra_time + waste_weight * waste


def construct_mandatory(instance: Instance, paths: ShortestPaths, rng: random.Random) -> Solution | None:
    routes = [Route(i, CAPACITY[vehicle_type]) for i, vehicle_type in enumerate(instance.vehicle_types)]
    unassigned = [street for street in instance.streets if street.category == "M"]

    # Regret insertion: at each iteration choose the street whose second best
    # placement is most painful, protecting scarce large vehicles first.
    while unassigned:
        selected: tuple[Street, int, int, Task] | None = None
        selected_key: tuple[float, float, float, float] | None = None
        for street in unassigned:
            placements: list[tuple[float, int, int, Task]] = []
            for route_index, route in enumerate(routes):
                if route.capacity < street.requirement:
                    continue
                for extra, position, task in _insertion_candidates(route, street, instance, paths):
                    placements.append((_mandatory_insertion_cost(extra, route, street, instance), route_index, position, task))
            if not placements:
                return None
            placements.sort(key=lambda item: item[0])
            best = placements[0]
            second = placements[1][0] if len(placements) > 1 else best[0] + 1_000_000_000
            compatible = sum(capacity >= street.requirement for capacity in instance.capacities)
            # Random noise breaks ties across multi-starts without overriding
            # material regret or capability scarcity.
            key = (second - best[0], -compatible, street.requirement, rng.random())
            if selected_key is None or key > selected_key:
                selected_key = key
                selected = (street, best[1], best[2], best[3])
        assert selected is not None
        street, route_index, position, task = selected
        _insert(routes[route_index], position, task)
        unassigned.remove(street)
    return Solution(routes)


def construct_mandatory_large(instance: Instance, paths: ShortestPaths, rng: random.Random) -> Solution | None:
    """Linear-in-task construction for the large public instances.

    Regret insertion is excellent on the small instances but quadratic in the
    number of streets. This version keeps exact insertion deltas, processes
    scarce heavy tasks first, and introduces only a tiny randomized tie-break.
    """
    routes = [Route(i, CAPACITY[vehicle_type]) for i, vehicle_type in enumerate(instance.vehicle_types)]
    route_times = [0.0] * len(routes)
    mandatory = [street for street in instance.streets if street.category == "M"]
    rng.shuffle(mandatory)
    mandatory.sort(
        key=lambda street: (
            -street.requirement,
            -street.travel_time,
            -street.length,
        )
    )
    for street in mandatory:
        placements: list[tuple[float, float, int, int, Task]] = []
        for route_index, route in enumerate(routes):
            if route.capacity < street.requirement:
                continue
            for extra, position, task in _insertion_candidates(route, street, instance, paths, route_times[route_index]):
                cost = _mandatory_insertion_cost(extra, route, street, instance)
                # Avoid consuming the final sliver of a route for a task that
                # could be placed almost as well on a less-loaded vehicle.
                projected_load = (route_times[route_index] + extra) / instance.time_limit
                cost += max(0.0, projected_load - 0.82) ** 3 * 30_000
                placements.append((cost, extra, route_index, position, task))
        if not placements:
            return None
        placements.sort(key=lambda placement: placement[0])
        shortlist = placements[: min(3, len(placements))]
        cost, extra, route_index, position, task = shortlist[0 if rng.random() < 0.90 else rng.randrange(len(shortlist))]
        _insert(routes[route_index], position, task)
        route_times[route_index] += extra
    return Solution(routes)


def construct_mandatory_randomized(instance: Instance, paths: ShortestPaths, rng: random.Random) -> Solution | None:
    """Diversified fallback when regret insertion paints itself into a corner.

    The random choice is deliberately restricted to good insertion candidates:
    it changes clustering, rather than simply accepting arbitrarily bad routes.
    """
    routes = [Route(i, CAPACITY[vehicle_type]) for i, vehicle_type in enumerate(instance.vehicle_types)]
    unassigned = [street for street in instance.streets if street.category == "M"]
    rng.shuffle(unassigned)
    unassigned.sort(key=lambda street: (-street.requirement, rng.random()))
    for street in unassigned:
        placements: list[tuple[float, int, int, Task]] = []
        for route_index, route in enumerate(routes):
            if route.capacity < street.requirement:
                continue
            for extra, position, task in _insertion_candidates(route, street, instance, paths):
                placements.append((_mandatory_insertion_cost(extra, route, street, instance), route_index, position, task))
        if not placements:
            return None
        placements.sort(key=lambda placement: placement[0])
        # Most of the time keep within the three best placements. Occasionally
        # widen the candidate list to escape a bad geographic clustering.
        width = 3 if rng.random() < 0.70 else 10
        _cost, route_index, position, task = placements[rng.randrange(min(width, len(placements)))]
        _insert(routes[route_index], position, task)
    return Solution(routes)


def _marginal_gain(street: Street, capacity: int, instance: Instance) -> float:
    coverage = street.length / instance.cleanable_length if instance.cleanable_length else 0.0
    waste = (capacity - street.requirement) * street.length / 1000
    penalty = waste / instance.max_waste if instance.max_waste > EPS else 0.0
    return instance.alpha * coverage - (1.0 - instance.alpha) * penalty


def insert_optionals(solution: Solution, instance: Instance, paths: ShortestPaths, rng: random.Random) -> None:
    """Greedy score-per-added-second insertion of optional streets."""
    already_cleaned = {task.street_id for route in solution.routes for task in route.tasks}
    remaining = {street.id for street in instance.streets if street.category == "O" and street.id not in already_cleaned}
    while remaining:
        best: tuple[float, Street, int, int, Task] | None = None
        for street_id in list(remaining):
            street = instance.streets[street_id]
            for route_index, route in enumerate(solution.routes):
                if route.capacity < street.requirement:
                    continue
                gain = _marginal_gain(street, route.capacity, instance)
                # With alpha=0, zero-waste optionals do not change score. Do
                # not spend route time on them; they cannot improve the result.
                if gain <= EPS:
                    continue
                candidates = _insertion_candidates(route, street, instance, paths)
                if not candidates:
                    continue
                extra, position, task = min(candidates, key=lambda item: item[0])
                density = gain / max(1.0, extra)
                key = density + rng.random() * 1e-12
                if best is None or key > best[0]:
                    best = (key, street, route_index, position, task)
        if best is None:
            break
        _density, street, route_index, position, task = best
        _insert(solution.routes[route_index], position, task)
        remaining.remove(street.id)


def insert_optionals_large(solution: Solution, instance: Instance, paths: ShortestPaths, rng: random.Random) -> None:
    """Scalable alpha-aware optional insertion with two adaptive passes."""
    cleaned = {task.street_id for route in solution.routes for task in route.tasks}
    candidates = [street for street in instance.streets if street.category == "O" and street.id not in cleaned]

    def static_priority(street: Street) -> float:
        best_gain = max(
            (_marginal_gain(street, capacity, instance) for capacity in instance.capacities if capacity >= street.requirement),
            default=-math.inf,
        )
        return best_gain / max(1, street.travel_time)

    candidates.sort(key=lambda street: (static_priority(street), rng.random()), reverse=True)
    route_times = [route_time(route, instance, paths) for route in solution.routes]
    deferred: list[Street] = []
    for _pass in range(2):
        work = candidates if _pass == 0 else deferred
        deferred = []
        for street in work:
            best: tuple[float, float, int, int, Task] | None = None
            for route_index, route in enumerate(solution.routes):
                if route.capacity < street.requirement:
                    continue
                gain = _marginal_gain(street, route.capacity, instance)
                if gain <= EPS:
                    continue
                options = _insertion_candidates(route, street, instance, paths, route_times[route_index])
                if not options:
                    continue
                extra, position, task = min(options, key=lambda option: option[0])
                density = gain / max(1.0, extra)
                candidate = (density, extra, route_index, position, task)
                if best is None or candidate[0] > best[0] + EPS:
                    best = candidate
            if best is None:
                deferred.append(street)
                continue
            _density, extra, route_index, position, task = best
            _insert(solution.routes[route_index], position, task)
            route_times[route_index] += extra


def improve_optional_exchanges_large(solution: Solution, instance: Instance, paths: ShortestPaths, rounds: int = 12) -> None:
    """Replace low-value optional tasks with better unserved opportunities.

    Static insertion fills the last time slack quickly. These bounded exchange
    searches reopen that slack, which is where large instances otherwise tend
    to get trapped. Every accepted exchange improves the official objective.
    """
    for _ in range(rounds):
        cleaned = {task.street_id for route in solution.routes for task in route.tasks}
        uncleaned = [street for street in instance.streets if street.category == "O" and street.id not in cleaned]
        current: list[tuple[float, int, int, Task]] = []
        for route_index, route in enumerate(solution.routes):
            for task_index, task in enumerate(route.tasks):
                street = instance.streets[task.street_id]
                if street.category == "O":
                    current.append((_marginal_gain(street, route.capacity, instance), route_index, task_index, task))

        def optimistic_gain(street: Street) -> float:
            return max(
                (_marginal_gain(street, capacity, instance) for capacity in instance.capacities if capacity >= street.requirement),
                default=-math.inf,
            )

        current.sort(key=lambda item: item[0])
        uncleaned.sort(key=optimistic_gain, reverse=True)
        route_times = [route_time(route, instance, paths) for route in solution.routes]
        best: tuple[float, int, int, int, int, Task, list[Task], list[Task] | None] | None = None

        for removed_gain, source_index, task_index, removed_task in current[:24]:
            source = solution.routes[source_index]
            source_without = source.tasks[:task_index] + source.tasks[task_index + 1 :]
            source_without_time = route_time(Route(source.vehicle_id, source.capacity, source_without), instance, paths)
            for added_street in uncleaned[:72]:
                for target_index, target in enumerate(solution.routes):
                    if target.capacity < added_street.requirement:
                        continue
                    added_gain = _marginal_gain(added_street, target.capacity, instance)
                    score_delta = added_gain - removed_gain
                    if score_delta <= EPS:
                        continue
                    base_tasks = source_without if target_index == source_index else target.tasks
                    base_time = source_without_time if target_index == source_index else route_times[target_index]
                    options = _insertion_candidates(
                        Route(target.vehicle_id, target.capacity, base_tasks),
                        added_street,
                        instance,
                        paths,
                        base_time,
                    )
                    if not options:
                        continue
                    extra, position, added_task = min(options, key=lambda option: option[0])
                    new_target = base_tasks[:position] + [added_task] + base_tasks[position:]
                    candidate = (
                        score_delta,
                        source_index,
                        target_index,
                        task_index,
                        position,
                        added_task,
                        source_without,
                        new_target if target_index != source_index else None,
                    )
                    if best is None or candidate[0] > best[0] + EPS:
                        best = candidate
        if best is None:
            return
        _delta, source_index, target_index, _task_index, _position, _added, source_without, target_after = best
        if target_index == source_index:
            # Here source_without already excludes the old optional and the
            # insertion is represented by target_after being None.
            added_street = instance.streets[_added.street_id]
            options = _insertion_candidates(
                Route(solution.routes[source_index].vehicle_id, solution.routes[source_index].capacity, source_without),
                added_street,
                instance,
                paths,
                route_time(Route(solution.routes[source_index].vehicle_id, solution.routes[source_index].capacity, source_without), instance, paths),
            )
            _extra, position, task = min(options, key=lambda option: option[0])
            solution.routes[source_index].tasks = source_without[:position] + [task] + source_without[position:]
        else:
            solution.routes[source_index].tasks = source_without
            assert target_after is not None
            solution.routes[target_index].tasks = target_after


def construct_mandatory_vrp(instance: Instance, seconds: int) -> Solution | None:
    """OR-Tools mandatory-only constructor for large, tightly packed fleets."""
    if seconds <= 0:
        return None
    try:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    except ImportError:
        return None

    paths = ShortestPaths(instance)
    tasks: list[Task] = []
    groups: list[tuple[Street, list[int]]] = []
    for street in instance.streets:
        if street.category != "M":
            continue
        group: list[int] = []
        for start, end in street.orientations():
            group.append(len(tasks) + 1)
            tasks.append(Task(street.id, start, end))
        groups.append((street, group))
    manager = pywrapcp.RoutingIndexManager(len(tasks) + 1, len(instance.vehicle_types), 0)
    routing = pywrapcp.RoutingModel(manager)
    unreachable = 10**12

    def travel(from_index: int, to_index: int) -> int:
        source, destination = manager.IndexToNode(from_index), manager.IndexToNode(to_index)
        start = instance.depot if source == 0 else tasks[source - 1].end
        if destination == 0:
            distance = paths.distance(start, instance.depot)
            return unreachable if math.isinf(distance) else int(distance)
        task = tasks[destination - 1]
        distance = paths.distance(start, task.start)
        return unreachable if math.isinf(distance) else int(distance + instance.streets[task.street_id].travel_time)

    time_callback = routing.RegisterTransitCallback(travel)
    routing.AddDimension(time_callback, 0, instance.time_limit, True, "Time")
    score_scale = 1_000_000_000
    for vehicle_id, capacity in enumerate(instance.capacities):
        def cost(from_index: int, to_index: int, capacity: int = capacity) -> int:
            value = travel(from_index, to_index)
            node = manager.IndexToNode(to_index)
            if node:
                street = instance.streets[tasks[node - 1].street_id]
                waste = (capacity - street.requirement) * street.length / 1000
                if instance.alpha < 1.0 - EPS and instance.max_waste > EPS:
                    value += int(round((1.0 - instance.alpha) * waste / instance.max_waste * score_scale))
            return value

        routing.SetArcCostEvaluatorOfVehicle(routing.RegisterTransitCallback(cost), vehicle_id)
    for street, group in groups:
        indices = [manager.NodeToIndex(node) for node in group]
        # A very large drop penalty lets the first-solution heuristic find a
        # route while still making any omitted mandatory task unacceptable.
        routing.AddDisjunction(indices, 10**15, 1)
        eligible = [vehicle_id for vehicle_id, capacity in enumerate(instance.capacities) if capacity >= street.requirement]
        if len(eligible) < len(instance.capacities):
            forbidden = [v for v in range(len(instance.capacities)) if v not in set(eligible)]
            cp_solver = routing.solver()
            for index in indices:
                for v in forbidden:
                    cp_solver.Add(routing.VehicleVar(index) != v)
    parameters = pywrapcp.DefaultRoutingSearchParameters()
    parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    parameters.time_limit.seconds = seconds
    parameters.log_search = False
    result = routing.SolveWithParameters(parameters)
    if result is None:
        return None
    routes: list[Route] = []
    for vehicle_id, capacity in enumerate(instance.capacities):
        route = Route(vehicle_id, capacity)
        index = routing.Start(vehicle_id)
        while not routing.IsEnd(index):
            index = result.Value(routing.NextVar(index))
            node = manager.IndexToNode(index)
            if node:
                route.tasks.append(tasks[node - 1])
        routes.append(route)
    candidate = Solution(routes)
    valid, _errors, _score = validate_solution(candidate, instance, paths)
    return candidate if valid else None


def improve_route_orientations(solution: Solution, instance: Instance, paths: ShortestPaths) -> bool:
    changed = False
    for route in solution.routes:
        current_time = route_time(route, instance, paths)
        for index, task in enumerate(route.tasks):
            street = instance.streets[task.street_id]
            if street.direction != 2:
                continue
            flipped = Task(task.street_id, task.end, task.start)
            trial = Route(route.vehicle_id, route.capacity, route.tasks[:index] + [flipped] + route.tasks[index + 1 :])
            candidate_time = route_time(trial, instance, paths)
            if candidate_time + EPS < current_time and candidate_time <= instance.time_limit:
                route.tasks[index] = flipped
                current_time = candidate_time
                changed = True
    return changed


def optimize_route_order_exact(route: Route, instance: Instance, paths: ShortestPaths, max_tasks: int = 14) -> bool:
    """Exactly reorder/orient a short route while keeping the same cleaned set.

    The construction heuristics are insertion based, so they can preserve a
    locally awkward ordering. For the public small/medium instances a route
    usually contains only a handful of cleaned streets; Held-Karp over those
    tasks is cheap and can release enough time for another optional street.
    Large routes are skipped deliberately.
    """
    task_count = len(route.tasks)
    if task_count <= 1 or task_count > max_tasks:
        return False

    current_time = route_time(route, instance, paths)
    options: list[tuple[Task, ...]] = []
    for task in route.tasks:
        street = instance.streets[task.street_id]
        if route.capacity < street.requirement:
            return False
        options.append(tuple(_task_options(street)))

    # State is (visited_mask, last_task_index, last_orientation_index).
    dp: dict[tuple[int, int, int], float] = {}
    parent: dict[tuple[int, int, int], tuple[int, int, int] | None] = {}
    for task_index, task_options in enumerate(options):
        bit = 1 << task_index
        for orientation_index, task in enumerate(task_options):
            start_cost = paths.distance(instance.depot, task.start)
            if math.isinf(start_cost):
                continue
            cost = start_cost + instance.streets[task.street_id].travel_time
            state = (bit, task_index, orientation_index)
            if cost < dp.get(state, math.inf):
                dp[state] = cost
                parent[state] = None

    full_mask = (1 << task_count) - 1
    for _depth in range(1, task_count):
        next_dp: dict[tuple[int, int, int], float] = {}
        next_parent: dict[tuple[int, int, int], tuple[int, int, int] | None] = {}
        for state, cost in dp.items():
            mask, last_index, last_orientation = state
            last_task = options[last_index][last_orientation]
            for task_index, task_options in enumerate(options):
                if mask & (1 << task_index):
                    continue
                new_mask = mask | (1 << task_index)
                street = instance.streets[route.tasks[task_index].street_id]
                for orientation_index, task in enumerate(task_options):
                    connection = paths.distance(last_task.end, task.start)
                    if math.isinf(connection):
                        continue
                    candidate = cost + connection + street.travel_time
                    new_state = (new_mask, task_index, orientation_index)
                    if candidate < next_dp.get(new_state, math.inf):
                        next_dp[new_state] = candidate
                        next_parent[new_state] = state
        if not next_dp:
            return False
        parent.update(next_parent)
        dp = next_dp

    best_state: tuple[int, int, int] | None = None
    best_total = math.inf
    for state, cost in dp.items():
        mask, last_index, last_orientation = state
        if mask != full_mask:
            continue
        last_task = options[last_index][last_orientation]
        finish = paths.distance(last_task.end, instance.depot)
        if math.isinf(finish):
            continue
        total = cost + finish
        if total < best_total:
            best_total = total
            best_state = state
    if best_state is None or best_total + EPS >= current_time:
        return False

    reordered: list[Task] = []
    state = best_state
    while state is not None:
        _mask, task_index, orientation_index = state
        reordered.append(options[task_index][orientation_index])
        state = parent[state]
    reordered.reverse()
    route.tasks = reordered
    return True


def improve_assignments(solution: Solution, instance: Instance, paths: ShortestPaths) -> bool:
    """Move a task to a better-capacity route when this improves exact score.

    Optional tasks are never moved to a worse vehicle. Mandatory moves are
    accepted when they reduce waste, then use route duration as a tie-breaker.
    """
    changed = False
    baseline = score_solution(solution, instance)
    for source_index, source in enumerate(solution.routes):
        for task_index in range(len(source.tasks) - 1, -1, -1):
            task = source.tasks[task_index]
            street = instance.streets[task.street_id]
            original_source_tasks = list(source.tasks)
            source.tasks.pop(task_index)
            source_time = route_time(source, instance, paths)
            best_move: tuple[float, float, int, int, Task] | None = None
            for target_index, target in enumerate(solution.routes):
                if target_index == source_index or target.capacity < street.requirement:
                    continue
                # Never increase a task's cleaning waste in a local move.
                if target.capacity > source.capacity:
                    continue
                for extra, position, candidate_task in _insertion_candidates(target, street, instance, paths):
                    trial_waste_delta = (target.capacity - source.capacity) * street.length / 1000
                    trial_time = source_time + route_time(target, instance, paths) + extra
                    key = (trial_waste_delta, trial_time)
                    if best_move is None or key < (best_move[0], best_move[1]):
                        best_move = (trial_waste_delta, trial_time, target_index, position, candidate_task)
            if best_move is None:
                source.tasks = original_source_tasks
                continue
            delta_waste, _time, target_index, position, candidate_task = best_move
            if delta_waste < -EPS:
                _insert(solution.routes[target_index], position, candidate_task)
                candidate_score = score_solution(solution, instance)
                if candidate_score.value + EPS >= baseline.value:
                    baseline = candidate_score
                    changed = True
                else:
                    solution.routes[target_index].tasks.pop(position)
                    source.tasks = original_source_tasks
            else:
                source.tasks = original_source_tasks
    return changed


def improve_relocation(solution: Solution, instance: Instance, paths: ShortestPaths) -> bool:
    """Apply the best capacity-feasible one-task relocation.

    This is intentionally broader than the waste-only reassignment operator:
    in coverage mode a Large vehicle may be the right place for a light street
    if it materially shortens the routing skeleton and makes room elsewhere.
    """
    baseline_score = score_solution(solution, instance)
    old_times = [route_time(route, instance, paths) for route in solution.routes]
    baseline_total_time = sum(old_times)
    best: tuple[float, float, int, int, list[Task], list[Task]] | None = None

    for source_index, source in enumerate(solution.routes):
        for task_index, task in enumerate(source.tasks):
            street = instance.streets[task.street_id]
            source_without = source.tasks[:task_index] + source.tasks[task_index + 1 :]
            source_without_time = route_time(Route(source.vehicle_id, source.capacity, source_without), instance, paths)
            for target_index, target in enumerate(solution.routes):
                if target.capacity < street.requirement:
                    continue
                base_target = source_without if target_index == source_index else target.tasks
                for position in range(len(base_target) + 1):
                    new_target = base_target[:position] + [task] + base_target[position:]
                    new_target_time = route_time(Route(target.vehicle_id, target.capacity, new_target), instance, paths)
                    if new_target_time > instance.time_limit:
                        continue
                    if target_index == source_index:
                        total_time = baseline_total_time - old_times[source_index] + new_target_time
                        new_source = new_target
                    else:
                        total_time = baseline_total_time - old_times[source_index] - old_times[target_index]
                        total_time += source_without_time + new_target_time
                        new_source = source_without
                    waste_delta = (target.capacity - source.capacity) * street.length / 1000
                    score_value = baseline_score.value
                    if instance.alpha < 1.0 - EPS and instance.max_waste > EPS:
                        score_value -= (1.0 - instance.alpha) * waste_delta / instance.max_waste
                    improves = score_value > baseline_score.value + EPS
                    ties_and_shorter = abs(score_value - baseline_score.value) <= EPS and total_time + EPS < baseline_total_time
                    if not (improves or ties_and_shorter):
                        continue
                    candidate = (score_value, total_time, source_index, target_index, new_source, new_target)
                    if best is None or candidate[0] > best[0] + EPS or (
                        abs(candidate[0] - best[0]) <= EPS and candidate[1] + EPS < best[1]
                    ):
                        best = candidate
    if best is None:
        return False
    _score, _time, source_index, target_index, new_source, new_target = best
    solution.routes[source_index].tasks = new_source
    if target_index != source_index:
        solution.routes[target_index].tasks = new_target
    return True


def local_search(solution: Solution, instance: Instance, paths: ShortestPaths, rng: random.Random) -> None:
    # A compact ALNS-style cycle: cheap deterministic route improvements followed
    # by another optional insertion pass. Repeating this is surprisingly useful
    # on the small/medium public instances and avoids heavy dependencies.
    for _ in range(12):
        moved = improve_assignments(solution, instance, paths)
        relocated = improve_relocation(solution, instance, paths)
        ordered = any(optimize_route_order_exact(route, instance, paths) for route in solution.routes)
        oriented = improve_route_orientations(solution, instance, paths)
        before = len({task.street_id for route in solution.routes for task in route.tasks})
        insert_optionals(solution, instance, paths, rng)
        after = len({task.street_id for route in solution.routes for task in route.tasks})
        if not moved and not relocated and not ordered and not oriented and before == after:
            break


def ortools_polish(
    instance: Instance,
    seed_solution: Solution,
    seconds: int,
    start_from_seed: bool = True,
    metaheuristic: int | None = None,
) -> Solution | None:
    """Use OR-Tools Routing as a high-intensity exact-score neighbourhood.

    A cleanable street is represented by one node per allowed orientation; a
    disjunction permits exactly one orientation. Mandatory disjunctions are
    constrained active, while optional disjunction penalties encode the actual
    coverage term of the official objective. Starting from a known valid route
    makes the CP search improve rather than spend its budget finding feasibility.
    """
    if seconds <= 0:
        return None
    try:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    except ImportError:
        return None

    paths = ShortestPaths(instance)
    tasks: list[Task] = []
    groups: list[tuple[Street, list[int]]] = []
    for street in instance.streets:
        if street.category == "C":
            continue
        group: list[int] = []
        for start, end in street.orientations():
            group.append(len(tasks) + 1)  # Node zero is the shared depot.
            tasks.append(Task(street.id, start, end))
        groups.append((street, group))

    manager = pywrapcp.RoutingIndexManager(len(tasks) + 1, len(instance.vehicle_types), 0)
    routing = pywrapcp.RoutingModel(manager)
    unreachable = 10**12

    def travel_callback(from_index: int, to_index: int) -> int:
        from_node, to_node = manager.IndexToNode(from_index), manager.IndexToNode(to_index)
        start = instance.depot if from_node == 0 else tasks[from_node - 1].end
        if to_node == 0:
            distance = paths.distance(start, instance.depot)
            return unreachable if math.isinf(distance) else int(distance)
        task = tasks[to_node - 1]
        distance = paths.distance(start, task.start)
        if math.isinf(distance):
            return unreachable
        return int(distance + instance.streets[task.street_id].travel_time)

    time_callback_index = routing.RegisterTransitCallback(travel_callback)
    routing.AddDimension(time_callback_index, 0, instance.time_limit, True, "Time")

    # Objective units are intentionally lexicographic: a score improvement is
    # worth much more than a few seconds, while travel time breaks score ties.
    score_scale = 1_000_000_000
    for vehicle_id, capacity in enumerate(instance.capacities):
        def objective_callback(from_index: int, to_index: int, capacity: int = capacity) -> int:
            value = travel_callback(from_index, to_index)
            to_node = manager.IndexToNode(to_index)
            if to_node == 0:
                return value
            street = instance.streets[tasks[to_node - 1].street_id]
            waste = (capacity - street.requirement) * street.length / 1000
            if instance.alpha <= EPS:
                # The numerator is integral, preserving exact waste ordering.
                value += (capacity - street.requirement) * street.length * 1_000
            elif instance.alpha < 1.0 - EPS and instance.max_waste > EPS:
                value += int(round((1.0 - instance.alpha) * waste / instance.max_waste * score_scale))
            return value

        cost_callback_index = routing.RegisterTransitCallback(objective_callback)
        routing.SetArcCostEvaluatorOfVehicle(cost_callback_index, vehicle_id)

    for street, group in groups:
        indices = [manager.NodeToIndex(node) for node in group]
        if street.category == "M":
            # A disjunction prevents both orientations being selected. The
            # equality turns it into an actual hard constraint.
            routing.AddDisjunction(indices, 0, 1)
            routing.solver().Add(sum(routing.ActiveVar(index) for index in indices) == 1)
        else:
            if instance.alpha <= EPS:
                reward = 0
            elif instance.alpha >= 1.0 - EPS:
                reward = street.length * 1_000_000
            else:
                reward = int(round(instance.alpha * street.length / instance.cleanable_length * score_scale))
            routing.AddDisjunction(indices, reward, 1)
        eligible = [vehicle_id for vehicle_id, capacity in enumerate(instance.capacities) if capacity >= street.requirement]
        if len(eligible) < len(instance.capacities):
            forbidden = [v for v in range(len(instance.capacities)) if v not in set(eligible)]
            cp_solver = routing.solver()
            for index in indices:
                for v in forbidden:
                    cp_solver.Add(routing.VehicleVar(index) != v)

    node_for_task = {task: node for node, task in enumerate(tasks, start=1)}
    seed_routes = [[node_for_task[task] for task in route.tasks] for route in seed_solution.routes]
    parameters = pywrapcp.DefaultRoutingSearchParameters()
    parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    parameters.local_search_metaheuristic = (
        metaheuristic
        if metaheuristic is not None
        else routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    parameters.time_limit.seconds = seconds
    parameters.log_search = False
    if start_from_seed:
        assignment = routing.ReadAssignmentFromRoutes(seed_routes, True)
        if assignment is None:
            return None
        result = routing.SolveFromAssignmentWithParameters(assignment, parameters)
    else:
        result = routing.SolveWithParameters(parameters)
    if result is None:
        return None

    routes: list[Route] = []
    for vehicle_id, capacity in enumerate(instance.capacities):
        route = Route(vehicle_id, capacity)
        index = routing.Start(vehicle_id)
        while not routing.IsEnd(index):
            index = result.Value(routing.NextVar(index))
            node = manager.IndexToNode(index)
            if node:
                route.tasks.append(tasks[node - 1])
        routes.append(route)
    candidate = Solution(routes)
    valid, _errors, _score = validate_solution(candidate, instance, paths)
    return candidate if valid else None


def solve(
    instance: Instance,
    time_limit: float,
    seeds: int,
    polish_seconds: int = 0,
    mandatory_vrp_seconds: int = 0,
) -> tuple[Solution, Score]:
    started = time.monotonic()
    paths = ShortestPaths(instance)
    best_solution: Solution | None = None
    best_score: Score | None = None
    if sum(street.category == "M" for street in instance.streets) > 100 and mandatory_vrp_seconds > 0:
        mandatory_seed = construct_mandatory_vrp(instance, mandatory_vrp_seconds)
        if mandatory_seed is not None:
            for seed in range(max(1, seeds)):
                candidate = mandatory_seed.copy()
                rng = random.Random(70_001 + seed * 104_729)
                insert_optionals_large(candidate, instance, paths, rng)
                improve_optional_exchanges_large(candidate, instance, paths)
                valid, _errors, candidate_score = validate_solution(candidate, instance, paths)
                if valid and (best_score is None or candidate_score.value > best_score.value + EPS):
                    best_solution, best_score = candidate, candidate_score
            if best_solution is not None and best_score is not None:
                if polish_seconds > 0:
                    guided_seconds = max(1, polish_seconds // 2)
                    tabu_seconds = polish_seconds - guided_seconds
                    polished = ortools_polish(instance, best_solution, guided_seconds)
                    if polished is not None:
                        polished_score = score_solution(polished, instance)
                        if polished_score.value > best_score.value + EPS:
                            best_solution, best_score = polished, polished_score
                    if tabu_seconds > 0:
                        try:
                            from ortools.constraint_solver import routing_enums_pb2

                            polished = ortools_polish(
                                instance,
                                best_solution,
                                tabu_seconds,
                                metaheuristic=routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH,
                            )
                        except ImportError:
                            polished = None
                        if polished is not None:
                            polished_score = score_solution(polished, instance)
                            if polished_score.value > best_score.value + EPS:
                                best_solution, best_score = polished, polished_score
                return best_solution, best_score
    attempts = max(1, seeds)
    for seed in range(attempts):
        if seed and time.monotonic() - started >= time_limit:
            break
        rng = random.Random(1009 + seed * 7919)
        big = sum(street.category == "M" for street in instance.streets) > 100
        # Try every constructor as a fallback: the regret heuristic can fail on
        # instances with many heavy-mandatory streets and few Large vehicles,
        # whereas the heavy-first "large" constructor succeeds there (and the
        # reverse on other instances). Robustness here matters for unseen data.
        order = (
            [construct_mandatory_large, construct_mandatory, construct_mandatory_randomized]
            if big else
            [construct_mandatory, construct_mandatory_randomized, construct_mandatory_large]
        )
        candidate = None
        for make in order:
            candidate = make(instance, paths, rng)
            if candidate is not None:
                break
        if candidate is None:
            continue
        if len(instance.streets) > 300:
            insert_optionals_large(candidate, instance, paths, rng)
        else:
            insert_optionals(candidate, instance, paths, rng)
            local_search(candidate, instance, paths, rng)
        valid, errors, candidate_score = validate_solution(candidate, instance, paths)
        if not valid:
            continue
        if best_score is None or candidate_score.value > best_score.value + EPS:
            best_solution, best_score = candidate.copy(), candidate_score
    if best_solution is None or best_score is None:
        raise RuntimeError(f"No feasible mandatory solution found for {instance.name}")
    if polish_seconds > 0:
        # Guided local search is good at filling routes; tabu search is better
        # at escaping the resulting basin. Split the budget deterministically
        # and retain only genuine score improvements after each phase.
        guided_seconds = max(1, polish_seconds // 2)
        tabu_seconds = polish_seconds - guided_seconds
        polished = ortools_polish(instance, best_solution, guided_seconds)
        if polished is not None:
            polished_score = score_solution(polished, instance)
            if polished_score.value > best_score.value + EPS:
                best_solution, best_score = polished, polished_score
        if tabu_seconds > 0:
            try:
                from ortools.constraint_solver import routing_enums_pb2

                polished = ortools_polish(
                    instance,
                    best_solution,
                    tabu_seconds,
                    metaheuristic=routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH,
                )
            except ImportError:
                polished = None
            if polished is not None:
                polished_score = score_solution(polished, instance)
                if polished_score.value > best_score.value + EPS:
                    best_solution, best_score = polished, polished_score
    return best_solution, best_score


def write_submission(path: Path, solution: Solution, instance: Instance, paths: ShortestPaths) -> None:
    valid, errors, _score = validate_solution(solution, instance, paths)
    if not valid:
        raise RuntimeError("Refusing to write invalid solution: " + "; ".join(errors))
    lines = [str(len(solution.routes))]
    for route in solution.routes:
        expanded = expand_route(route, instance, paths)
        assert expanded is not None
        nodes, cleaned, _total = expanded
        # The public validator defines n as the number of traversed arcs, so a
        # route with n arcs has n+1 junctions. An unused depot-only route is 0.
        lines.extend((str(len(nodes) - 1), " ".join(map(str, nodes)), " ".join(map(str, cleaned))))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def read_submission(path: Path, instance: Instance) -> Solution:
    # Used by --validate and unit tests. Blank cleaning rows are significant.
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if not lines:
        raise ValueError("empty submission")
    count = int(lines[0])
    if count != len(instance.vehicle_types) or len(lines) != 1 + 3 * count:
        raise ValueError("wrong submission line count or vehicle count")
    routes: list[Route] = []
    for i in range(count):
        edge_count = int(lines[1 + 3 * i])
        nodes = [int(value) for value in lines[2 + 3 * i].split()]
        if edge_count + 1 != len(nodes) or not nodes or nodes[0] != instance.depot or nodes[-1] != instance.depot:
            raise ValueError(f"route {i}: invalid junction row")
        cleaned = [] if not lines[3 + 3 * i].strip() else [int(value) for value in lines[3 + 3 * i].split()]
        route = Route(i, CAPACITY[instance.vehicle_types[i]])
        # A listed street must occur at least once on the output route. When
        # parsing external output, use its first matching traversal direction.
        for street_id in cleaned:
            if not 0 <= street_id < len(instance.streets):
                raise ValueError(f"route {i}: unknown cleaned street")
            street = instance.streets[street_id]
            task: Task | None = None
            for a, b in zip(nodes, nodes[1:]):
                if (a, b) in street.orientations():
                    task = Task(street_id, a, b)
                    break
            if task is None:
                raise ValueError(f"route {i}: cleaned street {street_id} is not traversed")
            route.tasks.append(task)
        routes.append(route)
    return Solution(routes)


def instance_summary(instance: Instance, solution: Solution, score: Score, paths: ShortestPaths) -> dict[str, object]:
    category_count = Counter(street.category for street in instance.streets)
    mandatory = {street.id for street in instance.streets if street.category == "M"}
    cleaned = {task.street_id for route in solution.routes for task in route.tasks}
    return {
        "instance": instance.name,
        "alpha": instance.alpha,
        "vehicles": dict(Counter(instance.vehicle_types)),
        "mandatory_streets": category_count["M"],
        "optional_streets": category_count["O"],
        "connector_streets": category_count["C"],
        "score": score.value,
        "coverage": score.coverage,
        "efficiency": score.efficiency,
        "cleaned_length": score.cleaned_length,
        "total_waste": score.waste,
        "mandatory_cleaned": len(mandatory & cleaned),
        "optional_cleaned": len(cleaned - mandatory),
        "duplicates": score.duplicates,
        "route_times": [int(route_time(route, instance, paths)) for route in solution.routes],
    }


def find_inputs(input_path: Path | None, input_dir: Path | None) -> list[Path]:
    if input_path is not None:
        return [input_path]
    if input_dir is None:
        raise ValueError("provide --input or --input-dir")
    files = sorted(input_dir.glob("*.txt"))
    if not files:
        raise ValueError(f"no .txt files found in {input_dir}")
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description="CLIPP Street Cleaning solver")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="one instance file")
    source.add_argument("--input-dir", type=Path, help="directory of instance files")
    parser.add_argument("--output", type=Path, help="output file (required with --input)")
    parser.add_argument("--output-dir", type=Path, help="directory for all outputs")
    parser.add_argument("--time-limit", type=float, default=30.0, help="seconds per instance")
    parser.add_argument("--seeds", type=int, default=12, help="multi-start attempts")
    parser.add_argument("--polish-seconds", type=int, default=0, help="OR-Tools polishing seconds per instance (0 disables it)")
    parser.add_argument("--mandatory-vrp-seconds", type=int, default=0, help="OR-Tools mandatory construction seconds for large instances")
    parser.add_argument("--validate", type=Path, help="validate an existing output for --input and exit")
    args = parser.parse_args()

    if args.validate:
        if args.input is None:
            parser.error("--validate requires --input")
        instance = parse_instance(args.input)
        solution = read_submission(args.validate, instance)
        valid, errors, score = validate_solution(solution, instance, ShortestPaths(instance))
        print(json.dumps({"valid": valid, "errors": errors, **score.__dict__}, indent=2))
        return 0 if valid else 2

    inputs = find_inputs(args.input, args.input_dir)
    if args.input and args.output is None:
        parser.error("--output is required with --input")
    output_dir = args.output_dir or Path("outputs") / "best"
    all_summaries: list[dict[str, object]] = []
    for input_file in inputs:
        instance = parse_instance(input_file)
        try:
            solution, score = solve(
                instance,
                args.time_limit,
                args.seeds,
                args.polish_seconds,
                args.mandatory_vrp_seconds,
            )
        except RuntimeError as error:
            summary = {"instance": instance.name, "status": "no_feasible_solution_found", "error": str(error)}
            all_summaries.append(summary)
            print(f"{instance.name}: {error}")
            continue
        paths = ShortestPaths(instance)
        output_file = args.output if args.input else output_dir / f"{input_file.stem}.out"
        assert output_file is not None
        write_submission(output_file, solution, instance, paths)
        summary = instance_summary(instance, solution, score, paths)
        all_summaries.append(summary)
        print(
            f"{instance.name}: score={score.value:.6f} coverage={score.coverage:.6f} "
            f"efficiency={score.efficiency:.6f} waste={score.waste:.3f} -> {output_file}"
        )
    summary_path = output_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(all_summaries, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
