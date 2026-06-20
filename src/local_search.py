from __future__ import annotations

from copy import deepcopy

from .construction import route_task_time
from .graph_utils import Graph
from .models import Solution, Task
from .scorer import optional_net_gain, score_solution
from .shortest_paths import ShortestPaths


def cleanup_bad_optionals(solution: Solution) -> Solution:
    instance = solution.instance
    if instance.alpha >= 0.999:
        return solution
    for route in solution.routes:
        route.tasks = [
            t
            for t in route.tasks
            if t.category == "M" or optional_net_gain(instance, route.vehicle.capacity, t.street_id) > 1e-12
        ]
    return solution


def reorient(solution: Solution, graph: Graph, sp: ShortestPaths) -> Solution:
    instance = solution.instance
    for route in solution.routes:
        changed = True
        while changed:
            changed = False
            base = route_task_time(route.tasks, instance.depot, sp)
            for i, task in enumerate(route.tasks):
                street = instance.streets[task.street_id]
                if street.direction != 2:
                    continue
                flipped = Task(task.street_id, task.end, task.start, task.time, task.length, task.category, task.requirement)
                trial = route.tasks[:]
                trial[i] = flipped
                ttime = route_task_time(trial, instance.depot, sp)
                if ttime < base and ttime <= instance.time_limit:
                    route.tasks = trial
                    changed = True
                    break
    return solution


def intra_relocate(solution: Solution, graph: Graph, sp: ShortestPaths, max_rounds: int = 2) -> Solution:
    instance = solution.instance
    for _ in range(max_rounds):
        any_change = False
        for route in solution.routes:
            base = route_task_time(route.tasks, instance.depot, sp)
            n = len(route.tasks)
            for i in range(n):
                task = route.tasks[i]
                rest = route.tasks[:i] + route.tasks[i + 1 :]
                for j in range(len(rest) + 1):
                    trial = rest[:j] + [task] + rest[j:]
                    ttime = route_task_time(trial, instance.depot, sp)
                    if ttime < base and ttime <= instance.time_limit:
                        route.tasks = trial
                        any_change = True
                        break
                if any_change:
                    break
        if not any_change:
            break
    return solution


def improve(solution: Solution, graph: Graph, sp: ShortestPaths) -> Solution:
    solution = cleanup_bad_optionals(solution)
    solution = reorient(solution, graph, sp)
    solution = intra_relocate(solution, graph, sp)
    return score_solution(solution)
