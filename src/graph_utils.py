from __future__ import annotations

from .models import Arc, Instance, Street, Task


class Graph:
    def __init__(self, instance: Instance):
        self.instance = instance
        self.adj: list[list[Arc]] = [[] for _ in range(instance.n)]
        self.best_arc: dict[tuple[int, int], Arc] = {}
        self.by_id = {s.id: s for s in instance.streets}
        for street in instance.streets:
            self._add(street.a, street.b, street)
            if street.direction == 2:
                self._add(street.b, street.a, street)

    def _add(self, u: int, v: int, street: Street) -> None:
        arc = Arc(u, v, street.id, street.time)
        self.adj[u].append(arc)
        key = (u, v)
        if key not in self.best_arc or arc.time < self.best_arc[key].time:
            self.best_arc[key] = arc

    def street_allows(self, street_id: int, u: int, v: int) -> bool:
        s = self.by_id[street_id]
        return (s.a == u and s.b == v) or (s.direction == 2 and s.b == u and s.a == v)

    def tasks_for_street(self, street: Street) -> list[Task]:
        if not street.cleanable:
            return []
        tasks = [
            Task(street.id, street.a, street.b, street.time, street.length, street.category, street.requirement)
        ]
        if street.direction == 2:
            tasks.append(
                Task(street.id, street.b, street.a, street.time, street.length, street.category, street.requirement)
            )
        return tasks

    def all_tasks(self, category: str | None = None) -> list[Task]:
        tasks: list[Task] = []
        for street in self.instance.streets:
            if not street.cleanable:
                continue
            if category is not None and street.category != category:
                continue
            tasks.extend(self.tasks_for_street(street))
        return tasks
