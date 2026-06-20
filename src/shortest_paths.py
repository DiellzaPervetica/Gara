from __future__ import annotations

import heapq
from dataclasses import dataclass

from .graph_utils import Graph


INF = 10**18


@dataclass
class Path:
    nodes: list[int]
    streets: list[int]
    time: int


class ShortestPaths:
    def __init__(self, graph: Graph):
        self.graph = graph
        self.cache: dict[int, tuple[list[int], list[int | None], list[int | None]]] = {}

    def run(self, source: int) -> tuple[list[int], list[int | None], list[int | None]]:
        if source in self.cache:
            return self.cache[source]
        n = self.graph.instance.n
        dist = [INF] * n
        parent = [None] * n
        parent_street = [None] * n
        dist[source] = 0
        heap = [(0, source)]
        while heap:
            du, u = heapq.heappop(heap)
            if du != dist[u]:
                continue
            for arc in self.graph.adj[u]:
                nd = du + arc.time
                if nd < dist[arc.to_node]:
                    dist[arc.to_node] = nd
                    parent[arc.to_node] = u
                    parent_street[arc.to_node] = arc.street_id
                    heapq.heappush(heap, (nd, arc.to_node))
        self.cache[source] = (dist, parent, parent_street)
        return self.cache[source]

    def distance(self, source: int, target: int) -> int:
        return self.run(source)[0][target]

    def path(self, source: int, target: int) -> Path | None:
        dist, parent, parent_street = self.run(source)
        if dist[target] >= INF:
            return None
        if source == target:
            return Path([source], [], 0)
        nodes = [target]
        streets: list[int] = []
        cur = target
        while cur != source:
            p = parent[cur]
            sid = parent_street[cur]
            if p is None or sid is None:
                return None
            streets.append(sid)
            nodes.append(p)
            cur = p
        nodes.reverse()
        streets.reverse()
        return Path(nodes, streets, dist[target])
