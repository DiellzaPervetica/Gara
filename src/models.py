from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


CAPACITY = {"S": 10, "M": 20, "L": 30}


@dataclass(frozen=True)
class Street:
    id: int
    a: int
    b: int
    direction: int
    time: int
    length: int
    category: str
    requirement: int

    @property
    def cleanable(self) -> bool:
        return self.category in {"M", "O"}


@dataclass(frozen=True)
class Vehicle:
    id: int
    kind: str

    @property
    def capacity(self) -> int:
        return CAPACITY[self.kind]


@dataclass(frozen=True)
class Arc:
    from_node: int
    to_node: int
    street_id: int
    time: int


@dataclass(frozen=True)
class Task:
    street_id: int
    start: int
    end: int
    time: int
    length: int
    category: str
    requirement: int


@dataclass
class Instance:
    name: str
    n: int
    m: int
    time_limit: int
    depot: int
    alpha: float
    streets: list[Street]
    vehicles: list[Vehicle]
    coordinates: Optional[list[tuple[float, float]]] = None

    @property
    def mandatory_ids(self) -> set[int]:
        return {s.id for s in self.streets if s.category == "M"}

    @property
    def optional_ids(self) -> set[int]:
        return {s.id for s in self.streets if s.category == "O"}

    @property
    def cleanable_ids(self) -> set[int]:
        return {s.id for s in self.streets if s.cleanable}


@dataclass
class Route:
    vehicle: Vehicle
    tasks: list[Task] = field(default_factory=list)
    extra_cleaned: list[int] = field(default_factory=list)
    time: int = 0
    nodes: list[int] = field(default_factory=list)
    traversed_streets: list[int] = field(default_factory=list)

    def cleaned_ids(self) -> list[int]:
        return [t.street_id for t in self.tasks] + self.extra_cleaned


@dataclass
class Solution:
    instance: Instance
    routes: list[Route]
    valid: bool = False
    score: float = 0.0
    coverage: float = 0.0
    efficiency: float = 0.0
    total_waste: float = 0.0
    reason: str = ""

    def cleaned_ids(self) -> list[int]:
        ids: list[int] = []
        for route in self.routes:
            ids.extend(route.cleaned_ids())
        return ids
