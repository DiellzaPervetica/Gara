from pathlib import Path

from src.graph_utils import Graph
from src.models import Solution
from src.parser import parse_instance
from src.scorer import score_solution
from src.shortest_paths import ShortestPaths
from src.validator import validate_solution
from src.construction import build_empty_solution


NO_COORDS = """\
3 3 100 2 0 0.5
0 1 2 10 100 M 10
1 2 1 20 200 O 20
2 0 2 30 0 C 0
S M
"""


WITH_COORDS = """\
3 3 100 2 0 1.0
0.0 0.0
1.0 0.0
2.0 0.0
0 1 2 10 100 M 10
1 2 1 20 200 O 20
2 0 2 30 0 C 0
S M
"""


def test_parser_without_coordinates(tmp_path: Path):
    p = tmp_path / "x.txt"
    p.write_text(NO_COORDS)
    inst = parse_instance(p)
    assert inst.n == 3
    assert len(inst.streets) == 3
    assert inst.coordinates is None
    assert [v.kind for v in inst.vehicles] == ["S", "M"]


def test_parser_with_coordinates(tmp_path: Path):
    p = tmp_path / "x.txt"
    p.write_text(WITH_COORDS)
    inst = parse_instance(p)
    assert inst.coordinates == [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]


def test_direction_and_validation(tmp_path: Path):
    p = tmp_path / "x.txt"
    p.write_text(NO_COORDS)
    inst = parse_instance(p)
    graph = Graph(inst)
    sp = ShortestPaths(graph)
    sol = build_empty_solution(inst)
    sol.routes[0].tasks.append(graph.tasks_for_street(inst.streets[0])[0])
    sol = validate_solution(sol, graph, sp)
    assert sol.valid
    assert sol.routes[0].nodes[0] == 0
    assert sol.routes[0].nodes[-1] == 0


def test_missing_mandatory_caught(tmp_path: Path):
    p = tmp_path / "x.txt"
    p.write_text(NO_COORDS)
    inst = parse_instance(p)
    graph = Graph(inst)
    sp = ShortestPaths(graph)
    sol = validate_solution(build_empty_solution(inst), graph, sp)
    assert not sol.valid
    assert "missing mandatory" in sol.reason


def test_connector_cannot_be_cleaned(tmp_path: Path):
    p = tmp_path / "x.txt"
    p.write_text(NO_COORDS)
    inst = parse_instance(p)
    graph = Graph(inst)
    sp = ShortestPaths(graph)
    sol = build_empty_solution(inst)
    # Forge a connector-shaped task to make sure validator rejects it.
    c = inst.streets[2]
    from src.models import Task

    sol.routes[0].tasks.append(Task(c.id, c.a, c.b, c.time, c.length, c.category, c.requirement))
    sol = validate_solution(sol, graph, sp)
    assert not sol.valid
    assert "connector" in sol.reason
