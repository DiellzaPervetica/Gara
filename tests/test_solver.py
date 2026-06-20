import tempfile
import unittest
from pathlib import Path

from src.main import ShortestPaths, parse_instance, solve, validate_solution, write_submission


class SolverTests(unittest.TestCase):
    def test_training_instances_are_solved_and_written_validly(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for name in ("train_a.txt", "train_b.txt", "train_n.txt"):
            input_file = root / "data" / name
            instance = parse_instance(input_file)
            solution, _score = solve(instance, time_limit=5, seeds=3)
            paths = ShortestPaths(instance)
            valid, errors, _ = validate_solution(solution, instance, paths)
            self.assertTrue(valid, errors)
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "solution.out"
                write_submission(output, solution, instance, paths)
                self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
