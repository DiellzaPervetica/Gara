# CLIPP Street Cleaning solver

## Train output results

`train_d` is intentionally omitted because it was removed/replaced.

| Instance | Output | Score | Coverage | Efficiency | Mandatory cleaned | Optional cleaned | Validator |
|---|---|---:|---:|---:|---:|---:|---|
| `train_a` | `outputs/best/train_a.out` | 0.7009432879827266 | 0.7009432879827266 | 0.8290679538176147 | 20 | 14 | VALID |
| `train_b` | `outputs/best/train_b.out` | 0.9677375207087805 | 0.6881802701069598 | 0.9677375207087805 | 28 | 0 | VALID |
| `m` | `outputs/best/m.out` | 0.7420075423062024 | 0.6990948291030462 | 0.8421372064469 | 356 | 157 | VALID |
| `train_n` | `outputs/best/train_n.out` | 0.9472492291765129 | 0.8944984583530259 | 1.0 | 10 | 14 | VALID |

Run every supplied instance:

```powershell
python -m src.main --input-dir data --output-dir outputs/best --time-limit 60 --seeds 20
```

For a longer competition run, add `--polish-seconds 240` to invoke guided and
tabu OR-Tools routing polish on the best valid multi-start solution.

For a large instance such as `m.txt`, add `--mandatory-vrp-seconds 180`; it
uses OR-Tools to construct the mandatory routes before optional scoring begins.

The solver writes one submission per input plus `outputs/summary.json`. It
accepts both the public train-file encoding (no coordinate rows) and the
coordinate-row variant described by the official README. Before a file is
written, routes are expanded and checked for direction, time, vehicle capacity,
connector cleaning, and mandatory coverage.

The public validator's first route value is the number of traversed arcs (so it
is one less than the number of junctions on the following line); the writer
uses that convention. Check a generated file with the same validator service:

```powershell
node tools/official_validator.mjs data/train_a.txt outputs/train_a.out
```


Validate an output independently:

```powershell
python -m src.main --input data/train_a.txt --validate outputs/train_a.out
```
