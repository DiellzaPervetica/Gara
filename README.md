# CLIPP Street Cleaning solver

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
