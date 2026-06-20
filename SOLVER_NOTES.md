# CLIPP Street Cleaning — solver notes (max-score pipeline)

This repo now contains a strong, OR-Tools–backed engine plus an iterated
boosting driver, validated against the official netlify semantics.

## Files

- `src/clipp_solver.py` — the single-file engine (construction + alpha-aware
  optional insertion + exact short-route reordering + OR-Tools VRP construction
  and GLS/Tabu polish). Patched for **OR-Tools 9.15** (the upstream
  `SetAllowedVehiclesForIndex` is broken on 9.15; replaced with `VehicleVar`
  inequality constraints) and made BOM-tolerant (`utf-8-sig`).
- `src/boost.py` — iterated, re-seeded boosting: collects seeds (existing
  `.out` files + fresh multistarts + a from-scratch CP solution), then loops
  OR-Tools polish over GLS / Tabu / Generic-Tabu / Simulated-Annealing,
  re-seeding from the current best each round. Monotonic: never returns worse
  than the best seed. Every candidate is validated before being kept.
- `src/lns.py` — ruin-and-recreate (Large Neighbourhood Search) over optional
  streets, for an extra coverage pass on large instances.
- `tools/official_validator.py` — Python reimplementation of the official
  node-list validator (matches the netlify result and the README worked
  example to 1e-6).
- `tools/score_test.py` — validate the scored submissions in `outputs/test/`.
- `tools/score_all.py` — validate the train references in `outputs/best/`.
- `seeds/` — best-known starting solutions used to seed the boost.

## Submission files (the scored set)

The competition is evaluated on `instances/test/` (c, e, o). The validated
submissions are:

| file | α | score | mandatory |
|---|---|---|---|
| `outputs/test/test_c.out` | 0.3 | 0.8145 | 16/16 |
| `outputs/test/test_e.out` | 1.0 | 0.7323 | 68/68 |
| `outputs/test/test_o.out` | 0.5 | 0.8053 | 11/11 |

`outputs/best/` holds the (non-scored) training references a, b, n, m.

## Requirements

```
pip install ortools        # 9.15 is patched-for in clipp_solver.py
```

## Reproduce the best outputs

```bash
# large / coverage-heavy instance — VRP construction matters most here
python -m src.boost --input instances/m.txt --output outputs/best/m.out \
    --seed-output seeds/m_colleague.out --total-seconds 540 --round-seconds 70 --fresh-seeds 4

# small/medium — fresh multistarts + long alternating polish
python -m src.boost --input instances/train_a.txt --output outputs/best/train_a.out \
    --seed-output seeds/train_a_colleague.out --total-seconds 360 --round-seconds 30 --fresh-seeds 20

# validate everything
python tools/score_all.py
```

## Key facts

- **Score = α·Coverage + (1−α)·Efficiency**, summed over best-per-instance.
- α per instance: train_a=1.0 (coverage only), train_b=0.0 (efficiency only),
  train_n=0.5, m=0.7.
- **train_d is provably infeasible**: 9 heavy mandatory streets (req 30) need
  ≥4 Large vehicles within T, but the fleet has only 2 (Held–Karp proof). No
  valid solution exists; it scores 0 for everyone. Not part of the scored set
  in the colleague's data (`a, b, n, m`).
