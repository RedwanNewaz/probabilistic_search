# Probabilistic Search: Busting Multiple Ghosts with Naive Bayes

A robot with a square sensor footprint searches a 50×50 grid for 10 hidden ghosts.
It starts with a **large** footprint (cheap but noisy) and incrementally **shrinks** it
(expensive but accurate), using Naive Bayes inference to decide where to look and when
to shrink, until every ghost is localized.

## Run

```bash
.venv/Scripts/python.exe main.py
```

Headless (no window, prints a summary only):

```bash
NO_ANIM=1 .venv/Scripts/python.exe main.py
```

Requires Python ≥ 3.13, `matplotlib` and `numpy` (see `pyproject.toml`).

## What you see

| Element | Meaning |
|---|---|
| Heatmap | Belief P(ghost in cell), from blue (0.0) through green/yellow to red (1.0) |
| Cyan rectangle | Current sensor footprint (the robot) |
| White circles | True ghost positions (hidden from the algorithm; shown for comparison) |
| Red dots | Cells the robot is confident contain a ghost (belief > `CONFIDENT`) |

## The model

This follows the *Ghostbusters* model from lecture, extended to many ghosts and a
variable-size sensor.

**Variables**

- `G_c ∈ {0, 1}`: is there a ghost in cell `c`? There is one such variable per cell.
  The prior is `P(G_c = 1) = NUM_OBJECTS / (GRID_SIZE²)`, the same for every cell.
- `D ∈ {hit, none}`: the sensor reading over a footprint `F` of side length `s`.

**Sensor model.** The reading is a noisy "is any ghost in the footprint?" detector:

```
P(D = hit | ghost in F)    = hit(s)  = max(0.60, 0.95 − 0.02·(s−1))
P(D = hit | no ghost in F) = fa(s)   = min(0.30, 0.02 + 0.015·(s−1))
```

The larger the footprint, the lower the hit rate and the higher the false-alarm rate.
So a big footprint covers a lot of ground but its answer is uncertain, just as the
far-away squares in Ghostbusters give unreliable colors. A 1×1 footprint is
almost exact (0.95 / 0.02).

| Footprint `s` | hit(s) | fa(s) |
|---|---|---|
| 1 | 0.95 | 0.02 |
| 2 | 0.93 | 0.035 |
| 4 | 0.89 | 0.065 |
| 8 | 0.81 | 0.125 |
| 16 | 0.65 | 0.245 |

## Inference: Naive Bayes update

The full joint over 2500 binary cells has 2²⁵⁰⁰ entries, which is hopeless. As in the
Ghostbusters slides, we use the **Naive Bayes** conditional-independence assumption to keep
inference *linear* in the number of cells: the belief is stored as one independent
probability `b_c = P(G_c = 1)` per cell, and each reading is treated as evidence about
each cell in its footprint.

For every cell `c` in the footprint, Bayes' rule gives

```
P(G_c | D) ∝ P(G_c) · P(D | G_c)
```

The likelihood `P(D | G_c)` has to account for the *other* cells in the footprint,
because one of them might be the ghost that triggered the hit. Let

```
r_c = ∏_{j ∈ F, j ≠ c} (1 − b_j)      # P(no ghost in the other cells)
```

Then, for a **hit** reading:

```
P(hit | G_c = 1) = hit(s)
P(hit | G_c = 0) = fa(s)·r_c + hit(s)·(1 − r_c)
```

and for a **no-hit** reading:

```
P(none | G_c = 1) = 1 − hit(s)
P(none | G_c = 0) = (1 − fa(s))·r_c + (1 − hit(s))·(1 − r_c)
```

The posterior is `b_c' = b_c·L1 / (b_c·L1 + (1 − b_c)·L0)`, where `L1`/`L0` are the two
likelihoods above. Cells outside the footprint are unchanged.

Intuition:

- A **hit** raises the belief of every cell in the footprint a little (any could be the
  ghost); the more cells that are already unlikely, the more the remaining ones gain.
- A **no-hit** lowers the whole footprint. With a small footprint this nearly rules the
  cells out; with a large footprint it only weakens them (it could be a miss).
- Evidence accumulates across overlapping readings, and a cell is only confirmed after
  small, accurate footprints agree.

**Approximation.** Each cell is updated using the *previous* beliefs of its neighbors
rather than solving for the exact joint posterior. This is the Naive Bayes
simplification: cells are treated as independent given the evidence model. It is cheap
(vectorized with NumPy) and works well in practice.

## Policy: large → small footprint

Where to measure and what size to use is decided by information gain.

1. **Schedule.** The footprint size follows `SIZE_SCHEDULE = [16, 8, 4, 2, 1]`.
2. **Choose a window.** For the current size `s`, every window position is scored by the
   expected information gain of its reading, computed from the current beliefs:

   ```
   p_x = 1 − ∏_{c ∈ F} (1 − b_c)             # P(any ghost in window)
   p_D = p_x·hit(s) + (1 − p_x)·fa(s)        # P(reading = hit)
   IG  = H(p_D) − p_x·H(hit(s)) − (1 − p_x)·H(fa(s))
   ```

   (`H` is binary entropy in bits, and `IG` is the mutual information between the
   reading and "a ghost is in this window".) Window sums use an integral image, so all
   positions are scored at once. The robot goes to the window with the best
   `IG / (1 + 0.05·distance)`, which prefers informative windows that are also nearby.
3. **Shrink.** When the best window at the current size has `IG < IG_THRESHOLD`
   (0.03 bits), the coarse sensor has told us what it can, so the footprint moves down to
   the next size. Early on, large windows have high gain (about a third of them contain a
   ghost); later they are mostly either clearly empty or too noisy to be useful, and the
   robot refines.
4. **Move.** The robot drives to the chosen window with 8-neighbour steps, as in the
   original controls.
5. **Stop.** When `NUM_OBJECTS` cells have belief > `CONFIDENT` (0.97) and the belief
   mass of all other cells is below 1, or after `MAX_STEPS`.

The coarse stages quickly clear large empty regions (one no-hit over 16×16 = 256 cells
is a lot of progress), and the fine stages pin down exact cells only where the belief
is concentrated.

## Parameters (top of `main.py`)

| Name | Default | Effect |
|---|---|---|
| `GRID_SIZE` | 50 | Grid is `GRID_SIZE × GRID_SIZE` cells |
| `NUM_OBJECTS` | 10 | Number of hidden ghosts (also sets the prior) |
| `SIZE_SCHEDULE` | `[16, 8, 4, 2, 1]` | Footprint sizes, large → small |
| `IG_THRESHOLD` | 0.03 | Lower = spend longer at each footprint size before shrinking |
| `CONFIDENT` | 0.97 | Belief to declare a ghost. Higher = fewer false positives, more measurements |
| `MAX_STEPS` | 3000 | Safety cap on measurements |
| `SEED` | `None` | Set an integer for reproducible runs |
| `hit_prob`, `false_alarm` | see above | Edit to change how quickly noise grows with footprint size |

## Results

Headless runs with the default settings typically take about 800–1100 measurements
(versus 2500 for visiting every cell with a 1×1 sensor). In my test runs the algorithm
located 9–10 of the 10 ghosts, with at most one false positive. With a lower
`CONFIDENT` (0.90) more runs ended with 1–2 false positives, which is the
speed-vs-accuracy trade-off controlled by that parameter.

## Limitations / ideas

- The detector is binary ("any ghost in footprint?"); it does not report how many ghosts
  there are, so a footprint with two ghosts looks the same as one with one.
- Beliefs are a per-cell approximation (Naive Bayes), not the exact joint posterior.
- The stopping rule uses the known ghost count. Without it, use a belief threshold only.
- Possible extensions: a travel-cost-aware planner, a count-valued sensor, and a
  moving-ghost (HMM / particle filter) version.

## Files

- `main.py`: simulation, Naive Bayes inference, planning and animation
- `pyproject.toml`, `uv.lock`: dependencies
