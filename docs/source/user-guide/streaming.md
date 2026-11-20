# Streaming Knox statistics

`SlidingWindowKnox` maintains the Knox space-time pair statistics for the
events that fall inside a sliding time window.  Unlike {class}`Knox` and
{class}`KnoxLocal`, it does not take a complete dataset: events arrive one at
a time and are identified by a stable id, so the processor can keep its
counters exact without recomputing every pair each time.

Three kinds of records are supported:

`add`
:   Introduce a new event.  Replaying an identical `add` is an idempotent
    no-op, so at-least-once delivery is safe.

`correct`
:   Change the position and/or time of an event.  The old neighbour links
    are retracted before the new ones are built, so pairs are never
    double-counted.

`revoke`
:   Remove an event and all of its pairs.  Revocation is terminal.

## Setup

```{code-cell} ipython3
import numpy as np
from pointpats import SlidingWindowKnox

rng = np.random.default_rng(7)
processor = SlidingWindowKnox(
    delta=2.0,          # spatial threshold
    tau=1.0,            # temporal threshold
    window=20.0,        # sliding window width
    allowed_lateness=0.5,
)
```

## Ingesting records

A compact space-time cluster followed by a distant event:

```{code-cell} ipython3
records = [
    ("a", 0.0, 0.0, 0.0),
    ("b", 0.8, 0.0, 0.5),
    ("c", 0.4, 0.7, 1.0),
    ("d", 9.0, 9.0, 1.2),
]
for event_id, x, y, t in records:
    outcome = processor.add(event_id, x, y, t)
    print(event_id, outcome.applied, outcome.reason)
```

The outcome records whether the record was admitted (`accepted`), whether it
changed the window (`applied`) and why not when it did not.  Replaying the
same record is accepted but not reapplied:

```{code-cell} ipython3
event_id, x, y, t = records[0]
replay = processor.add(event_id, x, y, t)
print(replay.accepted, replay.applied, replay.reason)
```

## Watermark policy

Each record is compared against the watermark `W`:

- `t > W` advances the watermark when `auto_advance=True` (the default),
  which can expire other events;
- `t <= W - window` is outside the window and is rejected as
  `"out_of_window"`;
- `t < W - allowed_lateness` is too late and is rejected as `"late"`.

```{code-cell} ipython3
late = processor.add("late", 1.0, 1.0, 0.0)
print(late.accepted, late.reason, late.watermark)
```

Events expire when `t <= W - window`.  Their ids are tombstoned, so replaying
an expired event is rejected rather than resurrected.

## Counting and corrections

`nst`, `ns` and `nt` are the global pair counts; `observed` and `expected`
mirror the contingency tables of {class}`Knox`.

```{code-cell} ipython3
print("events:", processor.n)
print("spatial pairs:", processor.ns)
print("temporal pairs:", processor.nt)
print("space-time pairs:", processor.nst)
print(processor.observed)
```

Corrections keep the counts exact.  Here `a` is moved far outside the cluster,
so its two space-time pairs are retracted:

```{code-cell} ipython3
before = processor.nst
processor.correct("a", x=50.0, y=50.0)
print("pairs before/after:", before, processor.nst)
```

## Consistent snapshots and inference

Permutation inference needs a fixed dataset, so the processor hands a
consistent snapshot to the batch algorithms.  The local counts in the snapshot
are the incrementally maintained ones, so they can be compared directly with
the batch result.

```{code-cell} ipython3
import numpy as np

np.random.seed(12345)
snapshot = processor.snapshot()
local = snapshot.to_knox_local(permutations=99, keep=False)
print(snapshot.n, local.statistic_.sum() / 2)
```

`hotspots()` runs {class}`KnoxLocal` on a snapshot and returns the same table
that `KnoxLocal.hotspots` produces, so downstream hotspot consumers can be
fed directly from a stream:

```{code-cell} ipython3
hotspots = processor.hotspots(inference="analytic", permutations=0)
hotspots.head()
```

`processor.adjacency()` exposes the space-time edges in the same
`focal`/`neighbor`/`orientation` layout as `KnoxLocal.adjlist`.

## Incremental work

Spatial candidates are found through a uniform grid whose cell size defaults
to `delta`, and temporal candidates through a sorted time index, so adding or
correcting an event only examines its own neighbourhood.  The
`candidate_checks` and `distance_checks` counters expose that work:

```{code-cell} ipython3
processor.reset_counters()
processor.add("probe", 5.0, 5.0, processor.watermark)
processor.candidate_checks, processor.distance_checks, processor.n
```
