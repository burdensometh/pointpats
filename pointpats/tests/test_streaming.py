import numpy as np
import pandas as pd
import pytest

from pointpats.spacetime import Knox, KnoxLocal
from pointpats.streaming import (
    EventOutcome,
    SlidingWindowKnox,
    StreamEvent,
    WindowSnapshot,
)


def _reference_pairs(records, delta, tau):
    """Brute force Knox pair set from (id, x, y, t) records."""
    pairs = set()
    n = len(records)
    for i in range(n):
        id_i, x_i, y_i, t_i = records[i]
        for j in range(i + 1, n):
            id_j, x_j, y_j, t_j = records[j]
            spatial = (x_i - x_j) ** 2 + (y_i - y_j) ** 2 <= delta * delta
            temporal = abs(t_i - t_j) <= tau
            if spatial and temporal:
                pairs.add(frozenset((id_i, id_j)))
    return pairs


def _processor(delta=2.0, tau=1.0, window=100.0, **kwargs):
    # tests that do not care about watermark policy accept any in-window record
    kwargs.setdefault("allowed_lateness", float("inf"))
    return SlidingWindowKnox(delta=delta, tau=tau, window=window, **kwargs)


def _pair_ids(proc):
    return {frozenset(pair) for pair in proc.st_pairs}


class TestSlidingWindowKnox:
    def test_matches_bruteforce_after_adds(self):
        # accept anything still inside the window so the cyclic timestamps
        # below remain admissible
        proc = _processor(allowed_lateness=float("inf"))
        records = []
        for i in range(40):
            x = (i % 7) * 0.9
            y = (i % 5) * 0.8
            t = float(i % 11)
            proc.add(i, x, y, t)
            records.append((i, x, y, t))

        assert _pair_ids(proc) == _reference_pairs(records, 2.0, 1.0)
        assert proc.nst == len(_pair_ids(proc))
        assert proc.ns == len(
            {
                frozenset((a[0], b[0]))
                for i, a in enumerate(records)
                for b in records[i + 1 :]
                if (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2 <= 4.0
            }
        )
        assert proc.nt == len(
            {
                frozenset((a[0], b[0]))
                for i, a in enumerate(records)
                for b in records[i + 1 :]
                if abs(a[3] - b[3]) <= 1.0
            }
        )

    def test_local_counts_match_snapshot(self):
        proc = _processor(allowed_lateness=float("inf"))
        records = []
        for i in range(30):
            x, y = float(i % 4), float(i % 3)
            t = float(i % 6)
            proc.add(i, x, y, t)
            records.append((i, x, y, t))

        snap = proc.snapshot()
        df = snap.local_counts()
        pair_ids = _reference_pairs(records, 2.0, 1.0)
        expected_nsti = {r[0]: 0 for r in records}
        for pair in pair_ids:
            for event_id in pair:
                expected_nsti[event_id] += 1
        for event_id, value in expected_nsti.items():
            assert df.loc[event_id, "nsti"] == value
        assert df["nsti"].sum() == 2 * len(pair_ids)

    def test_global_statistics_match_batch_knox(self):
        rng = np.random.default_rng(7)
        proc = _processor(
            delta=3.0, tau=2.0, window=10.0, allowed_lateness=float("inf")
        )
        s_coords = rng.uniform(0, 12, size=(60, 2))
        t_coords = rng.uniform(0, 10, size=(60, 1))
        for i in range(60):
            proc.add(i, *s_coords[i], t_coords[i, 0])

        batch = Knox(s_coords, t_coords, delta=3.0, tau=2.0, permutations=0)
        assert proc.ns == batch.observed[0, 1] + batch.observed[0, 0]
        assert proc.nt == batch.observed[1, 0] + batch.observed[0, 0]
        assert proc.nst == batch.statistic_
        assert proc.pairs == batch.observed.sum()
        np.testing.assert_allclose(proc.observed, batch.observed)
        np.testing.assert_allclose(proc.expected, batch.expected)
        assert proc.p_poisson == pytest.approx(batch.p_poisson)

    def test_duplicate_add_is_idempotent(self):
        proc = _processor()
        first = proc.add("a", 0.0, 0.0, 1.0)
        proc.add("b", 1.0, 0.0, 1.0)
        nst = proc.nst
        replay = proc.add("a", 0.0, 0.0, 1.0)
        assert isinstance(replay, EventOutcome)
        assert replay.accepted and not replay.applied
        assert replay.reason == "duplicate"
        assert first.applied
        assert proc.nst == nst
        assert proc.n == 2

    def test_conflicting_readd_is_rejected(self):
        proc = _processor()
        proc.add("a", 0.0, 0.0, 1.0)
        outcome = proc.add("a", 5.0, 5.0, 1.0)
        assert not outcome.accepted
        assert outcome.reason == "exists"
        assert proc.n == 1

    def test_correction_retracts_and_rebuilds_pairs(self):
        proc = _processor(delta=1.5, tau=1.0)
        proc.add("a", 0.0, 0.0, 1.0)
        proc.add("b", 0.5, 0.0, 1.5)
        proc.add("c", 8.0, 8.0, 1.0)
        assert proc.nst == 1
        assert _pair_ids(proc) == {frozenset({"a", "b"})}

        # move "a" far away from "b" and next to "c"
        outcome = proc.correct("a", x=8.5, y=8.0)
        assert outcome.accepted and outcome.applied
        assert _pair_ids(proc) == {frozenset({"a", "c"})}
        assert proc.nst == 1

        # move "a" back
        proc.correct("a", x=0.0, y=0.0, t=1.0)
        assert _pair_ids(proc) == {frozenset({"a", "b"})}
        assert proc.st_pairs == {("a", "b")} or proc.st_pairs == {("b", "a")}

    def test_time_correction_changes_temporal_neighbourhood(self):
        proc = _processor(delta=10.0, tau=1.0)
        proc.add("a", 0.0, 0.0, 0.0)
        proc.add("b", 0.0, 0.0, 5.0)
        assert proc.nst == 0
        proc.correct("a", t=5.5)
        assert proc.nst == 1
        proc.correct("a", t=20.0)
        assert proc.nst == 0
        assert proc.nt == 0
        assert proc.ns == 1

    def test_unknown_correction_is_rejected(self):
        proc = _processor()
        outcome = proc.correct("nope", x=1.0, y=1.0)
        assert not outcome.accepted
        assert outcome.reason == "unknown"

    def test_correction_requires_payload(self):
        proc = _processor()
        proc.add("a", 0.0, 0.0, 1.0)
        with pytest.raises(ValueError, match="provided together"):
            proc.correct("a", x=1.0)
        with pytest.raises(ValueError, match="at least one"):
            proc.correct("a")

    def test_revoke_removes_event_and_pairs(self):
        proc = _processor(delta=1.0, tau=1.0)
        for i in range(3):
            proc.add(i, float(i), 0.0, 1.0)
        assert proc.nst == 2
        outcome = proc.revoke(0)
        assert outcome.accepted and outcome.applied
        assert proc.n == 2
        assert _pair_ids(proc) == {frozenset({1, 2})}
        assert proc.ns == 1

        again = proc.revoke(0)
        assert again.accepted and not again.applied
        assert again.reason == "already_revoked"

        # revoked ids cannot come back
        readd = proc.add(0, 0.0, 0.0, 1.0)
        assert not readd.accepted
        assert readd.reason == "revoked"

    def test_revoke_unknown_is_noop(self):
        proc = _processor()
        outcome = proc.revoke("ghost")
        assert outcome.accepted and not outcome.applied
        assert outcome.reason == "unknown"
        assert proc.n == 0

    def test_watermark_expiry(self):
        proc = _processor(window=10.0, allowed_lateness=0.0)
        proc.add("a", 0.0, 0.0, 0.0)
        proc.add("b", 0.0, 0.0, 1.0)
        assert proc.n == 2
        expired = proc.advance_watermark(10.5)
        assert expired == ("a",)
        assert proc.n == 1
        assert "a" not in proc
        assert proc.watermark == 10.5

    def test_event_on_window_boundary_expires(self):
        proc = _processor(window=10.0, allowed_lateness=0.0)
        proc.add("a", 0.0, 0.0, 0.0)
        # W - window = 0.0, so t=0 is outside the left-open window
        assert proc.advance_watermark(10.0) == ("a",)
        # W - window < t <= W keeps t strictly greater than the boundary
        proc.add("b", 0.0, 0.0, 10.0)
        assert proc.n == 1
        assert proc.advance_watermark(20.0) == ("b",)

    def test_auto_advance_expires_on_newest_event(self):
        proc = _processor(window=5.0, auto_advance=True, allowed_lateness=0.0)
        proc.add("old", 0.0, 0.0, 0.0)
        proc.add("new", 0.0, 0.0, 6.0)
        assert proc.watermark == 6.0
        assert "old" not in proc
        assert proc.n == 1

    def test_late_event_rejected(self):
        proc = _processor(window=100.0, allowed_lateness=2.0)
        proc.add("a", 0.0, 0.0, 10.0)
        late = proc.add("b", 0.0, 0.0, 7.0)
        assert not late.accepted
        assert late.reason == "late"
        assert proc.n == 1

        accepted = proc.add("c", 0.0, 0.0, 8.5)
        assert accepted.accepted and accepted.applied
        assert proc.n == 2

    def test_future_event_rejected_without_auto_advance(self):
        proc = _processor(auto_advance=False, window=100.0)
        proc.add("a", 0.0, 0.0, 1.0)
        future = proc.add("b", 0.0, 0.0, 5.0)
        assert not future.accepted
        assert future.reason == "future"
        assert proc.watermark == 1.0

    def test_out_of_window_rejected(self):
        proc = _processor(window=5.0, auto_advance=False, allowed_lateness=100.0)
        proc.advance_watermark(10.0)
        outcome = proc.add("a", 0.0, 0.0, 5.0)
        assert not outcome.accepted
        assert outcome.reason == "out_of_window"

    def test_infinite_lateness_accepts_anything_in_window(self):
        proc = _processor(window=10.0, allowed_lateness=float("inf"))
        proc.advance_watermark(10.0)
        outcome = proc.add("a", 0.0, 0.0, 0.5)
        # 0.5 > 10 - 10 = 0, so it is inside the window and accepted
        assert outcome.accepted and outcome.applied

    def test_replay_of_expired_event_is_rejected(self):
        proc = _processor(window=5.0, auto_advance=True)
        proc.add("a", 0.0, 0.0, 0.0)
        proc.add("b", 0.0, 0.0, 10.0)
        assert "a" not in proc
        replay = proc.add("a", 0.0, 0.0, 0.0)
        assert not replay.accepted
        assert replay.reason == "expired"
        assert proc.n == 1

    def test_expired_event_cannot_be_corrected_or_added(self):
        proc = _processor(window=5.0, auto_advance=True)
        proc.add("a", 0.0, 0.0, 0.0)
        proc.advance_watermark(6.0)
        assert proc.correct("a", x=1.0, y=1.0).reason == "expired"
        assert proc.add("a", 1.0, 1.0, 1.0).reason == "expired"

    def test_watermark_is_monotone(self):
        proc = _processor()
        proc.advance_watermark(5.0)
        with pytest.raises(ValueError, match="monotonic"):
            proc.advance_watermark(4.0)

    def test_correction_that_advances_watermark_expires_target(self):
        proc = _processor(window=5.0, auto_advance=True)
        proc.add("a", 0.0, 0.0, 0.0)
        outcome = proc.correct("a", t=10.0)
        assert not outcome.accepted
        assert outcome.reason == "expired"
        assert outcome.expired == ("a",)

    def test_incremental_work_is_local(self):
        proc = SlidingWindowKnox(
            delta=1.0, tau=1.0, window=1e9, cell_size=1.0, allowed_lateness=float("inf")
        )
        for i in range(2000):
            proc.add(i, float(i), 0.0, float(i))
        proc.reset_counters()
        before = proc.nst
        proc.add("new", 500.5, 0.0, 500.5)
        # only the two immediate grid neighbours can be inspected, and only
        # the two new pairs are attributed to the new event
        assert proc.candidate_checks <= 6
        assert proc.distance_checks <= 6
        assert proc.nst - before == 2

    def test_process_accepts_stream_event_and_mapping(self):
        proc = _processor(delta=1.0, tau=1.0)
        proc.process(StreamEvent("a", "add", 0.0, 0.0, 1.0))
        proc.process({"id": "b", "x": 0.5, "y": 0.0, "t": 1.0})
        proc.process({"op": "correct", "id": "b", "x": 5.0, "y": 0.0})
        assert proc.nst == 0
        proc.process({"op": "revoke", "id": "a"})
        assert proc.n == 1
        with pytest.raises(ValueError, match="op"):
            proc.process({"op": "zap", "id": "c"})
        with pytest.raises(TypeError):
            proc.process(object())

    def test_process_many(self):
        proc = _processor(delta=1.0, tau=1.0)
        events = [{"id": i, "x": float(i), "y": 0.0, "t": 1.0} for i in range(5)]
        outcomes = proc.process_many(events)
        assert len(outcomes) == 5
        assert all(outcome.applied for outcome in outcomes)
        assert proc.nst == 4

    def test_snapshot_is_consistent(self):
        proc = _processor(delta=1.0, tau=1.0)
        proc.add("b", 0.0, 0.0, 2.0)
        proc.add("a", 0.0, 0.0, 1.0)
        proc.add("c", 0.0, 0.0, 1.0)
        snap = proc.snapshot()
        assert isinstance(snap, WindowSnapshot)
        assert snap.n == 3
        # ordered by (time, ingestion order)
        assert snap.ids == ("a", "c", "b")
        assert list(snap.t_coords[:, 0]) == [1.0, 1.0, 2.0]
        # pairs are oriented by ingestion order (b was added first)
        assert {frozenset(pair) for pair in snap.st_pairs} == {
            frozenset(("a", "c")),
            frozenset(("a", "b")),
            frozenset(("c", "b")),
        }

    def test_snapshot_feeds_knox_local(self):
        rng = np.random.default_rng(3)
        proc = _processor(delta=3.0, tau=2.0, window=1e9)
        for i in range(50):
            proc.add(i, rng.uniform(0, 10), rng.uniform(0, 10), rng.uniform(0, 10))
        snap = proc.snapshot()
        local = snap.to_knox_local(permutations=0)
        assert isinstance(local, KnoxLocal)
        np.testing.assert_array_equal(
            np.sort(local.nsti), np.sort(snap.nsti)
        )
        # adjacency edge counts agree with the incremental bookkeeping
        assert local.adjlist.shape[0] == 2 * len(snap.st_pairs)
        expected = proc.adjacency()
        pd.testing.assert_frame_equal(
            local.adjlist.sort_values(["focal", "neighbor"]).reset_index(drop=True)
            .astype({"focal": object, "neighbor": object}),
            expected.astype({"focal": object, "neighbor": object}),
            check_dtype=False,
        )

    def test_to_dataframe_roundtrip(self):
        proc = _processor(delta=1.0, tau=1.0)
        proc.add("a", 0.0, 0.0, 1.0)
        proc.add("b", 0.5, 0.0, 1.0)
        gdf = proc.to_dataframe()
        assert list(gdf.index) == ["a", "b"]
        assert "time" in gdf.columns
        local = KnoxLocal.from_dataframe(gdf, time_col="time", delta=1.0, tau=1.0)
        assert local.nsti.tolist() == [1.0, 1.0]

    def test_hotspots_consumer(self):
        rng = np.random.default_rng(11)
        proc = _processor(delta=2.0, tau=1.0, window=1e9)
        # tight space-time cluster plus background
        for i in range(10):
            proc.add(f"c{i}", rng.normal(0, 0.2), rng.normal(0, 0.2), 1.0)
        for i in range(30):
            proc.add(
                f"b{i}", rng.uniform(-20, 20), rng.uniform(-20, 20), rng.uniform(0, 50)
            )
        hotspots = proc.hotspots(inference="analytic", permutations=0)
        assert not hotspots.empty
        assert {"focal", "neighbor", "cluster"}.issubset(hotspots.columns)

    def test_empty_snapshot(self):
        proc = _processor()
        snap = proc.snapshot()
        assert snap.n == 0
        assert snap.s_coords.shape == (0, 2)
        assert snap.local_counts().empty
        assert proc.adjacency().empty
        assert proc.pairs == 0
        assert np.isnan(proc.p_poisson)

    def test_validation(self):
        with pytest.raises(ValueError, match="delta"):
            SlidingWindowKnox(delta=-1, tau=1, window=1)
        with pytest.raises(ValueError, match="tau"):
            SlidingWindowKnox(delta=1, tau=-1, window=1)
        with pytest.raises(ValueError, match="window"):
            SlidingWindowKnox(delta=1, tau=1, window=0)
        with pytest.raises(ValueError, match="allowed_lateness"):
            SlidingWindowKnox(delta=1, tau=1, window=1, allowed_lateness=-1)
        with pytest.raises(ValueError, match="cell_size"):
            SlidingWindowKnox(delta=1, tau=1, window=1, cell_size=0)
        proc = _processor()
        with pytest.raises(ValueError, match="required"):
            proc.add("a", None, 0.0, 0.0)

    def test_incremental_matches_rebuild_after_random_updates(self):
        rng = np.random.default_rng(42)
        proc = _processor(delta=2.0, tau=1.5, window=1e9, allowed_lateness=1e9)
        records = {}
        next_id = 0
        for _ in range(300):
            action = rng.choice(["add", "add", "add", "correct", "revoke"])
            if action == "add" or not records:
                x, y, t = rng.uniform(0, 10, 3)
                proc.add(next_id, x, y, t)
                records[next_id] = (x, y, t)
                next_id += 1
            elif action == "correct":
                event_id = int(rng.choice(list(records)))
                x, y, t = rng.uniform(0, 10, 3)
                proc.correct(event_id, x=x, y=y, t=t)
                records[event_id] = (x, y, t)
            else:
                event_id = int(rng.choice(list(records)))
                proc.revoke(event_id)
                del records[event_id]

            reference = _reference_pairs(
                [(k, *v) for k, v in records.items()], 2.0, 1.5
            )
            assert _pair_ids(proc) == reference
            assert proc.nst == len(reference)
