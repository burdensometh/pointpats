"""
Incremental Knox statistics over a sliding time window.

The classes in this module process a stream of events identified by a stable
id, keep all space-time neighbor bookkeeping for the events currently inside a
sliding time window, and only ever call the batch algorithms in
:mod:`pointpats.spacetime` when inference on a consistent snapshot is required.

Pair semantics match :func:`pointpats.spacetime._knox`:

* two events are *spatial* neighbors when their Euclidean distance is less than
  or equal to ``delta``;
* two events are *temporal* neighbors when their absolute time difference is
  less than or equal to ``tau``;
* they form a *space-time* (Knox) pair when they are both.

Window semantics are explicit and left-open/right-closed: with a watermark
``W`` and a window width ``window``, an event is active when
``W - window < t <= W``.  Consequently an event expires when
``t <= W - window``.
"""

__author__ = "PySAL contributors"

__all__ = [
    "EventOutcome",
    "SlidingWindowKnox",
    "StreamEvent",
    "WindowSnapshot",
]

import bisect
import math
import threading
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import poisson

from .spacetime import Knox, KnoxLocal

_OPERATIONS = ("add", "correct", "revoke")


@dataclass(frozen=True)
class StreamEvent:
    """A single event in the stream.

    Parameters
    ----------
    event_id : hashable
        Stable identifier of the event.  Re-using an identifier identifies the
        same event, so replays are idempotent and corrections/revocations
        target the event previously ingested under this id.
    op : str, optional
        One of ``"add"``, ``"correct"`` or ``"revoke"``, by default ``"add"``.
    x, y : float, optional
        Planar coordinates of the event.  Required for ``"add"``; either both
        or neither must be given for ``"correct"``.
    t : float, optional
        Event time (in the same numeric units as ``tau`` and ``window``).
        Required for ``"add"``.
    """

    event_id: Hashable
    op: str = "add"
    x: float | None = None
    y: float | None = None
    t: float | None = None


@dataclass(frozen=True)
class EventOutcome:
    """Result of applying a single event to a :class:`SlidingWindowKnox`.

    Attributes
    ----------
    event_id : hashable
        Identifier of the ingested event.
    op : str
        Operation that was requested.
    accepted : bool
        Whether the record was admitted by the watermark rules.  A rejected
        record did not change any statistic.
    applied : bool
        Whether the record changed the window (a replay of an identical record
        is accepted but not applied).
    reason : str or None
        Why the record was not applied.  ``None`` when ``applied`` is True.
        Possible values are ``"duplicate"``, ``"unchanged"``,
        ``"already_revoked"``, ``"unknown"``, ``"exists"``, ``"expired"``,
        ``"revoked"``, ``"future"``, ``"late"`` and ``"out_of_window"``.
    expired : tuple
        Ids that were evicted because ingesting this record advanced the
        watermark.
    watermark : float
        Watermark after processing the record.
    """

    event_id: Hashable
    op: str
    accepted: bool
    applied: bool
    reason: str | None
    expired: tuple
    watermark: float


@dataclass(frozen=True, eq=False)
class WindowSnapshot:
    """An immutable, consistent view of the active window.

    The arrays are ordered by ``(time, ingestion order)`` and are therefore
    aligned with ``ids``, ``nsti``, ``nsi`` and ``nti``.  The snapshot can be
    consumed directly by the batch algorithms through :meth:`to_knox_local`,
    :meth:`to_knox` and :meth:`to_dataframe`.

    Attributes
    ----------
    ids : tuple
        External ids in snapshot order.
    s_coords : ndarray of shape (n, 2)
        Spatial coordinates.
    t_coords : ndarray of shape (n, 1)
        Temporal coordinates.
    nsti, nsi, nti : ndarray of shape (n,)
        Local space-time, spatial and temporal neighbor counts for each event,
        matching ``KnoxLocal.nsti``, ``nsi`` and ``nti``.
    st_pairs : tuple of tuple
        The space-time (Knox) pairs, oriented by ingestion order.
    delta, tau : float
        Thresholds used to build the window.
    crs : optional
        Coordinate reference system carried over from the processor.
    """

    ids: tuple
    s_coords: np.ndarray
    t_coords: np.ndarray
    nsti: np.ndarray
    nsi: np.ndarray
    nti: np.ndarray
    st_pairs: tuple
    delta: float
    tau: float
    crs: Any = None

    @property
    def n(self):
        """Number of events in the snapshot."""
        return len(self.ids)

    def local_counts(self):
        """Local neighbor counts as a DataFrame indexed by event id.

        Returns
        -------
        pandas.DataFrame
            Columns ``x``, ``y``, ``time``, ``nsti``, ``nsi`` and ``nti``.
        """
        return pd.DataFrame(
            {
                "x": self.s_coords[:, 0],
                "y": self.s_coords[:, 1],
                "time": self.t_coords[:, 0],
                "nsti": self.nsti,
                "nsi": self.nsi,
                "nti": self.nti,
            },
            index=pd.Index(self.ids, name="id"),
        )

    def adjacency(self):
        """Space-time adjacency edges as a two-way edge table.

        The layout matches ``KnoxLocal.adjlist``: each undirected pair appears
        once as ``focal``/``neighbor`` and once as ``neighbor``/``focal``, and
        ``orientation`` describes the neighbour's time relative to the focal
        time: ``"lead"`` when the neighbour is later, ``"lag"`` when it is
        earlier and ``"coincident"`` otherwise.

        Returns
        -------
        pandas.DataFrame
            Columns ``focal``, ``neighbor`` and ``orientation``.
        """
        rows = []
        times = {event_id: self.t_coords[i, 0] for i, event_id in enumerate(self.ids)}
        for focal, neighbor in self.st_pairs:
            for a, b in ((focal, neighbor), (neighbor, focal)):
                if times[a] < times[b]:
                    orientation = "lead"
                elif times[a] > times[b]:
                    orientation = "lag"
                else:
                    orientation = "coincident"
                rows.append((a, b, orientation))
        out = pd.DataFrame(rows, columns=["focal", "neighbor", "orientation"])
        return out.sort_values(by=["focal", "neighbor"]).reset_index(drop=True)

    def to_dataframe(self):
        """Snapshot as a GeoDataFrame consumable by ``KnoxLocal.from_dataframe``.

        Returns
        -------
        geopandas.GeoDataFrame
            Point geometries indexed by event id with a numeric ``time`` column
            and the local counts as additional columns.
        """
        df = self.local_counts()
        geom = gpd.points_from_xy(self.s_coords[:, 0], self.s_coords[:, 1])
        return gpd.GeoDataFrame(df, geometry=geom, crs=self.crs)

    def to_knox_local(self, permutations=99, keep=False, crit=0.05):
        """Run :class:`pointpats.spacetime.KnoxLocal` on this snapshot.

        Parameters
        ----------
        permutations : int, optional
            Permutations for conditional inference, by default 99.  Set to 0
            to skip permutation inference.
        keep : bool, optional
            Whether to keep the realized permutation statistics, by default
            False.
        crit : float, optional
            Significance level stored on the returned object, by default 0.05.

        Returns
        -------
        pointpats.spacetime.KnoxLocal
            Fitted local Knox object built from this snapshot.
        """
        return KnoxLocal(
            self.s_coords.copy(),
            self.t_coords.copy(),
            self.delta,
            self.tau,
            permutations=permutations,
            keep=keep,
            crit=crit,
            crs=self.crs,
            ids=list(self.ids),
        )

    def to_knox(self, permutations=99, keep=False):
        """Run :class:`pointpats.spacetime.Knox` on this snapshot.

        Parameters
        ----------
        permutations : int, optional
            Permutations for inference, by default 99.
        keep : bool, optional
            Whether to keep the realized permutation statistics, by default
            False.

        Returns
        -------
        pointpats.spacetime.Knox
            Fitted global Knox object built from this snapshot.
        """
        return Knox(
            self.s_coords.copy(),
            self.t_coords.copy(),
            self.delta,
            self.tau,
            permutations=permutations,
            keep=keep,
        )


class _Event:
    """Internal mutable record for an active event."""

    __slots__ = (
        "cell",
        "event_id",
        "nsti",
        "s_neighbors",
        "slot",
        "st_neighbors",
        "t",
        "t_neighbors",
        "x",
        "y",
    )

    def __init__(self, event_id, x, y, t, slot):
        self.event_id = event_id
        self.x = x
        self.y = y
        self.t = t
        self.slot = slot
        self.cell = None
        self.s_neighbors = set()
        self.t_neighbors = set()
        self.st_neighbors = set()


class SlidingWindowKnox:
    """Incrementally maintained Knox statistics over a sliding time window.

    The processor accepts events identified by a stable id and supports three
    operations:

    ``add``
        Introduce a new event.  Re-adding an identical event is an idempotent
        no-op; re-adding an id that is already active with a different payload
        is rejected (``"exists"``) and must be sent as a ``correct`` instead.
    ``correct``
        Change the position and/or time of an active event.  The old
        neighbour links are retracted and the new ones are established, so the
        global counts, local counts and adjacency edges remain exact.
    ``revoke``
        Remove an active event and all of its links.  Revoking is terminal: a
        later ``add`` of the same id is rejected.

    Accepted records modify counters in ``O(k)`` where ``k`` is the number of
    spatial candidates in the surrounding grid cells plus the number of
    temporal neighbours, never ``O(n^2)``.  Spatial candidates come from a
    uniform grid with cell size ``cell_size`` (``delta`` by default); the
    temporal filter uses a sorted list of ``(time, slot)`` entries.

    Watermark rules
    ---------------
    Let ``W`` be the watermark, ``L`` the ``allowed_lateness`` and ``w`` the
    window width.  A new (or time-corrected) event with time ``t`` is admitted
    only when

    1. ``t <= W`` unless ``auto_advance`` is set, in which case the watermark
       is first advanced to ``t`` (which may expire other events);
    2. ``t > W - w`` (otherwise it is outside the current window);
    3. ``t >= W - L`` (otherwise it is late).

    Events expire when ``t <= W - w``.  Expiry retracts their pairs and records
    the id in a tombstone set, so replaying an expired event is rejected rather
    than resurrected.

    Parameters
    ----------
    delta : float
        Spatial threshold (inclusive) for the Knox pair relation.
    tau : float
        Temporal threshold (inclusive) for the Knox pair relation.
    window : float
        Sliding window width.  Active events satisfy
        ``watermark - window < t <= watermark``.
    allowed_lateness : float, optional
        How far behind the watermark a record may be and still be accepted,
        by default ``0.0`` (only events at or after the watermark are
        accepted until the watermark is advanced explicitly).  Use
        ``float("inf")`` to accept any record that is still inside the window.
    auto_advance : bool, optional
        Whether ingesting an event newer than the watermark advances the
        watermark, by default True.
    cell_size : float, optional
        Spatial grid cell size for the candidate index, by default ``delta``.
        Smaller values shrink candidate sets at the cost of scanning more
        cells.
    crs : optional
        Coordinate reference system attached to snapshots, by default None.

    Attributes
    ----------
    delta, tau, window, allowed_lateness : float
        Configuration values.
    distance_checks : int
        Number of exact spatial distances computed since the last
        :meth:`reset_counters`.  Useful to verify incremental behaviour.
    candidate_checks : int
        Number of grid candidates examined since the last
        :meth:`reset_counters`.
    """

    def __init__(
        self,
        delta,
        tau,
        window,
        allowed_lateness=0.0,
        auto_advance=True,
        cell_size=None,
        crs=None,
    ):
        delta = float(delta)
        tau = float(tau)
        window = float(window)
        allowed_lateness = float(allowed_lateness)
        if not math.isfinite(delta) or delta < 0:
            raise ValueError("`delta` must be a non-negative finite number")
        if not math.isfinite(tau) or tau < 0:
            raise ValueError("`tau` must be a non-negative finite number")
        if not math.isfinite(window) or window <= 0:
            raise ValueError("`window` must be a positive finite number")
        if math.isnan(allowed_lateness) or allowed_lateness < 0:
            raise ValueError("`allowed_lateness` must be non-negative")
        if cell_size is None:
            cell_size = delta if delta > 0 else 1.0
        cell_size = float(cell_size)
        if not math.isfinite(cell_size) or cell_size <= 0:
            raise ValueError("`cell_size` must be a positive finite number")

        self.delta = delta
        self.tau = tau
        self.window = window
        self.allowed_lateness = allowed_lateness
        self.auto_advance = bool(auto_advance)
        self.cell_size = cell_size
        self.crs = crs

        self._watermark = float("-inf")
        self._events = {}  # external id -> _Event
        self._slots = {}  # internal slot -> _Event
        self._next_slot = 0
        self._grid = {}  # (ix, iy) -> set of slots
        self._times = []  # sorted list of (t, slot)
        self._st_pairs = set()  # canonical (low slot, high slot)
        self._revoked = set()
        self._expired = set()

        # directed link counts; the number of undirected pairs is half
        self._n_s = 0
        self._n_t = 0
        self._n_st = 0

        self._lock = threading.RLock()
        self.distance_checks = 0
        self.candidate_checks = 0

    # ------------------------------------------------------------------ basic
    @property
    def watermark(self):
        """Current watermark (``-inf`` before any event or advancement)."""
        return self._watermark

    @property
    def n(self):
        """Number of active events."""
        return len(self._events)

    @property
    def ns(self):
        """Number of spatial neighbour pairs in the window."""
        return self._n_s // 2

    @property
    def nt(self):
        """Number of temporal neighbour pairs in the window."""
        return self._n_t // 2

    @property
    def nst(self):
        """Number of space-time (Knox) pairs in the window."""
        return self._n_st // 2

    @property
    def pairs(self):
        """Number of unordered event pairs in the window."""
        n = self.n
        return n * (n - 1) // 2

    @property
    def expected_nst(self):
        """Expected number of space-time pairs under independence."""
        if self.pairs == 0:
            return float("nan")
        return self.ns * self.nt / self.pairs

    @property
    def observed(self):
        """Observed 2x2 Knox contingency table (same layout as ``Knox``)."""
        nst, ns, nt = self.nst, self.ns, self.nt
        return np.array(
            [
                [nst, ns - nst],
                [nt - nst, self.pairs - (ns + nt - nst)],
            ],
            dtype=float,
        )

    @property
    def expected(self):
        """Expected 2x2 Knox contingency table (same layout as ``Knox``)."""
        enst = self.expected_nst
        ns, nt = self.ns, self.nt
        expected = np.zeros((2, 2))
        expected[0, 0] = enst
        expected[0, 1] = ns - enst
        expected[1, 0] = nt - enst
        expected[1, 1] = self.pairs - expected.sum()
        return expected

    @property
    def p_poisson(self):
        """Analytic one-sided p-value for the global count."""
        enst = self.expected_nst
        if math.isnan(enst):
            return float("nan")
        return 1 - poisson.cdf(self.nst, enst)

    @property
    def ids(self):
        """Ids of the active events, in snapshot order."""
        return tuple(
            rec.event_id
            for rec in sorted(self._events.values(), key=lambda r: (r.t, r.slot))
        )

    @property
    def st_pairs(self):
        """Space-time pairs as a set of id tuples oriented by ingestion order."""
        with self._lock:
            return {
                (
                    self._slots[low].event_id,
                    self._slots[high].event_id,
                )
                for low, high in self._st_pairs
            }

    def reset_counters(self):
        """Reset the ``distance_checks`` and ``candidate_checks`` counters."""
        with self._lock:
            self.distance_checks = 0
            self.candidate_checks = 0

    # ------------------------------------------------------------- operations
    def add(self, event_id, x, y, t):
        """Add an event identified by ``event_id``.

        Parameters
        ----------
        event_id : hashable
            Stable event identifier.
        x, y : float
            Planar coordinates.
        t : float
            Event time.

        Returns
        -------
        EventOutcome
            Outcome of the operation.
        """
        x, y, t = self._validate_payload(x, y, t)
        with self._lock:
            if event_id in self._revoked:
                return self._reject(event_id, "add", "revoked")
            rec = self._events.get(event_id)
            if rec is not None:
                if (rec.x, rec.y, rec.t) == (x, y, t):
                    return self._outcome(event_id, "add", True, False, "duplicate")
                return self._reject(event_id, "add", "exists")
            if event_id in self._expired:
                return self._reject(event_id, "add", "expired")

            # the first accepted record seeds the watermark even when
            # auto-advance is off
            if self._watermark == float("-inf"):
                self._watermark = t
            expired = ()
            if t > self._watermark and self.auto_advance:
                expired = self._advance(t)
            if t > self._watermark:
                return self._reject(event_id, "add", "future", expired)
            if t <= self._watermark - self.window:
                return self._reject(event_id, "add", "out_of_window", expired)
            if t < self._watermark - self.allowed_lateness:
                return self._reject(event_id, "add", "late", expired)

            rec = self._new_record(event_id, x, y, t)
            self._attach(rec)
            return self._outcome(event_id, "add", True, True, None, expired)

    def correct(self, event_id, x=None, y=None, t=None):
        """Correct the position and/or time of an active event.

        The old neighbour links are retracted before the new ones are added.
        At least one of ``x``, ``y`` or ``t`` must be supplied.  Coordinates
        must be given as a pair.

        Parameters
        ----------
        event_id : hashable
            Stable event identifier.
        x, y : float, optional
            New planar coordinates; provide both or neither.
        t : float, optional
            New event time.

        Returns
        -------
        EventOutcome
            Outcome of the operation.

        Raises
        ------
        ValueError
            If no field is provided or only one coordinate is provided.
        """
        if (x is None) != (y is None):
            raise ValueError("`x` and `y` must be provided together")
        if x is None and t is None:
            raise ValueError("at least one of `x`, `y` or `t` must be provided")
        with self._lock:
            if event_id in self._revoked:
                return self._reject(event_id, "correct", "revoked")
            if event_id in self._expired:
                return self._reject(event_id, "correct", "expired")
            rec = self._events.get(event_id)
            if rec is None:
                return self._reject(event_id, "correct", "unknown")

            new_x = rec.x if x is None else float(x)
            new_y = rec.y if y is None else float(y)
            new_t = rec.t if t is None else float(t)
            if not (math.isfinite(new_x) and math.isfinite(new_y)):
                raise ValueError("coordinates must be finite numbers")
            if not math.isfinite(new_t):
                raise ValueError("`t` must be a finite number")
            if (new_x, new_y, new_t) == (rec.x, rec.y, rec.t):
                return self._outcome(
                    event_id, "correct", True, False, "unchanged"
                )

            expired = ()
            if t is not None and new_t != rec.t:
                if new_t > self._watermark and self.auto_advance:
                    expired = self._advance(new_t)
                    if event_id not in self._events:
                        # the correction advanced the watermark past the
                        # event's own old time, expiring it
                        return self._reject(
                            event_id, "correct", "expired", expired
                        )
                if new_t > self._watermark:
                    return self._reject(event_id, "correct", "future", expired)
                if new_t <= self._watermark - self.window:
                    return self._reject(
                        event_id, "correct", "out_of_window", expired
                    )
                if new_t < self._watermark - self.allowed_lateness:
                    return self._reject(event_id, "correct", "late", expired)

            self._detach(rec)
            rec.x = new_x
            rec.y = new_y
            rec.t = new_t
            self._attach(rec)
            return self._outcome(
                event_id, "correct", True, True, None, expired
            )

    def revoke(self, event_id):
        """Revoke an active event and retract all of its pairs.

        Revocation is terminal: subsequent ``add`` or ``correct`` records for
        the same id are rejected.  Revoking an unknown or already revoked id is
        accepted as a no-op.

        Parameters
        ----------
        event_id : hashable
            Stable event identifier.

        Returns
        -------
        EventOutcome
            Outcome of the operation.
        """
        with self._lock:
            if event_id in self._revoked:
                return self._outcome(
                    event_id, "revoke", True, False, "already_revoked"
                )
            rec = self._events.get(event_id)
            if rec is None:
                if event_id in self._expired:
                    return self._outcome(
                        event_id, "revoke", True, False, "expired"
                    )
                return self._outcome(event_id, "revoke", True, False, "unknown")
            self._detach(rec)
            del self._events[event_id]
            del self._slots[rec.slot]
            self._revoked.add(event_id)
            return self._outcome(event_id, "revoke", True, True, None)

    def advance_watermark(self, t):
        """Advance the watermark to ``t`` and expire events that fall out.

        Parameters
        ----------
        t : float
            New watermark.  Must not be smaller than the current watermark.

        Returns
        -------
        tuple
            Ids of the events that expired.

        Raises
        ------
        ValueError
            If ``t`` is not finite or would move the watermark backwards.
        """
        t = float(t)
        if not math.isfinite(t):
            raise ValueError("watermark must be a finite number")
        with self._lock:
            if t < self._watermark:
                raise ValueError("watermark must be monotonic")
            return self._advance(t)

    def process(self, event):
        """Apply a :class:`StreamEvent` or an equivalent mapping.

        Parameters
        ----------
        event : StreamEvent or Mapping
            Mapping keys are ``"op"``, ``"id"``, ``"x"``, ``"y"`` and
            ``"t"``; ``"op"`` defaults to ``"add"``.

        Returns
        -------
        EventOutcome
            Outcome of the operation.

        Raises
        ------
        TypeError
            If ``event`` is neither a :class:`StreamEvent` nor a mapping.
        ValueError
            If the operation is not one of ``"add"``, ``"correct"`` or
            ``"revoke"``.
        """
        if isinstance(event, StreamEvent):
            op = event.op
            event_id = event.event_id
            x, y, t = event.x, event.y, event.t
        elif isinstance(event, Mapping):
            op = event.get("op", "add")
            if "id" in event:
                event_id = event["id"]
            elif "event_id" in event:
                event_id = event["event_id"]
            else:
                raise ValueError("event mapping requires an 'id' key")
            x, y, t = event.get("x"), event.get("y"), event.get("t")
        else:
            raise TypeError("event must be a StreamEvent or a mapping")

        if op not in _OPERATIONS:
            raise ValueError(f"`op` must be one of {_OPERATIONS}, got {op!r}")
        if op == "revoke":
            return self.revoke(event_id)
        if op == "correct":
            return self.correct(event_id, x=x, y=y, t=t)
        return self.add(event_id, x, y, t)

    def process_many(self, events):
        """Apply an iterable of events, returning their outcomes in order.

        Parameters
        ----------
        events : iterable
            Iterable of :class:`StreamEvent` or mapping records.

        Returns
        -------
        list of EventOutcome
        """
        return [self.process(event) for event in events]

    # --------------------------------------------------------------- snapshots
    def snapshot(self):
        """Return a consistent :class:`WindowSnapshot` of the active window.

        Returns
        -------
        WindowSnapshot
            Arrays and local counts aligned by ``(time, ingestion order)``.
        """
        with self._lock:
            recs = sorted(self._events.values(), key=lambda rec: (rec.t, rec.slot))
            ids = tuple(rec.event_id for rec in recs)
            if recs:
                s_coords = np.array(
                    [[rec.x, rec.y] for rec in recs], dtype=float
                )
                t_coords = np.array([[rec.t] for rec in recs], dtype=float)
                nsti = np.array(
                    [len(rec.st_neighbors) for rec in recs], dtype=float
                )
                nsi = np.array(
                    [len(rec.s_neighbors) for rec in recs], dtype=float
                )
                nti = np.array(
                    [len(rec.t_neighbors) for rec in recs], dtype=float
                )
            else:
                s_coords = np.empty((0, 2))
                t_coords = np.empty((0, 1))
                nsti = np.empty(0)
                nsi = np.empty(0)
                nti = np.empty(0)
            st_pairs = tuple(
                (self._slots[low].event_id, self._slots[high].event_id)
                for low, high in sorted(self._st_pairs)
            )
            return WindowSnapshot(
                ids=ids,
                s_coords=s_coords,
                t_coords=t_coords,
                nsti=nsti,
                nsi=nsi,
                nti=nti,
                st_pairs=st_pairs,
                delta=self.delta,
                tau=self.tau,
                crs=self.crs,
            )

    def local_counts(self):
        """Local counts of the active window as a DataFrame indexed by id.

        Returns
        -------
        pandas.DataFrame
            Columns ``x``, ``y``, ``time``, ``nsti``, ``nsi`` and ``nti``.
        """
        return self.snapshot().local_counts()

    def to_dataframe(self):
        """Active window as a GeoDataFrame consumable by ``KnoxLocal``.

        Returns
        -------
        geopandas.GeoDataFrame
            Point geometries indexed by event id.
        """
        return self.snapshot().to_dataframe()

    def adjacency(self):
        """Space-time adjacency edges for the active window.

        Returns
        -------
        pandas.DataFrame
            Two-way edge table as described in
            :meth:`WindowSnapshot.adjacency`.
        """
        return self.snapshot().adjacency()

    def to_knox_local(self, permutations=99, keep=False, crit=0.05):
        """Run ``KnoxLocal`` on a consistent snapshot of the window."""
        return self.snapshot().to_knox_local(
            permutations=permutations, keep=keep, crit=crit
        )

    def to_knox(self, permutations=99, keep=False):
        """Run ``Knox`` on a consistent snapshot of the window."""
        return self.snapshot().to_knox(permutations=permutations, keep=keep)

    def hotspots(
        self,
        crit=0.05,
        inference="permutation",
        keep_neighbors=True,
        permutations=99,
    ):
        """Significant space-time clusters for the current window.

        A consistent snapshot is handed to :class:`pointpats.spacetime.KnoxLocal`,
        which performs permutation inference when required.

        Parameters
        ----------
        crit : float, optional
            Significance level, by default 0.05.
        inference : str, optional
            ``"permutation"`` or ``"analytic"``, by default ``"permutation"``.
        keep_neighbors : bool, optional
            Whether to include non-significant members of hotspot components,
            by default True.
        permutations : int, optional
            Permutations passed to ``KnoxLocal``, by default 99.

        Returns
        -------
        pandas.DataFrame
            Table of hotspots as produced by ``KnoxLocal.hotspots``.
        """
        local = self.to_knox_local(permutations=permutations, keep=False)
        return local.hotspots(
            crit=crit, inference=inference, keep_neighbors=keep_neighbors
        )

    # ---------------------------------------------------------------- internals
    def _validate_payload(self, x, y, t):
        if x is None or y is None or t is None:
            raise ValueError("`x`, `y` and `t` are required for `add`")
        x, y, t = float(x), float(y), float(t)
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(t)):
            raise ValueError("`x`, `y` and `t` must be finite numbers")
        return x, y, t

    def _outcome(self, event_id, op, accepted, applied, reason, expired=()):
        return EventOutcome(
            event_id=event_id,
            op=op,
            accepted=accepted,
            applied=applied,
            reason=reason,
            expired=tuple(expired),
            watermark=self._watermark,
        )

    def _reject(self, event_id, op, reason, expired=()):
        return self._outcome(event_id, op, False, False, reason, expired)

    def _advance(self, t):
        self._watermark = t
        threshold = t - self.window
        expired = []
        while self._times and self._times[0][0] <= threshold:
            _, slot = self._times[0]
            rec = self._slots[slot]
            expired.append(rec.event_id)
            self._detach(rec)
            del self._events[rec.event_id]
            del self._slots[slot]
        self._expired.update(expired)
        return tuple(expired)

    def _new_record(self, event_id, x, y, t):
        rec = _Event(event_id, x, y, t, self._next_slot)
        self._next_slot += 1
        self._events[event_id] = rec
        self._slots[rec.slot] = rec
        return rec

    def _cell_of(self, x, y):
        return (
            math.floor(x / self.cell_size),
            math.floor(y / self.cell_size),
        )

    def _spatial_candidates(self, x, y, exclude):
        candidates = set()
        cell = self.cell_size
        ix0 = math.floor((x - self.delta) / cell)
        ix1 = math.floor((x + self.delta) / cell)
        iy0 = math.floor((y - self.delta) / cell)
        iy1 = math.floor((y + self.delta) / cell)
        delta2 = self.delta * self.delta
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                for slot in self._grid.get((ix, iy), ()):
                    if slot == exclude:
                        continue
                    self.candidate_checks += 1
                    other = self._slots[slot]
                    dx = other.x - x
                    dy = other.y - y
                    self.distance_checks += 1
                    if dx * dx + dy * dy <= delta2:
                        candidates.add(slot)
        return candidates

    def _temporal_candidates(self, t, exclude):
        candidates = set()
        lo = bisect.bisect_left(self._times, (t - self.tau, -math.inf))
        hi = bisect.bisect_right(self._times, (t + self.tau, math.inf))
        for _, slot in self._times[lo:hi]:
            if slot != exclude:
                candidates.add(slot)
        return candidates

    def _index_add(self, rec):
        cell = self._cell_of(rec.x, rec.y)
        rec.cell = cell
        self._grid.setdefault(cell, set()).add(rec.slot)
        bisect.insort(self._times, (rec.t, rec.slot))

    def _index_remove(self, rec):
        bucket = self._grid[rec.cell]
        bucket.discard(rec.slot)
        if not bucket:
            del self._grid[rec.cell]
        index = bisect.bisect_left(self._times, (rec.t, rec.slot))
        del self._times[index]
        rec.cell = None

    def _attach(self, rec):
        """Find the event's neighbours, link them symmetrically and index it."""
        slot = rec.slot
        spatial = self._spatial_candidates(rec.x, rec.y, slot)
        temporal = self._temporal_candidates(rec.t, slot)
        space_time = spatial & temporal

        for other_slot in spatial:
            other = self._slots[other_slot]
            other.s_neighbors.add(slot)
            rec.s_neighbors.add(other_slot)
        for other_slot in temporal:
            other = self._slots[other_slot]
            other.t_neighbors.add(slot)
            rec.t_neighbors.add(other_slot)
        for other_slot in space_time:
            other = self._slots[other_slot]
            other.st_neighbors.add(slot)
            rec.st_neighbors.add(other_slot)
            self._st_pairs.add(_pair_key(slot, other_slot))

        self._n_s += 2 * len(spatial)
        self._n_t += 2 * len(temporal)
        self._n_st += 2 * len(space_time)
        self._index_add(rec)

    def _detach(self, rec):
        """Retract the event's links, counters and index entries."""
        slot = rec.slot
        for other_slot in rec.s_neighbors:
            self._slots[other_slot].s_neighbors.discard(slot)
        for other_slot in rec.t_neighbors:
            self._slots[other_slot].t_neighbors.discard(slot)
        for other_slot in rec.st_neighbors:
            self._slots[other_slot].st_neighbors.discard(slot)
            self._st_pairs.discard(_pair_key(slot, other_slot))

        self._n_s -= 2 * len(rec.s_neighbors)
        self._n_t -= 2 * len(rec.t_neighbors)
        self._n_st -= 2 * len(rec.st_neighbors)
        rec.s_neighbors.clear()
        rec.t_neighbors.clear()
        rec.st_neighbors.clear()
        self._index_remove(rec)

    def __len__(self):
        return self.n

    def __contains__(self, event_id):
        with self._lock:
            return event_id in self._events


def _pair_key(a, b):
    return (a, b) if a < b else (b, a)
