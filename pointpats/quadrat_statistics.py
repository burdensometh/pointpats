"""
Quadrat statistics for planar point patterns

- First argument everywhere is a NumPy ndarray of point coordinates (n, 2).
- The study window is used to define the grid: if `window` is omitted it is the
  MBB rectangle of the points.
- The grid origin, cell identifiers and clipped cell geometry are computed once,
  from the observed pattern, and reused for every simulated realization.
- Cells are clipped to the study window; cells with zero exposed area are
  excluded from the test (and therefore from the degrees of freedom).
- Expected counts are proportional to each cell's exposed (clipped) area.
- Cell boundary points follow a deterministic, unique ownership rule: a point on
  a shared boundary is assigned to the cell with the smallest ``cell_id``.
- Points outside the study window are handled by an explicit policy and are
  reported on the statistic object.
- Rectangle and hexagon grids supported.
- Plotting is self-contained (matplotlib only): scatters points, draws window,
  draws the clipped grid, annotates counts, and optionally annotates per-cell
  chi-square contributions. Counts, contributions and plots all follow the same
  cell order.

"""  # noqa: E501

__author__ = "Serge Rey, Wei Kang, Hu Shao"
__all__ = ["RectangleM", "HexagonM", "QStatistic"]

import math

import geopandas as gpd
import numpy as np
import scipy.stats
import shapely
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import MultiPolygon, Polygon, box

from .random import poisson

# Relative tolerance used to decide whether a clipped cell has any exposed area.
_AREA_TOL = 1e-12


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _as_points_array(points) -> np.ndarray:
    """
    Normalize input into an (n, 2) float ndarray.

    Accepts:
      - array-like shaped (n, 2)
      - GeoPandas GeoSeries / GeometryArray of Points (and optionally MultiPoints)
      - PointPattern

    Returns:
      - np.ndarray of shape (n, 2)

    Raises:
      - ValueError for empty inputs or non-2D coordinate inputs
      - TypeError for unsupported geometry types
    """
    if isinstance(points, gpd.GeoDataFrame):
        points = points.geometry
    if isinstance(points, gpd.GeoSeries):
        points = points.dropna()
        if not points.geom_type.isin(["Point", "MultiPoint"]).all():
            raise TypeError(
                "GeoSeries must contain Point geometries (optionally MultiPoint). "
                "Got geometry type: "
                f"{getattr(points, 'geom_type', type(points).__name__)}"
            )

        coords = points.get_coordinates()

        if len(coords) == 0:
            raise ValueError("no non-empty Point geometries.")

        return coords.values.astype(float)

    # ---- PointPattern ---
    if hasattr(points, "points"):
        return points.points.values

    # ---- Generic array-like path ----
    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must be a (n, 2) array-like. Got shape {pts.shape}.")
    if pts.size == 0:
        raise ValueError("points is empty; cannot compute mbb.")
    return pts


def _compute_mbb(points: np.ndarray) -> np.ndarray:
    x_min = float(np.min(points[:, 0]))
    y_min = float(np.min(points[:, 1]))
    x_max = float(np.max(points[:, 0]))
    y_max = float(np.max(points[:, 1]))
    return np.array([x_min, y_min, x_max, y_max], dtype=float)


def _ensure_window(window, mbb):
    if window is None:
        return box(mbb[0], mbb[1], mbb[2], mbb[3])
    return window


def _coerce_rng(rng):
    """Coerce rng input to a NumPy Generator.

    Parameters
    ----------
    rng : None | int | numpy.random.Generator | numpy.random.RandomState | numpy.random.BitGenerator
        Random number generator specification.

    Returns
    -------
    numpy.random.Generator
    """  # noqa: E501
    if rng is None:
        return np.random.default_rng()

    if isinstance(rng, np.random.Generator):
        return rng

    # Accept legacy RandomState by deriving a seed from it.
    if isinstance(rng, np.random.RandomState):
        # Draw a seed deterministically from the RandomState stream.
        seed = int(rng.randint(0, 2**32 - 1, dtype=np.uint32))
        return np.random.default_rng(seed)

    # If a BitGenerator is passed, wrap it.
    if isinstance(rng, np.random.BitGenerator):
        return np.random.Generator(rng)

    # Int-like seeds
    if isinstance(rng, (int, np.integer)):
        return np.random.default_rng(int(rng))

    # Fall back to NumPy’s coercion (e.g., SeedSequence)
    return np.random.default_rng(rng)


def _window_to_paths(window_geom):
    """Return list of (x, y) arrays for plotting boundary(ies)."""
    paths = []
    if window_geom is None:
        return paths

    if isinstance(window_geom, Polygon):
        x, y = window_geom.exterior.xy
        paths.append((np.asarray(x), np.asarray(y)))
        for ring in window_geom.interiors:
            x, y = ring.xy
            paths.append((np.asarray(x), np.asarray(y)))
        return paths

    if isinstance(window_geom, MultiPolygon):
        for poly in window_geom.geoms:
            paths.extend(_window_to_paths(poly))
        return paths

    if hasattr(window_geom, "geoms"):
        for g in window_geom.geoms:
            paths.extend(_window_to_paths(g))
    return paths


def _scatter_points(ax, points):
    ax.scatter(points[:, 0], points[:, 1], s=10)


def _draw_window(ax, window_geom):
    for x, y in _window_to_paths(window_geom):
        ax.plot(x, y, lw=1.5)


def _iter_polygon_parts(geom):
    """Yield the polygonal pieces of a (possibly multi-part) geometry."""
    if geom is None or geom.is_empty:
        return
    if isinstance(geom, Polygon):
        yield geom
    elif hasattr(geom, "geoms"):
        for part in geom.geoms:
            yield from _iter_polygon_parts(part)


def _realization_points(realizations, n_realizations):
    """Normalize the output of ``poisson`` into a list of (n, 2) arrays."""
    if isinstance(realizations, (list, tuple)):
        items = list(realizations)
    else:
        array = np.asarray(realizations, dtype=float)
        if array.ndim == 3:
            items = [array[i] for i in range(array.shape[0])]
        elif n_realizations > 1 and array.ndim == 2:
            # ``poisson`` squeezes the singleton observation axis, so R
            # realizations of a single point arrive as an (R, 2) array.
            items = [array[i] for i in range(array.shape[0])]
        else:
            items = [array]

    if len(items) != n_realizations:
        raise ValueError(
            "The simulation returned "
            f"{len(items)} realization(s) but {n_realizations} were requested."
        )

    points = []
    for realization in items:
        pts = getattr(realization, "points", realization)
        points.append(np.asarray(pts, dtype=float).reshape(-1, 2))
    return points


# -----------------------------------------------------------------------------
# Shared grid plan
# -----------------------------------------------------------------------------
class _GridPlan:
    """Grid geometry shared by observations, simulations, counts and plots.

    The plan is built once from the study window. It stores every grid cell
    clipped to the window, the exposed area of each cell, and the (deterministic)
    list of cells that actually take part in the statistic.

    Cells with zero exposed area (entirely outside the window, or touching it
    only along a line/point) are dropped, so they never contribute to counts,
    chi-square contributions, plots or degrees of freedom.

    Parameters
    ----------
    kind : {"rectangle", "hexagon"}
        Tessellation type (informational).
    window : shapely geometry
        Study window.
    cell_ids : sequence of int
        Cell identifiers for every full (unclipped) grid cell.
    cell_polygons : sequence of shapely geometry
        Full grid cell polygons, aligned with `cell_ids`.
    """

    def __init__(self, kind, window, cell_ids, cell_polygons):
        self.kind = kind
        self.window = window

        cell_ids = np.asarray(cell_ids, dtype=np.int64)
        clipped = shapely.intersection(np.asarray(cell_polygons, dtype=object), window)
        areas = shapely.area(clipped)

        # A cell whose intersection with the window has (numerically) no area is
        # a zero-exposure cell: it cannot contain an observable event.
        tol = max(float(window.area), 0.0) * _AREA_TOL
        keep = areas > tol

        self.all_cell_ids = cell_ids.tolist()
        order = np.argsort(cell_ids[keep], kind="stable")
        self.cell_ids = cell_ids[keep][order].tolist()
        self.geometries = [clipped[i] for i in np.flatnonzero(keep)[order]]
        self.exposed_area = np.asarray(areas[keep][order], dtype=float)

        self.total_exposed_area = float(self.exposed_area.sum())
        if self.total_exposed_area > 0.0:
            self.area_fraction = self.exposed_area / self.total_exposed_area
        else:
            self.area_fraction = np.full_like(self.exposed_area, np.nan)

        self._tree = shapely.STRtree(self.geometries) if self.geometries else None
        self._positions = {cid: i for i, cid in enumerate(self.cell_ids)}

    @property
    def n_cells(self):
        return len(self.cell_ids)

    def counts(self, points):
        """Count points per included cell using the shared plan.

        Parameters
        ----------
        points : array_like of shape (n, 2)
            Point coordinates.

        Returns
        -------
        counts : numpy.ndarray of shape (n_cells,)
            Counts aligned with :attr:`cell_ids`.
        outside_mask : numpy.ndarray of bool, shape (n,)
            Points outside the study window.
        unassigned_mask : numpy.ndarray of bool, shape (n,)
            Points inside the window that could not be assigned to any included
            cell. With the nearest-cell fallback this is normally empty; it is
            retained for reporting geometric degeneracies.
        """
        pts = np.asarray(points, dtype=float)
        if pts.size == 0:
            pts = pts.reshape(0, 2)
        else:
            pts = pts.reshape(-1, 2)

        n_points = pts.shape[0]
        counts = np.zeros(self.n_cells, dtype=np.int64)
        if n_points == 0 or self.n_cells == 0:
            empty = np.zeros(n_points, dtype=bool)
            return counts, empty, empty.copy()

        x = pts[:, 0]
        y = pts[:, 1]

        # A point is "in" the window when it lies in the window's interior or on
        # its boundary. Points inside a hole are therefore outside.
        in_window = np.asarray(shapely.intersects_xy(self.window, x, y), dtype=bool)

        # Deterministic unique ownership: among all (clipped) cells whose
        # geometry intersects the point -- including shared boundaries -- take
        # the one with the smallest position in the plan's cell order. Since
        # cell_ids are sorted, this is the smallest cell_id.
        assignment = np.full(n_points, -1, dtype=np.intp)
        if self._tree is not None:
            input_idx, cell_idx = self._tree.query(
                shapely.points(pts), predicate="intersects"
            )
            if input_idx.size:
                sentinel = np.iinfo(np.intp).max
                best = np.full(n_points, sentinel, dtype=np.intp)
                np.minimum.at(best, input_idx, cell_idx)
                found = best != sentinel
                assignment[found] = best[found]

        assigned = in_window & (assignment >= 0)

        # Degenerate fallback: a point may be inside the window yet fall in a
        # gap not covered by any clipped cell (for example, sliver gaps left by
        # the finite hexagon lattice at the window edge). Such a point is still
        # observable, so assign it to the nearest included cell. The rule is
        # deterministic and ties are broken by smallest ``cell_id`` because the
        # plan order is sorted by ``cell_id``.
        leftover = in_window & (assignment < 0)
        if leftover.any() and self._tree is not None:
            nearest = self._tree.nearest(shapely.points(pts[leftover]))
            assignment[leftover] = np.asarray(nearest, dtype=np.intp)
            assigned = in_window & (assignment >= 0)

        if assigned.any():
            counts = np.bincount(assignment[assigned], minlength=self.n_cells)

        outside_mask = ~in_window
        unassigned_mask = in_window & (assignment < 0)
        return counts, outside_mask, unassigned_mask


def _build_plan(kind, window, cells):
    """Build a :class:`_GridPlan` from ``(cell_id, polygon)`` pairs."""
    pairs = sorted(cells, key=lambda item: item[0])
    cell_ids = [cell_id for cell_id, _ in pairs]
    polygons = [poly for _, poly in pairs]
    return _GridPlan(kind, window, cell_ids, polygons)


def _plot_grid(ax, plan, counts=None, chi2_contrib=None, show="counts"):
    """Draw the plan's clipped cells using a single, shared cell order."""
    patches = []
    piece_values = []
    annotations = []

    for position, (cell_id, geom) in enumerate(
        zip(plan.cell_ids, plan.geometries, strict=True)
    ):
        representative = geom.representative_point()
        annotations.append((representative.x, representative.y, position, cell_id))
        for part in _iter_polygon_parts(geom):
            patches.append(MplPolygon(np.asarray(part.exterior.coords), closed=True))
            if show == "chi2":
                piece_values.append(float(chi2_contrib[position]))

    pc = PatchCollection(patches, linewidths=1.0, edgecolor="red", facecolor="none")
    ax.add_collection(pc)

    if show == "counts":
        for cx, cy, position, _cell_id in annotations:
            ax.text(
                cx, cy, str(int(counts[position])), ha="center", va="center"
            )
    else:
        pc.set_array(np.asarray(piece_values, dtype=float))
        pc.set_alpha(0.6)
        pc.set_facecolor(None)
        plt.colorbar(pc, ax=ax, label="Chi-square contribution")
        for cx, cy, position, _cell_id in annotations:
            ax.text(
                cx, cy, f"{chi2_contrib[position]:.2f}", ha="center", va="center"
            )

    return pc


# -----------------------------------------------------------------------------
# Rectangle grid
# -----------------------------------------------------------------------------
class RectangleM:
    """
    Rectangle grid structure for quadrat-based method.

    The grid spans the study window (its bounding box) rather than the points'
    MBB, so the origin and cell layout do not change when a simulated pattern has
    a different MBB. Cells are clipped to the window; zero-exposure cells are
    omitted from the plan used for counting and plotting.
    """

    def __init__(
        self,
        pp,
        count_column=3,
        count_row=3,
        rectangle_width=0,
        rectangle_height=0,
        window=None,
    ):
        self.points = _as_points_array(pp)
        self.mbb = _compute_mbb(self.points)
        self.window = _ensure_window(window, self.mbb)

        x_min, y_min, x_max, y_max = self.window.bounds
        x_range = x_max - x_min
        y_range = y_max - y_min

        if rectangle_width and rectangle_height:
            self.rectangle_width = float(rectangle_width)
            self.rectangle_height = float(rectangle_height)
            self.count_column = max(1, int(math.ceil(x_range / self.rectangle_width)))
            self.count_row = max(1, int(math.ceil(y_range / self.rectangle_height)))
        else:
            self.count_column = int(count_column)
            self.count_row = int(count_row)
            self.rectangle_width = (
                x_range / float(self.count_column) if self.count_column else 0.0
            )
            self.rectangle_height = (
                y_range / float(self.count_row) if self.count_row else 0.0
            )

        self.origin = (x_min, y_min)
        self.num = self.count_column * self.count_row

        cells = []
        for iy in range(self.count_row):
            for ix in range(self.count_column):
                cell_id = ix + iy * self.count_column
                cells.append((cell_id, self._rect_cell_polygon(ix, iy)))
        self.plan = _build_plan("rectangle", self.window, cells)

    def point_location_sta(self):
        """Count points per cell, excluding points outside the study window."""
        counts, _outside, _unassigned = self.plan.counts(self.points)
        return dict(zip(self.plan.cell_ids, counts.tolist(), strict=True))

    def _rect_cell_bounds(self, ix, iy):
        x0 = self.origin[0] + ix * self.rectangle_width
        y0 = self.origin[1] + iy * self.rectangle_height
        x1 = x0 + self.rectangle_width
        y1 = y0 + self.rectangle_height
        return x0, y0, x1, y1

    def _rect_cell_polygon(self, ix, iy):
        x0, y0, x1, y1 = self._rect_cell_bounds(ix, iy)
        return box(x0, y0, x1, y1)

    def _iter_cells(self):
        for cell_id, geom in zip(
            self.plan.cell_ids, self.plan.geometries, strict=True
        ):
            representative = geom.representative_point()
            yield cell_id, geom, (representative.x, representative.y)

    def plot(
        self,
        title="Quadrat Count",
        show="counts",
        chi2_contrib=None,
    ):
        fig, ax = plt.subplots()
        ax.set_title(title)

        _scatter_points(ax, self.points)
        _draw_window(ax, self.window)

        counts, _outside, _unassigned = self.plan.counts(self.points)

        if show == "counts":
            _plot_grid(ax, self.plan, counts=counts, show="counts")
        elif show == "chi2":
            if chi2_contrib is None:
                raise ValueError('chi2_contrib must be provided when show="chi2".')
            chi2_contrib = np.asarray(chi2_contrib, dtype=float)
            if chi2_contrib.shape[0] != self.plan.n_cells:
                raise ValueError(
                    "chi2_contrib length must match number of plotted cells "
                    f"({self.plan.n_cells}). Got {chi2_contrib.shape[0]}."
                )
            _plot_grid(
                ax, self.plan, chi2_contrib=chi2_contrib, show="chi2"
            )
        else:
            raise ValueError('show must be "counts" or "chi2".')

        ax.set_aspect("equal", adjustable="box")
        return ax, list(self.plan.cell_ids)


# -----------------------------------------------------------------------------
# Hexagon grid
# -----------------------------------------------------------------------------
class HexagonM:
    """
    Hexagon grid structure for quadrat-based method.

    As with :class:`RectangleM`, the grid is anchored to the study window's
    bounding box, clipped to the window, and shared by observations and
    simulations. Cell ownership of boundary points is resolved deterministically
    by smallest ``cell_id``.
    """

    def __init__(self, pp, lh, window=None):
        self.points = _as_points_array(pp)
        self.h_length = float(lh)
        self.mbb = _compute_mbb(self.points)
        self.window = _ensure_window(window, self.mbb)

        if self.h_length <= 0:
            raise ValueError("lh must be positive.")

        x_min, y_min, x_max, y_max = self.window.bounds
        range_x = x_max - x_min
        range_y = y_max - y_min

        self.semi_height = self.h_length * math.cos(math.pi / 6.0)
        step_x = 1.5 * self.h_length
        sh = self.semi_height

        # Build a full staggering lattice that covers the window's bounding box,
        # then clip to the window. One extra ring of columns/rows is laid down on
        # each side: because hexagons recede to a vertex at the edge of their
        # column, the extreme columns alone would leave uncovered triangular
        # notches along the window boundary. Extra wholly-outside cells are later
        # dropped as zero-exposure cells.
        n_cols = int(math.ceil(range_x / step_x)) + 2 if range_x > 0 else 1
        n_rows = int(math.ceil(range_y / (2.0 * sh))) + 2 if range_y > 0 else 1
        col_ids = range(-1, n_cols + 1)
        row_ids = range(-1, n_rows + 1)

        self.count_column = len(col_ids)
        self.count_row_even = len(row_ids)
        self.count_row_odd = len(row_ids)

        cells = []
        for ix in col_ids:
            cx = x_min + self.h_length + ix * step_x
            offset = sh if ix % 2 == 0 else 0.0
            for iy in row_ids:
                cy = y_min + offset + iy * 2.0 * sh
                cell_id = (ix + 1) + (iy + 1) * self.count_column
                cells.append((cell_id, Polygon(self._hex_vertices(cx, cy)[:-1])))
        self.num = len(cells)
        self.plan = _build_plan("hexagon", self.window, cells)

    def point_location_sta(self):
        """Count points per cell, excluding points outside the study window."""
        counts, _outside, _unassigned = self.plan.counts(self.points)
        return dict(zip(self.plan.cell_ids, counts.tolist(), strict=True))

    def _hex_vertices(self, cx, cy):
        sh = self.semi_height
        lh = self.h_length
        return np.array(
            [
                [cx + lh, cy],
                [cx + lh / 2.0, cy + sh],
                [cx - lh / 2.0, cy + sh],
                [cx - lh, cy],
                [cx - lh / 2.0, cy - sh],
                [cx + lh / 2.0, cy - sh],
                [cx + lh, cy],
            ],
            dtype=float,
        )

    def _iter_cells(self):
        for cell_id, geom in zip(
            self.plan.cell_ids, self.plan.geometries, strict=True
        ):
            representative = geom.representative_point()
            yield cell_id, geom, (representative.x, representative.y)

    def plot(
        self,
        title="Quadrat Count",
        show="counts",
        chi2_contrib=None,
    ):
        fig, ax = plt.subplots()
        ax.set_title(title)

        _scatter_points(ax, self.points)
        _draw_window(ax, self.window)

        counts, _outside, _unassigned = self.plan.counts(self.points)

        if show == "counts":
            _plot_grid(ax, self.plan, counts=counts, show="counts")
        elif show == "chi2":
            if chi2_contrib is None:
                raise ValueError('chi2_contrib must be provided when show="chi2".')
            chi2_contrib = np.asarray(chi2_contrib, dtype=float)
            if chi2_contrib.shape[0] != self.plan.n_cells:
                raise ValueError(
                    "chi2_contrib length must match number of plotted cells "
                    f"({self.plan.n_cells}). Got {chi2_contrib.shape[0]}."
                )
            _plot_grid(
                ax, self.plan, chi2_contrib=chi2_contrib, show="chi2"
            )
        else:
            raise ValueError('show must be "counts" or "chi2".')

        ax.set_aspect("equal", adjustable="box")
        return ax, list(self.plan.cell_ids)


# -----------------------------------------------------------------------------
# Quadrat statistic
# -----------------------------------------------------------------------------
class QStatistic:
    """Pearson chi-square quadrat test for complete spatial randomness (CSR).

    This class partitions the study window into quadrats (rectangles or
    hexagons), counts the number of events falling in each quadrat, and evaluates
    departure from CSR using a Pearson chi-square statistic under a homogeneous
    Poisson null.

    The grid is defined once, from the observed pattern and the study window, as
    a shared :class:`_GridPlan`:

    - the grid origin is the lower-left corner of the window's bounding box;
    - every grid cell is clipped to the window;
    - each cell's expected count is proportional to its **exposed area** (the
      area of the clipped cell), which matters for windows with holes or several
      parts;
    - cells with zero exposed area are excluded from the statistic, so they do
      not enter the degrees of freedom;
    - the same origin, cell identifiers, clipped geometry and cell order are used
      for the observed pattern, every simulated realization, the chi-square
      contributions and plotting. A simulated pattern's own bounding box is
      never used to build a grid.

    Points on a shared cell boundary are assigned to exactly one cell: the cell
    with the smallest ``cell_id`` among those whose clipped geometry intersects
    the point. Points on the window boundary are counted; points strictly
    outside the window (including inside a hole) are handled by `outside`.

    Parameters
    ----------
    pp : :class: `.PointPattern` or array_like
        Event coordinates as an (n, 2) array-like of floats (x, y).
    shape : {"rectangle", "hexagon"}, default="rectangle"
        Quadrat tessellation type.
    nx : int, default=3
        Number of columns when `shape="rectangle"` and `rectangle_width` /
        `rectangle_height` are not provided.
    ny : int, default=3
        Number of rows when `shape="rectangle"` and `rectangle_width` /
        `rectangle_height` are not provided.
    rectangle_width : float, default=0
        Target rectangle width. Must be provided together with
        `rectangle_height`. When both are provided and non-zero, `nx` and `ny`
        are ignored and the grid dimensions are computed from the window's
        bounding box.
    rectangle_height : float, default=0
        Target rectangle height. Must be provided together with
        `rectangle_width`. When both are provided and non-zero, `nx` and `ny`
        are ignored and the grid dimensions are computed from the window's
        bounding box.
    lh : float, default=10
        Hexagon side length when `shape="hexagon"`.
    realizations : int, default=0
        Number of CSR simulations used to compute a Monte Carlo p-value.
        If 0, no simulation-based inference is performed.
    window : shapely geometry, optional
        Study window (e.g., Polygon or MultiPolygon, possibly with holes). If
        None, the window is the axis-aligned MBB rectangle of `points`.
    outside : {"exclude", "raise"}, default="exclude"
        Policy for points that fall outside the study window (or, in degenerate
        cases, inside the window but outside every included cell). ``"exclude"``
        drops them from the counts and records how many were dropped;
        ``"raise"`` raises a :class:`ValueError`.
    rng : None | int | numpy.random.Generator | numpy.random.RandomState, optional
        Random number generator control for reproducible CSR simulations when
        `realizations > 0`.

        - None: create a new ``numpy.random.default_rng()``
        - int: use as a seed for ``numpy.random.default_rng(seed)``
        - Generator: used as-is
        - RandomState: wrapped via ``numpy.random.default_rng(RandomState)``

    Attributes
    ----------
    points : numpy.ndarray
        (n, 2) array of event coordinates.
    mbb : numpy.ndarray
        Bounding box of `points` as ``[xmin, ymin, xmax, ymax]``.
    window : shapely geometry
        Study window used for the grid, simulation and plotting.
    shape : str
        Quadrat tessellation type: "rectangle" or "hexagon".
    rng : numpy.random.Generator
        RNG used for CSR simulations (always stored as a Generator).
    mr : RectangleM or HexagonM
        Tessellation manager instance (its :attr:`plan` is the shared grid).
    plan : _GridPlan
        Shared grid plan used for observations, simulations and plots.
    cell_ids : list of int
        Identifiers of the included (positive exposed area) cells, sorted
        ascending. Counts, contributions and plots follow this order.
    exposed_area : numpy.ndarray
        Clipped cell area for each included cell.
    area_fraction : numpy.ndarray
        ``exposed_area / total_exposed_area``; the null probability of a point
        falling in each included cell.
    expected_counts : numpy.ndarray
        Expected counts under CSR, ``n * area_fraction``.
    observed_counts : numpy.ndarray
        Observed counts aligned with `cell_ids`.
    df : int
        Degrees of freedom, ``k - 1`` where ``k`` is the number of included
        cells. Zero-exposure cells do not contribute.
    chi2 : float
        Observed Pearson chi-square statistic.
    chi2_pvalue : float
        Analytical p-value from the chi-square distribution.
    chi2_contrib : numpy.ndarray
        Per-cell chi-square contributions aligned with `cell_ids`, computed as
        ``(O - E)^2 / E`` with area-proportional `E`.
    n_outside : int
        Number of observed points excluded by the `outside` policy.
    outside_points : numpy.ndarray
        Observed points excluded by the `outside` policy.
    chi2_realizations : numpy.ndarray
        Simulated chi-square statistics for CSR realizations. Present only when
        `realizations > 0`.
    chi2_r_pvalue : float
        Monte Carlo p-value based on `chi2_realizations`, computed using the
        standard +1 correction:
        ``( #{T_sim >= T_obs} + 1 ) / (realizations + 1)``.
        Present only when `realizations > 0`.
    sim_outside_counts : numpy.ndarray or None
        Number of simulated points excluded per realization (same policy).

    Notes
    -----
    - The analytical test uses ``scipy.stats.chisquare`` with expected counts
      proportional to clipped cell area. For a rectangular window without holes
      this reduces to equal expected counts.
    - The simulation-based null generates CSR realizations within `window` with
      intensity ``n_included / area(window)``. Reproducibility depends on passing
      `rng` through to the underlying CSR generator (``poisson(..., rng=...)``).
    - Realizations are counted with the *same* plan, so a realization with a
      different sample bounding box cannot shift the grid origin or cell ids.
      Each realization's statistic uses its own point total with the same
      area-proportional expected counts.

    See Also
    --------
    RectangleM : Rectangular tessellation clipped to the study window.
    HexagonM : Hexagonal tessellation clipped to the study window.

    """

    def __init__(
        self,
        pp,
        shape="rectangle",
        nx=3,
        ny=3,
        rectangle_width=0,
        rectangle_height=0,
        lh=10,
        realizations=0,
        window=None,
        outside="exclude",
        rng=None,
    ):
        if outside not in ("exclude", "raise"):
            raise ValueError('outside must be either "exclude" or "raise".')

        self.points = _as_points_array(pp)
        self.mbb = _compute_mbb(self.points)
        self.window = _ensure_window(window, self.mbb)
        self.shape = shape
        self.rng = _coerce_rng(rng)
        self.outside_policy = outside

        if shape == "rectangle":
            self.mr = RectangleM(
                self.points,
                count_column=nx,
                count_row=ny,
                rectangle_width=rectangle_width,
                rectangle_height=rectangle_height,
                window=self.window,
            )
        elif shape == "hexagon":
            self.mr = HexagonM(self.points, lh, window=self.window)
        else:
            raise ValueError('shape must be either "rectangle" or "hexagon".')

        # The single grid plan shared by observations, simulations and plots.
        self.plan = self.mr.plan
        if self.plan.n_cells == 0 or self.plan.total_exposed_area <= 0:
            raise ValueError(
                "The grid has no cells with positive exposed area inside the "
                "study window; cannot compute a quadrat statistic."
            )

        self.cell_ids = list(self.plan.cell_ids)
        self.exposed_area = np.asarray(self.plan.exposed_area, dtype=float)
        self.area_fraction = np.asarray(self.plan.area_fraction, dtype=float)
        self.total_exposed_area = float(self.plan.total_exposed_area)

        obs_counts, outside_mask, unassigned_mask = self.plan.counts(self.points)
        excluded = outside_mask | unassigned_mask
        self.n_outside = int(excluded.sum())
        self.outside_points = self.points[excluded]
        if self.n_outside:
            if outside == "raise":
                raise ValueError(
                    f"{self.n_outside} observed point(s) fall outside the study "
                    "window or are not covered by the grid (outside='raise')."
                )

        self.observed_counts = np.asarray(obs_counts, dtype=float)
        self.n_included = int(self.observed_counts.sum())

        self.expected_counts = self.n_included * self.area_fraction
        self.chi2_contrib = np.full(self.plan.n_cells, np.nan, dtype=float)
        positive = self.expected_counts > 0
        self.chi2_contrib[positive] = (
            (self.observed_counts[positive] - self.expected_counts[positive]) ** 2
            / self.expected_counts[positive]
        )

        self.df = self.plan.n_cells - 1
        if self.n_included > 0 and self.df >= 1:
            self.chi2, self.chi2_pvalue = scipy.stats.chisquare(
                self.observed_counts, f_exp=self.expected_counts
            )
        else:
            self.chi2 = float(np.nansum(self.chi2_contrib))
            self.chi2_pvalue = float("nan")
            if self.df < 0:
                self.df = 0

        # simulation-based inference under CSR (reproducible via rng)
        self.sim_outside_counts = None
        if realizations and realizations > 0:
            self._simulate(realizations)

    def _simulate(self, realizations):
        if self.n_included == 0:
            raise ValueError("Cannot simulate: no observed points inside the window.")
        if self.window.area <= 0:
            raise ValueError("Cannot simulate: the study window has zero area.")

        intensity = self.n_included / float(self.window.area)
        simulated = poisson(self.window, intensity, realizations, rng=self.rng)
        realizations_points = _realization_points(simulated, realizations)

        chi2_realizations = []
        sim_outside_counts = []

        for pts_i in realizations_points:
            counts_i, outside_i, unassigned_i = self.plan.counts(pts_i)
            n_excluded = int(outside_i.sum() + unassigned_i.sum())
            sim_outside_counts.append(n_excluded)
            if n_excluded and self.outside_policy == "raise":
                raise ValueError(
                    f"{n_excluded} simulated point(s) fall outside the study "
                    "window or are not covered by the grid (outside='raise')."
                )

            counts_i = counts_i.astype(float)
            n_i = counts_i.sum()
            if n_i > 0 and self.plan.n_cells >= 1:
                expected_i = n_i * self.area_fraction
                chi2_i, _p = scipy.stats.chisquare(counts_i, f_exp=expected_i)
            else:
                chi2_i = 0.0
            chi2_realizations.append(chi2_i)

        self.chi2_realizations = np.asarray(chi2_realizations, dtype=float)
        self.sim_outside_counts = np.asarray(sim_outside_counts, dtype=int)
        self.chi2_r_pvalue = (np.sum(self.chi2_realizations >= self.chi2) + 1.0) / (
            realizations + 1.0
        )

    def plot(self, title="Quadrat Count", show="counts"):
        if show == "counts":
            ax, _cell_ids = self.mr.plot(title=title, show="counts")
            return ax

        if show == "chi2":
            # Contributions are already aligned with the shared plan order,
            # which is exactly the order used by ``mr.plot``.
            ax, _cell_ids = self.mr.plot(
                title=title, show="chi2", chi2_contrib=self.chi2_contrib
            )
            return ax

        raise ValueError('show must be either "counts" or "chi2".')
