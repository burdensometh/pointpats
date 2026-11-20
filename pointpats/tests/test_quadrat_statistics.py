import math

import geopandas as gpd
import numpy as np
import pytest
import scipy.stats
from shapely.geometry import MultiPoint, MultiPolygon, Point, Polygon, box

import pointpats
import pointpats.quadrat_statistics as m


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def pointpattern():
    """
    Return a simple pointpats.PointPattern with 3 points.
    """
    pts = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
        ]
    )
    return pointpats.PointPattern(pts)


@pytest.fixture
def pts_simple():
    # Four points spanning a 2x2 square (mbb = [0,0,2,2])
    return np.array(
        [
            [0.0, 0.0],
            [0.5, 0.5],
            [1.5, 1.5],
            [2.0, 2.0],  # on max boundary
        ],
        dtype=float,
    )


@pytest.fixture
def pts_random_small():
    rng = np.random.default_rng(123)
    return rng.random((50, 2), dtype=float) * 100.0


@pytest.fixture
def window_poly():
    # Simple rectangular polygon window
    return box(0.0, 0.0, 10.0, 5.0)


@pytest.fixture
def window_poly_with_hole():
    outer = box(0.0, 0.0, 10.0, 10.0)
    hole = box(4.0, 4.0, 6.0, 6.0)
    return Polygon(outer.exterior.coords, holes=[hole.exterior.coords])


@pytest.fixture
def window_multipoly():
    a = box(0.0, 0.0, 1.0, 1.0)
    b = box(2.0, 2.0, 3.0, 3.0)
    return MultiPolygon([a, b])


# -----------------------------------------------------------------------------
# Helper tests
# -----------------------------------------------------------------------------
def test_as_points_array_accepts_arraylike():
    pts = m._as_points_array([[0, 1], [2, 3.5]])
    assert isinstance(pts, np.ndarray)
    assert pts.shape == (2, 2)
    assert pts.dtype.kind in ("f", "i")  # will often be float, but allow int input


def test_as_points_array_rejects_bad_shape():
    with pytest.raises(ValueError, match=r"points must be a \(n, 2\)"):
        m._as_points_array([1, 2, 3])

    with pytest.raises(ValueError, match=r"points must be a \(n, 2\)"):
        m._as_points_array(np.zeros((3, 3)))


def test_as_points_array_rejects_empty():
    with pytest.raises(ValueError, match="points is empty"):
        m._as_points_array(np.empty((0, 2)))


def test_compute_mbb():
    pts = np.array([[1, 2], [3, 4], [-1, 10]], dtype=float)
    mbb = m._compute_mbb(pts)
    assert mbb.shape == (4,)
    assert np.allclose(mbb, [-1.0, 2.0, 3.0, 10.0])


def test_ensure_window_none_uses_box():
    mbb = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    w = m._ensure_window(None, mbb)
    assert w.equals(box(0.0, 1.0, 2.0, 3.0))


def test_coerce_rng_none():
    rng = m._coerce_rng(None)
    assert isinstance(rng, np.random.Generator)


def test_coerce_rng_int():
    rng = m._coerce_rng(123)
    assert isinstance(rng, np.random.Generator)

    # Determinism check
    rng2 = m._coerce_rng(123)
    assert rng.random() == rng2.random()


def test_coerce_rng_numpy_integer():
    rng = m._coerce_rng(np.int64(456))
    assert isinstance(rng, np.random.Generator)


def test_coerce_rng_generator_passthrough():
    gen = np.random.default_rng(789)
    out = m._coerce_rng(gen)
    assert out is gen


def test_coerce_rng_bitgenerator():
    bitgen = np.random.PCG64(123)
    rng = m._coerce_rng(bitgen)
    assert isinstance(rng, np.random.Generator)
    assert rng.bit_generator is bitgen


def test_coerce_rng_randomstate():
    rs = np.random.RandomState(321)
    rng = m._coerce_rng(rs)

    assert isinstance(rng, np.random.Generator)

    # Reproduce the seed used by _coerce_rng: it is the FIRST randint draw from RS(321).
    rs2 = np.random.RandomState(321)
    seed = int(rs2.randint(0, 2**32 - 1, dtype=np.uint32))

    expected = np.random.default_rng(seed)

    assert rng.random() == expected.random()


def test_coerce_rng_seedsequence():
    ss = np.random.SeedSequence(999)
    rng = m._coerce_rng(ss)
    assert isinstance(rng, np.random.Generator)


def test_window_to_paths_polygon_and_hole(window_poly_with_hole):
    paths = m._window_to_paths(window_poly_with_hole)
    # exterior + 1 hole -> 2 paths
    assert len(paths) == 2
    for x, y in paths:
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert x.shape == y.shape
        assert x.ndim == 1


def test_window_to_paths_multipolygon(window_multipoly):
    paths = m._window_to_paths(window_multipoly)
    assert len(paths) == 2  # two polygons -> two exterior rings


@pytest.mark.skipif(
    pytest.importorskip("geopandas", reason="geopandas not installed") is None,
    reason="geopandas not installed",
)
def test_as_points_array_geoseries_points_and_multipoints():
    import geopandas as gpd

    s = gpd.GeoSeries([Point(0, 0), MultiPoint([Point(1, 1), Point(2, 2)]), None])
    pts = m._as_points_array(s)
    assert pts.shape == (3, 2)
    assert np.allclose(pts, np.array([[0, 0], [1, 1], [2, 2]], dtype=float))


@pytest.mark.skipif(
    pytest.importorskip("geopandas", reason="geopandas not installed") is None,
    reason="geopandas not installed",
)
def test_as_points_array_geoseries_rejects_non_point_geoms():
    import geopandas as gpd

    poly = box(0, 0, 1, 1)
    s = gpd.GeoSeries([poly])
    with pytest.raises(TypeError, match="GeoSeries must contain Point geometries"):
        m._as_points_array(s)


@pytest.mark.skipif(
    pytest.importorskip("geopandas", reason="geopandas not installed") is None,
    reason="geopandas not installed",
)
def test_as_points_array_geoseries_all_empty_errors():
    import geopandas as gpd

    s = gpd.GeoSeries([None, Point()])
    # Point() is empty in shapely
    with pytest.raises(ValueError, match="no non-empty Point geometries"):
        m._as_points_array(s)


def test_as_points_array_pointpattern(pointpattern):
    assert isinstance(m._as_points_array(pointpattern), np.ndarray)


# -----------------------------------------------------------------------------
# RectangleM tests
# -----------------------------------------------------------------------------
def test_rectangle_grid_default_dimensions(pts_simple):
    r = m.RectangleM(pts_simple, count_column=2, count_row=2)
    assert r.count_column == 2
    assert r.count_row == 2
    assert r.num == 4
    assert np.allclose(r.mbb, [0.0, 0.0, 2.0, 2.0])
    # cell size should be 1x1
    assert math.isclose(r.rectangle_width, 1.0)
    assert math.isclose(r.rectangle_height, 1.0)


def test_rectangle_grid_width_height_override(pts_simple):
    # range_x=2, range_y=2, width=0.75 -> ceil(2/0.75)=3 columns
    r = m.RectangleM(pts_simple, rectangle_width=0.75, rectangle_height=1.0)
    assert r.count_column == 3
    assert r.count_row == 2
    assert r.num == 6


def test_rectangle_point_location_counts_sum_to_n(pts_random_small):
    r = m.RectangleM(pts_random_small, count_column=5, count_row=4)
    d = r.point_location_sta()
    assert len(d) == r.num
    assert sum(d.values()) == pts_random_small.shape[0]


def test_rectangle_boundary_point_in_last_cell(pts_simple):
    # The point [2,2] lies on max boundary; code clamps index==count to last cell
    r = m.RectangleM(pts_simple, count_column=2, count_row=2)
    d = r.point_location_sta()
    # bottom-left cell id 0 contains [0,0] and [0.5,0.5]
    assert d[0] == 2
    # top-right cell id 3 contains [1.5,1.5] and [2,2]
    assert d[3] == 2


def test_rectangle_plot_counts_smoke(pts_simple):
    r = m.RectangleM(pts_simple, count_column=2, count_row=2)
    ax, cell_ids = r.plot(show="counts")
    assert ax is not None
    assert len(cell_ids) == r.num


def test_rectangle_plot_chi2_requires_contrib(pts_simple):
    r = m.RectangleM(pts_simple, count_column=2, count_row=2)
    with pytest.raises(ValueError, match="chi2_contrib must be provided"):
        r.plot(show="chi2", chi2_contrib=None)


def test_rectangle_plot_chi2_length_mismatch(pts_simple):
    r = m.RectangleM(pts_simple, count_column=2, count_row=2)
    with pytest.raises(ValueError, match="length must match number of plotted cells"):
        r.plot(show="chi2", chi2_contrib=np.zeros(r.num - 1))


def test_rectangle_plot_invalid_show(pts_simple):
    r = m.RectangleM(pts_simple, count_column=2, count_row=2)
    with pytest.raises(ValueError, match='show must be "counts" or "chi2"'):
        r.plot(show="nope")


# -----------------------------------------------------------------------------
# HexagonM tests
# -----------------------------------------------------------------------------
def test_hexagon_grid_basic_invariants(pts_random_small):
    h = m.HexagonM(pts_random_small, lh=10.0)
    assert h.h_length == 10.0
    assert h.semi_height > 0
    assert h.count_column >= 1
    assert h.count_row_even >= 1
    assert h.count_row_odd >= 1
    assert h.num >= 1


def test_hexagon_point_location_counts_sum_to_n(pts_random_small):
    h = m.HexagonM(pts_random_small, lh=10.0)
    d = h.point_location_sta()
    assert sum(d.values()) == pts_random_small.shape[0]
    # dict keys correspond to the number of cells included
    assert len(d) >= 1


def test_hexagon_plot_counts_smoke(pts_simple):
    h = m.HexagonM(pts_simple, lh=1.0)
    ax, cell_ids = h.plot(show="counts")
    assert ax is not None
    assert len(cell_ids) == len(h.point_location_sta())


def test_hexagon_plot_chi2_requires_contrib(pts_simple):
    h = m.HexagonM(pts_simple, lh=1.0)
    with pytest.raises(ValueError, match="chi2_contrib must be provided"):
        h.plot(show="chi2", chi2_contrib=None)


def test_hexagon_plot_chi2_length_mismatch(pts_simple):
    h = m.HexagonM(pts_simple, lh=1.0)
    k = len(h.point_location_sta())
    with pytest.raises(ValueError, match="length must match number of plotted cells"):
        h.plot(show="chi2", chi2_contrib=np.zeros(k - 1))


def test_hexagon_plot_invalid_show(pts_simple):
    h = m.HexagonM(pts_simple, lh=1.0)
    with pytest.raises(ValueError, match='show must be "counts" or "chi2"'):
        h.plot(show="nope")


# -----------------------------------------------------------------------------
# QStatistic tests (analytical)
# -----------------------------------------------------------------------------
def test_qstatistic_rectangle_matches_scipy(pts_simple):
    qs = m.QStatistic(pts_simple, shape="rectangle", nx=2, ny=2, realizations=0)
    d = qs.mr.point_location_sta()
    obs = np.asarray(list(d.values()), dtype=float)

    chi2_ref, p_ref = scipy.stats.chisquare(obs)
    assert math.isclose(qs.chi2, chi2_ref, rel_tol=1e-12, abs_tol=0.0)
    assert math.isclose(qs.chi2_pvalue, p_ref, rel_tol=1e-12, abs_tol=0.0)
    assert qs.df == obs.size - 1
    assert len(qs.cell_ids) == obs.size


def test_qstatistic_hexagon_basic(pts_random_small):
    qs = m.QStatistic(pts_random_small, shape="hexagon", lh=10.0, realizations=0)
    assert qs.shape == "hexagon"
    assert qs.df == len(qs.cell_ids) - 1
    assert np.isfinite(qs.chi2)
    assert 0.0 <= qs.chi2_pvalue <= 1.0


def test_qstatistic_contrib_formula(pts_simple):
    qs = m.QStatistic(pts_simple, shape="rectangle", nx=2, ny=2, realizations=0)
    d = qs.mr.point_location_sta()
    obs = np.asarray(list(d.values()), dtype=float)
    expected = obs.mean()
    contrib_ref = (
        (obs - expected) ** 2 / expected if expected > 0 else np.full_like(obs, np.nan)
    )
    assert np.allclose(qs.chi2_contrib, contrib_ref, equal_nan=True)


def test_qstatistics_geopandas_input(pts_simple):
    gs = gpd.GeoSeries.from_xy(pts_simple[:, 0], pts_simple[:, 1])
    gdf = gs.to_frame("geometry")
    qs_gs = m.QStatistic(gs, shape="rectangle", nx=2, ny=2, realizations=0)
    qs_gdf = m.QStatistic(gdf, shape="rectangle", nx=2, ny=2, realizations=0)

    for qs in [qs_gs, qs_gdf]:
        d = qs.mr.point_location_sta()
        obs = np.asarray(list(d.values()), dtype=float)

        chi2_ref, p_ref = scipy.stats.chisquare(obs)
        assert math.isclose(qs.chi2, chi2_ref, rel_tol=1e-12, abs_tol=0.0)
        assert math.isclose(qs.chi2_pvalue, p_ref, rel_tol=1e-12, abs_tol=0.0)
        assert qs.df == obs.size - 1
        assert len(qs.cell_ids) == obs.size


def test_qstatistic_invalid_shape_raises(pts_simple):
    with pytest.raises(
        ValueError, match='shape must be either "rectangle" or "hexagon"'
    ):
        m.QStatistic(pts_simple, shape="triangle")


def test_qstatistic_plot_counts_smoke(pts_simple):
    qs = m.QStatistic(pts_simple, shape="rectangle", nx=2, ny=2, realizations=0)
    ax = qs.plot(show="counts")
    assert ax is not None


def test_qstatistic_plot_chi2_smoke(pts_simple):
    qs = m.QStatistic(pts_simple, shape="rectangle", nx=2, ny=2, realizations=0)
    ax = qs.plot(show="chi2")
    assert ax is not None


def test_qstatistic_plot_invalid_show(pts_simple):
    qs = m.QStatistic(pts_simple, shape="rectangle", nx=2, ny=2, realizations=0)
    with pytest.raises(ValueError, match='show must be either "counts" or "chi2"'):
        qs.plot(show="nope")


# -----------------------------------------------------------------------------
# QStatistic tests (simulation branch via poisson stub)
# -----------------------------------------------------------------------------
class _PoissonReturn:
    """Mimic an object with a `.points` attribute."""

    def __init__(self, points):
        self.points = points


def test_qstatistic_simulation_branch_calls_poisson_and_sets_outputs(
    monkeypatch, window_poly
):
    # Make points fit inside the window for clarity
    pts = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 2.0], [4.0, 4.0]], dtype=float)

    calls = {}

    def poisson_stub(window, intensity, realizations, rng=None):
        # record call args for assertions
        calls["window"] = window
        calls["intensity"] = intensity
        calls["realizations"] = realizations
        calls["rng_is_generator"] = isinstance(rng, np.random.Generator)

        # deterministic realizations: identical copies (chi2_sim == chi2_obs)
        return [_PoissonReturn(pts.copy()) for _ in range(realizations)]

    # Patch the imported poisson symbol in the module under test
    monkeypatch.setattr(m, "poisson", poisson_stub)

    qs = m.QStatistic(
        pts,
        shape="rectangle",
        nx=2,
        ny=2,
        realizations=5,
        window=window_poly,
        rng=123,
    )

    assert calls["realizations"] == 5
    assert calls["rng_is_generator"] is True
    assert math.isclose(
        calls["intensity"], pts.shape[0] / window_poly.area, rel_tol=1e-12
    )

    assert hasattr(qs, "chi2_realizations")
    assert qs.chi2_realizations.shape == (5,)
    # Because stub returns same points each time, simulated chi2 equals observed chi2.
    assert np.allclose(qs.chi2_realizations, qs.chi2)

    # With chi2_sim == chi2_obs, #{>=} == R, so p = (R+1)/(R+1) = 1
    assert math.isclose(qs.chi2_r_pvalue, 1.0, rel_tol=0.0, abs_tol=0.0)


def test_qstatistic_simulation_accepts_realizations_object_without_points_attr(
    monkeypatch, window_poly
):
    pts = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 2.0], [4.0, 4.0]], dtype=float)

    def poisson_stub(window, intensity, realizations, rng=None):  # noqa: ARG001
        # Return raw arrays (code path uses getattr(ri, "points", ri))
        return [pts.copy() for _ in range(realizations)]

    monkeypatch.setattr(m, "poisson", poisson_stub)

    qs = m.QStatistic(
        pts,
        shape="rectangle",
        nx=2,
        ny=2,
        realizations=3,
        window=window_poly,
        rng=np.random.default_rng(0),
    )
    assert qs.chi2_realizations.shape == (3,)
    assert 0.0 <= qs.chi2_r_pvalue <= 1.0


# -----------------------------------------------------------------------------
# Shared grid plan, area-proportional expectations and explicit policies
# -----------------------------------------------------------------------------
def test_rectangle_grid_origin_uses_window_not_points(window_poly):
    # Points span only a small part of the window; the grid must still be
    # anchored to the window's lower-left corner with the window's extent.
    pts = np.array([[2.0, 2.0], [3.0, 3.0]], dtype=float)
    r = m.RectangleM(pts, count_column=2, count_row=2, window=window_poly)
    assert np.allclose(r.origin, [0.0, 0.0])
    assert math.isclose(r.rectangle_width, 5.0)
    assert math.isclose(r.rectangle_height, 2.5)
    assert np.allclose(r.plan.total_exposed_area, window_poly.area)


def test_rectangle_zero_exposure_cells_are_excluded(window_poly_with_hole):
    # 10x10 window with a 2x2 hole -> area 96. A 10x10 grid gives 100 raw
    # cells, but only 96 have positive exposed area.
    pts = np.array([[1.0, 1.0], [5.0, 5.0], [9.0, 9.0]], dtype=float)
    r = m.RectangleM(pts, count_column=10, count_row=10, window=window_poly_with_hole)
    assert r.num == 100
    assert r.plan.n_cells == 96
    assert math.isclose(r.plan.total_exposed_area, window_poly_with_hole.area)
    # The four hole cells (ids for x,y in [4,6]) must not be in the plan.
    for iy in (4, 5):
        for ix in (4, 5):
            assert ix + iy * 10 not in r.plan.cell_ids


def test_rectangle_expected_counts_proportional_to_exposed_area(window_multipoly):
    pts = np.array([[0.5, 0.5], [2.5, 2.5], [0.1, 0.1], [2.9, 2.9]], dtype=float)
    qs = m.QStatistic(
        pts, shape="rectangle", nx=2, ny=2, window=window_multipoly, realizations=0
    )
    # Each part is 1x1; with a 2x2 grid over the 3x3 bounding box there are
    # four 1.5x1.5 cells but only those overlapping the two parts are kept.
    assert qs.plan.n_cells < 4
    assert np.allclose(
        qs.expected_counts.sum(), qs.n_included, equal_nan=True
    )
    assert np.allclose(qs.area_fraction.sum(), 1.0)
    assert qs.df == qs.plan.n_cells - 1


def test_observed_contributions_sum_to_chi2(window_poly_with_hole):
    rng = np.random.default_rng(2)
    pts = rng.random((40, 2)) * 10.0
    qs = m.QStatistic(pts, nx=4, ny=4, window=window_poly_with_hole, realizations=0)
    assert math.isclose(np.nansum(qs.chi2_contrib), qs.chi2, rel_tol=1e-12)


def test_boundary_points_assigned_to_smallest_cell_id():
    # All three points lie on shared cell boundaries of a 2x2 grid.
    pts = np.array([[1.0, 0.5], [0.5, 1.0], [1.0, 1.0]], dtype=float)
    r = m.RectangleM(pts, count_column=2, count_row=2, window=box(0, 0, 2, 2))
    counts = r.point_location_sta()
    assert counts[0] == 3  # smallest cell_id wins at each shared boundary
    assert sum(counts.values()) == 3


def test_outside_points_excluded_and_reported(window_poly):
    pts = np.array([[1.0, 1.0], [-1.0, 2.0], [11.0, 2.0]], dtype=float)
    qs = m.QStatistic(
        pts, nx=2, ny=2, window=window_poly, outside="exclude", realizations=0
    )
    assert qs.n_outside == 2
    assert qs.outside_points.shape == (2, 2)
    assert qs.n_included == 1
    assert np.allclose(qs.expected_counts.sum(), 1.0)


def test_outside_points_raise_policy(window_poly):
    pts = np.array([[1.0, 1.0], [-1.0, 2.0]], dtype=float)
    with pytest.raises(ValueError, match="outside='raise'"):
        m.QStatistic(pts, nx=2, ny=2, window=window_poly, outside="raise")


def test_invalid_outside_policy_raises(pts_simple):
    with pytest.raises(ValueError, match='outside must be either "exclude" or "raise"'):
        m.QStatistic(pts_simple, nx=2, ny=2, outside="ignore")


def test_point_inside_hole_is_outside(window_poly_with_hole):
    pts = np.array([[1.0, 1.0], [5.0, 5.0]], dtype=float)
    qs = m.QStatistic(
        pts, nx=2, ny=2, window=window_poly_with_hole, outside="exclude"
    )
    assert qs.n_outside == 1
    assert np.allclose(qs.outside_points, [[5.0, 5.0]])


def test_hexagon_exposed_area_covers_window(window_poly_with_hole):
    rng = np.random.default_rng(5)
    pts = rng.random((30, 2)) * 10.0
    h = m.HexagonM(pts, lh=1.5, window=window_poly_with_hole)
    assert math.isclose(
        h.plan.total_exposed_area, window_poly_with_hole.area, rel_tol=1e-9
    )
    in_hole = (pts[:, 0] > 4) & (pts[:, 0] < 6) & (pts[:, 1] > 4) & (pts[:, 1] < 6)
    counts, outside, unassigned = h.plan.counts(pts)
    assert outside.sum() == in_hole.sum()
    assert unassigned.sum() == 0
    assert counts.sum() == (~in_hole).sum()


def test_hexagon_cell_order_matches_plot_and_contributions(window_poly):
    rng = np.random.default_rng(6)
    pts = rng.random((25, 2)) * np.array([10.0, 5.0])
    qs = m.QStatistic(pts, shape="hexagon", lh=1.5, window=window_poly, realizations=0)
    ax, plotted_ids = qs.mr.plot(show="counts")
    assert plotted_ids == qs.cell_ids
    assert len(qs.chi2_contrib) == len(qs.cell_ids)


def test_simulation_reuses_observed_grid(monkeypatch, window_poly):
    rng = np.random.default_rng(0)
    pts = rng.random((20, 2)) * np.array([10.0, 5.0])

    class _Real:
        def __init__(self, points):
            self.points = points

    # Each realization has a deliberately tiny sample bounding box that would
    # produce a different grid if it were rebuilt from the points.
    def poisson_stub(window, intensity, realizations, rng=None):  # noqa: ARG001
        sim = np.array([[4.0, 2.0], [4.5, 2.5]], dtype=float)
        return [_Real(sim.copy()) for _ in range(realizations)]

    monkeypatch.setattr(m, "poisson", poisson_stub)
    qs = m.QStatistic(pts, nx=3, ny=2, window=window_poly, realizations=4, rng=0)

    assert qs.mr.plan.cell_ids == qs.cell_ids
    # The simulated statistics are all equal because the shared plan, not each
    # sample's bounding box, defines the counts.
    assert np.allclose(qs.chi2_realizations, qs.chi2_realizations[0])
    # All simulated points are inside the window.
    assert np.all(qs.sim_outside_counts == 0)


def test_simulation_expected_counts_are_area_weighted(monkeypatch, window_poly_with_hole):
    pts = np.array([[1.0, 1.0], [9.0, 9.0], [2.0, 8.0]], dtype=float)

    def poisson_stub(window, intensity, realizations, rng=None):  # noqa: ARG001
        # Deterministic realization with a known count distribution.
        sim = np.array([[1.0, 1.0], [9.0, 9.0], [1.0, 9.0], [9.0, 1.0]], dtype=float)
        return [sim.copy() for _ in range(realizations)]

    monkeypatch.setattr(m, "poisson", poisson_stub)
    qs = m.QStatistic(
        pts, nx=4, ny=4, window=window_poly_with_hole, realizations=2, rng=0
    )
    sim = np.array([[1.0, 1.0], [9.0, 9.0], [1.0, 9.0], [9.0, 1.0]], dtype=float)
    counts, _o, _u = qs.plan.counts(sim)
    expected = counts.sum() * qs.area_fraction
    chi2_ref, _p = scipy.stats.chisquare(counts, f_exp=expected)
    assert np.allclose(qs.chi2_realizations, chi2_ref)
