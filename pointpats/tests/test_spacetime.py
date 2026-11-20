import geopandas as gpd
import libpysal as lps
import matplotlib.pyplot as plt
import numpy
import pandas as pd
import pytest
from pytest import approx
from shapely import LineString

from pointpats import (
    Knox,
    KnoxLocal,
    SpaceTimeEvents,
    jacquez,
    knox,
    mantel,
    modified_knox,
)
from pointpats.spacetime import _spacetime_points_to_arrays


class TestKnox:
    def setup_method(self):
        path = lps.examples.get_path("burkitt.shp")
        self.gdf = gpd.read_file(path)

    def test_knox(self):
        global_knox = Knox(self.gdf[["X", "Y"]], self.gdf[["T"]], delta=20, tau=5)
        assert global_knox.statistic_ == 13
        assert global_knox.p_poisson == 0.14624558197140414
        assert not hasattr(global_knox, "sim")
        numpy.testing.assert_array_equal(
            global_knox.observed, [[1.300e01, 3.416e03], [3.900e01, 1.411e04]]
        )
        numpy.testing.assert_allclose(
            global_knox.expected,
            [[1.01438161e01, 3.41885618e03], [4.1856139e01, 1.41071438e04]],
            rtol=1e-5,
            atol=0,
        )

        numpy.random.seed(12345)
        global_knox = Knox(
            self.gdf[["X", "Y"]], self.gdf[["T"]], delta=20, tau=5, keep=True
        )
        assert global_knox.statistic_ == 13
        assert hasattr(global_knox, "sim")
        assert global_knox.p_sim == 0.21

    def test_knox_from_gdf(self):
        gdf = self.gdf.copy()
        # not technically the correct CRS...
        gdf = gdf.set_crs(21096)
        global_knox = Knox.from_dataframe(gdf, time_col="T", delta=20, tau=5)
        assert global_knox.statistic_ == 13
        assert global_knox.p_poisson == 0.14624558197140414
        assert not hasattr(global_knox, "sim")
        numpy.testing.assert_array_equal(
            global_knox.observed, [[1.300e01, 3.416e03], [3.900e01, 1.411e04]]
        )
        numpy.testing.assert_allclose(
            global_knox.expected,
            [[1.01438161e01, 3.41885618e03], [4.1856139e01, 1.41071438e04]],
            rtol=1e-5,
            atol=0,
        )

        # no CRS should raise a warning
        with pytest.warns(UserWarning, match="There is no CRS set on the dataframe"):
            global_knox = Knox.from_dataframe(self.gdf, time_col="T", delta=20, tau=5)
            numpy.testing.assert_allclose(
                global_knox.expected,
                [[1.01438161e01, 3.41885618e03], [4.1856139e01, 1.41071438e04]],
                rtol=1e-5,
                atol=0,
            )

        # unprojected coords
        with pytest.raises(
            ValueError,
            match="The input dataframe must be in a projected coordinate system",
        ):
            _gdf = gdf.copy().to_crs(4326)
            Knox.from_dataframe(_gdf, time_col="T", delta=20, tau=5)

        # non-numeric type for time
        with pytest.raises(
            ValueError,
            match="The time values must be stored as a numeric dtype",
        ):
            gdf["T"] = gdf["T"].astype(str)
            Knox.from_dataframe(gdf, time_col="T", delta=20, tau=5)

        numpy.random.seed(12345)
        global_knox = Knox(
            self.gdf[["X", "Y"]], self.gdf[["T"]], delta=20, tau=5, keep=True
        )
        assert global_knox.statistic_ == 13
        assert hasattr(global_knox, "sim")
        assert global_knox.p_sim == 0.21


class TestKnoxLocal:
    def setup_method(self):
        path = lps.examples.get_path("burkitt.shp")
        self.gdf = gpd.read_file(path)

    def test_knox_local(self):
        numpy.random.seed(12345)
        local_knox = KnoxLocal(
            self.gdf[["X", "Y"]].values,
            self.gdf[["T"]].values,
            delta=20,
            tau=5,
            keep=True,
        )
        assert local_knox.statistic_.shape == (188,)
        lres = local_knox
        gt0ids = numpy.where(lres.nsti > 0)
        numpy.testing.assert_array_equal(
            gt0ids,
            [
                [
                    25,
                    26,
                    30,
                    31,
                    35,
                    36,
                    41,
                    42,
                    46,
                    47,
                    51,
                    52,
                    102,
                    103,
                    116,
                    118,
                    122,
                    123,
                    137,
                    138,
                    139,
                    140,
                    158,
                    159,
                    162,
                    163,
                ]
            ],
        )
        numpy.testing.assert_allclose(
            lres.p_hypergeom[gt0ids],
            [
                0.1348993,
                0.14220663,
                0.07335085,
                0.08400282,
                0.1494317,
                0.21524073,
                0.0175806,
                0.04599869,
                0.17523687,
                0.18209188,
                0.19111321,
                0.16830444,
                0.13734428,
                0.14703242,
                0.06796364,
                0.03192559,
                0.13734428,
                0.17523687,
                0.12998154,
                0.1933476,
                0.13244507,
                0.13244507,
                0.12502644,
                0.14703242,
                0.12502644,
                0.12998154,
            ],
            rtol=1e-5,
            atol=0,
        )
        numpy.testing.assert_array_equal(
            lres.p_sims[gt0ids],
            [
                0.3,
                0.33,
                0.11,
                0.17,
                0.3,
                0.42,
                0.06,
                0.06,
                0.33,
                0.34,
                0.36,
                0.38,
                0.3,
                0.29,
                0.41,
                0.19,
                0.31,
                0.39,
                0.18,
                0.39,
                0.48,
                0.41,
                0.22,
                0.41,
                0.39,
                0.32,
            ],
        )

    def test_knox_local_from_gdf(self):
        gdf = self.gdf
        gdf = gdf.set_crs(21096)
        numpy.random.seed(12345)
        local_knox = KnoxLocal.from_dataframe(
            gdf, time_col="T", delta=20, tau=5, keep=True
        )
        assert local_knox.statistic_.shape == (188,)
        lres = local_knox
        gt0ids = numpy.where(lres.nsti > 0)
        numpy.testing.assert_array_equal(
            gt0ids,
            [
                [
                    25,
                    26,
                    30,
                    31,
                    35,
                    36,
                    41,
                    42,
                    46,
                    47,
                    51,
                    52,
                    102,
                    103,
                    116,
                    118,
                    122,
                    123,
                    137,
                    138,
                    139,
                    140,
                    158,
                    159,
                    162,
                    163,
                ]
            ],
        )
        numpy.testing.assert_allclose(
            lres.p_hypergeom[gt0ids],
            [
                0.1348993,
                0.14220663,
                0.07335085,
                0.08400282,
                0.1494317,
                0.21524073,
                0.0175806,
                0.04599869,
                0.17523687,
                0.18209188,
                0.19111321,
                0.16830444,
                0.13734428,
                0.14703242,
                0.06796364,
                0.03192559,
                0.13734428,
                0.17523687,
                0.12998154,
                0.1933476,
                0.13244507,
                0.13244507,
                0.12502644,
                0.14703242,
                0.12502644,
                0.12998154,
            ],
            rtol=1e-5,
            atol=0,
        )
        numpy.testing.assert_array_equal(
            lres.p_sims[gt0ids],
            [
                0.3,
                0.33,
                0.11,
                0.17,
                0.3,
                0.42,
                0.06,
                0.06,
                0.33,
                0.34,
                0.36,
                0.38,
                0.3,
                0.29,
                0.41,
                0.19,
                0.31,
                0.39,
                0.18,
                0.39,
                0.48,
                0.41,
                0.22,
                0.41,
                0.39,
                0.32,
            ],
        )

    def test_explore(self):
        gdf = self.gdf.copy()
        gdf = gdf.set_crs(21096)
        numpy.random.seed(12345)

        warn_empty_set = pytest.warns(UserWarning, match="empty neighbor set.")
        warn_empty_geoms = pytest.warns(
            UserWarning,
            match="The GeoSeries you are attempting to plot",
        )

        with warn_empty_set, warn_empty_geoms:
            m = KnoxLocal.from_dataframe(
                gdf, time_col="T", delta=20, tau=5, keep=True
            ).explore()

        numpy.testing.assert_array_almost_equal(
            m.get_bounds(),
            [
                [-0.0005034046601185694, 28.514258651567],
                [0.0008675512091255166, 28.514975377735905],
            ],
        )
        # old folium returns 5, new folium returns 3
        assert len(m.to_dict()["children"]) >= 3

    def test_hotspots_without_neighbors(self):
        gdf = self.gdf.copy()
        gdf = gdf.set_crs(21096)
        numpy.random.seed(1)
        knox = KnoxLocal.from_dataframe(
            gdf,
            time_col="T",
            delta=20,
            tau=5,
        ).hotspots(keep_neighbors=False, inference="analytic")
        assert knox.shape == (3, 7)

    def test_hotspots_with_neighbors(self):
        gdf = self.gdf.copy()
        gdf = gdf.set_crs(21096)
        knox = KnoxLocal.from_dataframe(
            gdf,
            time_col="T",
            delta=20,
            tau=5,
        ).hotspots(keep_neighbors=True, inference="analytic")
        assert knox.shape == (4, 7)

    @pytest.mark.mpl_image_compare
    def test_plot(self):
        gdf = self.gdf.copy()
        gdf = gdf.set_crs(21096)
        fig, ax2 = plt.subplots(figsize=(30, 18))
        lk = KnoxLocal.from_dataframe(gdf, time_col="T", delta=20, tau=5, keep=True)
        lk.plot(inference="analytic", ax=ax2)
        assert fig


class TestKnoxStratified:
    def setup_method(self):
        path = lps.examples.get_path("burkitt.shp")
        self.gdf = gpd.read_file(path)
        # two spatially separated clusters; within each cluster the two
        # events are close in time, across clusters they are not
        self.s_coords = numpy.array(
            [[0.0, 0.0], [1.0, 0.0], [100.0, 100.0], [101.0, 100.0]]
        )
        self.t_coords = numpy.array([[0.0], [1.0], [100.0], [101.0]])
        self.stratum = numpy.array(["north", "north", "south", "south"])

    def test_stratified_permutation_stays_within_strata(self):
        numpy.random.seed(12345)
        res = Knox(
            self.s_coords,
            self.t_coords,
            delta=2,
            tau=2,
            permutations=99,
            keep=True,
            stratum=self.stratum,
        )
        assert res.null == "stratified"
        assert res.nst == 2
        # times can only be exchanged within a stratum, so the two
        # space-time pairs are preserved under every permutation
        numpy.testing.assert_array_equal(res.sim, numpy.full(99, 2.0))
        assert res.p_sim == 1.0
        # the retained simulations record the null hypothesis used
        assert res.sim.null == "stratified"

        # the unconstrained null mixes times across strata
        numpy.random.seed(12345)
        res_unc = Knox(
            self.s_coords, self.t_coords, delta=2, tau=2, permutations=99, keep=True
        )
        assert res_unc.null == "unconstrained"
        assert res_unc.strata is None
        assert res_unc.sim.null == "unconstrained"
        assert (res_unc.sim != 2).any()

    def test_single_stratum_equals_unconstrained(self):
        # one stratum spanning all events reproduces the unconstrained null
        numpy.random.seed(12345)
        ref = Knox(self.gdf[["X", "Y"]], self.gdf[["T"]], delta=20, tau=5, keep=True)
        numpy.random.seed(12345)
        strat = Knox(
            self.gdf[["X", "Y"]],
            self.gdf[["T"]],
            delta=20,
            tau=5,
            keep=True,
            stratum=numpy.zeros(self.gdf.shape[0], dtype=int),
        )
        assert strat.null == "stratified"
        assert strat.strata["n_strata"] == 1
        assert strat.p_sim == ref.p_sim == 0.21
        numpy.testing.assert_array_equal(strat.sim, ref.sim)

    def test_strata_summary_singletons_and_missing(self):
        s_coords = numpy.array(
            [[0.0, 0.0], [1.0, 0.0], [50.0, 0.0], [60.0, 0.0],
             [100.0, 0.0], [101.0, 0.0]]
        )
        t_coords = numpy.array([[0.0], [1.0], [50.0], [60.0], [100.0], [101.0]])
        stratum = numpy.array(["a", "a", None, "only", "b", "b"], dtype=object)
        numpy.random.seed(12345)
        res = Knox(
            s_coords,
            t_coords,
            delta=2,
            tau=2,
            permutations=99,
            keep=True,
            stratum=stratum,
        )
        strata = res.strata
        assert strata["counts"] == {"a": 2, "only": 1, "b": 2}
        assert strata["n_strata"] == 3
        assert strata["singletons"] == ["only"]
        numpy.testing.assert_array_equal(strata["missing"], [2])
        numpy.testing.assert_array_equal(
            strata["exchangeable"], [True, True, False, False, True, True]
        )
        assert strata["n_exchangeable"] == 4
        # singleton and missing strata never exchange their times, and the
        # exchangeable pairs (0, 1) and (4, 5) are always preserved
        numpy.testing.assert_array_equal(res.sim, numpy.full(99, 2.0))
        assert res.p_sim == 1.0

    def test_degenerate_strata_warn(self):
        stratum = numpy.array(["a", "b", "c", "d"])
        with pytest.warns(UserWarning, match="degenerate"):
            res = Knox(
                self.s_coords,
                self.t_coords,
                delta=2,
                tau=2,
                permutations=99,
                keep=True,
                stratum=stratum,
            )
        assert res.strata["n_exchangeable"] == 0
        numpy.testing.assert_array_equal(res.sim, numpy.full(99, 2.0))
        assert res.p_sim == 1.0

    def test_stratum_validation(self):
        with pytest.raises(ValueError, match="one label per event"):
            Knox(self.s_coords, self.t_coords, delta=2, tau=2, stratum=["a", "b"])

    def test_from_dataframe_stratum(self):
        gdf = self.gdf.set_crs(21096)
        gdf.index = [f"ev{i}" for i in range(gdf.shape[0])]
        labels = numpy.where(gdf["T"] < 3500, "early", "late")

        # a series is aligned to the event ids, whatever its own order
        shuffled = pd.Series(labels, index=gdf.index).sample(frac=1, random_state=42)
        res = Knox.from_dataframe(
            gdf, time_col="T", delta=20, tau=5, permutations=0, stratum=shuffled
        )
        numpy.testing.assert_array_equal(res.strata["labels"], labels)
        assert res.null == "stratified"

        # a column name pulls the labels from the dataframe
        gdf["season"] = labels
        res = Knox.from_dataframe(
            gdf, time_col="T", delta=20, tau=5, permutations=0, stratum="season"
        )
        numpy.testing.assert_array_equal(res.strata["labels"], labels)
        with pytest.raises(ValueError, match="not found"):
            Knox.from_dataframe(
                gdf, time_col="T", delta=20, tau=5, permutations=0, stratum="nope"
            )

        # a series that shares no ids with the events is an error
        foreign = pd.Series(labels, index=[f"other{i}" for i in range(gdf.shape[0])])
        with pytest.raises(ValueError, match="could not be aligned"):
            Knox.from_dataframe(
                gdf, time_col="T", delta=20, tau=5, permutations=0, stratum=foreign
            )

        # ids not covered by the series are reported as missing strata
        partial = pd.Series(labels[10:], index=gdf.index[10:])
        res = Knox.from_dataframe(
            gdf, time_col="T", delta=20, tau=5, permutations=0, stratum=partial
        )
        numpy.testing.assert_array_equal(res.strata["missing"], numpy.arange(10))


class TestKnoxLocalStratified:
    def setup_method(self):
        # two spatially separated clusters; within each cluster the two
        # events are close in time, across clusters they are not
        self.s_coords = numpy.array(
            [[0.0, 0.0], [1.0, 0.0], [100.0, 100.0], [101.0, 100.0]]
        )
        self.t_coords = numpy.array([[0.0], [1.0], [100.0], [101.0]])
        self.stratum = numpy.array(["north", "north", "south", "south"])

    def test_focal_time_fixed_within_strata(self):
        numpy.random.seed(12345)
        res = KnoxLocal(
            self.s_coords,
            self.t_coords,
            delta=2,
            tau=2,
            permutations=99,
            keep=True,
            stratum=self.stratum,
        )
        assert res.null == "stratified"
        # each focal keeps its own time stamp and can only exchange with its
        # stratum partner, so its single space-time link is always preserved
        numpy.testing.assert_array_equal(res.sims, numpy.ones((4, 99)))
        numpy.testing.assert_array_equal(res.p_sims, numpy.ones(4))
        assert res.sims.null == "stratified"
        # the global permutation inference is stratified as well
        assert res.p_sim == 1.0
        # strata are exposed on the result geodataframe
        assert "stratum" in res._gdf.columns
        assert "exchangeable" in res._gdf.columns
        numpy.testing.assert_array_equal(
            res._gdf["exchangeable"].values, [True, True, True, True]
        )

        # the unconstrained conditional permutations can break the links
        numpy.random.seed(12345)
        res_unc = KnoxLocal(
            self.s_coords, self.t_coords, delta=2, tau=2, permutations=99, keep=True
        )
        assert res_unc.null == "unconstrained"
        assert res_unc.strata is None
        assert (res_unc.sims != 1).any()

    def test_local_degenerate_strata(self):
        stratum = numpy.array(["a", "b", "c", "d"])
        with pytest.warns(UserWarning, match="degenerate"):
            res = KnoxLocal(
                self.s_coords,
                self.t_coords,
                delta=2,
                tau=2,
                permutations=99,
                keep=True,
                stratum=stratum,
            )
        # no focal can exchange its time, so every permutation equals the
        # observed configuration
        numpy.testing.assert_array_equal(res.sims, numpy.ones((4, 99)))
        numpy.testing.assert_array_equal(res.p_sims, numpy.ones(4))

    def test_local_missing_stratum_fixed(self):
        stratum = numpy.array(["north", "north", None, None], dtype=object)
        numpy.random.seed(12345)
        res = KnoxLocal(
            self.s_coords,
            self.t_coords,
            delta=2,
            tau=2,
            permutations=99,
            keep=True,
            stratum=stratum,
        )
        numpy.testing.assert_array_equal(res.strata["missing"], [2, 3])
        numpy.testing.assert_array_equal(
            res.strata["exchangeable"], [True, True, False, False]
        )
        # focals with a missing stratum keep their time, and events in other
        # strata can never receive it
        numpy.testing.assert_array_equal(res.sims, numpy.ones((4, 99)))

    def test_stratum_aligned_to_ids(self):
        ids = ["e0", "e1", "e2", "e3"]
        ser = pd.Series(
            ["south", "north", "north", "south"], index=["e2", "e0", "e3", "e1"]
        )
        res = KnoxLocal(
            self.s_coords,
            self.t_coords,
            delta=2,
            tau=2,
            permutations=0,
            ids=ids,
            stratum=ser,
        )
        numpy.testing.assert_array_equal(
            res.strata["labels"], ["north", "south", "south", "north"]
        )

    def test_local_from_dataframe_stratum(self):
        gdf = gpd.GeoDataFrame(
            {"T": self.t_coords.ravel(), "region": self.stratum},
            geometry=gpd.points_from_xy(self.s_coords[:, 0], self.s_coords[:, 1]),
            crs="EPSG:21096",
        )
        numpy.random.seed(12345)
        res = KnoxLocal.from_dataframe(
            gdf,
            time_col="T",
            delta=2,
            tau=2,
            permutations=99,
            keep=True,
            stratum="region",
        )
        assert res.null == "stratified"
        numpy.testing.assert_array_equal(res.sims, numpy.ones((4, 99)))


class TestSpaceTimeEvents:
    def setup_method(self):
        path = lps.examples.get_path("burkitt.shp")
        self.events = SpaceTimeEvents(path, "T")

    def test_space_time_events(self):
        assert self.events.n == 188

    def test_knox(self):
        result = knox(self.events.space, self.events.t, delta=20, tau=5, permutations=1)
        assert result["stat"] == 13.0

    def test_mantel(self):
        result = mantel(
            self.events.space,
            self.events.time,
            1,
            scon=0.0,
            spow=1.0,
            tcon=0.0,
            tpow=1.0,
        )

        assert result["stat"] == approx(0.014154, rel=1e-4)

    def test_jacquez(self):
        with pytest.warns(
            UserWarning,
            match="The weights matrix is not fully connected",
        ):
            result = jacquez(self.events.space, self.events.t, k=3, permutations=1)

        assert result["stat"] == 12

    def test_modified_knox(self):
        result = modified_knox(
            self.events.space, self.events.t, delta=20, tau=5, permutations=1
        )

        assert result["stat"] == approx(2.810160, rel=1e-4)


def _spacetime_points_to_arrays_non_points():
    with pytest.raises(
        ValueError, match="The Knox statistic is only defined for Point geometries"
    ):
        _spacetime_points_to_arrays(
            gpd.GeoDataFrame(geometry=[LineString(((0, 0), (1, 1)))]),
            None,
        )
