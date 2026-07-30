"""Fiber-optic (RF-silent) drone detection: fusion, tether geometry, sensor mix."""
import pytest

from frontline_drones.fiber_optic import (
    LINK_MODALITIES,
    Modality,
    PHYSICAL_MODALITIES,
    SensorHit,
    TetherGeometry,
    assess_fiber_optic,
    assess_tether,
    recommend_sensors,
)


def hit(mod, conf):
    return SensorHit(mod, conf)


# ---- sensor hit ----

def test_hit_bad_confidence():
    with pytest.raises(ValueError):
        SensorHit(Modality.ACOUSTIC, 1.5)


def test_physical_vs_link_classification():
    assert hit(Modality.ACOUSTIC, 0.5).is_physical
    assert hit(Modality.RADAR, 0.5).is_physical
    assert hit(Modality.RF, 0.5).is_link
    assert hit(Modality.REMOTE_ID, 0.5).is_link
    assert not hit(Modality.RF, 0.5).is_physical


def test_modality_partition_complete():
    assert PHYSICAL_MODALITIES | LINK_MODALITIES == set(Modality)
    assert not (PHYSICAL_MODALITIES & LINK_MODALITIES)


# ---- assessment ----

def test_fiber_optic_strong_when_physical_and_silent():
    a = assess_fiber_optic([hit(Modality.ACOUSTIC, 0.8), hit(Modality.RADAR, 0.7)])
    assert a.rf_silent
    assert a.likelihood > 0.8
    assert "fiber-optic" in a.reasoning


def test_conventional_drone_low_likelihood():
    a = assess_fiber_optic([hit(Modality.ACOUSTIC, 0.8), hit(Modality.RF, 0.9),
                            hit(Modality.REMOTE_ID, 0.8)])
    assert not a.rf_silent
    assert a.likelihood < 0.2
    assert "conventional" in a.reasoning


def test_no_physical_not_assessable():
    a = assess_fiber_optic([hit(Modality.RF, 0.5)])
    assert a.physical_evidence == 0.0
    assert a.likelihood == 0.0
    assert "not assessable" in a.reasoning


def test_physical_evidence_noisy_or():
    a = assess_fiber_optic([hit(Modality.ACOUSTIC, 0.5), hit(Modality.OPTICAL, 0.5)])
    # 1 - (0.5*0.5) = 0.75
    assert a.physical_evidence == pytest.approx(0.75)


def test_likelihood_formula():
    a = assess_fiber_optic([hit(Modality.ACOUSTIC, 0.9), hit(Modality.RF, 0.3)])
    # physical=0.9, link=0.3 -> 0.9 * 0.7 = 0.63
    assert a.likelihood == pytest.approx(0.63, abs=1e-6)


def test_assessment_to_dict():
    d = assess_fiber_optic([hit(Modality.ACOUSTIC, 0.8)]).to_dict()
    for k in ("likelihood", "physical_evidence", "link_evidence", "rf_silent", "reasoning"):
        assert k in d


def test_empty_hits():
    a = assess_fiber_optic([])
    assert a.likelihood == 0.0 and a.physical_evidence == 0.0


# ---- tether geometry ----

def test_tether_within_and_monotonic():
    t = assess_tether([0, 500, 1200, 3000, 3000], max_tether_m=20000)
    assert t.within_tether and t.consistent
    assert t.payout_monotonicity == 1.0


def test_tether_exceeds_spool():
    t = assess_tether([0, 10000, 25000], max_tether_m=20000)
    assert not t.within_tether
    assert not t.consistent


def test_tether_retracting_inconsistent():
    # Range mostly decreasing -> not a fiber payout.
    t = assess_tether([5000, 4000, 3000, 2000, 1000], max_tether_m=20000)
    assert t.within_tether
    assert t.payout_monotonicity == 0.0
    assert not t.consistent


def test_tether_bad_max():
    with pytest.raises(ValueError):
        assess_tether([100], max_tether_m=0)


def test_tether_negative_range():
    with pytest.raises(ValueError):
        assess_tether([-1, 100])


def test_tether_max_range_reported():
    t = assess_tether([0, 800, 400, 1500], max_tether_m=20000)
    assert t.max_range_m == pytest.approx(1500)


def test_tether_to_dict():
    d = assess_tether([0, 500, 1000]).to_dict()
    for k in ("within_tether", "payout_monotonicity", "max_range_m", "consistent"):
        assert k in d


# ---- sensor recommendation ----

def test_heavy_jamming_prioritizes_physical():
    mix = recommend_sensors(0.8)
    assert Modality.RF not in mix and Modality.REMOTE_ID not in mix
    assert set(mix) == PHYSICAL_MODALITIES


def test_benign_rf_includes_link():
    mix = recommend_sensors(0.1)
    assert mix[0] is Modality.RF
    assert Modality.ACOUSTIC in mix


def test_recommend_bad_level():
    with pytest.raises(ValueError):
        recommend_sensors(1.5)


# ---- property sweeps ----

@pytest.mark.parametrize("pconf", [0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
@pytest.mark.parametrize("lconf", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
def test_likelihood_bounded_and_formula(pconf, lconf):
    a = assess_fiber_optic([hit(Modality.ACOUSTIC, pconf), hit(Modality.RF, lconf)])
    assert 0.0 <= a.likelihood <= 1.0
    assert a.likelihood == pytest.approx(pconf * (1 - lconf), abs=1e-6)


@pytest.mark.parametrize("lconf", [0.0, 0.1, 0.19, 0.2, 0.3, 0.5, 0.9])
def test_rf_silent_threshold(lconf):
    a = assess_fiber_optic([hit(Modality.ACOUSTIC, 0.8), hit(Modality.RF, lconf)],
                           rf_silent_threshold=0.2)
    assert a.rf_silent == (lconf <= 0.2)


@pytest.mark.parametrize("jam", [round(0.1 * i, 1) for i in range(11)])
def test_sensor_mix_valid_everywhere(jam):
    mix = recommend_sensors(jam)
    assert all(isinstance(m, Modality) for m in mix)
    assert set(PHYSICAL_MODALITIES) <= set(mix)   # physical always included
    if jam >= 0.5:
        assert not (set(LINK_MODALITIES) & set(mix))


@pytest.mark.parametrize("spool", [1000, 5000, 10000, 20000, 30000])
@pytest.mark.parametrize("peak", [500, 8000, 15000, 25000])
def test_within_tether_matches_spool(spool, peak):
    t = assess_tether([0, peak // 2, peak], max_tether_m=spool)
    assert t.within_tether == (peak <= spool)


@pytest.mark.parametrize("n", [2, 3, 5, 8])
def test_monotonic_payout_scores_one(n):
    ranges = [i * 500 for i in range(n)]
    t = assess_tether(ranges, max_tether_m=100000)
    assert t.payout_monotonicity == 1.0
    assert t.consistent
