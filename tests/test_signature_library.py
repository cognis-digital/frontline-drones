"""Tests for the cross-modality signature classifier / candidate fusion."""

from __future__ import annotations

import pytest

from frontline_drones import signature_library as sl
from frontline_drones.acoustic import AcousticCandidate
from frontline_drones.rf_classify import RFCandidate, RFObservation


def _rf(platform, score, name="RF"):
    return RFCandidate(
        platform=platform, name=name, vendor="", score=score,
        band_overlap=1, matched_on=("band",), source_url="",
    )


def _ac(platform, score, name="AC"):
    return AcousticCandidate(
        platform=platform, name=name, score=score, in_band=True,
        library_range_hz=(100.0, 200.0), source_url="",
    )


def test_noisy_or_single():
    assert sl._noisy_or([0.5]) == pytest.approx(0.5)


def test_noisy_or_two_independent():
    # 1 - (1-0.5)(1-0.5) = 0.75
    assert sl._noisy_or([0.5, 0.5]) == pytest.approx(0.75)


def test_noisy_or_clamps():
    assert sl._noisy_or([2.0]) == pytest.approx(1.0)
    assert sl._noisy_or([-1.0]) == pytest.approx(0.0)


def test_noisy_or_empty():
    assert sl._noisy_or([]) == 0.0


def test_fuse_single_modality():
    hyps = sl.fuse_candidates(rf=[_rf("dji-mavic-3", 1.0)])
    assert len(hyps) == 1
    assert hyps[0].num_modalities == 1
    assert hyps[0].corroborated is False


def test_fuse_corroboration_ranks_first():
    hyps = sl.fuse_candidates(
        rf=[_rf("dji-mavic-3", 0.5), _rf("dji-fpv", 0.9)],
        acoustic=[_ac("dji-mavic-3", 0.9)],
    )
    # dji-mavic-3 is seen by both modalities -> ranks above the single-sensor fpv.
    assert hyps[0].platform == "dji-mavic-3"
    assert hyps[0].corroborated is True


def test_fuse_corroborated_flag():
    hyps = sl.fuse_candidates(
        rf=[_rf("x", 0.8)], acoustic=[_ac("x", 0.8)]
    )
    assert hyps[0].num_modalities == 2
    assert set(hyps[0].modalities) == {"rf", "acoustic"}


def test_fuse_score_in_unit_interval():
    hyps = sl.fuse_candidates(rf=[_rf("x", 1.0)], acoustic=[_ac("x", 1.0)])
    assert 0.0 <= hyps[0].fused_score <= 1.0


def test_fuse_trust_weighting():
    # RF trust (0.85) > acoustic trust (0.60) for equal candidate scores.
    rf_only = sl.fuse_candidates(rf=[_rf("x", 1.0)])[0]
    ac_only = sl.fuse_candidates(acoustic=[_ac("y", 1.0)])[0]
    assert rf_only.fused_score > ac_only.fused_score


def test_fuse_keeps_strongest_duplicate():
    hyps = sl.fuse_candidates(rf=[_rf("x", 0.3), _rf("x", 0.9)])
    assert hyps[0].per_modality["rf"] == pytest.approx(0.85 * 0.9)


def test_fuse_limit():
    hyps = sl.fuse_candidates(
        rf=[_rf("a", 0.9), _rf("b", 0.8), _rf("c", 0.7)], limit=2
    )
    assert len(hyps) == 2


def test_fuse_empty_inputs():
    assert sl.fuse_candidates() == []


def test_fuse_skips_blank_platform():
    hyps = sl.fuse_candidates(rf=[_rf("", 0.9)])
    assert hyps == []


def test_hypothesis_to_dict():
    h = sl.fuse_candidates(rf=[_rf("x", 0.9)])[0]
    d = h.to_dict()
    assert set(d) == {
        "platform", "name", "fused_score", "modalities", "num_modalities",
        "corroborated", "per_modality",
    }


def test_hypothesis_name_from_candidate():
    h = sl.fuse_candidates(rf=[_rf("x", 0.9, name="DJI X")])[0]
    assert h.name == "DJI X"


def test_classify_observation_rf_only():
    obs = sl.MultiModalObservation(rf=RFObservation(bands=("2.4/5.8 GHz",)))
    hyps = sl.classify_observation(obs)
    assert hyps
    assert all(h.num_modalities >= 1 for h in hyps)


def test_classify_observation_acoustic_only():
    obs = sl.MultiModalObservation(acoustic_fundamental_hz=130.0)
    hyps = sl.classify_observation(obs)
    assert hyps
    assert all("acoustic" in h.modalities for h in hyps)


def test_classify_observation_fused_corroborates():
    obs = sl.MultiModalObservation(
        rf=RFObservation(bands=("2.4/5.8 GHz",), hopping="fhss"),
        acoustic_fundamental_hz=130.0,
    )
    hyps = sl.classify_observation(obs)
    # At least the top hypothesis should be corroborated by both modalities.
    assert hyps[0].corroborated is True


def test_classify_observation_empty_bundle():
    assert sl.classify_observation(sl.MultiModalObservation()) == []


def test_classify_observation_limit():
    obs = sl.MultiModalObservation(
        rf=RFObservation(bands=("2.4/5.8 GHz",)), acoustic_fundamental_hz=130.0
    )
    hyps = sl.classify_observation(obs, limit=2)
    assert len(hyps) <= 2


def test_classify_observation_deterministic():
    obs = sl.MultiModalObservation(
        rf=RFObservation(bands=("2.4/5.8 GHz",)), acoustic_fundamental_hz=130.0
    )
    a = [h.to_dict() for h in sl.classify_observation(obs)]
    b = [h.to_dict() for h in sl.classify_observation(obs)]
    assert a == b


def test_sort_order_num_modalities_then_score():
    hyps = sl.fuse_candidates(
        rf=[_rf("solo", 1.0), _rf("both", 0.1)],
        acoustic=[_ac("both", 0.1)],
    )
    # 'both' has 2 modalities so despite low scores it sorts before 'solo'.
    assert hyps[0].platform == "both"
