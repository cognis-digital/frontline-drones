"""Tests for the passive RF control-link classifier."""

from __future__ import annotations

from frontline_drones import rf_classify as rf
from frontline_drones.rf_classify import RFObservation


def test_normalize_bands_slash_form():
    assert rf.normalize_bands("2.4/5.8 GHz ISM") == {"2.4 GHz", "5.8 GHz"}


def test_normalize_bands_compact():
    assert "5.8 GHz" in rf.normalize_bands("5.8ghz")


def test_normalize_bands_mhz():
    assert "900 MHz" in rf.normalize_bands("915mhz")


def test_normalize_bands_empty():
    assert rf.normalize_bands("") == set()


def test_hop_token_fhss():
    assert rf._hop_token("adaptive FHSS") == "fhss"


def test_hop_token_fixed():
    assert rf._hop_token("fixed channel") == "fixed"


def test_hop_token_unknown():
    assert rf._hop_token("") == "unknown"


def test_observation_defaults():
    obs = RFObservation()
    assert obs.bands == ()
    assert obs.hopping == "unknown"
    assert obs.channel_bandwidth_mhz is None


def test_classify_returns_caveat():
    res = rf.classify(RFObservation(bands=("2.4 GHz",)))
    assert "Corroborate" in res.caveat
    assert "probabilistic aid" in res.caveat


def test_classify_band_match_scores():
    res = rf.classify(RFObservation(bands=("2.4/5.8 GHz",)))
    assert res.best is not None
    assert res.best.band_overlap >= 1


def test_classify_sorted_descending():
    res = rf.classify(RFObservation(bands=("2.4/5.8 GHz",), hopping="fhss"))
    scores = [c.score for c in res.candidates]
    assert scores == sorted(scores, reverse=True)


def test_classify_hopping_boosts_score():
    with_hop = rf.classify(RFObservation(bands=("2.4/5.8 GHz",), hopping="fhss"))
    without = rf.classify(RFObservation(bands=("2.4/5.8 GHz",), hopping="unknown"))
    assert with_hop.best.score >= without.best.score


def test_classify_protocol_hint_matches():
    res = rf.classify(RFObservation(bands=("2.4/5.8 GHz",), protocol_hint="ocusync"))
    assert res.best is not None
    assert "protocol" in res.best.matched_on


def test_classify_bandwidth_dimension():
    res = rf.classify(
        RFObservation(bands=("2.4/5.8 GHz",), channel_bandwidth_mhz=20.0)
    )
    assert res.best is not None


def test_classify_limit_respected():
    res = rf.classify(RFObservation(bands=("2.4/5.8 GHz",)), limit=2)
    assert len(res.candidates) <= 2


def test_classify_min_score_filters():
    # An implausible band nobody uses -> nothing survives min_score.
    res = rf.classify(RFObservation(bands=("77 GHz",)), min_score=0.5)
    assert res.candidates == ()
    assert res.best is None


def test_classify_no_bands_low_scores():
    res = rf.classify(RFObservation(hopping="fhss"))
    # Hopping alone can contribute but band overlap is zero.
    for c in res.candidates:
        assert c.band_overlap == 0


def test_candidate_to_dict_keys():
    res = rf.classify(RFObservation(bands=("2.4/5.8 GHz",)))
    d = res.best.to_dict()
    assert set(d) == {
        "platform", "name", "vendor", "score", "band_overlap", "matched_on", "source_url"
    }


def test_classification_to_dict():
    d = rf.classify(RFObservation(bands=("2.4/5.8 GHz",))).to_dict()
    assert "candidates" in d and "best" in d and "caveat" in d


def test_score_in_unit_interval():
    res = rf.classify(
        RFObservation(bands=("2.4/5.8 GHz",), hopping="fhss", protocol_hint="ocusync",
                      channel_bandwidth_mhz=20.0)
    )
    for c in res.candidates:
        assert 0.0 <= c.score <= 1.0


def test_classify_deterministic():
    obs = RFObservation(bands=("2.4/5.8 GHz",), hopping="fhss")
    assert rf.classify(obs).to_dict() == rf.classify(obs).to_dict()


def test_parse_bw_range_two():
    assert rf._parse_bw_range("~10-40") == (10.0, 40.0)


def test_parse_bw_range_single():
    assert rf._parse_bw_range("20") == (20.0, 20.0)


def test_parse_bw_range_none():
    assert rf._parse_bw_range("") is None


def test_custom_data_dir(tmp_path):
    csv = tmp_path / "rf-signatures.csv"
    csv.write_text(
        "id,name,platform,vendor,control_band,video_protocol,frequency_hopping,"
        "channel_bandwidth_mhz,detection_notes,source_url\n"
        "z1,Zeta One,z1,Zeta,2.4 GHz ISM,ZetaLink,fixed,10,notes,http://ex.test\n",
        encoding="utf-8",
    )
    res = rf.classify(RFObservation(bands=("2.4 GHz",)), data_dir=str(tmp_path))
    assert res.best is not None and res.best.platform == "z1"


def test_matched_on_lists_dimensions():
    res = rf.classify(
        RFObservation(bands=("2.4/5.8 GHz",), hopping="fhss", protocol_hint="ocusync")
    )
    assert "band" in res.best.matched_on
    assert "hopping" in res.best.matched_on
