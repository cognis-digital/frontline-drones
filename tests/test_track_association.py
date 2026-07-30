"""Tests for the multi-target data-association track manager."""

from __future__ import annotations

import pytest

from frontline_drones import track_association as ta
from frontline_drones.track_association import Detection, Track, TrackManager

# -- construction / validation ------------------------------------------------

def test_defaults_construct():
    m = TrackManager()
    assert m.tracks == []
    assert m.gate_radius_m > 0


def test_bad_gate_raises():
    with pytest.raises(ValueError):
        TrackManager(gate_radius_m=0)


def test_bad_confirm_hits_raises():
    with pytest.raises(ValueError):
        TrackManager(confirm_hits=0)


def test_confirm_window_smaller_than_hits_raises():
    with pytest.raises(ValueError):
        TrackManager(confirm_hits=5, confirm_window=3)


def test_bad_coast_raises():
    with pytest.raises(ValueError):
        TrackManager(max_coast_misses=0)


# -- spawning -----------------------------------------------------------------

def test_first_detection_spawns_tentative():
    m = TrackManager()
    m.update([(0.0, 0.0)], t=0.0)
    assert len(m.tracks) == 1
    assert m.tracks[0].state == "tentative"
    assert m.tracks[0].hits == 1


def test_unique_ids_assigned():
    m = TrackManager(gate_radius_m=5)
    m.update([(0, 0), (100, 100)], t=0.0)
    ids = {tr.track_id for tr in m.tracks}
    assert ids == {1, 2}


def test_far_apart_detections_spawn_separate_tracks():
    m = TrackManager(gate_radius_m=10)
    m.update([(0, 0), (500, 500)], t=0.0)
    assert len(m.live_tracks()) == 2


# -- association --------------------------------------------------------------

def test_continuing_track_associates():
    m = TrackManager(gate_radius_m=10, confirm_hits=2, confirm_window=4)
    m.update([(0.0, 0.0)], t=0.0)
    m.update([(1.0, 0.0)], t=1.0)
    assert len(m.live_tracks()) == 1
    assert m.tracks[0].hits == 2


def test_velocity_estimated():
    m = TrackManager(gate_radius_m=20)
    m.update([(0.0, 0.0)], t=0.0)
    m.update([(5.0, 0.0)], t=1.0)
    assert m.tracks[0].vel_east == pytest.approx(5.0)
    assert m.tracks[0].vel_north == pytest.approx(0.0)


def test_speed_and_heading_properties():
    m = TrackManager(gate_radius_m=20)
    m.update([(0.0, 0.0)], t=0.0)
    m.update([(0.0, 3.0)], t=1.0)
    assert m.tracks[0].speed_mps == pytest.approx(3.0)
    assert m.tracks[0].heading_deg == pytest.approx(0.0)  # moving north


def test_heading_east():
    m = TrackManager(gate_radius_m=20)
    m.update([(0.0, 0.0)], t=0.0)
    m.update([(4.0, 0.0)], t=1.0)
    assert m.tracks[0].heading_deg == pytest.approx(90.0)


def test_prediction_places_gate_ahead():
    # A fast track only stays associated because the gate follows its velocity.
    m = TrackManager(gate_radius_m=6, confirm_hits=2, confirm_window=6)
    m.update([(0.0, 0.0)], t=0.0)
    m.update([(5.0, 0.0)], t=1.0)    # within gate; establishes vel = 5 m/s east
    m.update([(12.0, 0.0)], t=2.0)   # 7 m from last fix (> gate) but 2 m from predicted
    assert len(m.live_tracks()) == 1
    assert m.tracks[0].hits == 3


def test_out_of_gate_spawns_new_track():
    m = TrackManager(gate_radius_m=5)
    m.update([(0.0, 0.0)], t=0.0)
    m.update([(100.0, 0.0)], t=1.0)
    assert len(m.live_tracks()) == 2


def test_greedy_nearest_neighbour_assignment():
    # Two tracks, two detections; each detection should bind to its nearer track.
    m = TrackManager(gate_radius_m=30)
    m.update([(0.0, 0.0), (100.0, 0.0)], t=0.0)
    m.update([(2.0, 0.0), (102.0, 0.0)], t=1.0)
    assert len(m.live_tracks()) == 2
    assert all(tr.hits == 2 for tr in m.tracks)


def test_one_detection_binds_one_track():
    # Two tracks but a single detection near track 1: track 2 must coast.
    m = TrackManager(gate_radius_m=30, confirm_hits=2, max_coast_misses=5)
    m.update([(0.0, 0.0), (100.0, 0.0)], t=0.0)
    m.update([(1.0, 0.0)], t=1.0)
    t1 = m.tracks[0]
    t2 = m.tracks[1]
    assert t1.hits == 2
    assert t2.misses == 1


# -- confirmation (M-of-N) ----------------------------------------------------

def test_promotes_to_confirmed():
    m = TrackManager(gate_radius_m=20, confirm_hits=3, confirm_window=5)
    for t in range(3):
        m.update([(t * 1.0, 0.0)], t=float(t))
    assert m.tracks[0].state == "confirmed"
    assert m.confirmed_tracks()


def test_not_confirmed_before_threshold():
    m = TrackManager(gate_radius_m=20, confirm_hits=3, confirm_window=5)
    m.update([(0.0, 0.0)], t=0.0)
    m.update([(1.0, 0.0)], t=1.0)
    assert m.tracks[0].state == "tentative"


# -- coasting / deletion ------------------------------------------------------

def test_confirmed_track_coasts_on_miss():
    m = TrackManager(gate_radius_m=20, confirm_hits=2, confirm_window=5, max_coast_misses=3)
    m.update([(0.0, 0.0)], t=0.0)
    m.update([(1.0, 0.0)], t=1.0)  # confirmed
    assert m.tracks[0].state == "confirmed"
    m.update([], t=2.0)            # miss
    assert m.tracks[0].state == "coasting"
    assert m.tracks[0].misses == 1


def test_track_deleted_after_max_coast():
    m = TrackManager(gate_radius_m=20, confirm_hits=2, confirm_window=5, max_coast_misses=2)
    m.update([(0.0, 0.0)], t=0.0)
    m.update([(1.0, 0.0)], t=1.0)  # confirmed
    m.update([], t=2.0)            # miss 1 -> coasting
    m.update([], t=3.0)            # miss 2 -> deleted
    assert m.tracks[0].state == "deleted"
    assert m.live_tracks() == []


def test_coasting_track_reacquired():
    m = TrackManager(gate_radius_m=30, confirm_hits=2, confirm_window=6, max_coast_misses=4)
    m.update([(0.0, 0.0)], t=0.0)
    m.update([(5.0, 0.0)], t=1.0)  # confirmed, vel 5 e
    m.update([], t=2.0)            # coast
    # Re-detect near predicted (10,0).
    m.update([(10.0, 0.0)], t=3.0)
    assert m.tracks[0].state == "confirmed"
    assert m.tracks[0].misses == 0


def test_tentative_track_dropped_quickly():
    # A one-shot tentative track that never re-appears should not linger forever.
    m = TrackManager(gate_radius_m=10, confirm_hits=3, confirm_window=5, max_coast_misses=3)
    m.update([(0.0, 0.0)], t=0.0)
    for t in range(1, 6):
        m.update([], t=float(t))
    assert m.tracks[0].state == "deleted"


# -- input coercion -----------------------------------------------------------

def test_accepts_detection_objects():
    m = TrackManager(gate_radius_m=10)
    m.update([Detection(east=0.0, north=0.0, t=0.0)], t=0.0)
    assert m.tracks[0].east == 0.0


def test_accepts_dicts():
    m = TrackManager(gate_radius_m=10)
    m.update([{"east": 3.0, "north": 4.0}], t=0.0)
    assert m.tracks[0].north == 4.0


def test_accepts_xy_dict_aliases():
    m = TrackManager(gate_radius_m=10)
    m.update([{"x": 1.0, "y": 2.0}], t=0.0)
    assert (m.tracks[0].east, m.tracks[0].north) == (1.0, 2.0)


def test_time_inferred_from_detections():
    m = TrackManager(gate_radius_m=10)
    m.update([Detection(0.0, 0.0, 5.0)])
    assert m.tracks[0].last_update_t == 5.0


# -- bookkeeping helpers ------------------------------------------------------

def test_prune_deleted():
    m = TrackManager(gate_radius_m=10, confirm_hits=2, max_coast_misses=1)
    m.update([(0.0, 0.0)], t=0.0)
    m.update([], t=1.0)  # tentative miss -> deleted (max_coast_misses=1)
    assert m.tracks[0].state == "deleted"
    removed = m.prune_deleted()
    assert removed == 1
    assert m.tracks == []


def test_history_capped():
    m = TrackManager(gate_radius_m=1000, history_len=3)
    for t in range(6):
        m.update([(t * 1.0, 0.0)], t=float(t))
    assert len(m.tracks[0].history) == 3


def test_to_dict_snapshot():
    m = TrackManager(gate_radius_m=20, confirm_hits=2)
    m.update([(0.0, 0.0)], t=0.0)
    m.update([(1.0, 0.0)], t=1.0)
    d = m.to_dict()
    assert d["num_tracks"] == 1
    assert d["num_confirmed"] == 1
    assert d["tracks"][0]["state"] == "confirmed"


def test_track_to_dict_keys():
    tr = Track(track_id=1, east=1.0, north=2.0, last_update_t=0.0)
    d = tr.to_dict()
    for k in ("track_id", "east", "north", "vel_east", "vel_north", "speed_mps",
              "heading_deg", "last_update_t", "hits", "misses", "age", "state"):
        assert k in d


def test_stationary_heading_zero():
    tr = Track(track_id=1, east=0.0, north=0.0)
    assert tr.heading_deg == 0.0


def test_deterministic_sequence():
    def run():
        m = TrackManager(gate_radius_m=15, confirm_hits=2)
        for t in range(5):
            m.update([(t * 2.0, 0.0), (t * 2.0, 100.0)], t=float(t))
        return m.to_dict()
    assert run() == run()


def test_states_constant_exposed():
    assert ta.TRACK_STATES == ("tentative", "confirmed", "coasting", "deleted")


def test_live_excludes_deleted():
    m = TrackManager(gate_radius_m=10, confirm_hits=2, max_coast_misses=1)
    m.update([(0.0, 0.0)], t=0.0)
    m.update([], t=1.0)
    assert all(tr.state != "deleted" for tr in m.live_tracks())
