from math import sqrt
import pytest

import numpy as np

from traccc.track import hungarian_matching, track
from traccc.trackers import Track, AccelTrack


def test_matching():
    """
    tests that the hungarian matching matches close tracks to close detections
    """
    tracks = [Track(0, np.array([0.92, 0, 0, 0, 0]), 0),
              Track(1, np.array([0.99, 20, 20, 0, 0]), 0),
              Track(2, np.array([0.39, 50, 40, 0, 0]), 0)]
    detections = np.array([[0.92, 1, 1, 2, 2],
                           [0.99, 50, 50, 2, 4],
                           [0.8, 20, 30, 2, 2]])
    cost, row_ind, col_ind = hungarian_matching(tracks, detections)
    assert np.all(row_ind == np.array([0, 1, 2]))
    assert np.all(col_ind == np.array([0, 2, 1]))
    assert cost == 10 + 10 + sqrt(2)


def test_reject_large_cost():
    """
    tests that the matching algorithm rejects a matching with a cost that is
    far too large
    """
    tracks = [Track(0, np.array([0.92, 0, 0, 0, 0]), 0),
              Track(1, np.array([0.99, 20, 20, 0, 0]), 0),
              Track(2, np.array([0.39, 50000, 400000, 0, 0]), 0)]
    detections = np.array([[0.92, 1, 1, 2, 2],
                           [0.99, 50, 50, 2, 4],
                           [0.8, 20, 30, 2, 2]])
    cost, row_ind, col_ind = hungarian_matching(
        tracks, detections, max_cost=100)

    # track 0 matches detection 0 (0, 0)
    # track 1 matchest detection 2 (1, 2)
    # track 2 matches nothing, as does detection 1
    assert np.all(row_ind == np.array([0, 1]))
    assert np.all(col_ind == np.array([0, 2]))
    assert cost == 10 + sqrt(2)


def test_more_detections_than_tracks():
    """
    This function tests that the correct matching is made when there are more
    detections than tracks.
    """
    tracks = [Track(0, np.array([0.92, 0, 0, 0, 0]), 0),
              Track(1, np.array([0.99, 20, 20, 0, 0]), 0)]
    detections = np.array([[0.92, 1, 1, 2, 2],
                           [0.99, 50, 50, 2, 4],
                           [0.91, 60, 50, 2, 4],
                           [0.8, 20, 30, 2, 2]])
    # track 0 matches detection 0 (0, 0)
    # track 1 matchest detection 3 (1, 3)
    # detection 1 and 2 are unmatched
    cost, row_ind, col_ind = hungarian_matching(
        tracks, detections, max_cost=100)
    assert np.all(row_ind == np.array([0, 1]))
    assert np.all(col_ind == np.array([0, 3]))
    assert cost == 10 + sqrt(2)


def test_more_tracks_than_detections():
    """
    This function tests that when there are more tracks than detections,
    some of the tracks go unmatched and the cost is still correctly computed.
    """
    tracks = [Track(0, np.array([0.92, 0, 0, 0, 0]), 0),
              Track(1, np.array([0.99, 20, 20, 0, 0]), 0),
              Track(2, np.array([0.99, 20, 10, 0, 0]), 0),
              Track(3, np.array([0.99, 50, 50, 0, 0]), 0),
              Track(4, np.array([0.39, 30, 30, 0, 0]), 0)]
    detections = np.array([[0.92, 21, 21, 2, 2],
                           [0.8, 31, 31, 2, 2]])
    # track 1 matches detection 0 (1, 0) with euclidean distance sqrt(2)
    # track 4 matchest detection 1 (4, 1) with euclidean distance sqrt(2)
    # tracks 0, 2, and 3 are unmatched
    cost, row_ind, col_ind = hungarian_matching(
        tracks, detections, max_cost=100)
    assert np.all(row_ind == np.array([1, 4]))
    assert np.all(col_ind == np.array([0, 1]))
    assert cost == sqrt(2) + sqrt(2)


@pytest.mark.parametrize("tracker", [Track, AccelTrack])
def test_track(tracker):
    """
    Simple test case to test that the tracker can take these detections and string
    them together.
    """
    detections = [np.array([[0.92, 0, 0, 5, 5]]),
                  np.array([[0.99, 20, 20, 5, 5]]),
                  np.array([[0.90, 41, 38, 5, 5]]),
                  np.array([[0.94, 58, 64, 5, 5]])]

    tracks = track(detections, tracker)
    assert len(tracks) == 1
    assert tracks[0].age == 3


@pytest.mark.parametrize("tracker", [Track, AccelTrack])
def test_no_track_switch(tracker):
    """
    Simple test case to ensure the tracker doesn't switch tracks when it shouldn't.
    In this test case, the trajectory of two objects cross, but the tracks shouldn't
    switch
    """
    detections = [np.array([[0.99, 0, 0, 0, 0], [0.99, 100, 0, 0, 0]]),
                  np.array([[0.99, 20, 20, 0, 0], [0.99, 80, 20, 0, 0]]),
                  np.array([[0.99, 40, 40, 0, 0], [0.99, 60, 40, 0, 0]]),
                  np.array([[0.99, 60, 60, 0, 0], [0.99, 40, 60, 0, 0]]),
                  np.array([[0.99, 80, 80, 0, 0], [0.99, 20, 80, 0, 0]]),
                  np.array([[0.99, 100, 100, 0, 0], [0.99, 0, 100, 0, 0]])]

    tracks = track(detections, tracker)
    assert len(tracks) == 2  # there should only be two tracks

    # and there should be a track that connects (0, 0) to (100, 100)
    # or connects (100, 0) to (0, 100)
    if np.all(tracks[0].prev_states[0] == np.array([0, 0, 0, 0])):
        assert np.allclose(tracks[0].prev_states[-1],
                      np.array([100, 100, 20, 20]), atol=1e-2, rtol=0)
    elif np.all(tracks[0].prev_states[0] == np.array([100, 0, 20, 20])):
        assert np.allclose(tracks[0].prev_states[-1],
                      np.array([0, 100, -20, 20]), atol=1e-2, rtol=0)
