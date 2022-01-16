"""
track.py. The purpose of this file is to generate tracks from a list of detections
"""
from filterpy.kalman import KalmanFilter
import numpy as np
from math import sqrt
from scipy.optimize import linear_sum_assignment
from trackers import Tracker


next_track_id = 0  # counter for track IDs

def euclidean_distance(track, detection):
    return sqrt((track.state[0] - detection[0]) ** 2 + (track.state[1] - detection[1]) ** 2)

def hungarian_matching(tracks, detections, cost_function=euclidean_distance):
    """
    Finds the minimum cost matching between tracks and detections, based on
    some distance metric. Returns a permutation of detections that orders them
    to be in the correct order with tracks.
    If there are more detections that tracks, the detections permuted to the end
    """
    cost_matrix = np.zeros((len(tracks), len(detections)))
    for i, track in enumerate(tracks):
        for j, detection in enumerate(detections):
            cost_matrix[i][j] = cost_function(track, detection)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)


def track(detections, death_time=5):
    tracks = []
    for frame_number, frame_detections in enumerate(detections):
        # handle deaths; if a track hasn't been seen in a few frames, delete it.
        tracks = [track for track in tracks if track.time_missing < death_time]
        for track in tracks:
            track.predict()  # advance the Kalman Filter, to get the prior for this timestep

        if len(frame_detections) == 0:
            for track in tracks:
                track.update(None)  # no data for any of the tracks
            continue  # no detections to process, continue

        else:  # there are some detections
            if len(tracks) > 0:
                matching = hungarian_matching(tracks, frame_detections)

            # handle births
            for detection in detections:
                tracks.append(Tracker(next_track_id, detection, frame_number))
                next_track_id += 1




if __name__ == "__main__":

