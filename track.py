"""
track.py. The purpose of this file is to generate tracks from a list of detections
"""
from filterpy.kalman import KalmanFilter
import numpy as np
from math import sqrt
from scipy.optimize import linear_sum_assignment
from trackers import Track
from tqdm import tqdm
import yaml


def euclidean_distance(track: Track, detection):
    return sqrt((track.kf.x[0] - detection[0]) ** 2 + (track.kf.x[1] - detection[1]) ** 2)

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
    return row_ind, col_ind


def track(detections, death_time=5):
    next_track_id = 0  # counter for track IDs
    inactive_tracks = []
    tracks = []
    for frame_number, frame_name in tqdm(enumerate(detections)):
        # handle deaths; if a track hasn't been seen in a few frames, delete it.
        frame_detections = detections[frame_name]

        inactive_tracks.extend([track for track in tracks if track.time_missing >= death_time])
        tracks = [track for track in tracks if track.time_missing < death_time]
        for track in tracks:
            track.predict()  # advance the Kalman Filter, to get the prior for this timestep

        if len(frame_detections) == 0:
            for track in tracks:
                track.update(None)  # no data for any of the tracks
            continue  # no detections to process, continue

        else:  # there are some detections
            if len(tracks) > 0:
                row_ind, col_ind = hungarian_matching(tracks, frame_detections)
                for i in range(len(row_ind)):
                    # update the matched tracks
                    tracks[row_ind[i]].update(frame_detections[col_ind[i]][:2])  # update using xy

                unmatched_track_indices = set(range(len(tracks))) - set(row_ind)
                for i in unmatched_track_indices:
                    tracks[i].update(None)

                # births
                unmatched_detection_indices = set(range(len(frame_detections))) - set(col_ind)
                for i in unmatched_detection_indices:
                    tracks.append(Track(next_track_id, frame_detections[i], frame_number))

            else:  # no tracks, but detections
                # births
                for detection in frame_detections:
                    tracks.append(Track(next_track_id, detection, frame_number))
                    next_track_id += 1

    inactive_tracks.extend(tracks)
    return inactive_tracks




if __name__ == "__main__":
    detections = np.load("io/living_room.npz")
    tracks = track(detections)

    track_lives = [track.encode_in_dictionary() for track in tracks]
    dictionary = {
        "detections_file": "io/living_room.npz",
        "tracks": track_lives
    }
    with open("io/living_room.yaml", 'w') as f:
        yaml.safe_dump(dictionary, f)

