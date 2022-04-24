"""
track.py. The purpose of this file is to generate tracks from a list of detections
"""
from filterpy.kalman import KalmanFilter
import numpy as np
from math import sqrt
from scipy.optimize import linear_sum_assignment
from trackers import Track
from tqdm import tqdm
import torch
from torchvision.ops import box_convert, nms
import yaml
import argparse


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
    """
    detections is a list of detections, every entry is a frame
    """
    next_track_id = 0  # counter for track IDs
    inactive_tracks = []
    tracks = []
    for frame_number, frame_detections in tqdm(enumerate(detections)):
        # handle deaths; if a track hasn't been seen in a few frames, delete it.

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


def filter_detections(detections, conf_threshold=0.0, iou_threshold=0.5) -> None:
    """
    Applies confidence filtering and Non-Max Suppression
    """
    for frame_number, frame_detections in tqdm(enumerate(detections)):
        conf = frame_detections[:,0]
        frame_detections = torch.Tensor(frame_detections[conf > conf_threshold])

        xyxy = box_convert(frame_detections[:, 1:], in_fmt="cxcywh", out_fmt="xyxy")
        best_candidates = nms(xyxy, frame_detections[:,0], iou_threshold=iou_threshold)
        detections[frame_number] = frame_detections[best_candidates, 1:].numpy()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Tracks objects using detections as input.")
    parser.add_argument("name", help="name of the project to be tracked.")
    parser.add_argument("--death_time", help="number of frames without an observation before track deletion", default=5)
    parser.add_argument("--iou_threshold", help="IoU threshold used in Non-Max Suppression filtering, must be in the range [0, 1].", default=0.5)
    parser.add_argument("--conf_threshold", help="confidence threshold for removing uncertain predictions, must be in the range [0, 1].", default=0.05)
    args = parser.parse_args()
    name = args.name

    detections = np.load(f"internal/{name}.npz")
    detections_list = []
    for frame_number, frame_name in tqdm(enumerate(detections)):
        detections_list.append(detections[frame_name])

    filter_detections(detections_list, float(args.conf_threshold), float(args.iou_threshold))
    tracks = track(detections_list, death_time=int(args.death_time))

    track_lives = [track.encode_in_dictionary() for track in tracks]
    dictionary = {
        "detections_file": f"internal/{name}.npz",
        "tracks": track_lives
    }
    with open(f"internal/{name}.yaml", 'w') as f:
        yaml.safe_dump(dictionary, f)

