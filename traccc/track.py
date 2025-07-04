"""
Module to generate tracks from a list of detections.
"""
import argparse
from math import sqrt
from typing import List, Tuple
import os

import numpy as np
import torch
import yaml
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_convert, nms
from tqdm import tqdm

from traccc.trackers import Track, AccelTrack


def euclidean_distance(track: Track, detection: np.ndarray) -> float:
    """
    Calculates euclidean distance between a track and a detection in pixel space.
    Args:
        track: The track to calculate the distance from.
        detection: A detection in (c, x, y, w, h) format.
    """
    assert len(detection) == 5  # confidence, x, y, w, h
    return sqrt((track.kf.x[0] - detection[1]) ** 2 + (track.kf.x[1] - detection[2]) ** 2)


def hungarian_matching(tracks, detections, cost_function=euclidean_distance, max_cost=np.infty):
    """
    Finds the minimum cost matching between tracks and detections, based on
    some distance metric. Returns a permutation of detections that orders them
    to be in the correct order with tracks.
    If there are more detections that tracks, the detections permuted to the end
    returns the scalar of the cost of all the matches, as well as two arrays of equal
    length. row_ind[x] is the track that matches with col_ind[x] detection.
    """
    cost_matrix = np.zeros((len(tracks), len(detections)))
    for i, track in enumerate(tracks):
        for j, detection in enumerate(detections):
            cost_matrix[i][j] = cost_function(track, detection)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # remove ridiculous matches; it's better to leave them unpaired
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] > max_cost:
            row_ind = np.delete(row_ind, np.where(row_ind == i))
            col_ind = np.delete(col_ind, np.where(col_ind == j))
    cost = cost_matrix[row_ind, col_ind].sum()
    return cost, row_ind, col_ind


def track(detections: List[np.ndarray], 
          track_class, death_time: int = 5, max_cost: float = np.infty):
    """
    Tracks objects in a sequence of detections using a Kalman Filter.
    Args:
        detections: List of ndarray, where each ndarray is a [N, 5] array, a list of detections.
            Each detection is a tuple of 5 elements: (c, x, y, w, h).
        track_class: The class of the track to be used (Track or AccelTrack).
        death_time: Number of frames without an observation before a track is deleted.
        max_cost: Maximum cost tolerated to match a track to a detection.
    Returns:
        List of tracks that were generated throughout the sequence.
    """
    next_track_id = 0  # counter for track IDs
    inactive_tracks = []
    tracks = []
    for frame_number, frame_detections in zip(range(len(detections)), tqdm(detections)):

        # handle deaths; if a track hasn't been seen in a few frames, deactivate it
        dead_tracks = [track for track in tracks if not track.active]
        active_tracks = [track for track in tracks if track.active]
        inactive_tracks.extend(dead_tracks)


        tracks = active_tracks
        for track in tracks:
            track.predict()  # advance the Kalman Filter, to get the prior for this timestep

        if len(frame_detections) == 0:
            for track in tracks:
                track.update(None)  # no data for any of the tracks

        else:  # there are some detections
            if len(tracks) > 0:
                _, row_ind, col_ind = hungarian_matching(
                    tracks, frame_detections, max_cost=max_cost)
                for i in range(len(row_ind)):
                    # update the matched tracks
                    tracks[row_ind[i]].update(frame_detections[col_ind[i]])

                unmatched_track_indices = set(
                    range(len(tracks))) - set(row_ind)
                for i in unmatched_track_indices:
                    tracks[i].update(None)

                # births
                unmatched_detection_indices = set(
                    range(len(frame_detections))) - set(col_ind)
                for i in unmatched_detection_indices:
                    tracks.append(
                        track_class(next_track_id, frame_detections[i], frame_number,
                                    death_time=death_time))
                    next_track_id += 1

            else:  # no tracks, but detections
                # births
                for detection in frame_detections:
                    tracks.append(track_class(next_track_id, detection,
                                  frame_number, death_time=death_time))
                    next_track_id += 1

    inactive_tracks.extend(tracks)
    return inactive_tracks


def filter_detections(detections: List[np.ndarray],
                      conf_threshold: float = 0.0,
                      iou_threshold: float =0.5) -> List[np.ndarray]:
    """
    Applies confidence filtering and Non-Max Suppression.
    Args:
        detections: List of ndarray, where each ndarray is a [N, 5] array, a list of detections.
            Each detection is a tuple of 5 elements: (c, x, y, w, h).
        conf_threshold: Confidence threshold for removing uncertain predictions.
        iou_threshold: IoU threshold used in Non-Max Suppression filtering.
    """
    filtered_detections = []
    for frame_detections in detections:
        conf = frame_detections[:, 0]
        frame_detections = torch.Tensor(
            frame_detections[conf > conf_threshold])

        xyxy = box_convert(
            frame_detections[:, 1:], in_fmt="cxcywh", out_fmt="xyxy")
        best_candidates = nms(
            xyxy, frame_detections[:, 0], iou_threshold=iou_threshold)
        filtered_detections.append(frame_detections[best_candidates].numpy())
    return filtered_detections


track_type_dict = {
    "Constant Velocity": Track,
    "Constant Acceleration": AccelTrack,
}


def run_track(name: str, track_type: str, death_time: int, 
              iou_threshold: float, conf_threshold: float, max_cost: float) -> str:
    """
    Runs the tracking portion of the pipeline.
    Args:
        name: Name of the project to be tracked.
        track_type: Type of the track to be used (Constant Velocity or Acceleration).
        death_time: Number of frames without an observation before a track is deleted.
        iou_threshold: IoU threshold used in Non-Max Suppression filtering.
        conf_threshold: Confidence threshold for removing uncertain predictions.
        max_cost: The maximum cost tolerated to match a track to a detection.
    Returns:
        A message indicating the success of the operation and the number of tracks saved.
    """

    # load the raw detections
    detections = np.load(f"internal/{name}.npz")
    detections_list = []
    for frame_name in detections:
        detections_list.append(detections[frame_name])

    # filter out the detections that won't be used
    detections_list = filter_detections(
        detections_list, conf_threshold, iou_threshold)

    # run the tracking algorithm
    tracks = track(
        detections_list, track_type_dict[track_type], death_time=death_time, max_cost=max_cost)

    # dump to yaml
    track_lives = [track.encode_in_dictionary() for track in tracks]
    dictionary = {
        "detections_file": f"internal/{name}.npz",
        "tracks": track_lives
    }
    with open(f"internal/{name}.yaml", 'w') as f:
        yaml.safe_dump(dictionary, f)

    return f"Successfully saved {len(track_lives)} tracks to internal/{name}.yaml"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tracks objects using detections as input.")
    parser.add_argument("name", help="name of the project to be tracked.")
    parser.add_argument(
        "--track_type", help="track type", default="Track")
    parser.add_argument(
        "--death_time", help="number of frames without an observation before track deletion", default=5)
    parser.add_argument(
        "--iou_threshold", help="IoU threshold used in Non-Max Suppression filtering, must be in the range [0, 1].", default=0.2)
    parser.add_argument(
        "--conf_threshold", help="confidence threshold for removing uncertain predictions, must be in the range [0, 1].", default=0.05)
    parser.add_argument(
        "--max_cost", help="the maximum cost tolerated to match a track to a detection.", default=200)
    args = parser.parse_args()
    name = args.name

    assert os.path.exists(
        f"internal/{name}.npz"), f"Could not find internal/{name}.npz"
    message = run_track(args.name, args.track_type, int(args.death_time), float(
        args.iou_threshold), float(args.conf_threshold), float(args.max_cost))
    print(message)
