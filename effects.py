from abc import ABC
import cv2
import numpy as np

class Effect(ABC):
    def __init__(self):
        pass

    def relevant(self, track: dict, frame_number) -> bool:
        """
        returns True if the frame needs to be modified because of this effect
        """
        if track["start_frame"] <= frame_number and frame_number < track["start_frame"] + track["age"]:
            return True
        else:
            return False

    def draw(self, frame, track, frame_number):
        raise NotImplementedError


class RedDot(Effect):
    def draw(self, frame, track, frame_number):
        i = frame_number - track["start_frame"]
        (x, y) = track["states"][i][:2]
        return cv2.circle(frame, (int(x), int(y)), radius=20,
                                color=(255, 0, 0), thickness=-1)

class LaggingBlueDot(Effect):
    def __init__(self, time_lag:int=8):
        self.time_lag = time_lag

    def relevant(self, track: dict, frame_number) -> bool:
        """
        returns True if the frame needs to be modified because of this effect
        """
        if track["age"] >= self.time_lag and \
            track["start_frame"] <= frame_number + self.time_lag and \
            frame_number < track["start_frame"] + track["age"] + self.time_lag:
            return True
        else:
            return False

    def draw(self, frame: np.ndarray, track: dict, frame_number: int) -> np.ndarray:
        i = frame_number - track["start_frame"] - self.time_lag
        if i >= 0:
            (x, y) = track["states"][i][:2]
        else:
            (x, y) = track["states"][0][:2]  # for times where the track is too young
        return cv2.circle(frame, (int(x), int(y)), radius=20,
                                color=(0, 0, 255), thickness=-1)

class Line(Effect):
    def __init__(self, length_in_frames:int=15):
        self.length_in_frames = length_in_frames

    def draw(self, frame: np.ndarray, track: dict, frame_number: int) -> np.ndarray:
        start_line = max(0, frame_number - self.length_in_frames - track["start_frame"])
        end_line = frame_number - track["start_frame"]

        for i in range(start_line, end_line):
            (x, y) = track["states"][i][:2]
            (x2, y2) = track["states"][i+1][:2]
            frame = cv2.line(frame, (int(x), int(y)), (int(x2), int(y2)),
                                color=(0, 0, 255), thickness=5)
        return frame


def aging_dot(video, track):
    """

    """
    max_frame = len(video)
    width=5
    for i in range(track["age"]):
        position = track["states"][i][:2]  # xy

        if i+track["start_frame"] >= max_frame:
            return  # end early, we can't write anything past the end of the vid
        frame = video[i+track["start_frame"]]
        video[i+track["start_frame"]] = cv2.circle(frame, (int(position[0]), int(position[1])), radius=int(width//1),
                                                   color=[min(255, i * 2)] * 3, thickness=-1)

def temporal_dot(video, track, frame_length=30):
    """

    """
    max_frame = len(video)
    position_queue = []
    for i in range(track["age"]):
        position = track["states"][i][:2]  # xy
        position_queue.append(position)
        if len(position_queue) > frame_length:
            position_queue.pop(0)

        for position in position_queue:
            if i+track["start_frame"] >= max_frame:
                return  # end early, we can't write anything past the end of the vid
            frame = video[i+track["start_frame"]]
            video[i+track["start_frame"]] = cv2.circle(frame, (int(position[0]), int(position[1])), radius=10,
                                                       color=[255, 0, 0], thickness=-1)

def temporal_line(video, track, frame_length=10):
    """

    """
    max_frame = len(video)
    position_queue = []
    for i in range(track["age"]):
        position = track["states"][i][:2]  # xy
        position_queue.append(position)
        if len(position_queue) > frame_length:
            position_queue.pop(0)

        position = position_queue[0]
        last_pos = (int(position[0]), int(position[1]))
        for position in position_queue:
            pos = (int(position[0]), int(position[1]))
            if i+track["start_frame"] >= max_frame:
                return  # end early, we can't write anything past the end of the vid
            frame = video[i+track["start_frame"]]
            video[i+track["start_frame"]] = cv2.line(frame, pos, last_pos,
                                                       color=[255, 0, 0], thickness=2)
            last_pos = pos
