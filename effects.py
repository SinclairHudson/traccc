from abc import ABC
import cv2
import numpy as np
from typing import Tuple

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


class Dot(Effect):
    def __init__(self, colour: Tuple[int], size:int=10):
        self.colour = colour
        self.size = size
    def draw(self, frame, track, frame_number):
        i = frame_number - track["start_frame"]
        (x, y) = track["states"][i][:2]
        return cv2.circle(frame, (int(x), int(y)), radius=self.size,
                                color=(255, 0, 0), thickness=-1)

class LaggingDot(Effect):
    def __init__(self, colour: Tuple[int], time_lag:int=8, size:int=10):
        self.colour = colour
        self.time_lag = time_lag
        self.size = size

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
    def __init__(self, colour=(0, 0, 255), length_in_frames:int=15, size:int=10):
        self.length_in_frames = length_in_frames
        self.colour = colour
        self.size = size

    def draw(self, frame: np.ndarray, track: dict, frame_number: int) -> np.ndarray:
        start_line = max(0, frame_number - self.length_in_frames - track["start_frame"])
        end_line = frame_number - track["start_frame"]

        for i in range(start_line, end_line):
            (x, y) = track["states"][i][:2]
            (x2, y2) = track["states"][i+1][:2]
            frame = cv2.line(frame, (int(x), int(y)), (int(x2), int(y2)),
                                color=self.colour, thickness=self.size)
        return frame

class HighlightLine(Effect):
    def __init__(self, colour=(255, 0, 0), length_in_frames:int=15, size:int=10):
        self.length_in_frames = length_in_frames
        self.colour = colour
        self.size = size

    def draw(self, frame: np.ndarray, track: dict, frame_number: int) -> np.ndarray:
        start_line = max(0, frame_number - self.length_in_frames - track["start_frame"])
        end_line = frame_number - track["start_frame"]

        blank = np.zeros_like(frame, dtype=np.uint8)

        for i in range(start_line, end_line):
            (x, y) = track["states"][i][:2]
            (x2, y2) = track["states"][i+1][:2]
            blank = cv2.line(blank, (int(x), int(y)), (int(x2), int(y2)),
                                color=self.colour, thickness=self.size)
            blank = cv2.GaussianBlur(blank, (self.size//2, self.size//2), 0)

        return cv2.addWeighted(frame, 1, blank, 1, 0)

class Contrail(Effect):
    def __init__(self, colour=(255, 0, 0), length_in_frames:int=15, size:int=10):
        self.length_in_frames = length_in_frames
        self.colour = colour
        self.size = size if size % 2 == 1 else size + 1  # must be odd

    def draw(self, frame: np.ndarray, track: dict, frame_number: int) -> np.ndarray:
        start_line = max(0, frame_number - self.length_in_frames - track["start_frame"])
        end_line = frame_number - track["start_frame"]

        blank = np.zeros_like(frame, dtype=np.uint8)

        for i in range(start_line, end_line):
            (x, y) = track["states"][i][:2]
            (x2, y2) = track["states"][i+1][:2]
            blank = cv2.line(blank, (int(x), int(y)), (int(x2), int(y2)),
                                color=self.colour, thickness=self.size)
            blank = cv2.GaussianBlur(blank, (self.size, self.size), self.size//2)

        return cv2.addWeighted(frame, 1, blank, 1, 0)
