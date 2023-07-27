from abc import ABC
import cv2
import numpy as np
from typing import Tuple, List

# pylint: disable=invalid-name, no-member

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

    def draw_tracks(self, frame, tracks: List[dict], frame_number: int):
        """
        The default is to draw every track independently
        """
        out_frame = frame
        for track in tracks:
            # print(f"drawing track {track['id']}")
            out_frame = self.draw(out_frame, track, frame_number)
        return out_frame

    def draw(self, frame, track, frame_number):
        raise NotImplementedError


class FullyConnected(Effect):
    def __init__(self, colour: Tuple[int], size: float = 1.0):
        self.colour = colour
        self.size = size

    def draw_tracks(self, frame, tracks: List[dict], frame_number: int):
        out_frame = frame
        state_indexes = [frame_number - track["start_frame"]
                         for track in tracks]
        states = [track["states"][max(0, i-1)]  # TODO investigate why this offset looks better
                     for track, i in zip(tracks, state_indexes)]
        # draw a line between each pair of points
        for i, (x1, y1, _, _, w1, h2) in enumerate(states):
            for _, (x2, y2, _, _, w2, h2) in enumerate(states[i+1:]):
                out_frame = cv2.line(out_frame, (int(x1), int(y1)), (int(
                    x2), int(y2)), self.colour, thickness=int(w2*self.size))
        return out_frame


class FullyConnectedNeon(Effect):
    def __init__(self, colour: Tuple[int], size: float = 1.0):
        self.colour = colour
        self.size = size

    def draw_tracks(self, frame, tracks: List[dict], frame_number: int):
        blank = np.zeros_like(frame, dtype=np.uint8)
        state_indexes = [frame_number - track["start_frame"]
                         for track in tracks]
        states = [track["states"][max(0, i-1)]  # TODO investigate why this offset looks better
                     for track, i in zip(tracks, state_indexes)]
        # draw a line between each pair of points
        if len(states) < 1:
            return frame

        width = states[0][4]
        for i, (x1, y1, _, _, w1, h1) in enumerate(states):
            for _, (x2, y2, _, _, w2, h2) in enumerate(states[i+1:]):
                blank = cv2.line(blank, (int(x1), int(y1)), (int(
                    x2), int(y2)), self.colour, thickness=int(self.size*w2*2))

        kernel_size = int(width * self.size * 3)
        if kernel_size % 2 == 0:
            kernel_size += 1

        blank = cv2.GaussianBlur(
            blank, (kernel_size, kernel_size), self.size//2)

        out_frame = cv2.addWeighted(frame, 1, blank, 1, 0)

        for i, (x1, y1, _, _, w1, h1) in enumerate(states):
            for _, (x2, y2, _, _, w2, h2) in enumerate(states[i+1:]):
                # white lines
                out_frame = cv2.line(out_frame, (int(x1), int(y1)), (int(
                    x2), int(y2)), (255, 255, 255), thickness=int(w2*self.size/2))

        return out_frame


class Dot(Effect):
    def __init__(self, colour: Tuple[int], size: float = 1.0):
        self.colour = colour
        self.size = size

    def draw(self, frame, track, frame_number):
        i = frame_number - track["start_frame"]
        (x, y, vx, vy, w, h) = track["states"][i]
        return cv2.circle(frame, (int(x), int(y)), radius=int(w*self.size/2),
                          color=self.colour, thickness=-1)


class LaggingDot(Effect):
    def __init__(self, colour: Tuple[int], time_lag: int = 8, size: float = 1.0):
        self.colour = colour
        self.time_lag = time_lag
        self.size = size

    def relevant(self, track: dict, frame_number: int) -> bool:
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
            (x, y, vx, vy, w, h) = track["states"][i]
        else:
            # for times where the track is too young
            (x, y, vx, vy, w, h) = track["states"][0]
        return cv2.circle(frame, (int(x), int(y)), radius=int(w*self.size/2),
                          color=(0, 0, 255), thickness=-1)


class Line(Effect):
    def __init__(self, colour=(0, 0, 255), length_in_frames: int = 15, size: float = 1.0):
        self.length_in_frames = length_in_frames
        self.colour = colour
        self.size = size

    def draw(self, frame: np.ndarray, track: dict, frame_number: int) -> np.ndarray:
        start_line = max(1, frame_number -
                         self.length_in_frames - track["start_frame"])
        end_line = frame_number - track["start_frame"]

        for i in range(start_line, end_line):
            (x, y, vx, vy, w, h) = track["states"][i]
            (x2, y2) = track["states"][i-1][:2]
            frame = cv2.line(frame, (int(x), int(y)), (int(x2), int(y2)),
                             color=self.colour, thickness=self.size*w)
        return frame


def draw_x(frame, x, y, colour, size):
    frame = cv2.line(frame, (int(x-size), int(y-size)), (int(x+size),
                                                         int(y+size)), color=colour, thickness=size//3)
    return cv2.line(frame, (int(x-size), int(y+size)), (int(x+size), int(y-size)), color=colour, thickness=size)


class Debug(Effect):
    def __init__(self, length_in_frames: int = 15, size: int = 10):
        self.length_in_frames = length_in_frames
        self.size = size
        self.colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                        (255, 0, 230), (252, 132, 0)]

    def draw(self, frame: np.ndarray, track: dict, frame_number: int) -> np.ndarray:
        start_line = max(1, frame_number -
                         self.length_in_frames - track["start_frame"])
        end_line = frame_number - track["start_frame"]

        colour = self.colours[track["id"] % len(self.colours)]
        for i in range(start_line, end_line):
            (x, y) = track["states"][i][:2]
            (x2, y2) = track["states"][i-1][:2]
            frame = cv2.line(frame, (int(x), int(y)), (int(x2), int(y2)),
                             color=colour, thickness=self.size)

            meas = track["measurements"][i]
            if meas is None:
                frame = draw_x(frame, x, y, colour, self.size * 2)
            else:  # a matched detection
                mx, my = meas[1:3]
                frame = cv2.circle(frame, (int(x), int(y)), color=colour,
                                   radius=self.size + 3, thickness=-1)
                frame = cv2.circle(frame, (int(mx), int(my)),
                                   color=(255, 255, 255), radius=self.size, thickness=-1)
                frame = cv2.line(frame, (int(mx), int(my)),
                                 (int(x), int(y)), color=(255, 255, 255),
                                 thickness=self.size)

        return frame


class HighlightLine(Effect):
    def __init__(self, colour=(255, 0, 0), length_in_frames: int = 15, size: int = 10):
        self.length_in_frames = length_in_frames
        self.colour = colour
        self.size = size

    def draw(self, frame: np.ndarray, track: dict, frame_number: int) -> np.ndarray:
        start_line = max(1, frame_number -
                         self.length_in_frames - track["start_frame"])
        end_line = frame_number - track["start_frame"]

        blank = np.zeros_like(frame, dtype=np.uint8)

        for i in range(start_line, end_line):
            (x, y) = track["states"][i][:2]
            (x2, y2) = track["states"][i-1][:2]
            blank = cv2.line(blank, (int(x), int(y)), (int(x2), int(y2)),
                             color=self.colour, thickness=self.size)

        blank = cv2.GaussianBlur(blank, (self.size//2, self.size//2), 0)

        return cv2.addWeighted(frame, 1, blank, 1, 0)


class Contrail(Effect):
    def __init__(self, colour=(255, 0, 0), length_in_frames: int = 15, size: int = 10):
        self.length_in_frames = length_in_frames
        self.colour = colour
        self.size = size if size % 2 == 1 else size + 1  # must be odd

    def draw(self, frame: np.ndarray, track: dict, frame_number: int) -> np.ndarray:
        start_line = max(1, frame_number -
                         self.length_in_frames - track["start_frame"])
        end_line = frame_number - track["start_frame"]

        blank = np.zeros_like(frame, dtype=np.uint8)

        for i in range(start_line, end_line):
            (x, y) = track["states"][i][:2]
            (x2, y2) = track["states"][i-1][:2]
            blank = cv2.line(blank, (int(x), int(y)), (int(x2), int(y2)),
                             color=self.colour, thickness=self.size)
            blank = cv2.GaussianBlur(
                blank, (self.size, self.size), self.size//2)

        return cv2.addWeighted(frame, 1, blank, 1, 0)
