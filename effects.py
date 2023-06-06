from abc import ABC
import cv2

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
