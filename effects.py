import cv2

def red_dot(video, track):
    """

    """
    for i in range(track["age"]):
        position = track["states"][i][:2]  # xy

        frame = video[i+track["start_frame"]]
        video[i+track["start_frame"]] = cv2.circle(frame, (int(position[0]), int(position[1])), radius=20,
                                                   color=(255, 0, 0), thickness=-1)


def aging_dot(video, track):
    """

    """
    for i in range(track["age"]):
        position = track["states"][i][:2]  # xy

        frame = video[i+track["start_frame"]]
        video[i+track["start_frame"]] = cv2.circle(frame, (int(position[0]), int(position[1])), radius=20,
                                                   color=[min(255, i * 2)] * 3, thickness=-1)

def temporal_dot(video, track, frame_length=30):
    """

    """
    position_queue = []
    for i in range(track["age"]):
        position = track["states"][i][:2]  # xy
        position_queue.append(position)
        if len(position_queue) > frame_length:
            position_queue.pop(0)

        for position in position_queue:
            frame = video[i+track["start_frame"]]
            video[i+track["start_frame"]] = cv2.circle(frame, (int(position[0]), int(position[1])), radius=10,
                                                       color=[255, 0, 0], thickness=-1)
