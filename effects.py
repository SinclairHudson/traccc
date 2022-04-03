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
