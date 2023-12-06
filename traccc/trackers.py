import numpy as np
from filterpy.kalman import KalmanFilter

class Track:
    """
    This is a class representing a single track, ideally a single object and its movements
    """

    def __init__(self, track_id: int, initial_pos: np.ndarray, start_frame: int, death_time: int = 5):
        self.id = track_id
        self.prev_states = []  # tracks all previous estimates of position, and velocity
        self.start_frame = start_frame
        self.death_time = death_time

        assert len(initial_pos) == 5  # cxywh
        self.kf = KalmanFilter(dim_x=6, dim_z=4)
        # initial pos is xywh
        # state (x vector) is [x, y, vx, vy]
        self.kf.x = np.array([initial_pos[1], initial_pos[2], 0, 0, initial_pos[3], initial_pos[4]])
        self.kf.F = np.array([[1, 0, 1, 0, 0, 0],  # x = x + vx
                              [0, 1, 0, 1, 0, 0],  # y = y + vy
                              [0, 0, 1, 0, 0, 0],  # vx = vx
                              [0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 1]
                              ])  # vy = vy + ay

        self.kf.H = np.array([[1, 0, 0, 0, 0, 0],  # we only measure position
                              [0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 1]
                              ])
        self.kf.P *= 1000

        self.age = 0  # in the first frame, age is 0
        self.time_missing = 0
        self.active = True
        self.prev_measurements = []

    def predict(self) -> None:
        """
        Advances the KalmanFilter, predicting the current state based on the prior
        """
        self.kf.predict()

    def update(self, measurement: np.ndarray) -> None:
        """
        Update our estimate of the state given the measurement. Calculate the posterior.
        """
        self.prev_states.append(self.kf.x)
        self.prev_measurements.append(measurement)
        self.age += 1
        if measurement is None:  # on this iteration, didn't see this object
            self.time_missing += 1
            if self.time_missing > self.death_time:
                self.active = False
        else:
            # measurement comes in as cxywh
            assert len(measurement) == 5
            measurement_xywh = measurement[1:5]
            self.kf.update(measurement_xywh)

    def encode_in_dictionary(self) -> dict:
        """
        Encodes the track in a dictionary to be saved and used downstream.
        Uses vanilla python datatypes to allow for YAML serialization.
        """
        life = {
            "id": self.id,
            "start_frame": self.start_frame,
            "states": [a.tolist() for a in self.prev_states],
            "measurements": [a.tolist() if a is not None else None
                             for a in self.prev_measurements],
            "age": self.age
        }
        return life


class AccelTrack(Track):
    def __init__(self, track_id: int, initial_pos: np.ndarray, start_frame: int, death_time: int = 5):
        self.id = track_id
        self.prev_states = []  # tracks all previous estimates of position, and velocity
        self.start_frame = start_frame
        self.death_time = death_time

        assert len(initial_pos) == 5  # cxywh
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        # initial pos is xywh
        # state (x vector) is [x, y, vx, vy]
        self.kf.x = np.array([initial_pos[1], initial_pos[2], 0, 0, initial_pos[3], initial_pos[4], 0, 0])
        self.kf.F = np.array([[1, 0, 1, 0, 0, 0, 0, 0],  # x = x + vx
                              [0, 1, 0, 1, 0, 0, 0, 0],  # y = y + vy
                              [0, 0, 1, 0, 0, 0, 1, 0],  # vx = vx
                              [0, 0, 0, 1, 0, 0, 0, 1],
                              [0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1],
                              ])  # vy = vy + ay

        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],  # we measure position and width and height
                              [0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0]
                              ])
        self.kf.P *= 1000

        self.age = 0  # in the first frame, age is 0
        self.time_missing = 0
        self.active = True
        self.prev_measurements = []
