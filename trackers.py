from filterpy.kalman import KalmanFilter
import numpy as np

DIM_X = 4  # position_x, position_y, velocity_x, velocity_y
DIM_Z = 2  # position, (x, y) in image coordinates (top left is origin)

class Track:
    """
    This is a class representing a single track, ideally a single object and its movements
    """
    def __init__(self, track_id, initial_pos, start_frame, death_time: int = 5):
        self.id = track_id
        self.prev_states = []  # tracks all previous estimates of position, and velocity
        self.start_frame = start_frame
        self.death_time = death_time

        assert len(initial_pos) == 5  # xywh
        self.kf = KalmanFilter(dim_x=DIM_X, dim_z=DIM_Z)
        # initial pos is xywh
        # state (x vector) is [x, y, vx, vy]
        self.kf.x = np.array([initial_pos[1], initial_pos[2], 0, 0])
        self.kf.F = np.array([[1, 0, 1, 0],  # x = x + vx
                              [0, 1, 0, 1],  # y = y + vy
                              [0, 0, 1, 0],  # vx = vx
                              [0, 0, 0, 1]])  # vy = vy + ay

        self.kf.H = np.array([[1, 0, 0, 0],  # we only measure position
                              [0, 1, 0, 0]])
        self.kf.P *= 1000

        self.age = 0
        self.time_missing = 0
        self.active=True

    def predict(self):
        """
        Advances the KalmanFilter, predicting the current state based on the prior
        """
        self.prev_states.append(self.kf.x)
        self.kf.predict()

    def update(self, measurement):
        """
        Update our estimate of the state given the measurement. Calculate the posterior.
        """
        self.age += 1
        if measurement is None:  # on this iteration, didn't see this object
            self.time_missing += 1
            if self.time_missing > self.death_time:
                self.active = False
        else:
            # measurement comes in as cxywh
            assert len(measurement) == 5
            measurement_xy = measurement[1:3]
            self.kf.update(measurement_xy)

    def encode_in_dictionary(self):
        # need to convert to vanilla python data types and dictionary for saving
        life = {
            "id": self.id,
            "start_frame": self.start_frame,
            "states": [a.tolist() for a in self.prev_states],
            "age": self.age
            }
        return life

