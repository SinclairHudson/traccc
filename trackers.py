from filterpy.kalman import KalmanFilter

DIM_X = 5  # position_x, position_y, velocity_x, velocity_y, acceleration_y
DIM_Z = 2  # position, (x, y) in image coordinates (top left is origin)

class Tracker:
    """
    This is a class representing a single track, ideally a single object and its movements
    """
    def __init__(self, track_id, initial_pos, start_frame):
        self.id = next_track_id
        self.prev_states = []  # tracks all previous estimates of position, and velocity
        next_track_id += 1
        self.start_frame = start_frame
        self.kf = KalmanFilter(dim_x=DIM_X, dim_z=DIM_Z)
        self.age = 0
        self.time_missing = 0

    def predict(self):
        """
        Advances the KalmanFilter, predicting the current state based on the prior
        """
        self.kf.predict()

    def update(self, measurement):
        """
        Update our estimate of the state given the measurement. Calculate the posterior.
        """
        if measurement is None:  # on this iteration, didn't see this object
            self.time_missing += 1
        else:
            self.kf.update(measurement)
