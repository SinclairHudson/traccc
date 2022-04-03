from trackers import Track



def standard_filter(track, min_age=30) -> bool:
    """
    A filter for determining if the track produced by the tracker is good for
    visualization.

    """
    if track["age"] < min_age:
        return False
    else:
        return True
