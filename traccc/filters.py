"""
Filters tracks themselves, before visualization.
"""
from traccc.trackers import Track

def standard_filter(track: dict, min_age: int=30) -> bool:
    """
    A filter for determining if the track produced by the tracker is good for
    visualization.
    """
    if track["age"] < min_age:
        return False
    else:
        return True
