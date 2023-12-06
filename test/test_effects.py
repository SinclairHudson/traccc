from traccc.effects import *
import os
import pytest

def get_stormy():
    if not os.path.exists("internal/stormy.npz"):
        os.system("python3 detect.py stormy --input test_assets/stormy.mp4")
        os.system("python3 track.py stormy")

def get_stormy_vid_generator():
    pass

def test_fully_connected():
    get_stormy()

@pytest.mark.parametrize("effect_class", [FullyConnected,
                                          FullyConnectedNeon,
                                          Dot,
                                          LaggingDot,
                                          Line,
                                          Debug,
                                          HighlightLine,
                                          Contrail])
def test_effect(effect_class):
    pass

