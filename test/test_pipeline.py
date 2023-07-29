
import os

def test_pipeline():
    """
    Tests that the whole pipeline can be run end to end, which checks quite a few
    things like imports being good, etc.
    """

    os.system("python3 detect.py stormy --input test_assets/stormy.mp4")
    os.system("python3 track.py stormy")
    os.system("python3 draw.py stormy --input test/assets/stormy.mp4 --effect line --length 8 --colour blue --output test_stormy.mp4")
