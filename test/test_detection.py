from balltracking.detectors import HuggingFaceDETR, PretrainedRN50Detector
import skvideo.io
import pytest

@pytest.mark.parametrize("DetectorClass", [HuggingFaceDETR, PretrainedRN50Detector])
def test_detector_single_ball_tarmac(DetectorClass):
    model = DetectorClass()
    # this has 120 frames
    vid_generator = skvideo.io.vreader(f"test_assets/ball_on_tarmac.mp4")
    result = model.detect_video(vid_generator)
    assert len(result) == 120
    breakpoint()
    for frame_detections in result:
        assert len(frame_detections) == 1  # there's only one ball
