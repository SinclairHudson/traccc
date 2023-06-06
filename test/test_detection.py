from balltracking.detectors import HuggingFaceDETR, PretrainedRN50Detector
import skvideo.io
import pytest

@pytest.mark.parametrize("DetectorClass", [HuggingFaceDETR, PretrainedRN50Detector])
def test_detector_single_ball_tarmac(DetectorClass):
    model = DetectorClass()
    # this has 120 frames
    vid_generator = skvideo.io.vreader(f"test_assets/ball_on_tarmac.mp4")
    result = model.detect_video(vid_generator)
    metadata = skvideo.io.ffprobe(f"test_assets/ball_on_tarmac.mp4")
    frame_count = int(metadata['video']['@nb_frames'])
    assert len(result) == frame_count
    for frame_detections in result:
        assert len(frame_detections) == 1  # there's only one ball
