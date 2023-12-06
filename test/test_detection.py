import pytest
import skvideo.io

from traccc.detectors import HuggingFaceDETR, PretrainedRN50Detector, OWLVITZeroShot


@pytest.mark.parametrize("DetectorClass", [HuggingFaceDETR, PretrainedRN50Detector])
def test_detector_single_ball_tarmac(DetectorClass):
    """
    Test that the detector can detect a single ball in a test video, in every single frame.
    Duplicate detections are acceptable because NMS is not being run.
    """
    model = DetectorClass()
    # this has 120 frames
    vid_generator = skvideo.io.vreader(f"test_assets/ball_on_tarmac.mp4")
    metadata = skvideo.io.ffprobe(f"test_assets/ball_on_tarmac.mp4")
    frame_count = int(metadata['video']['@nb_frames'])
    result = model.detect_video(vid_generator, frame_count=frame_count)
    assert len(result) == frame_count
    for frame_detections in result:
        # there must be a ball detected. there could be multiple boxes because no NMS
        assert len(frame_detections) >= 1

@pytest.mark.parametrize("DetectorClass", [OWLVITZeroShot])
def test_zeroshot_detector(DetectorClass):
    """
    Test that the detector can detect a single ball in a test video, in every single frame.
    Duplicate detections are acceptable because NMS is not being run.
    """
    model = DetectorClass()
    # this has 120 frames
    vid_generator = skvideo.io.vreader(f"test_assets/stormy.mp4")
    metadata = skvideo.io.ffprobe(f"test_assets/stormy.mp4")
    frame_count = int(metadata['video']['@nb_frames'])
    result = model.detect_video(vid_generator, prompts = ["man", "tennis ball", "small green ball"], frame_count=frame_count)
    assert len(result) == frame_count
    for frame_detections in result:
        # there must be a ball detected. there could be multiple boxes because no NMS
        assert len(frame_detections) >= 1
