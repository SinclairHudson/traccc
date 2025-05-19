"""
Used for detecting objects in a video, and saving to an output.
"""
import argparse
from typing import List
import os
import gradio as gr

import skvideo.io

from traccc.detectors import HuggingFaceDETR, PretrainedRN50Detector, OWLVITZeroShot

model_selector = {
    "DETR": HuggingFaceDETR,
    "RN50": PretrainedRN50Detector,
    "OWLVIT": OWLVITZeroShot
}

def run_detect(name: str, model: str, input_file: str, prompts: List[str] = None, progress=gr.Progress(track_tqdm=True)):
    vid_generator = skvideo.io.vreader(input_file)
    metadata = skvideo.io.ffprobe(input_file)
    frame_count = int(metadata['video']['@nb_frames'])

    print(f"frame_count: {frame_count}")
    detector = model_selector[model]()

    if not os.path.exists(f"internal"):
        os.system("mkdir internal")  # make internal if it doesn't exist
    detector.detect(
        vid_generator, filename=f"internal/{name}.npz", frame_count=frame_count, prompts=prompts)
    return f"Completed detection for project {name} using {model}."

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tracks objects using detections as input.")
    parser.add_argument("name", help="name of the project to be tracked.")
    parser.add_argument("--model", help="choice of model", default="DETR")
    parser.add_argument("--input", help="video file to be used", default=None)


    # input sanitization
    args = parser.parse_args()
    name = args.name
    input_file = args.input
    if input_file is None:
        input_file = f"io/{name}.mp4"


    assert os.path.exists(input_file), f"Input file {input_file} does not exist."
    assert args.model in model_selector, f"Model {args.model} isn't supported"

    run_detect(name, args.model, input_file)
