"""
Module for drawing effects on videos.
"""
import yaml
import argparse
import skvideo.io
from traccc import effects
from tqdm import tqdm
from traccc import filters
import cv2
from typing import Tuple
import gradio as gr

def hex_to_bgr(rgb_hex: str) -> Tuple[int, int, int]:
    """Converts a hex code to BGR."""
    rgb_hex = rgb_hex.lstrip('#')
    rgb = [int(rgb_hex[i:i+2], 16) for i in (0, 2, 4)]
    return rgb

def run_draw(name: str, input_video: str, output: str, effect_name: str,
             colour: str, size: float, length: int, min_age: int, progress=gr.Progress(track_tqdm=True)):
    """
    Runs the drawing portion of the pipeline.
    Inputs are already expected to be sanitized.
    Args:
        name: Name of the project to be drawn on.
        input_video: Path to the input video file.
        output: Path to the output video file.
        effect_name: Name of the effect to be used.
        colour: Colour of the effect in hex format.
        size: Size of the effect, relative to the width of the object.
        length: Length of the effect in frames.
        min_age: Minimum age (in frames) for tracks to be drawn.
    """
    vid_generator = skvideo.io.vreader(input_video)
    metadata = skvideo.io.ffprobe(input_video)
    frame_count = int(metadata['video']['@nb_frames'])
    fps = metadata['video']['@r_frame_rate']
    numerator, denominator = map(int, fps.split('/'))
    fps = numerator / denominator
    width = int(metadata['video']['@width'])
    height = int(metadata['video']['@height'])
    # rotation = int(metadata['video']['tag'][0]['@value'])
    # we also probably need to rotate
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # if rotation == 90 or rotation == 270: # TODO this is janky
    opencv_out = cv2.VideoWriter(
        output, fourcc, fps, (width, height))
    # else:
        # opencv_out = cv2.VideoWriter(
            # output, fourcc, fps, (width, height))

    with open(f"internal/{name}.yaml", 'r') as f:
        track_dictionary = yaml.safe_load(f)

    rgb_color = hex_to_bgr(colour)
    effect = {
        "dot": effects.Dot,
        "lagging_dot": effects.LaggingDot,
        "line": effects.Line,
        "highlight_line": effects.HighlightLine,
        "neon_line": effects.NeonLine,
        "contrail": effects.Contrail,
        "fully_connected": effects.FullyConnected,
        "fully_connected_neon": effects.FullyConnectedNeon,
        "debug": effects.Debug,
        "tricolor": effects.TriColor
    }[effect_name](rgb_color, length, size)

    tracks = track_dictionary["tracks"]

    # filter out all the tracks that we deem not good enough
    tracks = [track for track in tracks if filters.standard_filter(
        track, min_age=min_age)]

    print("adding effect")
    # TODO workaround to issue https://github.com/gradio-app/gradio/issues/3841
    # revert to the below line when bug is fixed
    # for i, frame in tqdm(enumerate(vid_generator), total=frame_count):
    for (frame, i) in zip(vid_generator, tqdm(range(frame_count))):
        relevant_tracks = [
            track for track in tracks if effect.relevant(track, i)]

        # loop through all tracks, draw each on the frame
        out_frame = effect.draw_tracks(frame, relevant_tracks, i)

        bgr_frame = cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR)
        opencv_out.write(bgr_frame)

    opencv_out.release()
    return f"successfully wrote video {output}."


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Draws effects on the video, based on the tracks")
    parser.add_argument("name", help="name of the project")
    parser.add_argument("--input", help="video file to be used", default=None)
    parser.add_argument(
        "--effect", help="name of effect you wish to use", default="line")
    parser.add_argument(
        "--min_age", help="tracks below this age don't get drawn", default=0)
    parser.add_argument("--colour", help="colour of the effect", default="#ff0000")
    parser.add_argument(
        "--length", help="length of the effect in frames", default=10)
    parser.add_argument(
        "--size", help="size or width of the effect", default=1.0)
    parser.add_argument(
        "--output", help="the output file", default=None)
    args = parser.parse_args()
    name = args.name
    effect = args.effect
    colour = args.colour
    length = int(args.length)
    size = float(args.size)
    if args.output == None:
        output = f"io/{name}_out.mp4"
    else:
        output = args.output

    if args.input == None:
        input_video = f"io/{name}.mp4"
    else:
        input_video = args.input

    run_draw(name, input_video, output, effect, args.colour,
             size,length, int(args.min_age))

