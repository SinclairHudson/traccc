import yaml
import argparse
import skvideo.io
from torchvision.io import read_video, write_video
from effects import *
from tqdm import tqdm
from filters import *
import cv2
from typing import Tuple


def run_draw(name: str, input_file: str, output:str, effect_name: str, colour: Tuple, size: float, length: int, min_age: int):
    """
    Inputs are already expected to be sanitized
    """
    vid_generator = skvideo.io.vreader(input_video)
    metadata = skvideo.io.ffprobe(input_video)
    frame_count = int(metadata['video']['@nb_frames'])
    fps = metadata['video']['@r_frame_rate']
    numerator, denominator = map(int, fps.split('/'))
    fps = numerator / denominator
    width = int(metadata['video']['@width'])
    # we also probably need to rotate
    height = int(metadata['video']['@height'])
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # TODO make sure width, height is correct in following line
    opencv_out = cv2.VideoWriter(
        output, fourcc, fps, (height, width))  # TODO explore why sometimes this needs to be flipped.

    with open(f"internal/{name}.yaml", 'r') as f:
        track_dictionary = yaml.safe_load(f)

    effect = {
        "dot": Dot(colour, size),
        "lagging_dot": LaggingDot(colour, length, size),
        "line": Line(colour, length, size),
        "highlight_line": HighlightLine(colour, length, size),
        "contrail": Contrail(colour, length, size),
        "fully_connected": FullyConnected(colour, size),
        "fully_connected_neon": FullyConnectedNeon(colour, size),
        "debug": Debug(length, size),
    }[effect_name]

    tracks = track_dictionary["tracks"]

    # filter out all the tracks that we deem not good enough
    tracks = [track for track in tracks if standard_filter(
        track, min_age=int(args.min_age))]

    print("adding effect")
    # for i, frame in tqdm(enumerate(vid_generator), total=frame_count):
    for i, frame in tqdm(enumerate(vid_generator), total=frame_count):
        relevant_tracks = [
            track for track in tracks if effect.relevant(track, i)]

        # loop through all tracks, draw each on the frame
        out_frame = effect.draw_tracks(frame, relevant_tracks, i)

        bgr_frame = cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR)
        opencv_out.write(bgr_frame)

    opencv_out.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Draws effects on the video, based on the tracks")
    parser.add_argument("name", help="name of the project")
    parser.add_argument("--input", help="video file to be used", default=None)
    parser.add_argument(
        "--effect", help="name of effect you wish to use", default="line")
    parser.add_argument(
        "--min_age", help="tracks below this age don't get drawn", default=0)
    parser.add_argument("--colour", help="colour of the effect", default="red")
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

    run_draw(name, input_video, effect, args.min_age, length, size, output)

    print(f"successfully wrote video io/{name}_out.mp4")
