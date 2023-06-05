import yaml
import argparse
import skvideo.io
from torchvision.io import read_video, write_video
from effects import *
from tqdm import tqdm
from filters import *

def draw(video, tracks, effect):
    for track in tqdm(tracks):
        effect(video, track)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draws effects on the video, based on the tracks")
    parser.add_argument("name", help="name of the project")
    parser.add_argument("--effect", help="name of effect you wish to use", default="red_dot")
    parser.add_argument("--min_age", help="tracks below this age don't get drawn", default=0)
    args = parser.parse_args()
    name = args.name
    effect = args.effect
    vid_generator = skvideo.io.vreader(f"io/{name}.mp4")
    vid_writer = skvideo.io.FFmpegWriter(f"io/{name}_out.mp4")
    metadata = skvideo.io.ffprobe(f"io/{name}.mp4")
    frame_count = int(metadata['video']['@nb_frames'])

    print("reading yaml")
    with open(f"internal/{name}.yaml", 'r') as f:
        track_dictionary = yaml.safe_load(f)

    effect = {
        "red_dot": RedDot()
        # add more here
    }[effect]
    tracks = track_dictionary["tracks"]

    # filter out all the tracks that we deem not good enough
    tracks = [track for track in tracks if standard_filter(track, min_age=int(args.min_age))]

    breakpoint()
    for i, frame in tqdm(enumerate(vid_generator), total=frame_count):
        relevant_tracks = [track for track in tracks if effect.relevant(track, i)]
        out_frame = frame

        # loop through all tracks, draw each on the frame
        for track in relevant_tracks:
            out_frame = effect.draw(out_frame, track, i)

        vid_writer.writeFrame(out_frame)

    vid_writer.close()
    print(f"successfully wrote video io/{name}_out.mp4")

