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
    parser = argparse.ArgumentParser(description="Tracks objects using detections as input.")
    parser.add_argument("name", help="name of the project to be tracked.")
    args = parser.parse_args()
    name = args.name
    print("reading video")
    vid = skvideo.io.vread(f"io/{name}.mp4", num_frames=100)

    print("reading yaml")
    with open(f"internal/{name}.yaml", 'r') as f:
        track_dictionary = yaml.safe_load(f)

    tracks = track_dictionary["tracks"]
    tracks = [track for track in tracks if standard_filter(track, min_age=2)]
    draw(vid, tracks, temporal_line)

    # write out video
    print("writing video")
    skvideo.io.vwrite(f"io/{name}_out.mp4", vid)

