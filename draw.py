import yaml
import skvideo.io
from torchvision.io import read_video, write_video
from effects import *
from tqdm import tqdm
from filters import *

def draw(video, tracks, effect):
    for track in tqdm(tracks):
        effect(video, track)

if __name__ == "__main__":
    print("reading video")
    vid = skvideo.io.vread("io/test.mp4")

    print("reading yaml")
    with open("io/living_room.yaml", 'r') as f:
        track_dictionary = yaml.safe_load(f)

    tracks = track_dictionary["tracks"]
    tracks = [track for track in tracks if standard_filter(track, min_age=50)]
    draw(vid, tracks, aging_dot)

    # write out video
    print("writing video")
    skvideo.io.vwrite("io/outputvideo.mp4", vid)
    # write_video("io/living_room_red_dot.mp4", vid, video_codec="h264", fps=60.0)

