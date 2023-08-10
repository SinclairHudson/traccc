import gradio as gr
from detectors import HuggingFaceDETR, PretrainedRN50Detector
import os

def detect(name: str, model: str = "DETR", input_file: str = None):
    if input_file is None:
        input_file = f"io/{name}.mp4"

    model_selector = {
        "DETR": HuggingFaceDETR,
        "RN50": PretrainedRN50Detector
    }

    assert os.path.exists(input_file), f"Input file {input_file} does not exist."
    vid_generator = skvideo.io.vreader(input_file)
    metadata = skvideo.io.ffprobe(input_file)
    frame_count = int(metadata['video']['@nb_frames'])

    detector = model_selector[args.model]()
    if not os.path.exists(f"internal"):
        os.system("mkdir internal")  # make internal if it doesn't exist
    detector.detect(
        vid_generator, filename=f"internal/{name}.npz", frame_count=frame_count)


with gr.Blocks() as demo:
    gr.Markdown("Create cool ball tracking videos!")
    with gr.Tab("Detect"):
        text_input = gr.Textbox()
        text_output = gr.Textbox()
        text_button = gr.Button("Detect")
    with gr.Tab("Track"):
        image_input = gr.Image()
        image_output = gr.Image()
        image_button = gr.Button("Track")
    with gr.Tab("Draw"):
        text_input = gr.Textbox()
        text_output = gr.Textbox()
        text_button = gr.Button("Draw Effect")

    # text_button.click(detect, inputs=text_input, outputs=text_output)

demo.launch(server_name="127.0.0.1", server_port=8080, share=True)
