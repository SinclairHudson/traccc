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

    assert os.path.exists(
        input_file), f"Input file {input_file} does not exist."
    vid_generator = skvideo.io.vreader(input_file)
    metadata = skvideo.io.ffprobe(input_file)
    frame_count = int(metadata['video']['@nb_frames'])

    detector = model_selector[args.model]()
    if not os.path.exists(f"internal"):
        os.system("mkdir internal")  # make internal if it doesn't exist
    detector.detect(
        vid_generator, filename=f"internal/{name}.npz", frame_count=frame_count)


with gr.Blocks() as demo:
    gr.Markdown("Create cool ball tracking videos with this one simple trick!")
    with gr.Tab("Detect"):
        text_input = gr.Textbox(placeholder="fireball", label="Project Name",
                                info="The name of the clip being processed. Remember \
                                this name and make it unique, because it's used in the \
                                next two steps as well. Using the same name will \
                                overwrite previous data!")
        # video_upload = gr.inputs.Video(label="Video File")
        input_file = gr.Textbox(placeholder="io/fireball.mp4", label="Input File")
        model_select = gr.inputs.Radio(["DETR", "RN50"], label="Model")
        text_button = gr.Button("Detect")
    with gr.Tab("Track"):
        text_input = gr.Textbox(placeholder="fireball", label="Project Name")
        death_time = gr.Slider(label="Death Time", minimum=1, maximum=20, value=5, interactive=True, step=1)
        iou_threshold = gr.Slider(label="IoU Threshold", minimum=0.01, maximum=1, value=0.20, interactive=True)
        confidence_treshold = gr.Slider(label="Confidence Threshold", minimum=0, maximum=1, value=0.05, interactive=True)
        image_button = gr.Button("Track")
    with gr.Tab("Draw"):
        text_input = gr.Textbox(placeholder="fireball", label="Project Name")
        input_file = gr.Textbox(placeholder="io/fireball.mp4", label="Input File", info="The input file \
                                should be the same as the one used in the Detect step.")
        output_file = gr.Textbox(placeholder="io/fireball_with_effect.mp4", label="Output File",
                                info="The video file to be created")
        death_time = gr.Slider(label="size", info="size of the effect, proportional to \
                               the width of the object being tracked.",
                               minimum=0, maximum=20, value=1, interactive=True)
        length = gr.Slider(label="length", info="length of the effect in frames",
                               minimum=1, maximum=50, value=7, interactive=True, step=1)
        min_age = gr.Slider(label="minimum age", info="minimum age requirement (in frames) for \
                            a track to be visualized. Increasing this value will remove tracks that \
                            are short-lived, possibly false-positives.",
                               minimum=1, maximum=50, value=7, interactive=True, step=1)
        model_select = gr.inputs.Radio(["Dot", "Lagging Dot",
                                        "Line", "Highlight Line",
                                        "Contrail", "Fully Connected",
                                        "Fully Connected Neon", "Debug"], label="Effect")
        colour = gr.ColorPicker(label="Colour")

        draw_button = gr.Button("Draw Effect")

    # text_button.click(detect, inputs=text_input, outputs=text_output)

demo.launch(server_name="0.0.0.0")
