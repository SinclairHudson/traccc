import gradio as gr
import os
from traccc.draw import run_draw
from traccc.track import run_track
from traccc.detect import run_detect

def sanitize_run_detect(project_name: str, model_select: str, input_file: str, prompts: str = None,
                        progress=gr.Progress(track_tqdm=True)):
    """
    Sanitizes the input for running detection. Runs detection if input is valid.
    Args:
        project_name: name of the project, the slug for internal files.
        model_select: string for
    """
    if not os.path.exists("io/" + input_file):
        raise gr.Error(f"Input file '{input_file}' does not exist. Is the file" + \
                       " in the specificed io folder? Is the folder mounted correctly?")

    if model_select == "OWLVIT" and prompts is None:
        raise gr.Error(f"In order to use a zero-shot detector, you must specify what you want detected via a prompt")

    prompts = prompts.split(",") if prompts is not None else None
    return run_detect(project_name, model_select, "io/" + input_file, prompts)

def sanitize_run_track(name: str, track_type: str, death_time: int, iou_threshold: float, conf_threshold: float, max_cost: float):
    if not os.path.exists(f"internal/{name}.npz"):
        raise gr.Error(f"Couldn't find detections for this project. Is the project name" + \
                       " correct?")

    return run_track(name, track_type, death_time, iou_threshold, conf_threshold, max_cost)

def sanitize_run_draw(name: str, input_video: str, output: str, effect_name: str,
             colour: str, size: float, length: int, min_age: int, progress=gr.Progress(track_tqdm=True)):

    if not os.path.exists(f"internal/{name}.yaml"):
        raise gr.Error(f"Couldn't find tracks for this project. Is the project name" + \
                       " correct?")

    if not os.path.exists("io/" + input_video):
        raise gr.Error(f"Couldn't find input video '{input_video}'. Is the input video path correct?")

    return run_draw(name, "io/" + input_video, "io/" + output, effect_name, colour, size, length, min_age)

with gr.Blocks() as demo:
    gr.Markdown("Create cool ball tracking videos with this one simple trick!")
    project_name_input = gr.Textbox(placeholder="fireball", label="Project Name",
                            info="The name of the clip being processed. Remember \
                            this name and make it unique, because it's used in the \
                            next two steps as well. Using the same name will \
                            overwrite previous data!")
    # video_upload = gr.inputs.Video(label="Video File")
    input_file = gr.Textbox(
        placeholder="fireball.mp4", label="Input File",
        info="The name of the file in the io directory.")
    with gr.Tab("Detect"):
        model_select = gr.components.Radio(["DETR", "RN50", "OWLVIT"], label="Model")
        prompts = gr.Textbox(placeholder="juggling ball, dog", label="Prompts (comma separated)")

        detect_button = gr.Button("Detect", variant="primary")
        debug_textbox = gr.Textbox(label="Output")
        detect_button.click(sanitize_run_detect, inputs=[
                            project_name_input, model_select, input_file, prompts], outputs=[debug_textbox])

    with gr.Tab("Track"):
        track_type_input = gr.components.Radio(
            ["Constant Acceleration", "Constant Velocity"], label="Track Type", value="Constant Acceleration")
        death_time = gr.Slider(label="Death Time", minimum=1,
                               maximum=20, value=5, interactive=True, step=1)
        iou_threshold = gr.Slider(
            label="IoU Threshold", minimum=0.01, maximum=1, value=0.20, interactive=True)
        confidence_treshold = gr.Slider(
            label="Confidence Threshold", minimum=0, maximum=1, value=0.05, interactive=True)
        max_cost = gr.Slider(label="Maximum Matching Cost",
                             minimum=0, maximum=1000, value=200, interactive=True)
        track_button = gr.Button("Track", variant="primary")
        track_debug_textbox = gr.Textbox(label="Output")
        track_button.click(sanitize_run_track, inputs=[project_name_input, track_type_input, death_time,
                           iou_threshold, confidence_treshold, max_cost], outputs=[track_debug_textbox])

    with gr.Tab("Draw"):
        output_file = gr.Textbox(placeholder="fireball_with_effect.mp4", label="Output File",
                                 info="The video file to be created")
        # TODO these need to be in a constants file.
        effect_name = gr.components.Radio(["dot", "lagging_dot",
                                           "line", "highlight_line", "neon_line",
                                           "contrail", "fully_connected",
                                           "fully_connected_neon", "debug", "tricolor"], label="Effect")
        colour = gr.ColorPicker(label="Colour", value="#ff0000")
        size = gr.Slider(label="size", info="size of the effect, proportional to \
                               the width of the object being tracked.",
                               minimum=0, maximum=20, value=1, interactive=True)
        length = gr.Slider(label="length", info="length of the effect in frames",
                           minimum=1, maximum=50, value=7, interactive=True, step=1)
        min_age = gr.Slider(label="minimum age", info="minimum age requirement (in frames) for \
                            a track to be visualized. Increasing this value will remove tracks that \
                            are short-lived, possibly false-positives.",
                            minimum=1, maximum=50, value=7, interactive=True, step=1)

        draw_button = gr.Button("Draw Effect", variant="primary")

        draw_debug_textbox = gr.Textbox(label="Output")
        draw_button.click(sanitize_run_draw, inputs=[project_name_input, input_file, output_file, effect_name, colour, size, length, min_age],
                          outputs=draw_debug_textbox)

demo.queue().launch(server_name="0.0.0.0")
