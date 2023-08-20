import gradio as gr
import os
from draw import run_draw
from track import run_track
from detect import run_detect

with gr.Blocks() as demo:
    gr.Markdown("Create cool ball tracking videos with this one simple trick!")
    with gr.Tab("Detect"):
        text_input = gr.Textbox(placeholder="fireball", label="Project Name",
                                info="The name of the clip being processed. Remember \
                                this name and make it unique, because it's used in the \
                                next two steps as well. Using the same name will \
                                overwrite previous data!")
        # video_upload = gr.inputs.Video(label="Video File")
        input_file = gr.Textbox(
            placeholder="io/fireball.mp4", label="Input File")
        model_select = gr.components.Radio(["DETR", "RN50"], label="Model")
        detect_button = gr.Button("Detect", variant="primary")
        debug_textbox = gr.Textbox(label="Output")
        detect_button.click(run_detect, inputs=[
                            text_input, model_select, input_file], outputs=[debug_textbox])

    with gr.Tab("Track"):
        track_name_input = gr.Textbox(
            placeholder="fireball", label="Project Name")
        track_type_input = gr.components.Radio(
            ["AccelTrack", "Track"], label="Track Type", value="AccelTrack")
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
        track_button.click(run_track, inputs=[track_name_input, track_type_input, death_time,
                           iou_threshold, confidence_treshold, max_cost], outputs=[track_debug_textbox])

    with gr.Tab("Draw"):
        draw_name = gr.Textbox(placeholder="fireball", label="Project Name")
        draw_input_file = gr.Textbox(placeholder="io/fireball.mp4", label="Input File", info="The input file \
                                should be the same as the one used in the Detect step.")
        output_file = gr.Textbox(placeholder="io/fireball_with_effect.mp4", label="Output File",
                                 info="The video file to be created")
        effect_name = gr.components.Radio(["dot", "lagging_dot",
                                           "line", "highlight_line",
                                           "contrail", "fully_connected",
                                           "fully_connected_neon", "debug"], label="Effect")
        colour = gr.ColorPicker(label="Colour")
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
        draw_button.click(run_draw, inputs=[draw_name, draw_input_file, output_file, effect_name, colour, size, length, min_age],
                          outputs=draw_debug_textbox)

demo.queue().launch(server_name="0.0.0.0")