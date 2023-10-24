import gradio as gr
import libs.foxutils.utils.core_utils as core_utils
from libs.foxutils.utils.gradio_utils import get_target_directory

from utils.map_utils import print_expressway_camera_locations
from utils.settings import DEFAULT_CAMERA_ID, DEFAULT_IMAGE_FILE, EXPRESSWAY_CAMERA_IDS, DEFAULT_DATASET_DIR, \
    DEFAULT_FILEPATH
import provide_insights
import time
from os.path import join as pathjoin
from PIL import Image

device = core_utils.device
HISTORY_LENGTH = int(core_utils.settings['MODELS']['total_vehicles_prediction_model_time_step'])


def get_insights(full_filename):
    image_file, folder, _ = provide_insights.split_filename_folder(full_filename)

    start_time = time.time()
    start_time_cpu = time.process_time()

    results = provide_insights.analyze(full_filename, delete_previous_results=False, history_length=HISTORY_LENGTH)
    end_time = time.time() - start_time
    end_time_cpu = time.process_time() - start_time_cpu
    print(f'\n\nFinished execution.\nRuntime: {end_time:.4f}sec\nCPU runtime: {end_time_cpu:.4f}sec\n\n')

    vehicle_img = results['object_detection']['image']
    object_df = results['object_detection']['dataframe']
    weather_label = {results['weather_detection']['label']: results['weather_detection']['prob']}
    anomaly_img = results['anomaly_detection']['heat_map_image']
    anomaly_label = {results['anomaly_detection']['label']: results['anomaly_detection']['prob']}

    vf_df = results['vehicle_forecasting']['dataframe']
    vf_predictions = results['vehicle_forecasting']['predictions']

    vehicle_prediction_str = "-Previous Counts: [" \
                             + ", ".join([str(round(x)) for x in vf_df['total_vehicles'].values]) \
                             + "]\n-Predicted Count (next 5min): " + str(round(vf_predictions[0]))
    return vehicle_img, object_df, weather_label, anomaly_label, anomaly_img, vehicle_prediction_str


def get_target_image(run_mode, camera_selection):
    if run_mode == "Live":
        raise NotImplementedError

    elif run_mode == "Test":
        image_file = pathjoin(DEFAULT_DATASET_DIR, DEFAULT_IMAGE_FILE)
        print(f'Reading image from {image_file}')
        img = Image.open(image_file)
        fig = print_expressway_camera_locations([camera_selection])
        return img, fig


def make_tab_expressway_camera():
    with gr.Row():
        with gr.Column(scale=1, min_width=600):
            gr.Markdown(
                """
                ###  Input from File
                """)

            with gr.Row():
                dropdown_button = gr.Dropdown(label="Camera ID", choices=EXPRESSWAY_CAMERA_IDS,
                                              allow_custom_value=False, value=DEFAULT_CAMERA_ID)

                mode_button = gr.Radio(label="Fetch Mode", choices=["Live", "Test"], value="Test")

            if mode_button.value == "Live":
                file1 = gr.File(label="Target file", scale=0, value=DEFAULT_FILEPATH)

                file_dir = gr.Textbox(label="Target Directory", value=DEFAULT_DATASET_DIR,
                                      info="Select directory from where the historical images will be fetched.",
                                      scale=0)
                with gr.Row():
                    directory_button = gr.Button("Select directory", size="sm", variant="primary", scale=0)

                directory_button.click(fn=get_target_directory, inputs=[], outputs=file_dir)

            else:
                pass

            with gr.Row():
                view_button = gr.Button("View", size="sm", variant="primary")
                submit_button = gr.Button("Submit", size="sm", variant="primary")

            img1 = gr.Image(label="Target image", scale=0)
            map_plot = gr.Plot(label="Camera Locations", scale=0)

        with gr.Column(scale=1, min_width=600):
            gr.Markdown(
                """
                ###  Results
                """)
            # Output components
            img2 = gr.Image(label="Object Detection")
            df1 = gr.DataFrame(label="Detected Objects")
            with gr.Row():
                vehicle_prediction = gr.Textbox(label="Vehicle Forecasting", interactive=False)
                label_weather = gr.Label(label="Weather Detection")

            with gr.Row():
                img3 = gr.Image(label="Anomaly Detection Heatmap")
                label_anomaly = gr.Label(label="Anomaly Detection")

    if mode_button.value == "Test":
        view_button.click(fn=get_target_image, inputs=[mode_button, dropdown_button], outputs=[img1, map_plot])

        submit_button.click(fn=lambda: get_insights(DEFAULT_FILEPATH), inputs=[],
                            outputs=[img2, df1, label_weather, label_anomaly, img3, vehicle_prediction])


def make_tab_dashcam():
    with gr.Row():
        with gr.Column(scale=1, min_width=600):
            gr.Markdown(
                """
                ###  Input from Dashcam
                """)


def make_tab_livestream():
    with gr.Row():
        with gr.Column(scale=1, min_width=600):
            gr.Markdown(
                """
                ###  Input from Livecam
                """)


with gr.Blocks() as demo:
    gr.Markdown(
        """
                # Digital twin of Singapore
                Describe the current situation on the road.
                """)

    with gr.Tab("Expressway Camera"):
        make_tab_expressway_camera()

    with gr.Tab("Dashcam"):
        make_tab_dashcam()

    with gr.Tab("Livestream"):
        make_tab_livestream()

if __name__ == "__main__":
    #demo.launch(inbrowser=False, show_error=True, debug=True, server_port=7861, favicon_path=pathjoin('assets', 'favicon.ico'))
    demo.launch(show_error=True, favicon_path=pathjoin('assets', 'favicon.ico'))
