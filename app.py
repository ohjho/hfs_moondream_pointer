import gradio as gr
import spaces, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from typing import Literal


@spaces.GPU
def load_model():
    return AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        revision="2025-04-14",
        trust_remote_code=True,
        device_map={"": "cuda"},
    )


@spaces.GPU
def detect(
    im: Image.Image, object_name: str, mode: Literal["point", "object_detection"]
):
    """
    Open Vocabulary Detection using moondream2

    Args:
        im: Pillow Image
        object_name: the object you would like to detect
        mode: point or object_detection
    Returns:
        list: a list of bounding boxes (xyxy) or points (xy) coordinates that are normalized
    """
    model = load_model()
    if mode == "point":
        return model.point(im, object_name)["points"]
    elif mode == "object_detection":
        return model.detect(im, object_name)["objects"]


demo = gr.Interface(
    fn=detect,
    inputs=[
        gr.Image(label="Input Image", type="pil"),
        gr.Textbox(label="Object to Detect"),
        gr.Dropdown(label="Mode", choices=["point", "object_detection"]),
    ],
    outputs=gr.Textbox(label="Output Text"),
)
demo.launch(
    mcp_server=True, app_kwargs={"docs_url": "/docs"}  # add FastAPI Swagger API Docs
)
