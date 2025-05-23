import gradio as gr
import spaces, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image


@spaces.GPU
def load_model():
    return AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        revision="2025-04-14",
        trust_remote_code=True,
        device_map={"": "cuda"},
    )


@spaces.GPU
def point(im: Image.Image, object_name: str):
    model = load_model()
    return model.detect(im, object_name)["objects"]


demo = gr.Interface(
    fn=point,
    inputs=[
        gr.Image(label="Input Image", type="pil"),
        gr.Textbox(label="Object to Detect"),
    ],
    outputs=gr.Textbox(label="Output Text"),
)
demo.launch()
