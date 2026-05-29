import gradio as gr
import spaces, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from typing import Literal


@spaces.GPU(duration=60)
def load_model():
    # moondream2
    # return AutoModelForCausalLM.from_pretrained(
    #     "vikhyatk/moondream2",
    #     revision="2025-04-14",
    #     trust_remote_code=True,
    #     device_map={"": "cuda"},
    # )

    # moondream3-preview
    moondream = AutoModelForCausalLM.from_pretrained(
        "moondream/moondream3-preview",
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map={"": "cuda"},
    )
    moondream.compile()
    return moondream


_MODEL = load_model()


@spaces.GPU(duration=30)
def detect(
    im: Image.Image,
    object_name: str,
    mode: Literal["point", "object_detection", "query"],
):
    """
    Open Vocabulary Detection and Visual Question Answering using moondream2

    Args:
        im: Pillow Image
        object_name: the object you would like to detect, or the question to ask when mode is "query"
        mode: point, object_detection, or query
    Returns:
        For "point" / "object_detection": a list of points (xy) or bounding boxes (xyxy) with normalized coordinates.
        For "query": a dict {"answer": str} with the answer to the question.
    """
    model = _MODEL  # load_model()
    if mode == "point":
        return model.point(im, object_name)["points"]
    elif mode == "object_detection":
        return model.detect(im, object_name)["objects"]
    elif mode == "query":
        return model.query(im, object_name)


demo = gr.Interface(
    fn=detect,
    inputs=[
        gr.Image(label="Input Image", type="pil"),
        gr.Textbox(
            label="Object / Question",
            info="object to detector (for points / object_detection) or question for a query",
        ),
        gr.Dropdown(label="Mode", choices=["point", "object_detection", "query"]),
    ],
    outputs=gr.JSON(label="Output JSON"),
)
demo.launch(
    mcp_server=True, app_kwargs={"docs_url": "/docs"}  # add FastAPI Swagger API Docs
)
