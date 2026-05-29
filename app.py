import json
import gradio as gr
import spaces, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from typing import Literal


# @spaces.GPU(duration=30)
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


_MODEL = (
    load_model()
)  # calling spaces.GPU decorated functions outside ZeroGPU scope will cause a PickingError


@spaces.GPU(duration=30)
def detect(
    im: Image.Image,
    object_name: str,
    mode: Literal["point", "object_detection", "query"],
    reasoning: bool = False,
    settings: dict = {"temperature": 0.0, "top_p": 0.95, "max_tokens": 512},
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
    if isinstance(settings, str):
        settings = json.loads(settings)
    if mode == "point":
        return model.point(im, object_name, settings=settings)["points"]
    elif mode == "object_detection":
        return model.detect(im, object_name, settings=settings)["objects"]
    elif mode == "query":
        return model.query(im, object_name, reasoning=reasoning, settings=settings)


demo = gr.Interface(
    fn=detect,
    title="moondream-pointer",
    description="using [moondream3-preview](https://huggingface.co/moondream/moondream3-preview) for object grounding",
    inputs=[
        gr.Image(label="Input Image", type="pil"),
        gr.Textbox(
            label="Object / Question",
            info="object to detector (for points / object_detection) or question for a query",
        ),
        gr.Dropdown(label="Mode", choices=["point", "object_detection", "query"]),
        gr.Checkbox(
            label="Reasoning",
            value=False,
            info="enable [chain-of-thought](https://huggingface.co/moondream/moondream3-preview#query) (query mode only)",
        ),
        gr.Textbox(
            label="Settings (JSON)",
            value='{"temperature": 0.0, "top_p": 0.95, "max_tokens": 512}',
            info="query: temperature / top_p / max_tokens · point & object_detection: max_objects",
        ),
    ],
    outputs=gr.JSON(label="Output JSON"),
)
demo.launch(
    mcp_server=True, app_kwargs={"docs_url": "/docs"}  # add FastAPI Swagger API Docs
)
