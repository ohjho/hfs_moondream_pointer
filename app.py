import json, tempfile
import gradio as gr
import spaces, torch
from gradio_client import Client, handle_file
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


_MODEL_ZOO = {
    "moondream3-preview": load_model(),  # calling spaces.GPU decorated functions outside ZeroGPU scope will cause a PickingError
    "moondream2": AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        revision="2025-06-21",
        trust_remote_code=True,
        device_map={"": "cuda"},
    ),
}


@spaces.GPU(duration=30)
def detect(
    im: Image.Image,
    object_name: str,
    mode: Literal["point", "object_detection", "query"],
    model_name: Literal["moondream2", "moondream3-preview"] = "moondream3-preview",
    reasoning: bool = False,
    settings: dict = {"temperature": 0.0, "top_p": 0.95, "max_tokens": 512},
) -> list[dict] | dict:
    """Open-vocabulary object localization and visual question answering with the Moondream vision-language model. The shape of the returned JSON depends on the "mode" argument. In "point" mode it returns a JSON list with one entry per located instance of "object_name"; each entry is an object {"x": float, "y": float} giving that instance's center as coordinates NORMALIZED to 0.0-1.0 of the input image (x is fraction of width from the left edge, y is fraction of height from the top edge). In "object_detection" mode it returns a JSON list with one entry per located instance; each entry is a bounding box {"x_min": float, "y_min": float, "x_max": float, "y_max": float} where all four values are NORMALIZED to 0.0-1.0 of the ORIGINAL input image ("x_min"/"y_min" is the top-left corner, "x_max"/"y_max" is the bottom-right corner) -- this is the albumentations "albumentations" bounding-box format (normalized [x_min, y_min, x_max, y_max]), documented at https://albumentations.ai/docs/3-basic-usage/bounding-boxes-augmentations/#bounding-box-formats -- values are NOT absolute pixels and the box is NOT [x, y, width, height]. In "query" mode it returns a JSON object {"answer": string} answering "object_name" as a natural-language question about the image.

    Args:
        im: The RGB image to run detection or answer a question about.
        object_name: In "point"/"object_detection" mode, the open-vocabulary name of the object to locate (e.g. "person", "red car"); in "query" mode, the natural-language question to ask about the image.
        mode: "point" returns normalized center points, "object_detection" returns normalized bounding boxes, "query" returns a text answer.
        model_name: Which Moondream checkpoint to run: "moondream3-preview" (default, more capable) or "moondream2".
        reasoning: Enable chain-of-thought reasoning; only applies to "query" mode on the moondream3-preview model and is ignored otherwise.
        settings: Inference parameters as a JSON object; for "query" mode use "temperature"/"top_p"/"max_tokens", for "point"/"object_detection" use "max_objects"; ignored for the moondream2 model.
    """

    model = _MODEL_ZOO[model_name]
    is_v2 = model_name == "moondream2"
    if isinstance(settings, str):
        settings = json.loads(settings)

    inf_params = {} if is_v2 else {"settings": settings}
    if not is_v2 and mode == "query":
        inf_params["reasoning"] = reasoning

    if mode == "point":
        return model.point(im, object_name, **inf_params)["points"]
    elif mode == "object_detection":
        return model.detect(im, object_name, **inf_params)["objects"]
    elif mode == "query":
        return model.query(im, object_name, **inf_params)


def annotate(im, result, mode):
    if not isinstance(result, list):
        return None  # query mode / cleared output
    boxes = [
        {
            "object_id": i,  # pilbox's default label_key / color_key
            "boundingBox": [r["x_min"], r["y_min"], r["x_max"], r["y_max"]]
            if mode == "object_detection"
            else [r["x"] - 0.01, r["y"] - 0.01, r["x"] + 0.01, r["y"] + 0.01],
        }
        for i, r in enumerate(result)
    ]
    im.save(path := tempfile.NamedTemporaryFile(suffix=".png", delete=False).name)
    return Client("GF-John/pilbox").predict(
        handle_file(path), json.dumps(boxes), "albumentations", api_name="/annotate"
    )


demo = gr.Interface(
    fn=detect,
    title="moondream-pointer",
    description="using [moondream3-preview](https://huggingface.co/moondream/moondream3-preview) and [moondream2](https://huggingface.co/vikhyatk/moondream2) for object grounding",
    inputs=[
        gr.Image(label="Input Image", type="pil"),
        gr.Textbox(
            label="Object / Question",
            info="object to detector (for points / object_detection) or question for a query",
        ),
        gr.Dropdown(label="Mode", choices=["point", "object_detection", "query"]),
        gr.Dropdown(label="Model Variant", choices=list(_MODEL_ZOO.keys())),
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
with demo, demo.output_components[0].parent:  # render inside the output column
    demo.output_components[0].change(
        annotate,
        [demo.input_components[0], demo.output_components[0], demo.input_components[2]],
        gr.Image(label="Annotated Image"),
        api_name=False,
    )
demo.launch(
    mcp_server=True, app_kwargs={"docs_url": "/docs"}  # add FastAPI Swagger API Docs
)
