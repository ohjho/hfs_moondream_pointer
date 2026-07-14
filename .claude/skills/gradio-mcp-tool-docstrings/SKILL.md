---
name: gradio-mcp-tool-docstrings
description: Write Gradio function docstrings that survive Gradio's MCP schema parser so the exposed tool is clear to LLM clients. Use this WHENEVER you write or edit a Gradio function that is (or will be) exposed over MCP, add mcp_server=True, set an api_name, or document a Gradio tool's description, parameters, or output schema — even if the user just says "make the tool description clearer" or "document the outputs". Especially load this before touching docstrings on functions wired into .click()/.submit()/gr.Interface in a Gradio app.
---

# Gradio MCP tool docstrings

When a Gradio app runs with `mcp_server=True`, every API endpoint becomes an MCP tool, and the
tool's schema — the description an LLM reads, and the per-parameter docs — is built **from the
Python function's docstring and type hints**. The catch: Gradio's docstring parser
(`gradio/utils.py::get_function_description`) is primitive and *silently* mangles anything not
shaped exactly right. A docstring that reads beautifully to a human can still produce a broken,
truncated, or URL-stripped tool schema. This skill encodes what actually survives the parser, so
your tools are unambiguous to the model calling them.

The rules below were verified empirically against a live `/gradio_api/mcp/schema` endpoint — trust
them over how the docstring "looks."

## How Gradio turns a function into an MCP tool

- **Tool name = the Python function name.** It is NOT the `api_name` you pass to `.click()` —
  `api_name` only names the REST endpoint. So if you want the MCP tool called `detect_objects`,
  **name the function `detect_objects`**. Renaming via `api_name` alone does nothing for MCP.
- **Tool description = the docstring prose *before* the `Args:` line.**
- **Per-parameter descriptions = the `Args:` section.**
- **Enable it:** `demo.launch(mcp_server=True)`. This needs **Gradio ≥ 5.28** and the
  **`gradio[mcp]`** extra installed (on HF Spaces: bump `sdk_version` in `README.md` and add
  `gradio[mcp]` to `requirements.txt`). On older Gradio, `mcp_server=True` is a no-op/error.

## The parser's three rules (and the traps)

Internalize these three — most broken schemas come from violating rule 2 or 3.

1. **Description block** (every line before `Args:`/`Returns:`): stripped and joined verbatim with
   spaces. **Safe for anything** — URLs, colons, parentheses, multi-sentence prose. This is the
   one place nothing gets mangled, so it is where the important content belongs.

2. **`Args:` section**: each line is split on the first `:` into `name: description`. A wrapped
   continuation line has no `:`, so **it is silently dropped**. Consequence: if a parameter
   description spans two lines, the second line vanishes and you won't be warned.
   → **Keep every parameter description on a single line**, however long.

3. **`Returns:` section**: each line is split on the first `:` and **everything before the colon is
   discarded**. This quietly destroys machine-facing detail:
   - a URL becomes garbage — `https://example.com` loses its `https:` and turns into
     `//example.com`;
   - field-labelled lines like `class_id (int): COCO index` lose the `class_id (int)` part.
   → **Do not put schema detail, field names, or URLs in `Returns:`.** Fold the output description
   into the description-block prose instead.

## Checklist for a rock-solid docstring

- Put the **full output schema in the description prose** (rule 1 keeps it intact), including any
  reference URLs.
- **One line per `Args:` entry** (rule 2).
- **Skip or minimize `Returns:`** (rule 3) — it can't hold URLs or field names safely.
- **Name the function** exactly what you want the MCP tool called.
- Give a real **return type hint** (e.g. `-> tuple[Image.Image, list[dict]]`) for a cleaner schema.
- Make outputs **JSON-serializable** (see below).
- After editing, **verify against the live schema** (see Verification) — never trust appearances.

## Coordinate / bounding-box outputs: always name the format

A bare number array like `[a, b, c, d]` is meaningless to a caller — the same four numbers mean
different boxes under different conventions. **Whenever a tool returns bounding boxes or any
coordinate/geometry, name the format explicitly in the description** and link a canonical
reference. Use the albumentations naming as the shared vocabulary:
https://albumentations.ai/docs/3-basic-usage/bounding-boxes-augmentations/#bounding-box-formats

Albumentations formats cheat-sheet — state which one you emit, and whether values are absolute
pixels or normalized 0–1:

| Format          | Coordinates                              | Values          |
|-----------------|------------------------------------------|-----------------|
| `pascal_voc`    | `[x_min, y_min, x_max, y_max]`           | absolute pixels |
| `albumentations`| `[x_min, y_min, x_max, y_max]`           | normalized 0–1  |
| `coco`          | `[x_min, y_min, width, height]`          | absolute pixels |
| `yolo`          | `[x_center, y_center, width, height]`    | normalized 0–1  |
| `cxcywh`        | `[x_center, y_center, width, height]`    | absolute pixels |

Also state the **coordinate space** (which image the pixels are relative to). In this repo,
detections come from supervision `detections.xyxy` → `[x_min, y_min, x_max, y_max]` in **absolute
pixels of the ORIGINAL input image** = albumentations **`pascal_voc`**. Say all of that; say what
it is NOT (not normalized, not `[x, y, width, height]`).

## JSON-serializability

MCP structured output must be plain JSON. Detection/ML libraries hand back numpy scalars, which can
break serialization. Convert before returning: `int(...)` for indices/ids, `float(...)` for scores,
`.tolist()` for arrays. In this repo see `detect_and_annotate` in `app.py`.

## Worked example (this repo's `detect_objects` in `app.py`)

Note: schema/URL/field detail lives in the prose; each `Args:` line is single-line; no `Returns:`.

```python
def detect_objects(input_image, confidence, resolution, checkpoint) -> tuple[Image.Image, list[dict]]:
    """Detect objects in an image using RF-DETR and return the annotated image plus structured detections.

    RF-DETR ... It produces two outputs. Output 1 is the annotated image. Output 2 is a JSON list
    of detections; each entry has: "class_id" (integer COCO index 0-79); "classname" (string);
    "confidence" (float 0.0-1.0); and "bounding_box". The "bounding_box" is [x_min, y_min, x_max,
    y_max] in ABSOLUTE PIXEL coordinates of the ORIGINAL input image (top-left & bottom-right
    corners) -- the albumentations "pascal_voc" format, documented at
    https://albumentations.ai/docs/3-basic-usage/bounding-boxes-augmentations/#bounding-box-formats .
    NOT normalized to 0-1 and NOT [x, y, width, height].

    Args:
        input_image: The RGB image to run object detection on.
        confidence: Minimum confidence score from 0.0 to 1.0; detections below this are discarded.
        resolution: Square inference resolution in pixels; snapped to the backbone's multiple.
        checkpoint: Which model to run: "base" (faster) or "large" (more accurate).
    """
```

## Verification (do this after every docstring change)

1. Run the app with `mcp_server=True` (startup logs should print an MCP SSE URL).
2. Dump the live schema with the bundled script:
   ```bash
   python .claude/skills/gradio-mcp-tool-docstrings/scripts/dump_mcp_schema.py --port 7860
   # or target one tool:
   python .../scripts/dump_mcp_schema.py --tool detect_objects
   ```
   Confirm: the **tool name** is what you expect, the **description** is complete with any **URLs
   intact**, and **every argument** has its full description (nothing truncated).
3. **Isolated-stub technique** (no model / GPU needed): to iterate fast on parsing, replicate just
   the function *signature + docstring* (a stub that returns dummy values) in a tiny app, wire it to
   a button `.click(...)`, `demo.launch(mcp_server=True, prevent_thread_lock=True)`, then run the
   dump script against it. This checks the docstring parsing in seconds without loading heavy deps.
   Requires only `pip install "gradio[mcp]"`.
