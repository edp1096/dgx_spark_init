"""Editor Test tab — minimal ImageEditor test from test_gradio_editor.py."""

import gradio as gr
import numpy as np


def _on_edit(editor_value):
    """Receive ImageEditor output and display info."""
    if editor_value is None:
        return "No image", None
    bg = editor_value.get("background")
    layers = editor_value.get("layers", [])
    composite = editor_value.get("composite")
    info = f"background: {type(bg).__name__}"
    if bg is not None and isinstance(bg, np.ndarray):
        info += f" shape={bg.shape} dtype={bg.dtype}"
    info += f"\nlayers: {len(layers)}"
    for i, layer in enumerate(layers):
        if isinstance(layer, dict):
            info += f"\n  layer[{i}] keys={list(layer.keys())}"
        elif isinstance(layer, np.ndarray):
            info += f"\n  layer[{i}] shape={layer.shape} dtype={layer.dtype}"
        else:
            info += f"\n  layer[{i}] type={type(layer).__name__}"
    info += f"\ncomposite: {type(composite).__name__}"
    if composite is not None and isinstance(composite, np.ndarray):
        info += f" shape={composite.shape} dtype={composite.dtype}"
    return info, composite


def build_editor_test_tab():
    with gr.Row():
        editor = gr.ImageEditor(
            label="Edit Image",
            type="numpy",
            sources=["upload", "clipboard"],
            brush=gr.Brush(colors=["#ffffff", "#000000"], default_size=20),
            eraser=gr.Eraser(default_size=20),
        )
        with gr.Column():
            info = gr.Textbox(label="Editor Output Info", lines=10)
            preview = gr.Image(label="Composite Preview", interactive=False)
    btn = gr.Button("Check Output", variant="primary")
    btn.click(fn=_on_edit, inputs=[editor], outputs=[info, preview])
