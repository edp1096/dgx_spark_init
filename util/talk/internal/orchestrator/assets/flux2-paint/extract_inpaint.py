"""Build a three-node image-inpainting subset from the pinned LanPaint source.

Keep upstream numerical implementations unchanged; exclude the UI, video/audio
nodes, routes, examples and alternative samplers. Shared sampler helpers remain.
"""
import ast
import shutil
import sys
from pathlib import Path

upstream, destination = map(Path, sys.argv[1:3])
source = upstream / "src" / "LanPaint"
text = (source / "nodes.py").read_text()
tree = ast.parse(text)
roots = {"LanPaint_ImageEncode", "LanPaint_SamplerCustomAdvanced", "LanPaint_ImageDecode"}
definitions = {}
for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        definitions[node.name] = node
    elif isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                definitions[target.id] = node
selected = set(roots)
pending = list(roots)
while pending:
    name = pending.pop()
    if name not in definitions:
        raise RuntimeError(f"missing required upstream definition: {name}")
    for child in ast.walk(definitions[name]):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            if child.id in definitions and child.id not in selected:
                selected.add(child.id)
                pending.append(child.id)
keep = {id(definitions[name]) for name in selected}
parts = ["# Extracted from scraed/LanPaint; see LICENSE and UPSTREAM.\n",
         "# MiniMax audio paths are not exposed by this image-only node package.\n",
         "time_shift_sigma = time_shift_slope = None\n"]
for node in tree.body:
    if id(node) in keep or isinstance(node, (ast.Import, ast.ImportFrom)):
        parts.append(ast.unparse(node) + "\n")
destination.mkdir(parents=True, exist_ok=True)
(destination / "nodes.py").write_text("\n".join(parts))
for filename in ("lanpaint.py", "earlystop.py", "types.py"):
    shutil.copy2(source / filename, destination / filename)
shutil.copy2(upstream / "LICENSE", destination / "LICENSE")
(destination / "UPSTREAM").write_text("https://github.com/scraed/LanPaint\nRevision: 32cf848e93971da380d868936e007f5611218bee\nImage node subset; numerical implementations preserved.\n")
(destination / "__init__.py").write_text(
    "from .nodes import " + ", ".join(sorted(roots)) + "\n\nNODE_CLASS_MAPPINGS = {\n" +
    "".join(f"    {name!r}: {name},\n" for name in sorted(roots)) + "}\n")
print(f"LanPaint subset: {len(roots)} image nodes, {len(selected)} definitions")
