"""BFS region geometry, preservation and graph contract without model loading."""
import ast
import asyncio
import base64
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
import uuid

from PIL import Image, ImageDraw, ImageOps, ImageFilter


ROOT = Path(__file__).resolve().parents[1] / 'internal/orchestrator/assets/flux2-paint'


class APIError(Exception):
    def __init__(self, status, detail): self.status_code = status; super().__init__(detail)


def functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)


def encoded(image):
    b = io.BytesIO(); image.save(b, 'PNG')
    return base64.b64encode(b.getvalue()).decode()


class HeadSwapTests(unittest.TestCase):
    def setUp(self):
        self.scope = dict(Image=Image, ImageDraw=ImageDraw, ImageOps=ImageOps, ImageFilter=ImageFilter,
                          io=io, base64=base64, HTTPException=APIError, uuid=uuid)
        functions(ROOT / 'api.py', {'png_bytes', 'checked_box', 'head_swap_inputs', 'restore_head_swap', 'generate_head_swap'}, self.scope)
        self.original = Image.new('RGB', (768, 1024), (40, 80, 120))
        self.head = Image.new('RGB', (435, 704), (200, 10, 40))
        self.scope['decode_image'] = lambda value: {'base':self.original, 'head':self.head}[value].copy()

    def request(self, **kwargs):
        fields = dict(source_image='base', head_image='head', mask_box=[96,64,384,400], reference_crop_box=[60,20,400,330], head_swap_strength=1.0, prompt='Keep the cartoon style.')
        fields.update(kwargs)
        return SimpleNamespace(**fields)

    def test_only_head_region_changes_and_original_dimensions_return(self):
        box = (96,64,384,400)
        result = self.scope['restore_head_swap'](encoded(Image.new('RGB',(512,608),(255,0,0))), self.original, box)
        out = Image.open(io.BytesIO(base64.b64decode(result)))
        self.assertEqual(out.size, self.original.size)
        self.assertEqual(out.getpixel((200,200)), (255,0,0))
        # Include immediate outside edges, other faces, and all frame corners.
        for x,y in [(95,64),(384,64),(100,63),(100,400),(600,200),(0,0),(767,1023)]:
            self.assertEqual(out.getpixel((x,y)), self.original.getpixel((x,y)))

    def test_target_and_reference_use_their_own_original_coordinates(self):
        original, target, head, box = self.scope['head_swap_inputs'](self.request())
        self.assertEqual(original.size, (768,1024))
        self.assertEqual(head.size, (340,310))
        self.assertEqual(box, (96,64,384,400))
        self.assertTrue(all(256 <= n <= 1024 and n%16 == 0 for n in target.size))

    def test_invalid_regions_fail_before_any_model_call(self):
        for box in [[0,0,769,1024], [0,0,10,10], [200,0,100,100], [0,0,100], [0,0,100.5,100]]:
            with self.assertRaises(APIError): self.scope['head_swap_inputs'](self.request(mask_box=box))
        with self.assertRaises(APIError): self.scope['head_swap_inputs'](self.request(reference_crop_box=[0,0,768,1024]))
        with self.assertRaises(APIError): self.scope['head_swap_inputs'](self.request(head_image=None))

    def test_graph_uses_two_references_and_only_bfs_v1_then_cleans_input_files(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            scope = {'Any':object, 'DIFFUSION_MODEL':'core', 'TEXT_ENCODER':'text', 'VAE':'vae'}
            functions(ROOT / 'base/api.py', {'workflow'}, scope)
            seen = []
            async def execute(graph):
                seen.append(graph)
                return encoded(Image.new('RGB',(512,512),(255,0,0)))
            self.scope['HEAD_SWAP_TRIGGER'] = 'head_swap: '
            self.scope['base'] = SimpleNamespace(INPUT_ROOT=Path(tmp), workflow=scope['workflow'], execute_workflow=execute)
            asyncio.run(self.scope['generate_head_swap'](self.request(), 42, 'test'))
            g = seen[0]
            self.assertEqual(g['40']['class_type'], 'SparkTalkBFSLoader')
            self.assertEqual(g['40']['inputs']['lora_name'], 'bfs-head-v1-flux2-klein-4b.safetensors')
            self.assertEqual(g['9']['inputs']['model'], ['40',0])
            self.assertEqual(sum(n['class_type']=='ReferenceLatent' for n in g.values()),2)
            self.assertEqual(g['4']['inputs']['text'],'head_swap: Keep the cartoon style.')
            self.assertEqual(list(Path(tmp).rglob('*.png')),[])


if __name__ == '__main__': unittest.main()
