"""Geometry and queue contracts, without loading CUDA or model weights."""
import asyncio
import base64
import io
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from PIL import Image
import api


def png(image):
    data = io.BytesIO()
    image.save(data, format="PNG")
    return data.getvalue()


def data_url(image):
    return "data:image/png;base64," + base64.b64encode(png(image)).decode()


class Geometry(unittest.TestCase):
    def test_masked_edit_preserves_rgba_and_original_size(self):
        source = Image.new("RGBA", (1501, 997), (22, 40, 90, 17))
        payload = api.ImageRequest(operation="inpaint", prompt="red", anypaint_image=data_url(source), mask_box=[10, 10, 110, 110])
        _, refs, original, mask, box, size = api.prepare(payload)
        self.assertEqual(refs[0].size, refs[1].size)
        self.assertLessEqual(max(refs[0].size), 1024)
        value = api.restore(png(Image.new("RGB", (64, 64), "red")), payload, original, mask, box, size)
        edited = Image.open(io.BytesIO(base64.b64decode(value)))
        self.assertEqual(edited.size, source.size)
        self.assertEqual(edited.getpixel((0, 0)), source.getpixel((0, 0)))
        self.assertEqual(edited.getpixel((500, 500)), source.getpixel((500, 500)))
        self.assertEqual(edited.getpixel((50, 50))[:3], (255, 0, 0))

    def test_outpaint_preserves_exact_interior(self):
        source = Image.new("RGBA", (99, 71), (50, 70, 90, 128))
        payload = api.ImageRequest(operation="outpaint", prompt="extend", source_image=data_url(source), outpaint_left=32, preserve_source=True)
        _, _, original, mask, box, size = api.prepare(payload)
        value = api.restore(png(Image.new("RGB", (64, 64), "red")), payload, original, mask, box, size)
        edited = Image.open(io.BytesIO(base64.b64decode(value)))
        self.assertEqual(edited.size, (131, 71))
        self.assertEqual(edited.crop(box).tobytes(), source.tobytes())

    def test_cutout_keeps_source_rgb_and_requires_real_alpha(self):
        source = Image.new("RGB", (80, 60), "blue")
        payload = api.ImageRequest(operation="background_remove", source_image=data_url(source))
        output = Image.new("RGBA", (80, 60), (255, 0, 0, 0))
        output.putpixel((20, 20), (255, 0, 0, 255))
        value = api.restore(png(output), payload, source, None, None, source.size)
        edited = Image.open(io.BytesIO(base64.b64decode(value)))
        self.assertEqual(edited.getpixel((20, 20)), (0, 0, 255, 255))
        with self.assertRaises(HTTPException):
            api.restore(png(source), payload, source, None, None, source.size)


class Queue(unittest.IsolatedAsyncioTestCase):
    async def test_dit_job_bridge_keeps_reference_names_and_cancellation_lease(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder in ("requests", "results", "output", "cancel"):
                (root / folder).mkdir()
            class Request:
                async def is_disconnected(self): return False
            async def worker():
                while not list((root / "requests").glob("*.json")):
                    await asyncio.sleep(.01)
                path = next((root / "requests").glob("*.json"))
                job = json.loads(path.read_text())
                self.assertEqual(job["reference_files"], ["reference.png"])
                self.assertEqual(job["operation"], "inpaint")
                (root / "output" / (job["case"] + ".png")).write_bytes(b"png-result")
                (root / "results" / path.name).write_text(json.dumps({"status":"success"}))
            with patch.object(api,"JOB",root), patch.object(api,"resident_state",lambda: {}):
                task = asyncio.create_task(worker())
                result = await api.execute_resident("edit", [Path("reference.png")], 1, (1024,1024), "inpaint", Request())
                await task
                self.assertEqual(result,b"png-result")
                self.assertEqual(list((root/"requests").iterdir()),[])
                class Disconnected:
                    async def is_disconnected(self): return True
                async def cancel_ack():
                    while not list((root/"cancel").iterdir()):
                        await asyncio.sleep(.01)
                    marker = next((root/"cancel").iterdir())
                    self.assertTrue((root/"requests"/(marker.name+".json")).exists())
                    (root/"results"/(marker.name+".json")).write_text(json.dumps({"status":"canceled"}))
                task = asyncio.create_task(cancel_ack())
                with self.assertRaises(HTTPException) as error:
                    await api.execute_resident("edit", [Path("reference.png")], 1, (1024,1024), "inpaint", Disconnected())
                await task
                self.assertEqual(error.exception.status_code,499)
                self.assertEqual(list((root/"cancel").iterdir()),[])

    async def test_dit_mode_has_stage_workspace_and_keeps_core(self):
        async def state(): return {"status":"ok","busy":False}
        with patch.object(api,"backend_state",state), patch.object(api,"DIT_RESIDENT",True):
            memory = await api.memory()
        self.assertEqual(memory["memory_gib"],14)
        self.assertEqual(memory["workspace_gib"],6)
        self.assertTrue(memory["keep_models_loaded"])

    async def test_resident_mode_reports_separate_memory_budget(self):
        async def state():
            return {"status": "ok", "busy": False, "active": 0, "queued": 0}
        for resident, budget in ((False, 14.0), (True, 24.0)):
            with patch.object(api, "backend_state", state), patch.object(api, "KEEP_MODELS_LOADED", resident):
                memory = await api.memory()
            self.assertEqual(memory["memory_gib"], budget)
            self.assertEqual(memory["keep_models_loaded"], resident)
            self.assertEqual(memory["workspace_gib"], 10.0)

    async def test_active_and_waiting_requests_refuse_quiesce(self):
        runtime = api.Runtime()
        started, finish = asyncio.Event(), asyncio.Event()
        async def first():
            async with runtime.job():
                started.set()
                await finish.wait()
        async def waiting():
            async with runtime.job():
                pass
        a = asyncio.create_task(first())
        await started.wait()
        b = asyncio.create_task(waiting())
        await asyncio.sleep(0)
        self.assertEqual(runtime.jobs, 2)
        async def state():
            return {"status": "ok", "busy": runtime.jobs > 0}
        with patch.object(api, "runtime", runtime), patch.object(api, "backend_state", state):
            with self.assertRaises(HTTPException) as error:
                await api.quiesce()
            self.assertEqual(error.exception.status_code, 409)
            b.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await b
            self.assertEqual(runtime.jobs, 1)
            finish.set()
            await a
            await api.quiesce()
            with self.assertRaises(HTTPException):
                async with runtime.job():
                    pass
            await api.resume()
            async with runtime.job():
                pass
        self.assertEqual(runtime.jobs, 0)


if __name__ == "__main__":
    unittest.main()
