from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from pydantic import ValidationError
import api


class WorkerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        api.busy = api.quiescing = False
        api.backend = "ace"
        self.payload = api.Generation(prompt="gentle solo piano")

    def test_only_ace_models_are_advertised_and_accepted(self):
        self.assertEqual(api.Generation.model_json_schema()["properties"]["model"]["enum"], ["ace-turbo", "ace-xl"])
        for model in ["ace-turbo", "ace-xl"]:
            self.assertEqual(api.Generation(model=model, prompt="solo piano").model, model)

    async def test_cancel_kills_child_before_releasing_busy_flag(self):
        process = SimpleNamespace(returncode=None, pid=12345)
        async def wait():
            process.returncode = -15
        process.wait = wait
        request = SimpleNamespace(is_disconnected=AsyncMock(return_value=True))
        with patch.object(api.asyncio, "create_subprocess_exec", AsyncMock(return_value=process)), patch.object(api.os, "killpg") as kill:
            with self.assertRaises(HTTPException) as error:
                await api.generate(self.payload, request)
        self.assertEqual(error.exception.status_code, 499)
        kill.assert_called_once()
        self.assertIsNotNone(process.returncode)
        self.assertFalse(api.busy)

    async def test_backend_failure_has_no_audio_fallback(self):
        async def spawn(*args, **kwargs):
            kwargs["stdout"].write(b"CUDA allocation failed")
            kwargs["stdout"].flush()
            return SimpleNamespace(returncode=1)
        with patch.object(api.asyncio, "create_subprocess_exec", spawn):
            with self.assertRaises(HTTPException) as error:
                await api.generate(self.payload, None)
        self.assertEqual(error.exception.status_code, 502)
        self.assertIn("CUDA allocation failed", error.exception.detail)
        self.assertFalse(api.busy)

    async def test_busy_and_wrong_model_never_launch_child(self):
        with patch.object(api.asyncio, "create_subprocess_exec", AsyncMock()) as spawn:
            api.busy = True
            with self.assertRaises(HTTPException) as error:
                await api.generate(self.payload, None)
            self.assertEqual(error.exception.status_code, 409)
            api.busy = False
            with self.assertRaises(ValidationError):
                api.Generation(model="heartmula", prompt="solo piano")
            spawn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
