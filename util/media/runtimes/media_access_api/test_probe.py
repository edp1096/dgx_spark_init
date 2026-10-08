import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import api


class ProbeTest(unittest.TestCase):
    @patch("api.subprocess.run")
    def test_probe_media_does_not_mix_stderr_with_json(self, run):
        payload = {"streams": [{"codec_type": "video"}], "format": {}}
        run.return_value = subprocess.CompletedProcess(
            args=["ffprobe"],
            returncode=0,
            stdout=json.dumps(payload),
            stderr="[mov,mp4] non-fatal diagnostic\n",
        )

        self.assertEqual(api.probe_media(Path("source.mp4")), payload)

    @patch("api.subprocess.run")
    def test_probe_media_reports_invalid_machine_output(self, run):
        run.return_value = subprocess.CompletedProcess(
            args=["ffprobe"], returncode=0, stdout="not-json", stderr=""
        )

        with self.assertRaisesRegex(RuntimeError, "invalid JSON"):
            api.probe_media(Path("source.mp4"))

    def test_audio_only_media_is_persisted_as_playable_asset(self):
        with tempfile.TemporaryDirectory() as root:
            asset_dir = Path(root) / "media"
            asset_dir.mkdir()
            source = Path(root) / "source.m4a"
            source.write_bytes(b"source-audio")

            def fake_run(command, timeout=None):
                Path(command[-1]).write_bytes(b"playable-audio")
                return ""

            probe = {"streams": [{"codec_type": "audio", "codec_name": "aac"}], "format": {}}
            with (
                patch.object(api, "ASSET_DIR", asset_dir),
                patch.object(api, "probe_media", return_value=probe),
                patch.object(api, "probe_duration", return_value=12.5),
                patch.object(api, "run", side_effect=fake_run) as run,
            ):
                asset = api.persist_media_asset(source, "https://soundcloud.com/example/track")

            self.assertEqual(asset["media_type"], "audio")
            self.assertEqual(asset["content_type"], "audio/mp4")
            self.assertEqual(asset["width"], 0)
            self.assertEqual(asset["height"], 0)
            self.assertTrue((asset_dir / asset["id"] / "audio.m4a").is_file())
            self.assertIn("copy", run.call_args.args[0])


class TemporaryStorageTest(unittest.TestCase):
    def test_cleanup_skips_active_prepare_directory(self):
        with tempfile.TemporaryDirectory() as root:
            data_dir = Path(root)
            stale = data_dir / "prepare-stale"
            active = data_dir / "prepare-active"
            stale.mkdir()
            active.mkdir()
            (stale / "fragment").write_bytes(b"stale")
            (active / "fragment").write_bytes(b"active")
            with patch.object(api, "DATA_DIR", data_dir):
                with api.active_prepare_lock:
                    api.active_prepare_dirs.add(active)
                try:
                    result = api.cleanup_media_storage()
                finally:
                    with api.active_prepare_lock:
                        api.active_prepare_dirs.discard(active)

            self.assertEqual(result["removed_directories"], 1)
            self.assertFalse(stale.exists())
            self.assertTrue(active.exists())

    def test_automatic_cleanup_respects_age(self):
        with tempfile.TemporaryDirectory() as root:
            data_dir = Path(root)
            old = data_dir / "prepare-old"
            recent = data_dir / "prepare-recent"
            old.mkdir()
            recent.mkdir()
            (old / "fragment").write_bytes(b"old")
            (recent / "fragment").write_bytes(b"recent")
            old_time = time.time() - 3 * 3600
            for path in (old, old / "fragment"):
                path.touch(exist_ok=True)
                Path(path).chmod(0o755 if Path(path).is_dir() else 0o644)
                os.utime(path, (old_time, old_time))
            with patch.object(api, "DATA_DIR", data_dir):
                result = api.cleanup_media_storage(older_than_hours=2)

            self.assertEqual(result["removed_directories"], 1)
            self.assertFalse(old.exists())
            self.assertTrue(recent.exists())


class RecoveryTest(unittest.TestCase):
    def test_duplicate_prepare_for_same_request_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            work_dir = Path(root) / "prepare-job-123"
            api.begin_prepare("job-123", work_dir)
            try:
                with self.assertRaisesRegex(api.HTTPException, "already active") as raised:
                    api.begin_prepare("job-123", work_dir)
                self.assertEqual(raised.exception.status_code, 409)
            finally:
                api.finish_prepare("job-123", work_dir)

    def test_corrupt_aac_partial_is_remuxed_for_recovery(self):
        with tempfile.TemporaryDirectory() as root:
            work_dir = Path(root)
            partial = work_dir / "source.mp4.part"
            partial.write_bytes(b"x" * (16 << 20))

            def fake_prepare(command, timeout, request_id):
                Path(command[-1]).write_bytes(b"recovered-media")
                return ""

            probe = {
                "streams": [
                    {"codec_type": "video", "codec_name": "h264"},
                    {"codec_type": "audio", "codec_name": "aac"},
                ]
            }
            with (
                patch.object(api, "probe_media", return_value=probe),
                patch.object(api, "probe_duration", return_value=120.0),
                patch.object(api, "run_prepare_command", side_effect=fake_prepare) as prepare,
            ):
                recovered = api.recover_corrupt_partial_download(
                    work_dir, "AAC: Error submitting packet to decoder"
                )

            self.assertEqual(recovered, work_dir / "source.recovered.mp4")
            command = prepare.call_args.args[0]
            self.assertIn("+discardcorrupt", command)
            self.assertIn("ignore_err", command)

    def test_network_failure_does_not_accept_partial_download(self):
        with tempfile.TemporaryDirectory() as root:
            work_dir = Path(root)
            (work_dir / "source.mp4.part").write_bytes(b"x" * (16 << 20))
            self.assertIsNone(
                api.recover_corrupt_partial_download(work_dir, "HTTP Error 403: Forbidden")
            )

    def test_request_id_uses_durable_work_directory(self):
        with tempfile.TemporaryDirectory() as root, patch.object(api, "DATA_DIR", Path(root)):
            self.assertEqual(
                api.request_work_dir("job-123"),
                Path(root) / "prepare-job-123",
            )

    def test_cancel_prepare_terminates_only_registered_process(self):
        class FakeProcess:
            def __init__(self):
                self.terminated = False
                self.waited = False

            def poll(self):
                return None

            def terminate(self):
                self.terminated = True

            def wait(self, timeout=None):
                self.waited = True
                return 0

        with tempfile.TemporaryDirectory() as root:
            data_dir = Path(root)
            progress_dir = data_dir / "progress"
            progress_dir.mkdir()
            work_dir = data_dir / "prepare-job-123"
            work_dir.mkdir()
            process = FakeProcess()
            with (
                patch.object(api, "DATA_DIR", data_dir),
                patch.object(api, "PROGRESS_DIR", progress_dir),
            ):
                with api.active_prepare_lock:
                    api.active_prepare_dirs.add(work_dir)
                    api.active_prepare_processes["job-123"] = process
                try:
                    result = api.cancel_media_prepare("job-123")
                    self.assertEqual(result["status"], "cancelling")
                    self.assertTrue(process.terminated)
                    self.assertTrue(process.waited)
                    self.assertEqual(
                        json.loads((progress_dir / "job-123.json").read_text())["stage"],
                        "cancelled",
                    )
                finally:
                    api.finish_prepare("job-123", work_dir)

    def test_delete_media_job_artifacts_removes_only_owned_state(self):
        with tempfile.TemporaryDirectory() as root:
            data_dir = Path(root)
            progress_dir = data_dir / "progress"
            progress_dir.mkdir()
            work_dir = data_dir / "prepare-job-123"
            work_dir.mkdir()
            (work_dir / "source.mp4.part").write_bytes(b"partial")
            (progress_dir / "job-123.json").write_text('{"stage":"failed"}')
            unrelated = data_dir / "prepare-job-1234"
            unrelated.mkdir()
            with (
                patch.object(api, "DATA_DIR", data_dir),
                patch.object(api, "PROGRESS_DIR", progress_dir),
            ):
                api.delete_media_job_artifacts("job-123")
            self.assertFalse(work_dir.exists())
            self.assertFalse((progress_dir / "job-123.json").exists())
            self.assertTrue(unrelated.exists())

    @patch("api.validate_audio_decode")
    @patch("api.probe_duration", return_value=12.5)
    def test_reusable_source_uses_completed_media(self, _probe_duration, validate_audio_decode):
        with tempfile.TemporaryDirectory() as root:
            work_dir = Path(root)
            source = work_dir / "source.mp4"
            source.write_bytes(b"complete-media")
            api.write_recovery(
                work_dir,
                source_name="https://example.com/video",
                source_file=source.name,
                stage="downloaded",
            )

            self.assertEqual(
                api.reusable_source(work_dir, "https://example.com/video"), source
            )
            self.assertIsNone(
                api.reusable_source(work_dir, "https://example.com/other")
            )
            validate_audio_decode.assert_called_once_with(source)

    @patch("api.validate_audio_decode", side_effect=RuntimeError("invalid AAC"))
    @patch("api.probe_duration", return_value=3600)
    def test_reusable_source_discards_corrupt_audio(self, _probe_duration, _validate):
        with tempfile.TemporaryDirectory() as root:
            work_dir = Path(root)
            source = work_dir / "source.mp4"
            source.write_bytes(b"container metadata is valid but audio is corrupt")
            api.write_recovery(
                work_dir,
                source_name="https://example.com/video",
                source_file=source.name,
                stage="downloaded",
            )

            self.assertIsNone(
                api.reusable_source(work_dir, "https://example.com/video")
            )
            self.assertFalse(source.exists())

    @patch("api.probe_media", return_value={"streams": [{"codec_type": "audio"}]})
    @patch("api.run_prepare_command")
    def test_audio_validation_decodes_entire_primary_stream(self, run_prepare, _probe):
        api.validate_audio_decode(Path("source.mp4"), "job-123")

        command, timeout, request_id = run_prepare.call_args.args
        self.assertEqual(command[-3:], ["-f", "null", "-"])
        self.assertIn("0:a:0", command)
        self.assertEqual(timeout, 7200)
        self.assertEqual(request_id, "job-123")


if __name__ == "__main__":
    unittest.main()
