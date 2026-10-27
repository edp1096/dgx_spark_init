import array
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import store
import speech
from audio_mix import timeline, narration_fit
from spec import Voice


def pcm(seconds, voiced=True):
    return array.array("h", [1000 if index % 2 else -1000 for index in range(round(seconds * 44100))]).tobytes() if voiced else b"\0\0" * round(seconds * 44100)


class AudioTests(unittest.TestCase):
    def test_redesign_rejects_unchanged_direction(self):
        from pipeline import check_redesign_brief
        project = store.Project(session_id="session", title="제목")
        with self.assertRaisesRegex(ValueError, "unchanged"):
            check_redesign_brief(project, {"design_brief":"small   title"}, {"design_brief":"small title"})
        check_redesign_brief(project, {"design_brief":"large opening hook"}, {"design_brief":"small title"})
        project.design_mode = "preserve"
        check_redesign_brief(project, {"design_brief":"small title"}, {"design_brief":"small title"})

    def test_redesign_clears_inherited_template_but_explicit_preserve_keeps_it(self):
        import api
        parent=store.Project(session_id="session",title="제목",template_id="a"*32).model_dump()
        with patch.object(store,"get",return_value={"session_id":"session","project":parent}), patch.object(store,"create",side_effect=lambda project:project.model_dump()):
            revised=api.project_tool(api.ProjectRequest(session_id="session",action="revise",job_id="b"*32,brief="제목 디자인을 새로 만들어"))
            self.assertEqual(revised["template_id"],"")
            self.assertEqual(revised["design_mode"],"redesign")
            preserved=api.project_tool(api.ProjectRequest(session_id="session",action="revise",job_id="b"*32,design_mode="preserve"))
            self.assertEqual(preserved["template_id"],"a"*32)

    def test_duration_requires_speech_coverage_and_exact_frames(self):
        voices = [10.8, 11.1, 10.9, 11.4, 10.7]
        scenes = [{"narration": "내용", "duration_seconds": 0} for _ in voices]
        items = [{"duration": x, "segments": []} for x in voices]
        result, duration, fit = timeline(scenes, items, 60, Voice().model_dump())
        self.assertEqual(duration, 60)
        self.assertEqual(sum(round(x["duration"] * 24) for x in result), 1440)
        self.assertGreater(fit["narration_coverage"], .9)
        for value in (4, 20):
            with self.assertRaisesRegex(ValueError, "내레이션"):
                timeline(scenes, [{"duration": value, "segments": []} for _ in voices], 60, Voice().model_dump())

    def test_internal_silence_is_measured_not_trimmed(self):
        metrics = speech.analyze(pcm(.5, False) + pcm(1) + pcm(2.2, False) + pcm(1) + pcm(.7, False), 44100)
        self.assertAlmostEqual(metrics["longest_internal_silence"], 2.2, places=1)
        self.assertGreater((metrics["trim_end_sample"] - metrics["trim_start_sample"]) / 44100, 4.2)

    def test_silence_retry_is_bounded_and_raw_failure_retained(self):
        response = MagicMock()
        response.headers = {}
        response.content = pcm(1) + pcm(2.2, False) + pcm(1)
        with tempfile.TemporaryDirectory() as directory, patch("speech.httpx.Client") as client:
            client.return_value.__enter__.return_value.post.return_value = response
            with self.assertRaisesRegex(ValueError, "시도 3회"):
                speech.synthesize("잘못된 무음을 검사합니다.", "http://example.test", "ref.wav", Path(directory))
            self.assertEqual(client.return_value.__enter__.return_value.post.call_count, 3)
            self.assertEqual(len(list(Path(directory).glob("*.wav"))), 3)

    def test_short_chunks_keep_text_and_actual_segment_timing(self):
        text = "첫 번째 문장입니다. 두 번째 문장입니다."
        self.assertEqual(speech.chunks(text), ["첫 번째 문장입니다.", "두 번째 문장입니다."])
        response = MagicMock(); response.headers = {}; response.content = pcm(1)
        with patch("speech.httpx.Client") as client:
            client.return_value.__enter__.return_value.post.return_value = response
            result = speech.synthesize(text, "http://example.test", "ref.wav")
        self.assertAlmostEqual(result["duration"], 2.18)
        self.assertAlmostEqual(result["segments"][1]["start"], 1.18)
        self.assertTrue(speech.analyze(pcm(41.8), 44100)["hit_cap"])


class DurableTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = patch.object(store, "ROOT", Path(self.directory.name)); self.root.start()
        store.initialize()

    def tearDown(self):
        self.root.stop(); self.directory.cleanup()

    def test_restart_requeues_interrupted_work_and_retains_completed_stages(self):
        project = store.create(store.Project(session_id="session", title="시험", brief="작업 복원"))
        self.assertEqual(store.claim(), project["id"])
        self.assertIsNone(store.claim())
        folder = store.ROOT / project["id"]; folder.mkdir(); (folder / "speech.json").write_text("{}")
        store.initialize()
        self.assertEqual(store.get(project["id"])["status"], "queued")
        self.assertTrue((folder / "speech.json").exists())

    def test_cancel_survives_restart_and_retry_is_explicit(self):
        project = store.create(store.Project(session_id="session", title="시험", brief="작업 중지"))
        store.claim(); store.action(project["id"], "cancel"); store.initialize()
        self.assertEqual(store.get(project["id"])["status"], "canceled")
        self.assertIsNone(store.claim())
        store.action(project["id"], "retry")
        self.assertEqual(store.claim(), project["id"])


if __name__ == "__main__":
    unittest.main()
