import json
from pathlib import Path
import tempfile
import unittest
from generation_progress import Tracker, stages, PROFILE, append_event, read_events

class Estimates(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.now=0
    def tearDown(self):self.temp.cleanup()
    def tracker(self,kind='h3'):
        return Tracker(self.root,self.root/'state','a'*32,kind,clock=lambda:self.now)
    def test_cold_run_reports_only_measured_stage_eta(self):
        t=self.tracker();self.assertIsNone(t.record('encode_start')['eta_seconds'])
        t.record('sample_start');self.now=10;p=t.record('progress',step=1,total=20)
        self.assertEqual((p['eta_seconds'],p['eta_scope']),(190,'stage'))
        self.now=20;t.record('progress',step=2,total=20)
        self.now=30;p=t.record('progress',step=3,total=20)
        self.assertEqual(p['eta_seconds'],170)
        t.record('sample_end');p=t.record('decode_video_start');self.assertIsNone(p['eta_seconds'])
    def test_learned_full_eta_includes_decode_audio_and_save(self):
        t=self.tracker();timings={s:1 for s in stages('h3')};timings.update(sample=200,decode_video=15,decode_audio=3,encode=8)
        t.remember(timings);t=self.tracker();t.record('sample_start');self.now=20;p=t.record('progress',step=2,total=20)
        self.assertEqual(p['eta_scope'],'total');self.assertEqual(p['eta_seconds'],202)
        self.now=200;t.record('sample_end');p=t.record('decode_video_start');self.assertEqual(p['eta_seconds'],21)
        self.now=500;p=t.record('save_start');self.now=510;p=t.record('conditioning_cache_hit');self.assertIsNone(p['eta_seconds'])
        self.assertEqual(t.record('idle')['eta_seconds'],0)
    def test_profiles_and_history_are_separate_and_bounded(self):
        t=self.tracker();t.remember({s:1 for s in stages('h3')})
        image=self.tracker('qwim');self.assertIsNone(image.record('encode_start')['eta_seconds'])
        for _ in range(12):image.remember({s:2 for s in stages('qwim')})
        h=json.loads((self.root/'state/timings.json').read_text());self.assertEqual(len(h[PROFILE['qwim']]),8);self.assertEqual(len(h[PROFILE['h3']]),1)
        self.assertNotIn('decode_audio',stages('qwim'))
    def test_cursor_does_not_lose_partial_append_or_other_request(self):
        append_event(self.root,'a'*32,{'event':'queued'});path=self.root/'progress'/('a'*32+'.jsonl')
        with path.open('a') as f:f.write('{"event":"sample')
        self.assertEqual(read_events(self.root,'a'*32,0),{'events':[{'event':'queued'}],'next':1})
        with path.open('a') as f:f.write('"}\n')
        self.assertEqual(read_events(self.root,'a'*32,1)['events'],[{'event':'sample'}])
        self.assertEqual(read_events(self.root,'b'*32,0)['events'],[])
