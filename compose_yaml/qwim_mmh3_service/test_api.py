import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from fastapi.testclient import TestClient
import api
from generation_progress import append_event
from test_image_inputs import inline_image
from image_inputs import decode_keyframe

class Contract(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();api.ROOT=Path(self.temp.name)
        for name in ['requests','results','output']:(api.ROOT/name).mkdir()
        (api.ROOT/'ready.json').write_text('{}')
        api.active=api.queued=0;api.quiescing=False
        self.client=TestClient(api.app)
    def tearDown(self):self.temp.cleanup()
    def test_health_reports_worker_acceleration(self):
        acceleration={'attention':'sol','spectrum':False,'fbc':False}
        (api.ROOT/'ready.json').write_text(json.dumps({'acceleration':acceleration}))
        self.assertEqual(self.client.get('/health').json()['acceleration'],acceleration)
    def fake_worker(self, keyframes=None, fail=False):
        for _ in range(200):
            jobs=list((api.ROOT/'requests').glob('*.json'))
            if jobs:break
            time.sleep(.01)
        else:raise AssertionError('request missing')
        job=json.loads(jobs[0].read_text());self.assertEqual(job['kind'],'h3')
        self.assertEqual(job['progress_id'],'a'*32)
        for label, value in (keyframes or {}).items():
            raw, info = decode_keyframe(value)
            self.assertEqual(Path(job[label+'_frame']).read_bytes(),raw)
            self.assertEqual(job[label+'_frame_sha256'],info['sha256'])
            self.assertEqual((job['width'],job['height']),(512,768))
        append_event(api.ROOT,job['progress_id'],{'event':'progress','stage':'sample','kind':'h3','step':1,'total':20})
        batch=self.client.get('/v1/runtime/progress/'+job['progress_id']).json()
        self.assertEqual([e['event'] for e in batch['events']],['queued','progress'])
        self.assertEqual(self.client.get('/v1/runtime/progress/'+'b'*32).json()['events'],[])
        self.assertEqual(self.client.post('/v1/runtime/quiesce').status_code,409)
        output=api.ROOT/'output'/(job['case']+'.mp4');output.write_bytes(b'valid-test-mp4')
        (api.ROOT/'results'/jobs[0].name).write_text(json.dumps({'status':'error' if fail else 'success'}))
    def test_actual_first_last_images_are_queued_and_cleaned(self):
        for fail in (False, True):
            frames={'first':inline_image(),'last':inline_image(color='blue')}
            thread=threading.Thread(target=self.fake_worker,kwargs={'keyframes':frames,'fail':fail});thread.start()
            response=self.client.post('/v1/videos/generations',json={'prompt':'run','first_frame':frames['first'],'last_frame':frames['last']},headers={'X-SparkTalk-Request-ID':'a'*32})
            thread.join();self.assertEqual(response.status_code,502 if fail else 200)
            self.assertEqual(list((api.ROOT/'inputs').iterdir()),[])
            self.assertFalse(api.active or api.queued)
            if not fail:
                self.assertEqual(response.headers['x-video-width'],'512')
                self.assertEqual(response.headers['x-video-height'],'768')
                self.assertEqual(response.headers['x-video-input-mode'],'i2v')
            for folder in ('requests','results','output','progress'):
                for path in (api.ROOT/folder).iterdir():path.unlink()
    def test_invalid_keyframes_do_not_queue_or_fall_back_to_text(self):
        for value in ['https://example.com/image.png','data:image/png;base64,broken']:
            response=self.client.post('/v1/videos/generations',json={'prompt':'run','first_frame':value})
            self.assertEqual(response.status_code,400)
        self.assertEqual(self.client.post('/v1/images/generations',json={'prompt':'run','first_frame':inline_image()}).status_code,400)
        self.assertEqual(list((api.ROOT/'requests').iterdir()),[])
        self.assertFalse(api.active or api.queued)
    def test_video_stream_and_output_cleanup(self):
        thread=threading.Thread(target=self.fake_worker);thread.start()
        response=self.client.post('/v1/videos/generations',json={'prompt':'red car rolls','seed':1,'output_format':'png'},headers={'X-SparkTalk-Request-ID':'a'*32})
        thread.join();self.assertEqual(response.status_code,200);self.assertEqual(response.content,b'valid-test-mp4')
        self.assertEqual(response.headers['content-type'],'video/mp4')
        self.assertEqual(list((api.ROOT/'output').iterdir()),[])
        self.assertFalse(self.client.get('/v1/runtime/memory').json()['busy'])
    def test_invalid_arguments_and_quiesce(self):
        self.assertEqual(self.client.post('/v1/videos/generations',json={'prompt':'car','duration':60}).status_code,422)
        self.assertEqual(self.client.post('/v1/videos/generations',json={'prompt':'   '}).status_code,400)
        self.assertEqual(self.client.post('/v1/images/generations',json={'prompt':'car','size':'2048x2048'}).status_code,400)
        self.assertEqual(self.client.get('/v1/runtime/progress/'+'a'*32+'?after=-1').status_code,400)
        self.assertEqual(self.client.post('/v1/videos/generations',json={'prompt':'car'},headers={'X-SparkTalk-Request-ID':'../bad'}).status_code,400)
        self.assertEqual(self.client.post('/v1/runtime/quiesce').status_code,200)
        self.assertEqual(self.client.post('/v1/videos/generations',json={'prompt':'car'}).status_code,503)
        self.assertEqual(list((api.ROOT/'requests').iterdir()),[])
        self.assertEqual(self.client.post('/v1/runtime/resume').status_code,200)
