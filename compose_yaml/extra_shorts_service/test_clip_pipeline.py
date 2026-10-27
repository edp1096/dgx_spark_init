import base64
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

import store
import clip_store
import clip_pipeline as pipeline


PLAN={'title':'움직이는 만화','scope':'image_to_video','unsupported_reason':'','edit_required':True,
      'image_prompt':'Restyle as a coherent hand-drawn cartoon, retaining the subject and setting.',
      'motion_prompt':'The subject smiles and waves the visible right hand. Keep the camera fixed.',
      'requested_motion':'Smile and wave the right hand.','preserve':['one subject','clothes','background'],
      'style':'hand-drawn cartoon','audio':'generated',
      'inspect_regions':[{'label':'hand','box':[100,100,400,500]}]}
PASS={'verdict':'pass','summary':'The actual image meets the requested style and composition.','issues':[],'repair_prompt':''}


def failure(target='image',category='hand'):
    return {'verdict':'revise','summary':'A visible defect needs a correction.',
            'issues':[{'category':category,'severity':'major','evidence':'The visible fingers are fused in the candidate image.',
                       'target':target,'region':{'label':'right hand','box':[100,100,400,500]},'times':[.75]}],
            'repair_prompt':'Separate the fused fingers while preserving the hand wave and the cartoon style.'}


def png(color='red'):
    b=io.BytesIO();Image.new('RGB',(128,192),color).save(b,format='PNG');return b.getvalue()


def inline(raw):return 'data:image/png;base64,'+base64.b64encode(raw).decode()


class FakeStudio:
    def __init__(self,image_reviews=None,video_reviews=None,video_failures=0):
        self.image_reviews=list(image_reviews or [PASS]);self.video_reviews=list(video_reviews or [PASS])
        self.video_failures=video_failures;self.calls=[];self.last_image=png('blue')
    def __call__(self,operation,payload):
        self.calls.append((operation,copy.deepcopy(payload)))
        if operation=='author':
            phase=payload['phase']
            if phase=='plan':return copy.deepcopy(PLAN)
            if phase=='motion_repair':return {'motion_prompt':'The subject smiles and gently waves the right hand while the original background stays still.','changes':'Replaced the defect description with the desired motion and stable background.'}
            values=self.image_reviews if phase=='image_review' else self.video_reviews
            return copy.deepcopy(values.pop(0) if len(values)>1 else values[0])
        if operation=='image':
            color='blue' if sum(op=='image' for op,_ in self.calls)==1 else 'green'
            if payload['operation']=='inpaint':
                source=Image.open(io.BytesIO(base64.b64decode(payload['source_image'].split(',')[1]))).convert('RGB')
                source.paste(color,tuple(payload['mask_box']));b=io.BytesIO();source.save(b,format='PNG');self.last_image=b.getvalue()
            else:self.last_image=png(color)
            return {'data':base64.b64encode(self.last_image).decode()}
        if operation=='video':
            if self.video_failures:
                self.video_failures-=1;raise RuntimeError('temporary generation outage')
            assert base64.b64decode(payload['first_frame'].split(',')[1])==self.last_image,'wrong image sent to H3'
            return {'data':base64.b64encode(b'video-fixture').decode(),'width':128,'height':192}
        raise AssertionError(operation)


class DurableProduction(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.old=store.ROOT;store.ROOT=Path(self.tmp.name)
        store.initialize();clip_store.initialize()
        self.request=clip_store.Request(session_id='session',source_image_id='photo',source_image=inline(png()),brief='이 사진 만화로 바꾸고 손 흔들게 해줘',request_key='request',seed=42)
        self.job=clip_store.create(self.request);self.root=clip_store.folder(self.job['id'])
        self.probe=patch.object(pipeline,'probe',return_value={'width':128,'height':192,'frames':124,'fps':24,'duration':124/24,'has_audio':True});self.probe.start()
        self.evidence=patch.object(pipeline,'video_evidence',side_effect=self.fake_evidence);self.evidence.start()
    def tearDown(self):
        self.probe.stop();self.evidence.stop();store.ROOT=self.old;self.tmp.cleanup()
    def fake_evidence(self,root,path,label):
        target=root/label;target.mkdir(exist_ok=True)
        for name in ['overview.jpg','temporal.jpg']:Image.new('RGB',(128,192),'blue').save(target/name)
        return {'overview':target/'overview.jpg','temporal':target/'temporal.jpg','times':[0,2.5,5.125],'consecutive_times':[2,2+1/24,2+2/24]}
    def run_studio(self,studio):pipeline.produce(self.job['id'],studio)

    def test_original_to_edit_to_h3_and_delivery_gated_by_reviews(self):
        studio=FakeStudio();self.run_studio(studio)
        result=clip_store.get(self.job['id'])
        self.assertEqual(result['status'],'completed');self.assertTrue(result['output_ready'])
        report=json.loads((self.root/'report.json').read_text())
        self.assertNotEqual(report['source_sha256'],report['final_image_sha256'])
        self.assertEqual(report['final_image_sha256'],report['video_attempts'][0]['generation']['first_frame_sha256'])
        self.assertEqual([p['phase'] for op,p in studio.calls if op=='author'],['plan','image_review','video_review'])

    def test_detected_hand_defect_uses_mask_and_video_failure_preserves_required_motion(self):
        studio=FakeStudio([failure(),PASS],[failure('video','motion'),PASS]);self.run_studio(studio)
        edits=[p for op,p in studio.calls if op=='image'];videos=[p for op,p in studio.calls if op=='video']
        self.assertEqual([p['operation'] for p in edits],['identity_edit','inpaint'])
        self.assertEqual(edits[1]['mask_box'],[12,19,52,96])
        self.assertEqual([p['seed'] for p in edits],[42,43]);self.assertEqual([p['seed'] for p in videos],[50,51])
        self.assertTrue(all(PLAN['requested_motion'] in p['prompt'] for p in videos))
        self.assertNotIn('Separate the fused fingers',videos[1]['prompt'])
        self.assertEqual(sum(op=='author' and p['phase']=='motion_repair' for op,p in studio.calls),1)
        report=json.loads((self.root/'report.json').read_text())
        self.assertTrue(report['image_attempts'][1]['generation']['outside_mask_identical'])
        self.assertEqual(report['retries'],{'image':1,'video':1})

    def test_repeated_local_failure_switches_to_original_instead_of_compounding_damage(self):
        studio=FakeStudio([failure(),failure(),PASS]);self.run_studio(studio)
        edits=[p for op,p in studio.calls if op=='image']
        self.assertEqual([p['operation'] for p in edits],['identity_edit','inpaint','identity_edit'])
        self.assertEqual(edits[0]['source_image'],edits[2]['source_image'])

    def test_transport_failure_resumes_without_regenerating_accepted_image(self):
        studio=FakeStudio(video_failures=1)
        with self.assertRaisesRegex(RuntimeError,'outage'):self.run_studio(studio)
        self.run_studio(studio)
        self.assertEqual(sum(op=='image' for op,_ in studio.calls),1)
        self.assertEqual(sum(op=='author' and p['phase']=='plan' for op,p in studio.calls),1)
        videos=[p for op,p in studio.calls if op=='video'];self.assertEqual(videos[0],videos[1])

    def test_resume_after_accepted_video_does_not_generate_or_review_again(self):
        studio=FakeStudio();original=pipeline.shutil.copyfile
        def interrupted(source,target):
            if Path(target).name=='final.mp4':raise OSError('disk temporarily unavailable')
            return original(source,target)
        with patch.object(pipeline.shutil,'copyfile',side_effect=interrupted),self.assertRaises(OSError):self.run_studio(studio)
        count=len(studio.calls);self.run_studio(studio);self.assertEqual(len(studio.calls),count)
        self.assertTrue(clip_store.get(self.job['id'])['output_ready'])

    def test_video_review_can_send_original_image_defect_back_to_edit_stage(self):
        studio=FakeStudio([PASS,PASS],[failure('image'),PASS]);self.run_studio(studio)
        videos=[p for op,p in studio.calls if op=='video'];self.assertEqual(len(videos),2)
        self.assertNotEqual(videos[0]['first_frame'],videos[1]['first_frame'])
        self.assertEqual(sum(op=='image' for op,_ in studio.calls),2)

    def test_quality_budget_never_resets_on_retry_and_failed_results_are_not_published(self):
        studio=FakeStudio([failure()])
        for _ in range(2):
            with self.assertRaisesRegex(ValueError,'이미지 수정 2회'):self.run_studio(studio)
        self.assertEqual(sum(op=='image' for op,_ in studio.calls),3)
        self.assertFalse(clip_store.get(self.job['id'])['output_ready']);self.assertFalse((self.root/'final.mp4').exists())

    def test_changed_accepted_input_is_rejected_on_resume(self):
        studio=FakeStudio(video_failures=1)
        with self.assertRaises(RuntimeError):self.run_studio(studio)
        (self.root/'image-0.png').write_bytes(png('yellow'))
        with self.assertRaisesRegex(ValueError,'변경'):self.run_studio(studio)

    def test_create_deduplicates_and_restart_requeues_only_unfinished_work(self):
        duplicate=clip_store.create(self.request.model_copy(update={'seed':99}))
        self.assertEqual(duplicate['id'],self.job['id'])
        self.assertEqual(duplicate['project']['seed'],42)
        with self.assertRaisesRegex(ValueError,'different production'):
            clip_store.create(self.request.model_copy(update={'brief':'different request'}))
        self.assertEqual(clip_store.claim(),self.job['id']);clip_store.initialize()
        self.assertEqual(clip_store.get(self.job['id'])['status'],'queued')
        clip_store.action(self.job['id'],'cancel');clip_store.initialize()
        self.assertEqual(clip_store.get(self.job['id'])['status'],'canceled')
        clip_store.action(self.job['id'],'retry');self.assertEqual(clip_store.claim(),self.job['id'])

    def test_cancel_wins_a_race_with_final_completion_and_keeps_its_turn_key(self):
        clip_store.claim();clip_store.action(self.job['id'],'cancel','a'*64)
        clip_store.update(self.job['id'],'completed','완료',100)
        job=clip_store.get(self.job['id']);self.assertEqual(job['status'],'canceled')
        self.assertEqual(job['project']['cancel_turn_key'],'a'*64)
        self.assertFalse(job['output_ready'])

    def test_contradictory_pass_and_invalid_region_are_not_accepted(self):
        with self.assertRaises(ValueError):pipeline.Review.model_validate({**failure(),'verdict':'pass'})
        with self.assertRaises(ValueError):pipeline.Region(label='hand',box=[900,0,100,1000])

    def test_masked_edit_with_collateral_changes_is_rejected(self):
        prior=self.root/'prior.png';prior.write_bytes(png('red'))
        def wrong_edit(operation,payload):return {'data':base64.b64encode(png('blue')).decode()}
        with self.assertRaisesRegex(ValueError,'영역 밖'):
            pipeline.generate_image(wrong_edit,self.job['project'],self.job['id'],self.root,PLAN,1,prior,failure(),set())

    def test_bridge_reconnects_after_app_restart_without_changing_request(self):
        import clip_render
        import httpx
        response=unittest.mock.MagicMock(status_code=200)
        response.json.return_value={'data':'result'}
        client=unittest.mock.MagicMock()
        client.__enter__.return_value=client
        client.post.side_effect=[httpx.ConnectError('app restarting'),response]
        payload={'session_id':'session','job_id':self.job['id'],'seed':42}
        with patch.object(clip_render.httpx,'Client',return_value=client),patch.object(clip_render.time,'sleep') as sleep:
            self.assertEqual(clip_render.bridge('image',payload),{'data':'result'})
            self.assertEqual(client.post.call_args_list[0],client.post.call_args_list[1]);sleep.assert_called_once_with(2)


if __name__=='__main__':unittest.main()
