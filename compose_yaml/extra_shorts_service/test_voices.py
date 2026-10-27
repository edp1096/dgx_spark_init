import json
import base64
import io
import wave
import array
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import store
import voice_tools
import voice_catalog
import pipeline
from spec import Voice
from pipeline import speech_key, fingerprint, plan_project, make_speech, make_visuals


class VoiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = patch.object(store, 'ROOT', Path(self.tmp.name)); self.root.start(); self.addCleanup(self.root.stop)
        store.initialize()

    def test_all_nine_ids_and_references(self):
        rows = voice_catalog.catalog()
        self.assertEqual(len(rows), 9)
        self.assertEqual(len({x['id'] for x in rows}), 9)
        for item in rows:
            self.assertEqual(Voice(preset=item['id']).preset, item['id'])
            path, text = voice_catalog.reference(item['id'])
            self.assertTrue(path.is_file()); self.assertTrue(text)
        with self.assertRaises(ValueError): Voice(preset='made_up')
        with self.assertRaises(ValueError): voice_catalog.get_voice('../api.py')

    def test_selection_is_session_scoped_and_snapshotted(self):
        voice_tools.execute('one', 'select', 'ryan')
        self.assertEqual(store.selected_voice('one'), 'ryan')
        self.assertIsNone(store.selected_voice('two'))
        first = store.create(store.Project(session_id='one', title='first'))
        self.assertEqual(first['project']['voice_preset'], 'ryan')
        explicit = store.create(store.Project(session_id='one', title='explicit', voice_preset='aiden'))
        self.assertEqual(explicit['project']['voice_preset'], 'aiden')
        voice_tools.execute('one', 'select', 'serena')
        self.assertEqual(store.get(first['id'])['project']['voice_preset'], 'ryan')
        self.assertEqual(len(store.list_jobs('one')), 2)
        voice_tools.execute('one', 'list')
        self.assertEqual(len(store.list_jobs('one')), 2)

    def test_apply_preserves_original_plan_and_sets_exact_voice(self):
        old = store.create(store.Project(session_id='one', title='original', duration_seconds=15))
        folder = store.ROOT / old['id'];folder.mkdir()
        plan={'title':'original','design_brief':'keep','scenes':[{'title':'scene','narration':'원래 대본입니다.','visual':'typography'}], 'voice':{'preset':'sohee'}, 'music':{'kind':'none'}}
        (folder/'storyboard.json').write_text(json.dumps(plan));store.update(old['id'], 'completed', 'done', 100)
        voice_tools.execute('one','select','aiden')
        revised = voice_tools.execute('one','apply','',old['id'])
        self.assertNotEqual(revised['id'],old['id'])
        p=revised['project'];self.assertEqual(p['voice_preset'],'aiden');self.assertEqual(p['design_mode'],'preserve')
        self.assertEqual(p['plan']['scenes'][0]['narration'],plan['scenes'][0]['narration'])
        self.assertEqual(p['plan']['scenes'][0]['reuse_scene'],0);self.assertTrue(p['plan']['verbatim_narration'])
        self.assertEqual(json.loads((folder/'storyboard.json').read_text()),plan)
        with self.assertRaisesRegex(ValueError,'대화'): voice_tools.execute('two','apply','ryan',old['id'])
        with self.assertRaisesRegex(ValueError,'완성'): voice_tools.execute('one','apply','ryan',revised['id'])

    def test_unchanged_old_references_keep_audio_cache_keys(self):
        for preset in ['sohee','uncle_fu']:
            voice=Voice(preset=preset).model_dump()
            old=fingerprint({'text':'same','preset':preset,'speed':voice['speed'],'pause':voice['sentence_pause'],'version':2})
            self.assertEqual(speech_key('same',voice),old)
            with patch.object(voice_catalog,'get_voice',return_value={'reference_sha256':'changed','cache_version':2}):
                self.assertNotEqual(speech_key('same',voice),old)
        v=Voice(preset='ryan').model_dump();a=speech_key('same',v)
        with patch.object(voice_catalog,'get_voice',return_value={'reference_sha256':'changed'}):
            self.assertNotEqual(a,speech_key('same',v))

    def test_selected_voice_wins_over_author_default(self):
        project=store.Project(session_id='one', title='title',voice_preset='eric',plan={'title':'title','design_brief':'design','scenes':[{'title':'s','narration':'hello','visual':'typography'}]})
        result=plan_project(project,store.ROOT,{},[],None)
        self.assertEqual(result['voice']['preset'],'eric')

    def test_wrong_returned_voice_is_rejected(self):
        project=store.Project(session_id='one',title='title',duration_seconds=15)
        voice=Voice(preset='ryan').model_dump()
        plan={'voice':voice,'scenes':[{'narration':'음성 검사'}]}
        def wrong_voice(*args):return {'voice_preset':'sohee','reference_sha256':'wrong','items':[]}
        with self.assertRaisesRegex(ValueError,'참조 음성'):
            make_speech(project,plan,store.ROOT,None,{},wrong_voice,lambda *args:None)

    def test_voice_only_pace_fit_reuses_raw_audio_without_rewriting_or_resynthesis(self):
        data=io.BytesIO()
        with wave.open(data,'wb') as w:
            w.setnchannels(1);w.setsampwidth(2);w.setframerate(44100)
            w.writeframes(array.array('h',[1000,-1000]*352800).tobytes())
        voice=Voice(preset='ryan').model_dump()
        plan={'title':'title','design_brief':'keep','voice':voice,'scenes':[{'title':'s','visual':'typography','narration':'원래 대본'}],'verbatim_narration':True}
        project=store.Project(session_id='one',title='title',duration_seconds=15,voice_only=True)
        calls=[]
        def bridge(kind,payload):
            calls.append(kind)
            return {'voice_preset':'ryan','reference_sha256':voice_catalog.get_voice('ryan')['reference_sha256'],'items':[{'audio':base64.b64encode(data.getvalue()).decode(),'duration':16,'segments':[{'text':'원래 대본','start':0,'end':16}],'records':[]}]}
        result=make_speech(project,plan,store.ROOT,None,{},bridge,lambda *args:None)
        self.assertEqual(calls,['speech']);self.assertTrue(result['pace_fitted'])
        self.assertGreater(result['voice']['speed'],1);self.assertLessEqual(result['voice']['speed'],1.2)
        self.assertEqual(plan['scenes'][0]['narration'],'원래 대본')
        self.assertEqual((store.ROOT/'voice-0-raw.wav').read_bytes(),data.getvalue())
        self.assertAlmostEqual(result['items'][0]['duration'],14.45,delta=.1)

    def test_voice_apply_reuses_real_image_bytes_without_generation(self):
        old=store.create(store.Project(session_id='one',title='original',duration_seconds=15))
        parent=store.ROOT/old['id'];parent.mkdir()
        from PIL import Image
        Image.new('RGB',(16,16),'navy').save(parent/'asset-0.png')
        original=(parent/'asset-0.png').read_bytes()
        plan={'title':'original','design_brief':'keep','voice':{'preset':'sohee'},'scenes':[{'title':'s','narration':'원래 대본입니다','visual':'image','image_prompt':'original image'}]}
        (parent/'storyboard.json').write_text(json.dumps(plan))
        (parent/'timing.json').write_text(json.dumps({'scenes':[{'kind':'image','asset':'asset-0.png'}]}))
        store.update(old['id'],'completed','done',100)
        new=voice_tools.execute('one','apply','aiden',old['id']);project=store.Project.model_validate(new['project']);folder=store.ROOT/new['id'];folder.mkdir()
        timeline=[{**project.plan['scenes'][0],'duration':15}]
        def no_generation(*args):raise AssertionError('voice change regenerated image')
        result=make_visuals(project,project.plan,timeline,folder,parent,[],no_generation,lambda *args:None)
        self.assertEqual((folder/result[0]['asset']).read_bytes(),original)
        self.assertEqual((parent/'asset-0.png').read_bytes(),original)

    def test_voice_only_design_uses_original_even_with_saved_template_or_failed_child(self):
        parent=store.ROOT/'parent';parent.mkdir();original={'css':'original css','javascript':'original animation'}
        (parent/'artwork.json').write_text(json.dumps(original))
        project=store.Project(session_id='one',title='test',voice_only=True,design_mode='preserve',template_id='a'*32)
        for stale in [False,True]:
            folder=store.ROOT/str(stale);folder.mkdir()
            if stale:(folder/'artwork.json').write_text(json.dumps({'css':'unwanted redesign'}))
            observed=[]
            def compile_only(folder,spec,artwork):
                observed.append(artwork);raise RuntimeError('checked source before render')
            with patch.object(pipeline.authoring,'specification',return_value={'caption_rules':{}}), patch.object(pipeline.authoring,'get_template',side_effect=AssertionError('saved template replaced original')), patch.object(pipeline.authoring,'compile_composition',side_effect=compile_only), patch.object(pipeline,'call_author',side_effect=AssertionError('voice-only changed design')):
                with self.assertRaisesRegex(RuntimeError,'checked source'):
                    pipeline.design_project(project,{},[{'asset':'','kind':'typography'}],15,[],folder,parent,None,lambda *args:None)
            self.assertEqual(observed,[original])

    def test_visual_review_reuse_requires_identical_content_code_and_fresh_audit(self):
        parent=store.ROOT/'parent';child=store.ROOT/'child';parent.mkdir();child.mkdir()
        artwork={'css':'approved css'}
        for folder in [parent,child]:(folder/'artwork.json').write_text(json.dumps(artwork))
        plan={'title':'same','design_brief':'same','voice':{'preset':'ryan'},'scenes':[{'title':'s','narration':'same words','visual':'image','reuse_scene':None}]}
        (parent/'storyboard.json').write_text(json.dumps(plan))
        quality={'passed':True,'applied':{'template':{'template_sha256':'approved'}},'verification':{'visual_review':{'passed':True}}}
        (parent/'quality.json').write_text(json.dumps(quality))
        changed=copy.deepcopy(plan);changed['voice']['preset']='aiden';changed['scenes'][0]['reuse_scene']=0
        project=store.Project(session_id='one',title='same',source_job_id='a'*32,voice_only=True)
        check=lambda p,a: pipeline.reused_visual_review(project,p,child,parent,{'template_sha256':'approved'},a)
        self.assertTrue(check(changed,{'passed':True})['passed'])
        self.assertIsNone(check(changed,{'passed':False}))
        changed['scenes'][0]['narration']='different words';self.assertIsNone(check(changed,{'passed':True}))
        changed['scenes'][0]['narration']='same words'
        (child/'artwork.json').write_text(json.dumps({'css':'changed'}));self.assertIsNone(check(changed,{'passed':True}))


if __name__=='__main__':unittest.main()
