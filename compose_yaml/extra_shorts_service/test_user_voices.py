import io
import base64
import hashlib
import json
import shutil
import tempfile
import time
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
import speech
import store
import voice_analysis
import voice_catalog
import voice_jobs
import voice_library
import voice_runner
from voice_quality import coverage


class LibraryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.patches=[patch.object(store,'ROOT',self.root/'jobs'),patch.object(voice_catalog,'LIBRARY',self.root/'voices')]
        for p in self.patches:p.start()
        store.initialize();voice_jobs.initialize()
        self.wav=self.root/'reference.wav';self.wav.write_bytes(speech.wav_bytes(b'\x10\x00'*44100))

    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.temp.cleanup()

    def staged(self,name='내 음성'):
        return voice_library.stage(self.wav,'안녕하세요. 참조 대사입니다.','one',name,{'duration_seconds':1,'warnings':[]})

    def ready(self,scope='persistent'):
        value=self.staged()
        return voice_library.finish(value['id'],'one',self.wav,'미리 듣기',{'passed':True},scope)

    def test_preparing_reference_expires_and_deleted_session_cancels_work(self):
        value=self.staged()
        current=voice_catalog._current(value['id'])
        self.assertGreater(current['expires_at'],time.time())
        job=voice_jobs.create(voice_jobs.Request(session_id='one',action='generate',voice_id='sohee',text='안녕하세요.',idempotency_key='cancel-test'))
        voice_jobs.cancel_session('one')
        self.assertEqual(voice_jobs.get(job['id'])['status'],'canceled')
        self.assertFalse((voice_catalog.LIBRARY/value['id']/'r1/reference.wav').exists())

    def test_source_reuse_rejects_corruption(self):
        job=voice_jobs.create(voice_jobs.Request(session_id='one',action='prepare',idempotency_key='source'),io.BytesIO(self.wav.read_bytes()))
        (voice_jobs.folder(job['id'])/'source.bin').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError,'해시'):
            voice_jobs.create(voice_jobs.Request(session_id='one',action='prepare',source_job_id=job['id'],idempotency_key='reuse'))

    def test_reference_reset_cannot_silently_generate_from_old_voice(self):
        with self.assertRaisesRegex(ValueError,'target_voice_id'):
            voice_jobs.Request(session_id='one',action='prepare',voice_id='sohee',source_url='https://example.com/new.wav',idempotency_key='bad-reset')
        with self.assertRaisesRegex(ValueError,'동시에'):
            voice_jobs.Request(session_id='one',action='prepare',source_url='https://example.com/new.wav',source_job_id='a'*32,idempotency_key='two-sources')

    def test_manifest_repair_after_commit_before_filesystem_publish(self):
        value=self.staged()
        original=voice_jobs.atomic_json
        def fail_current(path,data):
            if path.name=='current.json':raise OSError('simulated power loss')
            return original(path,data)
        with patch.object(voice_library,'atomic_json',side_effect=fail_current):
            with self.assertRaises(OSError):
                voice_library.finish(value['id'],'one',self.wav,'대사',{'passed':True},'persistent')
        voice_library.repair_manifests()
        recovered=voice_catalog.get_voice(value['id'],'two')
        self.assertEqual(recovered['scope'],'persistent')
        self.assertEqual(recovered['revision'],1)

    def test_builtins_unchanged_and_custom_private_until_ready(self):
        self.assertEqual(len(voice_catalog.catalog()),9)
        value=self.staged()
        self.assertEqual(len(voice_catalog.catalog('one')),9)
        with self.assertRaises(ValueError):voice_catalog.reference(value['id'],'two',allow_pending=True)
        self.assertEqual(voice_catalog.reference(value['id'],'one',allow_pending=True)[1],'안녕하세요. 참조 대사입니다.')
        voice=voice_library.finish(value['id'],'one',self.wav,'미리 듣기',{'passed':True},'persistent')
        self.assertEqual(voice_catalog.get_voice(voice['id'],'two')['name'],'내 음성')
        self.assertEqual(len(voice_catalog.public_catalog('two')),10)
        self.assertNotIn('source',voice_catalog.public_catalog('two')[-1])

    def test_temporary_is_not_library_default_and_expires_but_outputs_are_separate(self):
        voice=self.ready('temporary')
        self.assertEqual(len(voice_catalog.public_catalog('one')),9)
        self.assertEqual(len(voice_catalog.public_catalog('one',True)),10)
        with self.assertRaises(ValueError):voice_catalog.get_voice(voice['id'],'two')
        with self.assertRaises(ValueError):store.select_voice('one',voice['id'])
        current=voice_catalog._current(voice['id']);current['expires_at']=time.time()-1;voice_library._record(current)
        with self.assertRaises(ValueError):voice_catalog.get_voice(voice['id'],'one')
        voice_library.cleanup();self.assertFalse((voice_catalog.LIBRARY/voice['id']/'r1/reference.wav').exists())
        self.assertTrue(self.wav.exists())

    def test_promotion_rename_and_reference_revision_keep_old_bytes(self):
        old=self.ready('temporary')
        with self.assertRaises(ValueError):voice_library.manage('one',old['id'],'promote','bad\nname')
        self.assertEqual(voice_catalog.get_voice(old['id'],'one')['scope'],'temporary')
        voice_library.manage('one',old['id'],'promote','영구 이름')
        self.assertEqual(voice_catalog.get_voice(old['id'],'two')['name'],'영구 이름')
        voice_library.manage('two',old['id'],'rename','새 이름')
        before=voice_catalog.reference(old['id'],'two',1)[0].read_bytes()
        staged=self.staged('다른 참조');updated=voice_library.finish(staged['id'],'one',self.wav,'새 대사',{'passed':True},'persistent',target_id=old['id'])
        self.assertEqual(updated['revision'],2);self.assertEqual(updated['name'],'새 이름')
        self.assertEqual(voice_catalog.reference(old['id'],'two',1)[0].read_bytes(),before)
        again=voice_library.finish(staged['id'],'one',self.wav,'새 대사',{'passed':True},'persistent',target_id=old['id']);self.assertEqual(again['revision'],2)

    def test_delete_removes_selection_but_keeps_reference_used_by_queued_job(self):
        voice=self.ready();store.select_voice('one',voice['id'])
        job=voice_jobs.create(voice_jobs.Request(session_id='one',action='generate',voice_id=voice['id'],text='안녕하세요.',idempotency_key='use'))
        voice_library.manage('one',voice['id'],'delete');voice_library.cleanup()
        self.assertIsNone(store.selected_voice('one'))
        with self.assertRaises(ValueError):voice_catalog.get_voice(voice['id'],'one')
        path,_=voice_catalog.reference(voice['id'],'one',1,allow_history=True);self.assertTrue(path.exists())
        voice_jobs.action(job['id'],'one','cancel');voice_library.cleanup();self.assertFalse(path.exists())

    def test_reference_corruption_fails_and_manifest_can_recover_from_db(self):
        voice=self.ready();parent=voice_catalog.LIBRARY/voice['id'];(parent/'current.json').write_text('broken')
        voice_library.repair_manifests();self.assertEqual(voice_catalog.get_voice(voice['id'])['id'],voice['id'])
        (parent/'r1/reference.wav').write_bytes(b'wrong')
        with self.assertRaisesRegex(ValueError,'검증 실패'):voice_catalog.reference(voice['id'])

    def test_job_idempotency_scope_and_restart_recovery(self):
        request=voice_jobs.Request(session_id='one',action='prepare',scope='persistent',name='음성',idempotency_key='request')
        first=voice_jobs.create(request,io.BytesIO(self.wav.read_bytes()));again=voice_jobs.create(request,io.BytesIO(self.wav.read_bytes()));self.assertEqual(first['id'],again['id'])
        with self.assertRaises(ValueError):voice_jobs.create(request.model_copy(update={'name':'다른 이름'}),io.BytesIO(self.wav.read_bytes()))
        with self.assertRaises(ValueError):voice_jobs.get(first['id'],'two')
        self.assertEqual(voice_jobs.claim(),first['id']);voice_jobs.initialize();self.assertEqual(voice_jobs.get(first['id'])['status'],'queued')

    def test_clause_omission_is_not_hidden_by_long_transcript_and_small_errors_are_allowed(self):
        expected='여러 가지 내용을 충분히 길게 설명해서 전체 글자 수를 늘립니다. 끝내지 마세요.'
        self.assertFalse(coverage(expected,'여러 가지 내용을 충분히 길게 설명해서 전체 글자 수를 늘립니다.')['passed'])
        self.assertTrue(coverage('이제 평소처럼 마무리합니다.','이제 평수처럼 마무리합니다.')['passed'])

    def test_preview_regeneration_does_not_reuse_old_sentence_split_checkpoint(self):
        voice=self.ready();request=voice_jobs.Request(session_id='one',action='prepare',idempotency_key='preview')
        text=voice_runner.PREVIEW_TEXT
        oldkey=hashlib.sha256(json.dumps({'voice':voice['id'],'revision':voice['revision'],'reference':voice['reference_sha256'],'transcript':voice.get('reference_text_sha256'),'input':text,'policy':'native-tags-v1'},sort_keys=True).encode()).hexdigest()
        output=self.root/'preview.wav';output.write_bytes(self.wav.read_bytes())
        (self.root/'generation.json').write_text(json.dumps({'fingerprint':oldkey,'file':'preview.wav','sha256':voice_runner.digest(output)}))
        response={'voice_preset':voice['id'],'reference_sha256':voice['reference_sha256'],'items':[{'audio':base64.b64encode(self.wav.read_bytes()).decode(),'continuous':True}]}
        with patch.object(voice_runner,'bridge',return_value=response) as bridge,patch.object(voice_runner,'asr',return_value={'text':text}):
            result=voice_runner.generate({'session_id':'one','id':'test'},request,self.root,voice,False,lambda:None,lambda *a:None)
        self.assertEqual(bridge.call_count,1)
        payload=bridge.call_args.args[1]
        self.assertEqual(payload['texts'],[text])
        self.assertTrue(payload['voice']['continuous'])
        self.assertEqual(payload['voice']['reference_revision'],voice['revision'])
        self.assertNotEqual(result['fingerprint'],oldkey)
        self.assertTrue(result['validation']['passed'])

    def test_input_correction_is_validated_and_publication_ack_cannot_hide_new_version(self):
        request=voice_jobs.Request(session_id='one',action='prepare',idempotency_key='correction')
        job=voice_jobs.create(request,io.BytesIO(self.wav.read_bytes()))
        voice_jobs.update(job['id'],'needs_input','select',30,{'question':'choose'})
        voice_jobs.acknowledge(job['id'],1)
        with self.assertRaises(ValueError):
            voice_jobs.action(job['id'],'one','choose',{'start':4,'end':2})
        self.assertEqual(voice_jobs.get(job['id'])['status'],'needs_input')
        corrected=voice_jobs.action(job['id'],'one','choose',{'candidate':0,'transcript':'수정한 대사'})
        self.assertEqual(corrected['version'],2)
        self.assertEqual(corrected['request']['transcript'],'수정한 대사')
        voice_jobs.update(job['id'],'completed','done',100,{})
        voice_jobs.acknowledge(job['id'],1)
        self.assertIn(job['id'],[x['id'] for x in voice_jobs.pending_deliveries()])
        voice_jobs.acknowledge(job['id'],2)
        self.assertNotIn(job['id'],[x['id'] for x in voice_jobs.pending_deliveries()])


class AnalysisTests(unittest.TestCase):
    def test_long_silence_and_antiphase_stereo_choose_real_speech_without_downmix_cancellation(self):
        ref=voice_catalog.ROOT/'voices/references/serena.wav'
        with wave.open(str(ref)) as f:
            rate=f.getframerate();samples=np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').copy()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'renamed.bin';mono=np.concatenate([np.zeros(rate*20,dtype=np.int16),samples,np.zeros(rate,dtype=np.int16)])
            stereo=np.column_stack((mono,-mono)).astype('<i2')
            with wave.open(str(source),'wb') as f:f.setnchannels(2);f.setsampwidth(2);f.setframerate(rate);f.writeframes(stereo.tobytes())
            request=voice_jobs.Request(session_id='one',action='prepare',scope='temporary',idempotency_key='analysis')
            result=voice_analysis.analyze(source,root,request,lambda:None,lambda value:None)
            candidate=result['candidates'][0];self.assertGreater(candidate['start'],19);self.assertLessEqual(candidate['duration_seconds'],15);self.assertGreater(candidate['speech_seconds'],5)
            with wave.open(str(root/candidate['file'])) as f:
                self.assertEqual(f.getnchannels(),1);audio=np.frombuffer(f.readframes(f.getnframes()),dtype='<i2');self.assertGreater(np.max(np.abs(audio)),100)

    def test_too_short_and_silence_do_not_become_ready_references(self):
        for seconds in [.3,3]:
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);p=root/'input.wav';p.write_bytes(speech.wav_bytes(b'\0\0'*round(seconds*44100)))
                request=voice_jobs.Request(session_id='one',action='prepare',idempotency_key='silent')
                with self.assertRaises(voice_analysis.NeedsInput):voice_analysis.analyze(p,root,request,lambda:None,lambda value:None)


if __name__=='__main__':unittest.main()
