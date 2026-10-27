import array
import base64
import io
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import audio_mix
import pipeline
import speech
import store
from spec import Voice


class NativeTagDelivery(unittest.TestCase):
    def test_default_pacing_preserves_native_gain_and_silence_bit_for_bit(self):
        pcm = array.array('h', [2000, -2000] * 22050 + [0, 0] * 22050 + [12, -12] * 22050).tobytes()
        raw = speech.wav_bytes(pcm)
        text = '<whisper>작게 말합니다. [pause] 이어서 말합니다.</whisper>'
        item = {'audio':base64.b64encode(raw).decode(),'duration':3,'text':text,'native_tags':True,
                'segments':[{'text':speech.spoken_text(text),'start':0,'end':3}]}
        with tempfile.TemporaryDirectory() as directory, patch.object(audio_mix.subprocess,'run',side_effect=AssertionError('tagged audio was normalized')):
            folder=Path(directory)
            result=audio_mix.prepare_voice(item,folder,0,Voice().model_dump())
            self.assertEqual((folder/'voice-0.wav').read_bytes(),raw)
            self.assertEqual((folder/'voice-0-raw.wav').read_bytes(),raw)
            self.assertTrue(result['native_gain_preserved'])

    def test_asr_checks_only_spoken_text_and_tagged_cache_has_new_policy(self):
        voice=Voice(preset='serena').model_dump()
        text='[whisper] 첫 문장입니다. <slow>다음 문장입니다.</slow>'
        key=pipeline.speech_key(text,voice)
        old=pipeline.fingerprint({'text':text,'preset':voice['preset'],'speed':voice['speed'],
                                  'pause':voice['sentence_pause'],'version':3,
                                  'reference_sha256':pipeline.voice_catalog.get_voice('serena')['reference_sha256']})
        self.assertNotEqual(key,old)
        project=store.Project(session_id='test',title='test')
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);(folder/'voice-0.wav').write_bytes(speech.wav_bytes(b'\0\0'*44100))
            calls=[]
            def bridge(operation,payload):
                calls.append((operation,payload))
                return {'passed':True,'transcript':'첫 문장입니다. 다음 문장입니다.'}
            pipeline.verify_speech(project,{'scenes':[{'narration':text}]},{'items':[{'fingerprint':key}]},folder,bridge)
            self.assertEqual(calls[0][0],'transcribe')
            self.assertEqual(calls[0][1]['expected'],'첫 문장입니다. 다음 문장입니다.')


if __name__ == '__main__':
    unittest.main()
