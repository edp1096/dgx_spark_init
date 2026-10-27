import asyncio
import unittest
from unittest.mock import patch

import api
import voice_catalog


class Request:
    async def is_disconnected(self):return False


class Voices(unittest.TestCase):
    def test_entire_batch_is_validated_before_any_generation(self):
        api.busy=False;api.quiescing=False
        payload=api.Batch(texts=['정상 문장입니다.', '<whisper>닫히지 않은 지시입니다.'])
        with patch.object(api,'synthesize') as synth:
            with self.assertRaises(api.HTTPException) as error:
                asyncio.run(api.batch(payload,Request()))
            self.assertEqual(error.exception.status_code,422)
            synth.assert_not_called()
        self.assertFalse(api.busy)

    def test_nine_references_are_intact_and_transcripts_match(self):
        self.assertEqual(len(voice_catalog.catalog()),9)
        for v in voice_catalog.catalog():
            path,text=voice_catalog.reference(v['id'])
            self.assertTrue(path.is_file());self.assertEqual(text,v['reference_text'])
        with self.assertRaises(ValueError):api.Voice(preset='unknown')

    def test_multi_voice_preparation_uses_each_reference_without_training(self):
        api.busy=False;api.quiescing=False;calls=[]
        def synth(text,backend,path,**kwargs):
            calls.append((path,kwargs['reference_text']))
            return {'audio':'AA==','duration':1}
        payload=api.Batch(texts=['first','second'],voice=api.Voice(presets=['ryan','sohee']))
        with patch.object(api,'synthesize',side_effect=synth):result=asyncio.run(api.batch(payload,Request()))
        self.assertIsNone(result['voice_preset'])
        self.assertEqual([x['voice_preset'] for x in result['items']],['ryan','sohee'])
        self.assertNotEqual(calls[0][0],calls[1][0]);self.assertNotEqual(calls[0][1],calls[1][1])
        self.assertFalse(api.busy)

    def test_bad_batch_or_reference_does_not_leave_service_busy(self):
        api.busy=False;api.quiescing=False
        with self.assertRaises(Exception):asyncio.run(api.batch(api.Batch(texts=['one'],voice=api.Voice(presets=['ryan','sohee'])),Request()))
        self.assertFalse(api.busy)
        with patch.object(api,'reference',side_effect=ValueError('bad reference')):
            with self.assertRaises(api.HTTPException) as error:asyncio.run(api.batch(api.Batch(texts=['one']),Request()))
            self.assertEqual(error.exception.status_code,422)
        self.assertFalse(api.busy)


if __name__=='__main__':unittest.main()
