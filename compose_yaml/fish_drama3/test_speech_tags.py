import array
import base64
import io
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import speech


def pcm(seconds, amplitude=1000):
    return array.array('h', [amplitude, -amplitude] * round(seconds * 44100 / 2)).tobytes()


class NativeDirections(unittest.TestCase):
    def test_continuous_plain_preview_preserves_words_in_one_request(self):
        text='안녕하세요. 이 목소리로 새로운 이야기를 전해 드립니다.'
        with patch('speech.httpx.Client') as client:
            post=client.return_value.__enter__.return_value.post
            post.return_value=MagicMock(headers={},content=pcm(3))
            result=speech.synthesize(text,'http://example.test','ref.wav',continuous=True)
        self.assertEqual(post.call_count,1)
        self.assertEqual(post.call_args.kwargs['json']['input'],text)
        self.assertEqual(len(result['segments']),1)
        self.assertTrue(result['continuous'])
        for invalid in ['', '가'*161, '<whisper>안녕']:
            with patch('speech.httpx.Client') as client:
                with self.assertRaises(ValueError):speech.synthesize(invalid,'http://example.test','ref.wav',continuous=True)
                client.assert_not_called()

    def test_scopes_and_directions_survive_multiple_sentences_and_old_length_limit(self):
        for text in ['[whisper] 첫 문장입니다. 다음 문장도 속삭입니다.',
                     '<whisper>첫 문장입니다. 다음 문장입니다.</whisper> 평소 목소리입니다.',
                     '[조용히 속삭이며] ' + '여러 문장을 같은 목소리로 전달합니다. ' * 5,
                     '<slow><whisper>천천히 말합니다. 비밀입니다.</whisper></slow>']:
            self.assertEqual(speech.chunks(text), [text.strip()])
        self.assertEqual(speech.chunks('첫 문장입니다. 다음 문장입니다.'), ['첫 문장입니다.', '다음 문장입니다.'])

    def test_malformed_and_oversized_directions_fail_before_synthesis(self):
        for text in ['<whisper>안녕하세요.', '안녕하세요.</whisper>', '[whisper 안녕하세요.',
                     '<slow><whisper>안녕</slow></whisper>', '<whisper >안녕</whisper>',
                     '[whisper] ' + '가' * (speech.TAGGED_TEXT_LIMIT + 1), '[whisper]']:
            with self.subTest(text=text), patch('speech.httpx.Client') as client:
                with self.assertRaises(ValueError):
                    speech.synthesize(text, 'http://example.test', 'ref.wav')
                client.assert_not_called()

    def test_native_pause_and_audio_are_not_retried_trimmed_or_spliced(self):
        text = '<whisper>첫 문장입니다. [pause] 다음 문장입니다.</whisper>'
        original = pcm(.4, 0) + pcm(1) + pcm(2.4, 0) + pcm(1, 12) + pcm(.7, 0)
        response = MagicMock(headers={}, content=original)
        with patch('speech.httpx.Client') as client:
            post = client.return_value.__enter__.return_value.post
            post.return_value = response
            result = speech.synthesize(text, 'http://example.test', 'ref.wav')
        self.assertEqual(post.call_count, 1)
        self.assertEqual(post.call_args.kwargs['json']['input'], text)
        self.assertEqual(result['spoken_text'], '첫 문장입니다.  다음 문장입니다.'.replace('  ', ' '))
        self.assertEqual(result['segments'][0]['text'], result['spoken_text'])
        self.assertEqual(len(result['segments']), 1)
        with wave.open(io.BytesIO(base64.b64decode(result['audio']))) as wav:
            self.assertEqual(wav.readframes(wav.getnframes()), original)

    def test_near_silent_generated_tail_is_rejected_instead_of_trimmed_into_success(self):
        response = MagicMock(headers={}, content=pcm(1) + pcm(8, 0))
        with tempfile.TemporaryDirectory() as directory, patch('speech.httpx.Client') as client:
            client.return_value.__enter__.return_value.post.return_value = response
            with self.assertRaisesRegex(ValueError, 'trailing_silence'):
                speech.synthesize('[whisper] 첫 문장입니다. 마지막 문장입니다.', 'http://example.test', 'ref.wav', Path(directory))
            self.assertEqual(client.return_value.__enter__.return_value.post.call_count, 3)
            self.assertEqual(len(list(Path(directory).glob('*.wav'))), 3)

    def test_quiet_whisper_tail_and_explicit_final_pause_are_distinct_from_stall(self):
        quiet = speech.analyze(pcm(1) + pcm(.2, 0) + pcm(3, 12) + pcm(.2, 0), 44100)
        self.assertLess(quiet['trailing_silence_seconds'], .25)
        self.assertEqual(speech.rejection_reasons(quiet, '[whisper] 조용한 끝말입니다.'), [])
        pause = speech.analyze(pcm(1) + pcm(3, 0), 44100)
        self.assertEqual(speech.rejection_reasons(pause, '<whisper>잠시 쉽니다. [pause]</whisper>'), [])
        self.assertIn('trailing_silence', speech.rejection_reasons(pause, '[pause] 이어서 말합니다.'))
        self.assertIn('output_cap', speech.rejection_reasons(speech.analyze(pcm(41.8), 44100), '끝입니다. [pause]'))


if __name__ == '__main__':
    unittest.main()
