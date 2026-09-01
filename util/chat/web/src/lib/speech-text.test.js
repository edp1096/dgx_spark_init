import assert from 'node:assert/strict';
import test from 'node:test';
import { createSpeechBatcher, createSpeechChunker, normalizeSpeechNotation, speechTextFromMarkdown } from './speech-text.js';

test('speech text keeps block pauses and removes trailing source links', () => {
  const markdown = `## 🌦️ 신길동 내일 날씨

- **기온**: 최고 28°C / 최저 22°C
- **날씨**: 흐림, 소나기 ☁️
- **비**: 낮 시간대 약한 비 가능

> ☂️ 우산 챙기세요!

[AccuWeather - 신길동](https://example.com/weather) | [기상청](https://example.com/kma)`;
  const result = speechTextFromMarkdown(markdown);

  assert.equal(result.includes('AccuWeather'), false);
  assert.equal(result.includes('기상청'), false);
  assert.equal(result.includes('🌦️'), false);
  assert.equal(result.includes('☂️'), false);
  assert.match(result, /기온: 최고 28도, 최저 22도/u);
  assert.match(result, /가능\.\n우산 챙기세요!/u);
  assert.match(result, /신길동 내일 날씨\.\n기온:/u);
});

test('normalizes temperatures, ranges, percentages, and emoji for Korean speech', () => {
  assert.equal(
    normalizeSpeechNotation('🌡️ 22~28°C, 68～77°F, 강수 20~40% ☂️'),
    '22도에서 28도, 화씨 68도에서 77도, 강수 20퍼센트에서 40퍼센트',
  );
  assert.equal(normalizeSpeechNotation('버전 1~3 및 단위 °C'), '버전 1에서 3 및 단위 섭씨');
});

test('removes visual symbols instead of naming them in narration', () => {
  assert.equal(normalizeSpeechNotation('한국경제신문 → 韓國經濟新聞'), '한국경제신문, 韓國經濟新聞');
  assert.equal(normalizeSpeechNotation('← ↑ ↓ ↔'), '');
  assert.equal(speechTextFromMarkdown('**한국경제신문** → **韓國經濟新聞**'), '한국경제신문, 韓國經濟新聞.');
  assert.equal(normalizeSpeechNotation('정답 = 42 ★'), '정답, 42');
});

test('reads unambiguous arithmetic notation by meaning', () => {
  assert.equal(normalizeSpeechNotation('1+1=귀요미'), '1 더하기 1은 귀요미');
  assert.equal(normalizeSpeechNotation('2 × 3 = 6'), '2 곱하기 3은 6');
  assert.equal(normalizeSpeechNotation('10/2=5'), '10 나누기 2는 5');
  assert.equal(normalizeSpeechNotation('12 / 3 / 2 = 2'), '12 나누기 3 나누기 2는 2');
  assert.equal(normalizeSpeechNotation('1+1+1=3'), '1 더하기 1 더하기 1은 3');
});

test('reads slash-separated numeric status sequences as a list', () => {
  assert.equal(
    normalizeSpeechNotation('로드 평균: 0.94 / 0.93 / 1.00'),
    '로드 평균: 0.94, 0.93, 1.00',
  );
});

test('restores omitted Korean vowels in chat laughter for speech', () => {
  assert.equal(normalizeSpeechNotation('ㅎㅎㅎ 😄'), '흐흐흐');
  assert.equal(normalizeSpeechNotation('그렇네 ㅋㅋㅋㅋ'), '그렇네 크크크크');
  assert.equal(speechTextFromMarkdown('ㅎㅎㅎ 😄'), '흐흐흐.');
});

test('speaks conventional Korean chat shorthand naturally', () => {
  const examples = new Map([
    ['ㅇㅋ', '오키'], ['ㅇㅇ', '응응'], ['ㄴㄴ', '노노'], ['ㄱㄱ', '고고'],
    ['ㅂㅂ', '바이바이'], ['ㅂㅇ', '바이'], ['ㅎㅇ', '하이'], ['ㄱㅅ', '감사'],
    ['ㅈㅅ', '죄송'], ['ㅊㅋ', '축하'], ['ㅅㄱ', '수고'], ['ㄷㄷ', '덜덜'],
    ['ㅁㄹ', '몰라'], ['ㄱㅊ', '괜찮아'], ['ㄹㅇ', '리얼'], ['ㅇㅈ', '인정'],
    ['ㅇㄷ', '어디'], ['ㅉㅉ', '쯧쯧'], ['ㅍㅎㅎ', '푸하하'],
  ]);
  for (const [shorthand, spoken] of examples) {
    assert.equal(normalizeSpeechNotation(shorthand), spoken);
  }
  assert.equal(speechTextFromMarkdown('ㅇㅋ, 그렇게 하자'), '오키, 그렇게 하자.');
  assert.equal(normalizeSpeechNotation('속상해 ㅠ_ㅠ 그래도 괜찮아 ㅡㅡ'), '속상해 그래도 괜찮아');
  assert.equal(normalizeSpeechNotation('ㅈㄱ ㅁㅇ ㅇㄴ ㄱㄷ'), 'ㅈㄱ ㅁㅇ ㅇㄴ ㄱㄷ');
});

test('normalizes joined and Unicode variants of the thanks shorthand', () => {
  assert.equal(normalizeSpeechNotation('ㄳㄳ!!'), '감사감사!!');
  assert.equal(normalizeSpeechNotation('ㄱㅅ ㄳ ᆪ ﾣ ᄀᄉ'), '감사 감사 감사 감사 감사');
});

test('speaks common English chat shorthand as natural phrases', () => {
  assert.equal(
    normalizeSpeechNotation('lol idk, btw I am afk rn'),
    "laughing out loud I don't know, by the way I am away from keyboard right now",
  );
  assert.equal(normalizeSpeechNotation('OMG, tysm! brb'), 'oh my god, thank you so much! be right back');
  assert.equal(normalizeSpeechNotation('gg wp, lmao'), 'good game well played, laughing my ass off');
});

test('does not rewrite English shorthand inside links, email, or the LoL game name', () => {
  assert.equal(normalizeSpeechNotation('https://example.com/idk?next=lol'), 'https://example.com/idk?next=lol');
  assert.equal(normalizeSpeechNotation('lol@example.com'), 'lol@example.com');
  assert.equal(normalizeSpeechNotation('LoL 게임과 lol 반응'), 'LoL 게임과 laughing out loud 반응');
});

test('removes visual list bullets and ordinal markers without dropping content numbers', () => {
  const markdown = `• 첫 번째 항목
◦ 두 번째 항목
1. 세 번째 항목
(2) 네 번째 항목
③ 다섯 번째 항목
- [x] 수량은 3개
2026년 자료`;
  assert.equal(
    speechTextFromMarkdown(markdown),
    '첫 번째 항목.\n두 번째 항목.\n세 번째 항목.\n네 번째 항목.\n다섯 번째 항목.\n수량은 3개.\n2026년 자료.',
  );
});

test('speaks common weather speed units and tabular rows naturally', () => {
  assert.equal(normalizeSpeechNotation('바람\t1.0 m/s'), '바람, 초속 1미터');
  assert.equal(normalizeSpeechNotation('돌풍 12.5 m/s · 이동 36 km/h'), '돌풍 초속 12.5미터 · 이동 시속 36킬로미터');
  assert.equal(normalizeSpeechNotation('강수 2.0 mm/h'), '강수 시간당 2밀리미터');
});

test('speech text retains meaningful inline links and removes tool calls', () => {
  const markdown = '자세한 내용은 [사용 설명서](https://example.com)를 확인하세요.\n<tool_call>{}</tool_call>';
  assert.equal(speechTextFromMarkdown(markdown), '자세한 내용은 사용 설명서를 확인하세요.');
});

test('omits parenthetical asides only when requested for continuous voice', () => {
  const markdown = '**한국경제신문(한경)**은 한경닷컴(온라인), 한국경제TV와 함께합니다.';
  assert.equal(
    speechTextFromMarkdown(markdown, { omitParentheticals: true }),
    '한국경제신문은 한경닷컴, 한국경제TV와 함께합니다.',
  );
  assert.equal(
    speechTextFromMarkdown(markdown),
    '한국경제신문(한경)은 한경닷컴(온라인), 한국경제TV와 함께합니다.',
  );
});

test('continuous speech chunker omits full-width parenthetical asides', () => {
  const chunker = createSpeechChunker({ omitParentheticals: true });
  assert.deepEqual(chunker.push('한국경제신문（한경）은 경제 일간지야. '), ['한국경제신문은 경제 일간지야.']);
});

test('stream chunker emits completed visual lines and sentences once', () => {
  const chunker = createSpeechChunker();
  assert.deepEqual(chunker.push('## 날씨\n기온은 20도입니다. 다음'), ['날씨.','기온은 20도입니다.']);
  assert.deepEqual(chunker.push(' 문장입니다\n[기상청](https://example.com)\n'), ['다음 문장입니다.']);
  assert.deepEqual(chunker.finish(), []);
});

test('stream chunker waits for an unfinished sentence', () => {
  const chunker = createSpeechChunker();
  assert.deepEqual(chunker.push('아직 작성 중'), []);
  assert.deepEqual(chunker.finish(), ['아직 작성 중.']);
});

test('speech batcher reduces independent TTS requests while retaining an early threshold', () => {
  const batcher = createSpeechBatcher({ maxChunks: 3, targetCharacters: 100 });
  assert.deepEqual(batcher.push(['첫 문장.', '둘째 문장.']), []);
  assert.deepEqual(batcher.push(['셋째 문장.']), ['첫 문장.\n둘째 문장.\n셋째 문장.']);
  assert.deepEqual(batcher.push(['마지막 문장.']), []);
  assert.deepEqual(batcher.finish(), ['마지막 문장.']);
});
