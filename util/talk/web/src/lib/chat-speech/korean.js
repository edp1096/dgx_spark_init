const CHAT_SHORTHAND = new Map([
  ['ㅍㅎㅎ', '푸하하'],
  // ㄳ is also commonly entered as a single compound-final character.
  ['ㄳ', '감사'], ['ᆪ', '감사'], ['ﾣ', '감사'], ['ᄀᄉ', '감사'],
  ['ㅇㅋ', '오키'], ['ㅇㅇ', '응응'], ['ㄴㄴ', '노노'], ['ㄱㄱ', '고고'],
  ['ㅂㅂ', '바이바이'], ['ㅂㅇ', '바이'], ['ㅎㅇ', '하이'], ['ㄱㅅ', '감사'],
  ['ㅈㅅ', '죄송'], ['ㅊㅋ', '축하'], ['ㅅㄱ', '수고'], ['ㄷㄷ', '덜덜'],
  ['ㅁㄹ', '몰라'], ['ㄱㅊ', '괜찮아'], ['ㄹㅇ', '리얼'], ['ㅇㅈ', '인정'],
  ['ㅇㄷ', '어디'], ['ㅉㅉ', '쯧쯧'],
]);

const CHAT_SHORTHAND_PATTERN = /ㅍㅎㅎ|ᄀᄉ|ㄳ|ᆪ|ﾣ|ㅇㅋ|ㅇㅇ|ㄴㄴ|ㄱㄱ|ㅂㅂ|ㅂㅇ|ㅎㅇ|ㄱㅅ|ㅈㅅ|ㅊㅋ|ㅅㄱ|ㄷㄷ|ㅁㄹ|ㄱㅊ|ㄹㅇ|ㅇㅈ|ㅇㄷ|ㅉㅉ/gu;

export function normalizeKoreanChatSpeech(text) {
  return String(text || '')
    .replace(CHAT_SHORTHAND_PATTERN, (value) => CHAT_SHORTHAND.get(value))
    // Korean chat commonly omits the vowel in laughter. A single consonant
    // may be a literal letter, so only restore repeated laughter characters.
    .replace(/ㅎ{2,}/gu, (value) => value.replace(/ㅎ/gu, '흐'))
    .replace(/ㅋ{2,}/gu, (value) => value.replace(/ㅋ/gu, '크'));
}
