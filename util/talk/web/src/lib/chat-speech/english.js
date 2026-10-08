const CHAT_SHORTHAND = new Map([
  ['afk', 'away from keyboard'],
  ['brb', 'be right back'],
  ['btw', 'by the way'],
  ['fyi', 'for your information'],
  ['gg', 'good game'],
  ['idk', "I don't know"],
  ['ikr', 'I know, right'],
  ['imho', 'in my humble opinion'],
  ['imo', 'in my opinion'],
  ['lmao', 'laughing my ass off'],
  ['lol', 'laughing out loud'],
  ['ngl', 'not going to lie'],
  ['np', 'no problem'],
  ['omg', 'oh my god'],
  ['omw', 'on my way'],
  ['pls', 'please'],
  ['plz', 'please'],
  ['rn', 'right now'],
  ['rofl', 'rolling on the floor laughing'],
  ['tbh', 'to be honest'],
  ['thx', 'thanks'],
  ['ty', 'thank you'],
  ['tysm', 'thank you so much'],
  ['wp', 'well played'],
  ['yw', "you're welcome"],
]);

const CHAT_SHORTHAND_PATTERN = new RegExp(`\\b(?:${[...CHAT_SHORTHAND.keys()]
  .sort((left, right) => right.length - left.length)
  .join('|')})\\b`, 'giu');
const PROTECTED_PATTERN = /https?:\/\/\S+|www\.\S+|\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b/giu;

function protectedRanges(text) {
  return [...text.matchAll(PROTECTED_PATTERN)].map((match) => [match.index, match.index + match[0].length]);
}

export function normalizeEnglishChatSpeech(text) {
  const source = String(text || '');
  const protectedText = protectedRanges(source);
  return source.replace(CHAT_SHORTHAND_PATTERN, (value, offset) => {
    if (protectedText.some(([start, end]) => offset >= start && offset < end)) return value;
    // The mixed-case spelling is conventionally used for League of Legends.
    if (value === 'LoL') return value;
    return CHAT_SHORTHAND.get(value.toLowerCase()) || value;
  });
}
