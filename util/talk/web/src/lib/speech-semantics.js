// Conservative Korean narration rules. Classify notation before verbalizing it;
// an unclassified hyphen/colon is not evidence of subtraction or a clock time.
const digits = ['영','일','이','삼','사','오','육','칠','팔','구'];
export function sinoNumber(value) {
  const raw=String(value).replaceAll(',','');
  if(!/^-?\d{1,16}(?:\.\d+)?$/.test(raw))return value;
  const negative=raw.startsWith('-'),[integer,fraction]=raw.replace(/^-/,'').split('.');
  let n=BigInt(integer),groups=[],index=0;
  do {let group=Number(n%10000n),piece='';for(let i=3;i>=0;i--){const d=Math.floor(group/10**i)%10;if(d)piece+=(d===1&&i>0?'':digits[d])+['','십','백','천'][i];}if(piece)groups.unshift(piece+['','만','억','조'][index]);n/=10000n;index++;}while(n);
  return (negative?'마이너스 ':'')+(groups.join(' ')||'영')+(fraction?' 점 '+[...fraction].map(d=>digits[Number(d)]).join(' '):'');
}
function nativeNumber(value) {
 const n=Number(value);if(!Number.isInteger(n)||n<1||n>99)return sinoNumber(value);
 return ['','열','스물','서른','마흔','쉰','예순','일흔','여든','아흔'][Math.floor(n/10)]+['','한','두','세','네','다섯','여섯','일곱','여덟','아홉'][n%10] || '';
}
export function classifySpeechNotation(text) {
 // Sentence-local clues keep a score elsewhere from changing an unrelated range.
 return String(text).split(/(\n|[!?。])/u).map(sentence=>{
  let value=sentence;
  value=value.replace(/(?<![\w\d.])(-?\d+(?:\.\d+)?)\s*(<=|>=|!=|≤|≥|≠|<|>)\s*(-?\d+(?:\.\d+)?)(?![\w\d.])/gu,(_,a,op,b)=>{
   const particle=/[013678]$/.test(a)?'은':'는';
   const relation={'<':'보다 작다','>':'보다 크다','<=':' 이하이다','≤':' 이하이다','>=':' 이상이다','≥':' 이상이다','!=':'와 같지 않다','≠':'와 같지 않다'}[op];
   return `${sinoNumber(a)}${particle} ${sinoNumber(b)}${relation}`;
  });
  const sporting=/축구|야구|농구|배구|경기|스코어|득점|승리|패배|제압|승리|무승부/u.test(sentence);
  if(sporting)value=value.replace(/(?<![\d.\w-])(\d{1,3})\s*[-:]\s*(\d{1,3})(?![\d.\w-]|\s*(?:=|원|년|월|일|도|℃|%|시|분|초))/gu,'$1 대 $2');
  if(/비율|경쟁률|배합|비례/u.test(sentence))value=value.replace(/(?<![\d:])(\d+)\s*:\s*(\d+)(?![\d:])/gu,'$1 대 $2');
  else if(/오전|오후|새벽|아침|저녁|시각|출발|도착|시작|마감|시간/u.test(sentence))value=value.replace(/(?<![\d:])(\d{1,2}):(\d{2})(?![\d:])/gu,(m,h,min)=>Number(h)<24&&Number(min)<60?`${nativeNumber(Number(h)%12||12)} 시 ${sinoNumber(min)} 분`:m);
  else if(/영상|재생|경과|기록|타임/u.test(sentence))value=value.replace(/(?<![\d:])(\d+):(\d{2})(?![\d:])/gu,(m,min,sec)=>Number(sec)<60?`${sinoNumber(min)} 분 ${sinoNumber(sec)} 초`:m);
  // ISO dates must be recognized before any arithmetic processing.
  value=value.replace(/(?<![\w\d-])(\d{4})-(\d{1,2})-(\d{1,2})(?![\w\d-])/gu,(m,y,mo,d)=>Number(mo)>=1&&Number(mo)<=12&&Number(d)>=1&&Number(d)<=31?`${y}년 ${mo}월 ${d}일`:m);
  value=value.replace(/(?<![\w\d.-])(\d+)\s*-\s*(\d+)\s*(개월|년|일|도|원|개|명)(?![A-Za-z])/gu,'$1$3에서 $2$3');
  return value;
 }).join('');
}
export function readKoreanSpeech(text) {
 if(!/[가-힣]/u.test(text))return text;
 let value=String(text);
 const acronyms={KBS:'케이비에스',IAEA:'아이 에이 이 에이',MBC:'엠비씨',SBS:'에스비에스',AI:'에이아이',API:'에이피아이'};
 value=value.replace(/\b(?:KBS|IAEA|MBC|SBS|AI|API)\b/gu,m=>acronyms[m]);
 value=value.replace(/(?<![\w\d.])(\d{1,2})\s*월/gu,(m,n)=>Number(n)===6?'유월':Number(n)===10?'시월':`${sinoNumber(n)}월`);
 value=value.replace(/(?<![\w\d.])(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*(개월|퍼센트|킬로미터|밀리미터|미터|년|일|분|초|도|원|억|만|조|건|개|명|살|시)(?![A-Za-z])/gu,(m,n,unit)=>{
  const read=(['개','명','살'].includes(unit)||(unit==='시'&&Number(n)<=12))?(Number(n)===20?'스무':nativeNumber(n)):sinoNumber(n);
  return `${read} ${unit}`;
 });
 // Numbers in classified score/range/arithmetic expressions, including the result.
 value=value.replace(/(?<![\w\d.])(-?\d+(?:\.\d+)?)(?=\s*(?:대\s|에서\s|더하기\s|빼기\s|곱하기\s|나누기\s|은\s|는\s))/gu,m=>sinoNumber(m));
 value=value.replace(/((?:대|에서|더하기|빼기|곱하기|나누기|은|는)\s+)(-?\d+(?:\.\d+)?)(?![\w\d.])/gu,(_,word,n)=>word+sinoNumber(n));
 return value;
}
