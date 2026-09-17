import test from 'node:test';
import assert from 'node:assert/strict';
import {speechTextFromMarkdown as speech,normalizeSpeechNotation} from './speech-normalizer.js';
import {createSpeechChunker} from './speech-text.js';

test('news score and duration are not arithmetic or ambiguous digits',()=>{
 assert.equal(speech('한국 여자축구 방글라데시 6-0 제압, 8강 확정'),'한국 여자축구 방글라데시 육 대 영 제압, 8강 확정.');
 assert.equal(speech('1년 11개월 만에 뒤집힌 결론'),'일 년 십일 개월 만에 뒤집힌 결론.');
 assert.equal(speech('6-0=6'),'육 빼기 영은 육.');
 assert.equal(speech('2 * 3 = 6'),'이 곱하기 삼은 육.');
 assert.equal(speech('2 < 3'),'이는 삼보다 작다.');
 assert.equal(speech('3 >= 2'),'삼은 이 이상이다.');
 assert.equal(normalizeSpeechNotation('모델 X-11, 번호 6-0'),'모델 X-11, 번호 6-0');
});
test('dates, signs, decimals, counters and ranges survive markdown cleaning',()=>{
 for(const [input,expected] of [
 ['9월 18일','구월 십팔 일.'],['2026-09-18','이천이십육 년 구월 십팔 일.'],
 ['6월과 10월','유월과 시월.'],['15~22℃','십오 도에서 이십이 도.'],['15-22℃','십오 도에서 이십이 도.'],
 ['-3℃','영하 삼 도.'],['기온 -3~-1℃','기온 영하 삼 도에서 영하 일 도.'],
 ['3.14미터','삼 점 일 사 미터.'],['5m/s','초속 오 미터.'],['20개와 11개월','스무 개와 십일 개월.'],
 ['비율 3:1','비율 삼 대 일.'],['오후 3:10 출발','오후 세 시 십 분 출발.'],['영상 3:10','영상 삼 분 십 초.']
 ])assert.equal(speech(input),expected,input);
});
test('Korean acronym reading does not rewrite English sentences or identifiers',()=>{
 assert.equal(speech('KBS·IAEA 뉴스'),'케이비에스·아이 에이 이 에이 뉴스.');
 assert.equal(speech('IAEA reports on KBS.'),'IAEA reports on KBS.');
 assert.equal(speech('IP 192.168.1.1, 버전 v1.2.3, 모델 GPT-4'),'IP 192.168.1.1, 버전 v1.2.3, 모델 GPT-4.');
});
test('incremental speech retains full notation before normalization',()=>{
 const c=createSpeechChunker();const result=[];
 for(const chunk of ['한국 여자축구 6-','0 제압.\n','1년 1','1개월 만에 결론.\n','기온 -','3℃.\n'])result.push(...c.push(chunk));result.push(...c.finish());
 assert.deepEqual(result,['한국 여자축구 육 대 영 제압.','일 년 십일 개월 만에 결론.','기온 영하 삼 도.']);
});
