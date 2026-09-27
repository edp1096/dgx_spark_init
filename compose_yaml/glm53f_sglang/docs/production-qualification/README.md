# Talk 전체 세트 검증

Talk 내장 레시피에서 GLM을 시작하고, 워커의 Magpie TTS·Extra Media·SSH·Collector를 켠 상태에서 시험했습니다. 1M 입력 처리 도중 실제 TTS 요청도 성공했습니다. 메모리 수치는 호스트 전체 카운터이며 모델 한 프로세스의 쓰기로 단정하지 않습니다.

기능·이미지 12개, 대화·캐시 26개, Talk SSE 대화가 통과했습니다. 비교용 벤치 도구 조건과 HTTP 호환 패치는 [단독 검증 기록](../dflash-qualification/)과 같습니다. 128토큰 출력이며 code·structured는 깊이 0, filler만 0/2048/8192입니다.

KV 저장 기능을 켠 구성이 아닙니다. 보고된 스왑은 Linux의 /swap.img에 발생한 실제 쓰기이며, 무스왑 구성을 보장하지 않습니다.
