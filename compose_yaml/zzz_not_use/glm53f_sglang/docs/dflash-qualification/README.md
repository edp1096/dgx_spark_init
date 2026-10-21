# DFlash2 검증 조건

Huihui NVFP4, TP2, 1M, FP8 target·draft KV, DFlash 5토큰, 동시 요청 1개. CPU 70%·GPU 2100MHz 제한.

이 디렉터리는 작업 메모리 비율 0.2의 단독 시험입니다. Talk 전체 세트의 최종 시험은 별도로 기록합니다.

벤치 도구는 tool-eval-bench `6be685f0e6b9e0df05ed024848cf7fe1eca48752`. SGLang이 스트리밍 `return_token_ids`를 거부할 때 기존 재시도가 실행되도록 HTTP 오류 본문을 스트림이 닫히기 전에 읽는 패치만 적용했습니다. 패치를 함께 보존합니다. 실제 출력 토큰 수와 생성 시간을 측정했으며, acceptance rate는 이 도구에서 제공되지 않았습니다.

128토큰 출력이며, filler만 깊이 0/2048/8192를 사용합니다. code·structured 행은 모두 깊이 0입니다. 1M 시험은 캐시를 비우고 별도 salt를 사용해 1,047,622토큰을 입력한 후 서로 떨어진 코드 5개를 JSON으로 회수했습니다.
