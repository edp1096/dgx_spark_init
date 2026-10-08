## 마이크 설정

리버스프록시+사설인증서로도 안되니 사용하는 브라우저에서 허용하는 수 밖에 없다.

아래 둘 중 하나로 이동하여 서버 주소를 http(s)://아이피:포트 형식으로 추가하고 사용 설정 후 재시작.

* `edge://flags/#unsafely-treat-insecure-origin-as-secure`
* `chrome://flags/#unsafely-treat-insecure-origin-as-secure`


## vLLM 연결

* 관리자 설정 > 연결 > OpenAI API 예. http://192.168.100.60:8000/v1
* API Type: Chat Coimpletions (Responses로 하면 안됨)

## 웹검색 기능
* [searxng](searxng) 도커로 실행 후, openwebui 설정에서 웹검색 서비스로 선택
