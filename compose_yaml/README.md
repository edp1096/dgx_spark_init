# Compose 설정

`env.sample`이 있는 구성은 해당 디렉터리에서 `cp -n env.sample .env` 후 값을 수정한다. `.env`는 Git에서 제외하고 `env.sample`은 커밋한다. 토큰·비밀번호는 예제에 넣지 않는다.

`manage.sh`가 있는 모델은 스크립트가 `.env`를 자동 생성한다. 직접 Compose를 쓰는 구성은 같은 디렉터리의 `.env`를 자동으로 읽는다. 셸 스크립트를 직접 실행할 때는 `set -a; . ./.env; set +a`로 적용한다.

llama.cpp의 systemd 실행은 `env.sample`을 기존 이름인 `llama-server.env`로 복사한다. 별도 설정 변수가 없는 Compose는 `.env`가 필요 없다.
