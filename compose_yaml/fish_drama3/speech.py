"""Bounded Drama3 speech with observable retries; never remove internal silence."""
import array
import base64
import io
import math
import re
import time
import wave

import httpx

REFERENCE_TEXT = "안녕하세요. 오늘은 여러분과 함께 새로운 이야기를 시작하려고 합니다. 편안한 마음으로 끝까지 들어 주세요."
TAGGED_TEXT_LIMIT = 160
TAG_PATTERN = re.compile(r"\[[^\[\]]*\]|</?[A-Za-z][A-Za-z0-9_-]*>|<\|speaker:[0-9]+\|>")


def spoken_text(text: str) -> str:
    """Validate direction boundaries and return only words meant to be spoken."""
    stack = []
    for match in TAG_PATTERN.finditer(text):
        tag = match.group()
        if tag.startswith("["):
            if not tag[1:-1].strip():
                raise ValueError("비어 있는 음성 지시문입니다")
        elif tag.startswith("<|speaker:"):
            if tag != "<|speaker:0|>":
                raise ValueError("현재 참조 음성은 speaker:0만 지원합니다")
        elif tag.startswith("</"):
            if not stack or stack.pop() != tag[2:-1]:
                raise ValueError("음성 태그의 여닫는 순서가 맞지 않습니다")
        else:
            stack.append(tag[1:-1])
    if stack:
        raise ValueError("닫히지 않은 음성 태그가 있습니다")
    plain = TAG_PATTERN.sub("", text)
    if any(mark in plain for mark in ("[", "]", "<|")) or re.search(r"</?[A-Za-z]", plain):
        raise ValueError("음성 태그의 형식이나 여닫는 괄호를 확인해 주세요")
    return re.sub(r"\s+", " ", plain).strip()


def has_tags(text: str) -> bool:
    return TAG_PATTERN.search(text) is not None


def directed_text(text: str, instructions: str = "") -> str:
    instructions = instructions.strip()
    if any(mark in instructions for mark in '[]<>'):
        raise ValueError('전체 말투 지시는 괄호 없이 적고, 구간 태그는 대사에 넣어 주세요')
    value = f'[{instructions}] {text}' if instructions else text
    chunks(value)
    return value


def chunks(text: str, limit: int = 90, continuous: bool = False) -> list[str]:
    plain = spoken_text(text)
    if continuous:
        if not plain or len(plain) > TAGGED_TEXT_LIMIT:
            raise ValueError('연속 생성할 대사는 1~160자여야 합니다')
        return [text.strip()]
    if has_tags(text):
        if not plain:
            raise ValueError("음성 태그와 함께 읽을 대사를 입력해 주세요")
        if len(plain) > TAGGED_TEXT_LIMIT:
            raise ValueError(f"태그가 있는 대사는 한 장면에 {TAGGED_TEXT_LIMIT}자까지 입력해 주세요")
        # Directions and their return to normal share one model context. Never
        # split a scope, copy a direction to another chunk, or splice its audio.
        return [text.strip()]
    sentences = re.split(r"(?<=[.!?。！？])\s+|\n+", text.strip())
    result = []
    for sentence in sentences:
        while len(sentence) > limit:
            boundary = sentence.rfind(" ", 0, limit + 1)
            if boundary < limit // 2:
                boundary = limit
            result.append(sentence[:boundary].strip())
            sentence = sentence[boundary:].strip()
        if sentence:
            result.append(sentence)
    return result


def analyze(pcm: bytes, rate: int) -> dict:
    samples = array.array("h", pcm)
    step = max(1, rate // 50)
    frame_rms = []
    for offset in range(0, len(samples), step):
        frame = samples[offset:offset + step]
        frame_rms.append(math.sqrt(sum(value * value for value in frame) / max(1, len(frame))))
    # Keep quiet whispered tails above the noise floor. A fixed -60 dBFS gate
    # previously mistook them for padding; this is activity, not a speech VAD.
    threshold = min(32.768, max(1.0, max(frame_rms, default=0) * .003))
    active = [value > threshold for value in frame_rms]
    voiced = [index for index, value in enumerate(active) if value]
    first, last = (voiced[0], voiced[-1]) if voiced else (0, 0)
    run = longest = 0
    for value in active[first:last + 1]:
        run = 0 if value else run + 1
        longest = max(longest, run)
    duration = len(samples) / rate
    return {"duration_seconds": duration, "longest_internal_silence": longest / 50,
            "voiced_seconds": len(voiced) / 50, "hit_cap": duration >= 41.7,
            "trailing_silence_seconds": (len(samples) - min(len(samples), (last + 1) * step)) / rate if voiced else duration,
            "activity_threshold_dbfs": 20 * math.log10(threshold / 32768),
            "trim_start_sample": max(0, first * step - int(rate * .12)),
            "trim_end_sample": min(len(samples), (last + 1) * step + int(rate * .12))}


def rejection_reasons(metrics: dict, text: str) -> list[str]:
    reasons = []
    if metrics["hit_cap"]:
        reasons.append("output_cap")
    if metrics["voiced_seconds"] < .3:
        reasons.append("empty_audio")
    if not has_tags(text) and metrics["longest_internal_silence"] > 1.8:
        reasons.append("internal_silence")
    without_scopes = TAG_PATTERN.sub(lambda match: match.group() if match.group().startswith("[") else "", text)
    intentional_tail = re.search(r"\[(?:short |long )?pause\]\s*$", without_scopes, re.IGNORECASE)
    if metrics["trailing_silence_seconds"] > 2 and not intentional_tail:
        reasons.append("trailing_silence")
    return reasons


def wav_bytes(pcm: bytes, rate: int = 44100) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(pcm)
    return output.getvalue()


def synthesize(text: str, endpoint: str, reference: str, folder=None, prefix="speech", pause=.18, reference_text=REFERENCE_TEXT, continuous=False) -> dict:
    records, pieces, segments = [], [], []
    rate = 44100
    for index, sentence in enumerate(chunks(text, continuous=continuous)):
        accepted = False
        for attempt in range(3):
            payload = {"input": sentence, "references": [{"audio_path": reference, "text": reference_text}],
                       "response_format": "pcm", "stream": True, "temperature": 1.0, "top_p": .9,
                       "top_k": 2147483647, "max_new_tokens": 900}
            started = time.monotonic()
            with httpx.Client(timeout=180) as client:
                response = client.post(endpoint.rstrip("/") + "/v1/audio/speech", json=payload)
                response.raise_for_status()
            rate = int(response.headers.get("x-sample-rate", response.headers.get("x-audio-sample-rate", "44100")))
            if rate != 44100:
                raise ValueError("Unexpected Drama3 sample rate")
            pcm = response.content
            if len(pcm) % 2:
                raise ValueError("Truncated PCM response")
            metrics = analyze(pcm, rate)
            reasons = rejection_reasons(metrics, sentence)
            accepted = not reasons
            record = {"sentence": sentence, "attempt": attempt + 1, "accepted": accepted,
                      "rejection_reasons": reasons,
                      "generation_seconds": time.monotonic() - started, **metrics}
            records.append(record)
            if folder is not None:
                (folder / f"{prefix}-{index}-{attempt}.wav").write_bytes(wav_bytes(pcm, rate))
            if accepted:
                piece = pcm if has_tags(sentence) else pcm[metrics["trim_start_sample"] * 2:metrics["trim_end_sample"] * 2]
                start = sum(len(item) / 2 / rate for item in pieces) + len(pieces) * pause
                segments.append({"text": spoken_text(sentence), "start": start, "end": start + len(piece) / 2 / rate})
                pieces.append(piece)
                break
        if not accepted:
            raise ValueError(f"Fish 음성 검사 실패: {sentence[:60]} (시도 3회, {', '.join(reasons)})")
    if not pieces:
        raise ValueError("Empty speech input")
    pcm = (b"\x00\x00" * int(rate * pause)).join(pieces)
    return {"audio": base64.b64encode(wav_bytes(pcm, rate)).decode(), "duration": len(pcm) / 2 / rate,
            "records": records, "segments": segments, "sample_rate": rate, "text": text,
            "spoken_text": spoken_text(text), "native_tags": has_tags(text), "continuous": continuous}
