"""Generate model-backed BGM once; preserve its provenance across revisions."""
import base64
import hashlib
import json
import math
import shutil
import subprocess
import wave

import numpy as np


def validate_music(path, duration):
    with wave.open(str(path)) as audio:
        if audio.getsampwidth() != 2:
            raise ValueError("음악 WAV는 PCM16이어야 합니다")
        rate, channels = audio.getframerate(), audio.getnchannels()
        data = np.frombuffer(audio.readframes(audio.getnframes()), dtype='<i2').astype(np.float32)
    data = data.reshape(-1, channels) / 32768
    actual = len(data) / rate
    if actual < duration - .25:
        raise ValueError(f"생성 음악 길이가 부족합니다: {actual:.2f}초 / {duration:.2f}초")
    data = data[:round(duration * rate)]
    rms = float(np.sqrt(np.mean(data ** 2)))
    threshold = max(.0015, rms * .025)
    active = [np.sqrt(np.mean(data[i:i+rate//10] ** 2)) > threshold for i in range(0, len(data), rate//10)]
    indices = [i for i, value in enumerate(active) if value]
    if not indices:
        raise ValueError("생성 음악에 유효한 오디오가 없습니다")
    run = longest = 0
    for value in active[indices[0]:indices[-1]+1]:
        run = 0 if value else run + 1
        longest = max(longest, run)
    quality = {"duration": actual, "active_fraction": sum(active)/len(active), "rms": rms,
               "leading_quiet_seconds": indices[0]/10, "trailing_quiet_seconds": (len(active)-1-indices[-1])/10,
               "longest_internal_quiet_seconds": longest/10}
    if quality["active_fraction"] < .92 or quality["trailing_quiet_seconds"] > 2.5 or quality["leading_quiet_seconds"] > 1.5 or longest > 25:
        raise ValueError("생성 음악이 중간 또는 끝에서 너무 일찍 비어 있습니다: " + json.dumps(quality))
    return quality


def prepare(project, plan, folder, parent, duration, bridge):
    music = plan["music"]
    if music["kind"] != "generated":
        return {"kind": music["kind"], "notes": music["notes"],
                "source": "None" if music["kind"] == "none" else "Original additive synthesis"}
    request = {key: music[key] for key in ("model", "prompt", "seed", "bpm")}
    request["duration_seconds"] = max(10, math.ceil(duration))
    fingerprint = hashlib.sha256(json.dumps({"request":request,"continuous_bgm_policy":1}, sort_keys=True).encode()).hexdigest()
    for source in [folder, parent]:
        if source is None or not (source/"music.json").is_file() or not (source/"bgm.wav").is_file():
            continue
        record = json.loads((source/"music.json").read_text())
        if record.get("fingerprint") == fingerprint:
            if source != folder:
                shutil.copy2(source/"bgm.wav", folder/"bgm.wav")
                shutil.copy2(source/"music.json", folder/"music.json")
            return record
    attempts = []
    for attempt in range(3):
        # Give the model room for its natural ending, then take the requested
        # continuous excerpt. Only generated audio is used; no padding/looping.
        generated_duration = min(120, math.ceil(request["duration_seconds"] * (1.25 + .25 * attempt)))
        candidate = {**request, "seed": (request["seed"] + attempt) % 2147483648,
                     "duration_seconds": generated_duration,
                     "prompt": request["prompt"][:1700] + f" Continuous background music throughout all {generated_duration} seconds, keep playing without silent breaks; do not end or fade early."}
        raw = folder/f"music-candidate-{attempt}.wav"
        sidecar = raw.with_suffix('.json')
        cached = json.loads(sidecar.read_text()) if sidecar.exists() else {}
        if cached.get("fingerprint") == fingerprint and cached.get("request") == candidate and raw.exists():
            metadata = cached["metadata"]
        else:
            result = bridge("music", {"session_id": project.session_id, "music": candidate})
            raw.write_bytes(base64.b64decode(result["audio"], validate=True))
            metadata = result["metadata"]
        entry = {"fingerprint":fingerprint,"request":candidate,"metadata":metadata}
        try:
            quality = validate_music(raw, duration)
            entry.update(accepted=True,quality=quality)
        except ValueError as error:
            entry.update(accepted=False,error=str(error))
        sidecar.write_text(json.dumps(entry,ensure_ascii=False,indent=2))
        attempts.append(entry)
        if entry["accepted"]:
            break
    else:
        raise ValueError("BGM 연속성 검사 3회 실패: " + attempts[-1]["error"])
    shutil.copy2(raw, folder/"music-original.wav")
    # Normalize and fade real generated audio; never loop or pad missing music.
    filters = f"loudnorm=I=-18:TP=-2:LRA=9,afade=t=in:d=0.4,afade=t=out:st={max(0,duration-1.5)}:d=1.5"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(raw), "-af", filters,
                    "-t", str(duration), "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le", str(folder/"bgm.wav")], check=True)
    record = {"kind": "generated", "source": "Model-generated instrumental music",
              "fingerprint": fingerprint, "request": candidate, "metadata": metadata,
              "used_duration_seconds": duration, "quality": quality, "attempts": attempts}
    (folder/"music.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
    return record
