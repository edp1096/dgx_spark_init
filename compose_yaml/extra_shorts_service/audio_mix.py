"""Measured narration pacing and original score synthesis; music never counts as speech."""
import base64
import math
from pathlib import Path
import subprocess
import wave

import numpy as np

from speech import analyze, has_tags


def prepare_voice(item, folder, index, voice):
    raw = folder/f"voice-{index}-raw.wav"
    raw.write_bytes(base64.b64decode(item.pop("audio"), validate=True))
    target = folder/f"voice-{index}.wav"
    tagged = item.get("native_tags", False) or has_tags(item.get("text", ""))
    if tagged and voice['speed'] == 1:
        # Native delivery includes intentional gain and silence changes.
        target.write_bytes(raw.read_bytes())
    else:
        filters = f"atempo={voice['speed']}"
        if not tagged:
            filters += ",loudnorm=I=-18:TP=-2:LRA=7,afade=t=in:d=0.015"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(raw), "-af", filters,
                        "-ac", "1", "-ar", "44100", "-c:a", "pcm_s16le", str(target)], check=True)
    with wave.open(str(target)) as audio:
        duration = audio.getnframes()/audio.getframerate()
    ratio = duration/item["duration"]
    item["raw_duration"] = item["duration"]
    item["duration"] = duration
    item["segments"] = [{**segment, "start": segment["start"]*ratio, "end": segment["end"]*ratio} for segment in item["segments"]]
    item["speed"] = voice["speed"]
    item["native_gain_preserved"] = tagged
    return item


def retime_voice(item, folder, index, voice):
    """Reuse Fish output when a voice-only revision needs bounded pace fitting."""
    raw = folder / f"voice-{index}-raw.wav"
    if not raw.exists():
        raise ValueError("음성 속도를 맞출 원본 합성 파일이 없습니다")
    original = dict(item)
    ratio = item['raw_duration'] / item['duration']
    original['segments'] = [{**s, 'start':s['start']*ratio, 'end':s['end']*ratio} for s in item['segments']]
    original['duration'] = item['raw_duration']
    original['audio'] = base64.b64encode(raw.read_bytes()).decode()
    return prepare_voice(original, folder, index, voice)


def narration_fit(scenes, items, target, voice):
    speech = sum(item["duration"] for item in items)
    silent = sum(scene.get("duration_seconds") or 1.5 for scene in scenes if not scene["narration"].strip())
    spoken_scenes = sum(bool(scene["narration"].strip()) for scene in scenes)
    intended = target or speech + silent + spoken_scenes*.5
    coverage = speech/max(.01, intended)
    budget = max(1, intended-silent-spoken_scenes*.55)
    if speech + silent + spoken_scenes*.24 > intended:
        status = "too_long"
    elif coverage < voice["min_coverage"] or intended-speech-silent > spoken_scenes*voice["max_gap_seconds"]:
        status = "too_short"
    else:
        status = "fit"
    return {"status": status, "narration_seconds": speech, "duration_seconds": intended,
            "narration_coverage": coverage, "desired_narration_seconds": budget,
            "rewrite_length_multiplier": budget/max(.01, speech), "explicit_silent_seconds": silent}


def timeline(scenes, items, target, voice):
    fit = narration_fit(scenes, items, target, voice)
    if fit["status"] != "fit":
        raise ValueError(f"내레이션 길이 검증 실패: {fit['status']} (음성 {fit['narration_seconds']:.1f}초 / 영상 {fit['duration_seconds']:.1f}초)")
    target_frames = round(fit["duration_seconds"]*24)
    frames = [math.ceil((item["duration"]+.24)*24) if scene["narration"].strip() else round((scene.get("duration_seconds") or 1.5)*24) for scene, item in zip(scenes, items)]
    voiced = [index for index, scene in enumerate(scenes) if scene["narration"].strip()]
    extra, remainder = divmod(target_frames-sum(frames), len(voiced))
    if extra < 0:
        raise ValueError("내레이션이 최종 프레임 예산을 초과합니다")
    for ordinal, index in enumerate(voiced):
        frames[index] += extra + (ordinal < remainder)
    cursor, result = 0, []
    for index, (scene, item, count) in enumerate(zip(scenes, items, frames)):
        result.append({**scene, "start": cursor/24, "duration": count/24,
                       "voice_start": cursor/24+.12, "voice_duration": item["duration"], "segments": item["segments"]})
        cursor += count
    return result, cursor/24, fit


def narration_track(folder, timeline, duration):
    rate = 44100
    data = bytearray(round(duration*rate)*2)
    for index, scene in enumerate(timeline):
        if scene["voice_duration"] <= 0:
            continue
        with wave.open(str(folder/f"voice-{index}.wav")) as audio:
            if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, rate):
                raise ValueError("음성 WAV 규격이 잘못됐습니다")
            pcm = audio.readframes(audio.getnframes())
        start = round(scene["voice_start"]*rate)*2
        if start+len(pcm) > len(data):
            raise ValueError("음성이 영상 끝에서 잘립니다")
        data[start:start+len(pcm)] = pcm
    path = folder/"narration.wav"
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1); audio.setsampwidth(2); audio.setframerate(rate); audio.writeframes(data)
    metrics = analyze(data, rate)
    metrics.pop("hit_cap", None)
    metrics["energy_active_seconds"] = metrics.pop("voiced_seconds")
    return metrics


def synth_music(path, duration, music):
    rate = 22050
    samples = np.zeros(round(duration*rate), dtype=np.float64)
    notes = music["notes"]
    if not notes and music["kind"] == "synth_pad":
        notes = [{"start":0,"duration":duration,"midi":key,"velocity":.22} for key in (48,55,60,64)]
    if not notes:
        raise ValueError("건반 BGM에는 실제 음표·타이밍이 필요합니다")
    for note in notes:
        start = round(note["start"]*rate)
        length = min(round(note["duration"]*rate), len(samples)-start)
        if length <= 0:
            continue
        t = np.arange(length)/rate
        frequency = 440*2**((note["midi"]-69)/12)
        if music["kind"] == "synth_keys":
            # Original additive keyboard timbre, explicitly not a recorded piano.
            signal = sum((1/h**1.6)*np.sin(2*np.pi*frequency*h*t)*np.exp(-t*(.6+h*.17)) for h in range(1,7))
            envelope = np.minimum(1,t/.008)*np.minimum(1,np.maximum(0,(length/rate-t)/.12))
        else:
            signal = np.sin(2*np.pi*frequency*t)+.18*np.sin(4*np.pi*frequency*t)
            envelope = np.minimum(1,t/.8)*np.minimum(1,np.maximum(0,(length/rate-t)/1.2))
        samples[start:start+length] += signal*envelope*note["velocity"]
    peak = np.max(np.abs(samples))
    if peak > 0:
        samples *= .65/peak
    fade = np.minimum(1,np.arange(len(samples))/rate/1.5)*np.minimum(1,(len(samples)-np.arange(len(samples)))/rate/2)
    pcm = (np.clip(samples*fade,-1,1)*32767).astype('<i2').tobytes()
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1);audio.setsampwidth(2);audio.setframerate(rate);audio.writeframes(pcm)


def mux(folder, duration, music):
    if music["kind"] == "none":
        args = ["-i", "render.mp4", "-i", "narration.wav", "-map", "0:v", "-map", "1:a"]
    else:
        if music["kind"] == "generated":
            if not (folder/"bgm.wav").is_file():
                raise ValueError("생성한 BGM 파일이 없습니다")
        else:
            synth_music(folder/"bgm.wav", duration, music)
        gain = music["gain"]
        filters = f"[1:a]asplit=2[voice][side];[2:a]volume={gain}[music];[music][side]sidechaincompress=threshold=0.02:ratio=8:attack=20:release=250[bed];[voice][bed]amix=inputs=2:normalize=0[a]"
        args = ["-i", "render.mp4", "-i", "narration.wav", "-i", "bgm.wav", "-filter_complex", filters, "-map", "0:v", "-map", "[a]"]
    subprocess.run(["ffmpeg", "-y", "-v", "error", *args, "-c:v", "copy", "-c:a", "aac", "-t", str(duration), "-movflags", "+faststart", "final.mp4"], cwd=folder, check=True)
