"""Plan -> resources -> measured audio -> model-authored design -> preview -> render."""
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from PIL import Image, ImageDraw

import assets
import audio_mix
import authoring
import music
import store
import speech as speech_input
import voice_catalog
from spec import Plan, Artwork, Review


def save(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)); temp.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read(path, fallback=None):
    return json.loads(path.read_text()) if path.exists() else fallback


def parent_resources(project):
    if not project.source_job_id:
        return None, {}, []
    parent = store.get(project.source_job_id)
    if parent["session_id"] != project.session_id:
        raise ValueError("다른 대화의 제작물을 재사용할 수 없습니다")
    root = store.ROOT/project.source_job_id
    plan = read(root/"storyboard.json", {})
    timing = read(root/"timing.json", {}).get("scenes", [])
    catalog = [{"index": index, "title": scene["title"], "kind": scene.get("kind"),
                "image_prompt": scene.get("image_prompt", ""), "narration": scene.get("narration", ""),
                "has_asset": bool(scene.get("asset") and (root/scene["asset"]).is_file())} for index, scene in enumerate(timing)]
    return root, plan, catalog


def call_author(bridge, project, phase, data, images=None):
    return bridge("author", {"session_id": project.session_id, "phase": phase, "data": data, "images": images or []})


def narration_character_target(project, parent_plan):
    preset = project.voice_preset or parent_plan.get('voice', {}).get('preset') or 'sohee'
    sample = voice_catalog.get_voice(preset, project.session_id, project.voice_revision, allow_history=bool(project.voice_revision))
    duration = sample.get('preview_duration_seconds', 0)
    rate = len(sample.get('preview_text', '')) / duration if duration else 6
    return min(1800, round((project.duration_seconds or 60) * .94 * rate))


def plan_project(project, folder, parent_plan, catalog, bridge):
    existing = read(folder/"storyboard.json")
    if existing and existing.get("version") in (2, 3):
        result = Plan.model_validate(existing).model_dump()
        if result['voice'].get('reference_revision') is None:
            reference = voice_catalog.get_voice(result['voice']['preset'], project.session_id)
            result['voice']['reference_revision'] = reference['revision']
            save(folder/'storyboard.json', result)
        return result
    if project.plan:
        result = Plan.model_validate(project.plan).model_dump()
        check_redesign_brief(project, result, parent_plan)
    else:
        feedback = ""
        for attempt in range(3):
            raw = call_author(bridge, project, "plan", {
                "brief": project.brief, "title": project.title, "duration_seconds": project.duration_seconds or 60,
                "target_narration_characters": narration_character_target(project, parent_plan),
                "bgm_requested": project.bgm, "design_mode":project.design_mode, "source_plan": parent_plan, "reusable_scenes": catalog,
                "available_assets": assets.list_assets(project.session_id), "required_asset_ids": project.asset_ids,
                "schema": Plan.model_json_schema(), "feedback": feedback,
                "voice_catalog": voice_catalog.public_catalog(project.session_id), "selected_voice": project.voice_preset,
            })
            save(folder/f"plan-attempt-{attempt}.json", raw)
            try:
                result = Plan.model_validate(raw).model_dump()
                check_redesign_brief(project, result, parent_plan)
                if project.scenes:
                    result["scenes"] = [scene.model_dump() for scene in project.scenes]
                break
            except ValueError as error:
                feedback = str(error)[:3000]
        else:
            raise ValueError("제작 계획 형식을 고치지 못했습니다: "+feedback)
    if project.voice_preset:
        result["voice"]["preset"] = project.voice_preset
        result["voice"]["reference_revision"] = project.voice_revision
    reference = voice_catalog.get_voice(result['voice']['preset'], project.session_id, result['voice'].get('reference_revision'))
    result['voice']['reference_revision'] = reference['revision']
    result["asset_ids"] = list(dict.fromkeys(project.asset_ids+result["asset_ids"]))
    save(folder/"storyboard.json", result)
    return result


def check_redesign_brief(project, result, parent_plan):
    if project.design_mode == "redesign" and parent_plan.get("design_brief") and " ".join(result["design_brief"].split()) == " ".join(parent_plan["design_brief"].split()):
        raise ValueError("Redesign returned the rejected design_brief unchanged. Rewrite the visual direction, including the headline treatment and removal of obsolete decoration instructions. Preserve media/audio separately from visual design.")


def select_assets(project, plan, folder, bridge):
    cached = read(folder/"selected-assets.json")
    if cached is not None:
        for record in cached:
            record["scene_asset"]=any(scene.get("visual") in ("lottie","svg") and scene.get("asset_id")==record["id"] for scene in plan["scenes"])
        save(folder/"selected-assets.json",cached)
        return cached
    selected = list(plan["asset_ids"])
    searches = []
    if not selected:
        for query in plan["lottie_queries"]:
            searches.append(assets.search(query))
        if searches:
            candidates = {item["source_url"] for result in searches for item in result["results"]}
            choice = call_author(bridge, project, "select_assets", {"brief":project.brief, "design_brief":plan["design_brief"], "results":searches})
            for url in choice.get("source_urls", [])[:3]:
                if url not in candidates:
                    raise ValueError("검색 결과에 없는 Lottie 주소를 선택했습니다")
                selected.append(assets.import_lottie(project.session_id, url)["id"])
            if not selected:
                plan["limitations"].append("요청한 Lottie 검색에서 적합한 무료 소재를 선택하지 못했습니다.")
    for scene in plan["scenes"]:
        if scene.get("asset_id"):
            selected.append(scene["asset_id"])
    records = []
    for asset_id in dict.fromkeys(selected):
        record, source = assets.get_asset(project.session_id, asset_id)
        target = folder/"assets"/asset_id
        target.mkdir(parents=True, exist_ok=True)
        for name in (record["filename"], "source.json", "license.html"):
            if (source/name).exists():
                shutil.copyfile(source/name, target/name)
        records.append({**record, "path":str((target/record["filename"]).relative_to(folder)),
                        "scene_asset": any(scene.get("visual") in ("lottie","svg") and scene.get("asset_id")==asset_id for scene in plan["scenes"])})
    save(folder/"selected-assets.json", records)
    return records


def speech_key(text, voice, session_id=None):
    reference = voice_catalog.get_voice(voice["preset"], session_id, voice.get('reference_revision'), allow_history=bool(voice.get('reference_revision')))
    payload = {"text":text,"preset":voice["preset"],"speed":voice["speed"],"pause":voice["sentence_pause"],"version":2 if voice_catalog.LEGACY_REFERENCES.get(voice["preset"]) == reference["reference_sha256"] else 3}
    if speech_input.has_tags(text) or voice.get('instructions'):
        payload["version"] = 4
    if voice.get('instructions'):
        payload['instructions'] = voice['instructions']
    if reference.get('kind') == 'custom':
        payload['version'] = 5
        payload['reference_revision'] = reference['revision']
        payload['reference_text_sha256'] = reference['reference_text_sha256']
    if payload["version"] != 2:
        payload["reference_sha256"] = reference["reference_sha256"]
    return fingerprint(payload)


def make_speech(project, plan, folder, parent_root, parent_plan, bridge, stage):
    fit_attempts = []
    for attempt in range(3):
        stage(f"Fish 음성 합성·길이 확인 {attempt+1}/3", 18)
        voice, scenes = plan["voice"], plan["scenes"]
        current = read(folder/"speech.json", {"items":[]})
        old = read(parent_root/"speech.json", {"items":[]}) if parent_root else {"items":[]}
        items, missing = [], []
        for index, scene in enumerate(scenes):
            text = scene["narration"].strip(); key = speech_key(text, voice, project.session_id)
            item = None
            if not text:
                item = {"duration":0,"segments":[],"records":[],"fingerprint":key,"text":""}
            elif index < len(current["items"]) and current["items"][index].get("fingerprint")==key and (folder/f"voice-{index}.wav").exists():
                item = current["items"][index]
            elif parent_root:
                for old_index, old_item in enumerate(old["items"]):
                    if old_item.get("fingerprint")==key and (parent_root/f"voice-{old_index}.wav").exists():
                        shutil.copyfile(parent_root/f"voice-{old_index}.wav", folder/f"voice-{index}.wav")
                        item = dict(old_item); break
            items.append(item)
            if item is None:
                missing.append(index)
        if missing:
            raw = bridge("speech", {"session_id":project.session_id,"job_id":folder.name,"texts":[speech_input.directed_text(scenes[index]["narration"], voice.get('instructions','')) for index in missing],
                                    "voice":{"preset":voice["preset"],"reference_revision":voice.get('reference_revision'),"sentence_pause":voice["sentence_pause"]*voice["speed"]}})
            reference = voice_catalog.get_voice(voice["preset"], project.session_id, voice.get('reference_revision'), allow_history=bool(voice.get('reference_revision')))
            if raw.get("voice_preset") != voice["preset"] or raw.get("reference_sha256") != reference["reference_sha256"]:
                raise ValueError("요청한 음성과 실제 Fish 참조 음성이 일치하지 않습니다")
            if len(raw.get("items", [])) != len(missing):
                raise ValueError("음성 합성 결과 개수가 다릅니다")
            for index,item in zip(missing,raw["items"]):
                prepared = audio_mix.prepare_voice(item,folder,index,voice)
                prepared["fingerprint"] = speech_key(scenes[index]["narration"],voice,project.session_id)
                items[index] = prepared
        reference = voice_catalog.get_voice(voice["preset"], project.session_id, voice.get('reference_revision'), allow_history=bool(voice.get('reference_revision')))
        speech = {"items":items,"voice":voice,"reference_source":reference.get('reference_source', '음성 참조'),"reference_sha256":reference['reference_sha256'],"reference_revision":reference['revision']}
        save(folder/"speech.json", speech)
        fit = audio_mix.narration_fit(scenes, items, project.duration_seconds, voice)
        fit_attempts.append(fit); save(folder/"duration-fit.json",fit_attempts)
        if fit["status"]=="fit":
            return speech
        if project.voice_only and attempt == 0:
            old_speed = voice['speed']
            speed = max(.8, min(1.2, old_speed * fit['narration_seconds'] / max(1, fit['desired_narration_seconds'])))
            pause = voice['sentence_pause'] * old_speed / speed
            if abs(speed-old_speed) > .001 and .08 <= pause <= .8:
                voice['speed'], voice['sentence_pause'] = speed, pause
                for index, item in enumerate(items):
                    if not scenes[index]['narration'].strip():
                        continue
                    items[index] = audio_mix.retime_voice(item, folder, index, voice)
                    items[index]['fingerprint'] = speech_key(scenes[index]['narration'], voice, project.session_id)
                speech.update(items=items, voice=voice, pace_fitted=True)
                save(folder/'storyboard.json', plan)
                save(folder/'speech.json', speech)
                fit = audio_mix.narration_fit(scenes, items, project.duration_seconds, voice)
                fit_attempts.append(fit); save(folder/'duration-fit.json', fit_attempts)
                if fit['status'] == 'fit':
                    return speech
        if plan["verbatim_narration"] or any(speech_input.has_tags(scene["narration"]) for scene in scenes):
            raise ValueError(f"지정한 원문 음성이 {fit['narration_seconds']:.1f}초여서 요청 길이에 맞지 않습니다. 원문이나 영상 길이를 조정해야 합니다.")
        if attempt==2:
            raise ValueError(f"내레이션 길이를 맞추지 못했습니다: {fit}. 긴 빈 구간으로 채우지 않았습니다.")
        stage("실측 음성 길이에 맞춰 로컬 모델이 대본 수정",20)
        rewritten = call_author(bridge,project,"rewrite",{"brief":project.brief,"scenes":scenes,"measured":fit,"desired_narration_seconds":fit["desired_narration_seconds"],"rewrite_length_multiplier":fit["rewrite_length_multiplier"],"target_characters_per_scene":[round(len(scene["narration"])*fit["rewrite_length_multiplier"]) for scene in scenes],"verbatim_narration":False})
        save(folder/f"rewrite-attempt-{attempt}.json",rewritten)
        narrations = rewritten.get("narrations",[])
        if len(narrations)!=len(scenes) or any(not isinstance(text,str) for text in narrations):
            raise ValueError("수정 대본의 장면 수가 다릅니다")
        if narrations == [scene["narration"] for scene in scenes]:
            raise ValueError("길이 수정 요청에도 대본이 바뀌지 않았습니다. 수정 대본을 다시 작성해야 합니다.")
        save(folder/f"storyboard-before-fit-{attempt}.json",plan)
        for scene,text in zip(scenes,narrations):
            scene["narration"]=text
        plan = Plan.model_validate(plan).model_dump()
        save(folder/"storyboard.json",plan)
    raise AssertionError("speech fitting did not finish")


def verify_speech(project, plan, speech, folder, bridge):
    cache = read(folder/"coverage.json", [])
    checks=[]
    for index,(scene,item) in enumerate(zip(plan["scenes"],speech["items"])):
        if not scene["narration"].strip():
            checks.append({"passed":True,"intentional_title_scene":True});continue
        key=item["fingerprint"]
        if index<len(cache) and cache[index].get("fingerprint")==key:
            check=cache[index]
        else:
            check=bridge("transcribe",{"session_id":project.session_id,"audio":base64.b64encode((folder/f"voice-{index}.wav").read_bytes()).decode(),"expected":speech_input.spoken_text(scene["narration"])})
            check["fingerprint"]=key
        checks.append(check)
    save(folder/"coverage.json",checks)
    if any(not check["passed"] for check in checks):
        for index,check in enumerate(checks):
            if not check["passed"]:
                speech["items"][index]["fingerprint"]="failed_coverage"
        save(folder/"speech.json",speech)
        raise ValueError("음성 전사에서 대본 누락·불일치를 발견했습니다. 실패한 장면의 음성을 다시 합성해야 합니다.")
    return checks


def make_visuals(project, plan, timeline, folder, parent_root, records, bridge, stage):
    parent_timing = read(parent_root/"timing.json",{}).get("scenes",[]) if parent_root else []
    indexed = {record["id"]:record for record in records}
    for index,scene in enumerate(timeline):
        stage(f"장면 소재 준비 {index+1}/{len(timeline)}",40+25*index/len(timeline))
        cache=folder/f"asset-{index}.json"
        key=fingerprint({k:scene.get(k) for k in ("visual","image_prompt","attachment_id","asset_id","reuse_scene","manim","points","duration")})
        result=read(cache)
        if result and result.get("fingerprint")==key and (not result.get("asset") or (folder/result["asset"]).exists()):
            scene.update(kind=result["kind"],asset=result["asset"])
            if result.get("loop_asset"):
                scene["asset"]=result["loop_asset"];scene["repeated_clip"]=True
            continue
        visual=scene["visual"]
        if visual in ("lottie","svg"):
            record=indexed.get(scene["asset_id"])
            if not record or record["kind"]!=visual:
                raise ValueError("선택한 벡터 소재가 이 장면 형식과 맞지 않습니다")
            result={"kind":"lottie" if visual=="lottie" else "image","asset":record["path"],"source":record}
        elif visual=="typography":
            result={"kind":"typography","asset":""}
        elif scene.get("reuse_scene") is not None and parent_root:
            old_index=scene["reuse_scene"]
            if old_index>=len(parent_timing):raise ValueError("재사용할 장면이 없습니다")
            old=parent_timing[old_index]
            source=(parent_root/old.get("asset","")).resolve()
            expected="image" if visual=="image" else "video"
            if old.get("kind")!=expected or not source.is_file() or not source.is_relative_to(parent_root.resolve()):
                raise ValueError("원본 장면과 요청한 소재 형식이 다릅니다")
            target=f"asset-{index}"+source.suffix
            shutil.copyfile(source,folder/target)
            result={"kind":expected,"asset":target,"reused_from":{"job_id":project.source_job_id,"scene":old_index}}
        elif visual=="diagram":
            data_path=folder/f"scene-{index}.json";save(data_path,scene)
            filename=f"scene-{index}.mp4"
            subprocess.run(["manim","--renderer=cairo","--fps","24","-r","720,720","--media_dir",str(folder/"manim"),"-o",filename,str(Path(__file__).with_name("manim_scene.py")),"StoryScene"],cwd=folder,env={**os.environ,"SHORTS_SCENE_JSON":str(data_path)},check=True,timeout=180)
            outputs=list((folder/"manim/videos").rglob(filename))
            if len(outputs)!=1:raise ValueError("Manim 결과 파일이 없거나 중복됐습니다")
            shutil.copyfile(outputs[0],folder/filename)
            result={"kind":"video","asset":filename,"generator":"manim"}
        elif visual in ("image","video"):
            if not scene["image_prompt"] and not scene["attachment_id"]:
                raise ValueError("장면 이미지·영상의 생성 지시 또는 첨부 ID가 필요합니다")
            raw=bridge("asset",{"session_id":project.session_id,"scene":scene})
            target=f"asset-{index}."+("mp4" if raw["kind"]=="video" else "png")
            (folder/target).write_bytes(base64.b64decode(raw.pop("data"),validate=True))
            result={**raw,"asset":target}
        else:raise ValueError("지원하지 않는 장면 종류")
        result["fingerprint"]=key;save(cache,result)
        scene.update(kind=result["kind"],asset=result["asset"])
        if scene["kind"]=="video" and visual=="video":
            target=f"loop-{index}.mp4"
            subprocess.run(["ffmpeg","-y","-v","error","-stream_loop","-1","-i",scene["asset"],"-t",str(scene["duration"]),"-an","-c:v","libx264","-threads","2","-r","24",target],cwd=folder,check=True,timeout=180)
            scene["asset"]=target;scene["repeated_clip"]=True
            result["loop_asset"]=target;save(cache,result)
    return timeline


def previews(folder, audit):
    frames=audit["previews"]
    if not frames:raise ValueError("미리보기 프레임이 없습니다")
    thumb_w,thumb_h=240,427
    sheet=Image.new("RGB",(min(3,len(frames))*thumb_w,((len(frames)+2)//3)*(thumb_h+28)),"#121212")
    draw=ImageDraw.Draw(sheet)
    for index,frame in enumerate(frames):
        source=Image.open(folder/frame["file"]).convert("RGB");source.thumbnail((thumb_w,thumb_h))
        x,y=(index%3)*thumb_w,(index//3)*(thumb_h+28)
        sheet.paste(source,(x,y));draw.text((x+8,y+thumb_h+5),f"{frame['time']:.2f}s",fill="white")
    sheet.save(folder/"preview-sheet.jpg",quality=85)
    images=[]
    for frame in frames[::max(1,len(frames)//3)][:4]:
        image=Image.open(folder/frame["file"]).convert("RGB");image.thumbnail((360,640))
        import io
        data=io.BytesIO();image.save(data,format="JPEG",quality=82)
        images.append("data:image/jpeg;base64,"+base64.b64encode(data.getvalue()).decode())
    return images


def design_media_sheet(folder, timeline):
    selected = timeline[::max(1,len(timeline)//3)][:3]
    sheet = Image.new("RGB",(360*len(selected),640),"#202020")
    found = False
    for index,scene in enumerate(selected):
        path = folder/scene.get("asset", "missing")
        if path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp") or not path.is_file():
            continue
        picture = Image.open(path).convert("RGB"); picture.thumbnail((360,640))
        sheet.paste(picture,(index*360,0)); found=True
    if not found:
        return []
    import io
    data=io.BytesIO();sheet.save(data,format="JPEG",quality=80)
    return ["data:image/jpeg;base64,"+base64.b64encode(data.getvalue()).decode()]


def reused_visual_review(project, plan, folder, parent_root, authored, audit):
    """Reuse approved visuals only when their content/code are identical.

    Voice changes still run a fresh browser audit of the new caption/media
    timing. Do not re-author or re-judge the user's unchanged visual choices.
    """
    if not project.voice_only or not parent_root or not audit.get('passed'):
        return None
    prior_quality = read(parent_root/'quality.json', {})
    prior_review = prior_quality.get('verification', {}).get('visual_review', {})
    if not prior_quality.get('passed') or not prior_review.get('passed'):
        return None
    if authored.get('template_sha256') != prior_quality.get('applied', {}).get('template', {}).get('template_sha256'):
        return None
    if read(folder/'artwork.json') != read(parent_root/'artwork.json'):
        return None
    previous = read(parent_root/'storyboard.json', {})
    def visual_content(value):
        result = {key:value.get(key) for key in ['title','design_brief','captions','headline','music','asset_ids']}
        result['scenes'] = [{k:v for k,v in scene.items() if k != 'reuse_scene'} for scene in value.get('scenes', [])]
        return result
    if visual_content(plan) != visual_content(previous):
        return None
    return {'passed': True, 'issues': [],
            'summary': 'Original approved artwork, scene inputs and text are unchanged. Fresh browser layout/media/caption timing checks passed for this voice revision.',
            'reused_from': project.source_job_id,
            'scope': 'Unchanged source visual review reused; new output timing audited in the browser.'}


def design_project(project, plan, timeline, duration, records, folder, parent_root, bridge, stage):
    spec=authoring.specification(plan,timeline,duration,records)
    artwork=read(folder/"artwork.json")
    prior=read(parent_root/"artwork.json") if parent_root else None
    if project.template_id and artwork is None and not project.voice_only:
        prior=authoring.get_template(project.session_id,project.template_id)["artwork"]
    if project.voice_only:
        if prior is None: raise ValueError("음성만 바꿀 원본 디자인이 없습니다")
        artwork=prior
    elif project.design_mode == "preserve" and artwork is None:
        artwork=prior
    guide=read(Path(__file__).with_name("design_reference.json"))
    media_images=design_media_sheet(folder,timeline)
    feedback=read(folder/"design-feedback.json",[])
    previous_audit=read(folder/"audit.json")
    if previous_audit and not previous_audit.get("passed"):
        feedback=previous_audit.get("issues",[])[:20]
    for attempt in range(1 if project.voice_only else 3):
        save(folder/"design-feedback.json",feedback)
        stage(f"로컬 모델 템플릿 작성·검사 {attempt+1}/3",68)
        if feedback:
            rules=spec["caption_rules"]
            feedback=[*feedback[:12],{"code":"repair_guidance","message":f"Keep the entire caption text rectangle inside x={720*rules['safe_x']:.0f}..{720*(1-rules['safe_x']):.0f}, y={1280*rules['safe_top']:.0f}..{1280*rules['safe_bottom']:.0f}. For wrapping, set an explicit .caption width that fits the safe span; max-width alone with left:50% shrinks an absolutely positioned box to 360px. Use the supplied .line elements without extra wrapping, at least {rules['min_font_px']}px. Caption fades must start at each caption's supplied start and finish within .15 sec. Do not duplicate renderer-owned IDs or declare spec/tl."}]
            if spec.get("headline_rules"):
                rules=spec["headline_rules"]
                feedback.append({"code":"headline_repair_guidance","message":f"Entire #project-title box including padding must fit x={720*rules['safe_left']:.0f}..{720*rules['safe_right']:.0f}, y={1280*rules['safe_top']:.0f}..{1280*rules['safe_bottom']:.0f}. Maximum width {720*(rules['safe_right']-rules['safe_left']):.0f}px; text minimum {rules['min_font_px']}px. Use the .headline-line spans and respect the requested opening duration."})
        if artwork is None or (feedback and not project.voice_only):
            artwork=call_author(bridge,project,"design",{"brief":project.brief,"contract":authoring.CONTRACT,"spec":spec,"schema":Artwork.model_json_schema(),"design_mode":project.design_mode,"reference_guide":guide,"media_contact_sheet":"Actual scene media, in order; use subject placement when composing text.","previous_artwork":artwork or (prior if project.design_mode == "preserve" else None),"feedback":feedback},media_images)
        save(folder/f"artwork-attempt-{attempt}.json",artwork)
        try:
            authored=authoring.compile_composition(folder,spec,artwork)
            lint=subprocess.run(["hyperframes","lint",str(folder)],cwd=folder,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=60)
            (folder/"lint.txt").write_text(lint.stdout)
            if lint.returncode:
                raise ValueError("HyperFrames lint: "+lint.stdout[-6000:])
            (folder/"audit.json").unlink(missing_ok=True)
            subprocess.run(["node",str(Path(__file__).with_name("audit.mjs")),str(folder)],cwd=folder,check=False,timeout=90)
            audit=read(folder/"audit.json")
            if not audit or not audit["passed"]:
                feedback=(audit or {}).get("issues",[{"code":"audit_missing"}])[:20];continue
            images=previews(folder,audit)
            reused = reused_visual_review(project, plan, folder, parent_root, authored, audit)
            if reused is not None:
                stage("원본 디자인 검수 재사용·새 타이밍 검사 완료",72)
                save(folder/'visual-review.json', reused)
                save(folder/'design-feedback.json', [])
                return authored,audit,reused
            stage("로컬 비전 모델이 실제 미리보기 검사",72)
            review=call_author(bridge,project,"review",{"brief":project.brief,"design_brief":plan["design_brief"],"applied_caption_styles":[s["styles"] for s in audit["samples"] if s["styles"]][:4],"headline_rules":spec.get("headline_rules"),"measured_headline_styles":[s["headline"] for s in audit["samples"] if s.get("headline")][:4],"scene_kinds":[s["kind"] for s in timeline],"browser_verified_caption_constraints":spec["caption_rules"],"asset_visible_pixels":audit.get("asset_pixels",[]),"review_scope":"The browser has verified literal caption lengths, line counts, font sizes and bounds. Do not recount Korean characters from pixels or speculate about timing from still images. Imported Lottie assets are animated and can have blank frames; positive visible pixel measurements confirm use. Judge actual steady-state layout, contrast and clear visible problems; do not require an animated accent in every still."},images)
            review=Review.model_validate(review).model_dump()
            save(folder/"visual-review.json",review)
            if not review.get("passed"):
                feedback=[{"code":"visual_review","issues":review.get("issues",[])}];continue
            save(folder/"design-feedback.json",[])
            return authored,audit,review
        except (ValueError,subprocess.CalledProcessError) as error:
            feedback=[{"code":"composition_error","message":str(error)[:3000]}]
    save(folder/"design-feedback.json",feedback)
    raise ValueError("템플릿 검사에서 문제를 고치지 못했습니다: "+json.dumps(feedback,ensure_ascii=False))


def produce(job_id, bridge, render_frames):
    started=time.monotonic();job=store.get(job_id);project=store.Project.model_validate(job["project"])
    folder=store.ROOT/job_id;folder.mkdir(exist_ok=True)
    (folder/"error.txt").unlink(missing_ok=True)
    def stage(label,progress):
        store.update(job_id,"running",label,progress);print(label,flush=True)
    stage("요구사항·제작 계획 구성",5)
    parent_root,parent_plan,catalog=parent_resources(project)
    plan=plan_project(project,folder,parent_plan,catalog,bridge)
    stage("상용 사용 가능한 애니메이션 소재 준비",10)
    records=select_assets(project,plan,folder,bridge)
    save(folder/"storyboard.json",plan)
    speech=make_speech(project,plan,folder,parent_root,parent_plan,bridge,stage)
    plan=read(folder/"storyboard.json")
    stage("음성 전사·대본 누락 검사",35)
    coverage=verify_speech(project,plan,speech,folder,bridge)
    timeline,duration,fit=audio_mix.timeline(plan["scenes"],speech["items"],project.duration_seconds,plan["voice"])
    narration=audio_mix.narration_track(folder,timeline,duration)
    if narration["longest_internal_silence"]>plan["voice"]["max_gap_seconds"]+.25:
        raise ValueError(f"내레이션 전용 트랙에 긴 빈 구간이 있습니다: {narration['longest_internal_silence']:.2f}초")
    timeline=make_visuals(project,plan,timeline,folder,parent_root,records,bridge,stage)
    stage("배경음악 준비",60)
    music_record=music.prepare(project,plan,folder,parent_root,duration,bridge)
    save(folder/"timing.json",{"duration":duration,"scenes":timeline})
    authored,audit,review=design_project(project,plan,timeline,duration,records,folder,parent_root,bridge,stage)
    caption_styles=[row for sample in audit["samples"] for row in sample["styles"]]
    applied={"duration_seconds":duration,"scene_count":len(timeline),"scene_kinds":[scene["visual"] for scene in timeline],
             "template":authored,"template_id":project.template_id,"design_mode":project.design_mode,"headline":plan.get("headline"),"headline_styles":[s["headline"] for s in audit["samples"] if s.get("headline")][:8],"caption_styles":caption_styles[:8],
             "voice_preset":plan["voice"]["preset"],"voice_speed":plan["voice"]["speed"],"sentence_pause":plan["voice"]["sentence_pause"],
             "narration_seconds":fit["narration_seconds"],"narration_coverage":fit["narration_coverage"],"longest_narration_gap":narration["longest_internal_silence"],
             "music_kind":plan["music"]["kind"],"music_gain":plan["music"]["gain"],"music":music_record,
             "animation_assets":[{k:a.get(k) for k in ("id","name","kind","source_url","license","license_url","creator")} for a in records]}
    limitations=plan["limitations"]+read(folder/"artwork.json")["limitations"]
    verification={"browser_layout":audit["passed"],"animation_visible_pixels":audit.get("asset_pixels",[]),"visual_review":review,"speech_coverage":all(check["passed"] for check in coverage),"narration_track":narration,"duration_fit":fit,"caption_timing":"sentence timestamps with proportional phrase timing"}
    quality={"passed":True,"applied":applied,"verification":verification,"limitations":limitations,"speech":coverage,"elapsed_seconds":time.monotonic()-started}
    save(folder/"quality.json",quality)
    sources={"tts_model":"https://huggingface.co/watchpen/drama-3-int4-koen160","voice_reference":speech["reference_source"],"renderer":"https://github.com/heygen-com/hyperframes","template_author":"selected local model","animations":records,"content_sources":plan["sources"],"music":music_record}
    save(folder/"sources.json",sources)
    if project.preview_only:
        store.update(job_id,"preview_ready","미리보기 검사 완료",75);return
    stage("검사한 템플릿으로 영상 렌더링",75)
    render_frames(folder,job_id)
    audio_mix.mux(folder,duration,plan["music"])
    stage("완성 파일·오디오 트랙 검사",95)
    probe=json.loads(subprocess.check_output(["ffprobe","-v","error","-show_streams","-show_format","-of","json","final.mp4"],cwd=folder))
    video=next(stream for stream in probe["streams"] if stream["codec_type"]=="video")
    if (video["width"],video["height"])!=(720,1280) or abs(float(video["duration"])-duration)>.05 or not any(stream["codec_type"]=="audio" for stream in probe["streams"]):
        raise ValueError("완성본 해상도·길이·오디오 검증 실패")
    subprocess.run(["ffmpeg","-y","-v","error","-ss","1","-i","final.mp4","-frames:v","1","poster.png"],cwd=folder,check=True)
    quality.update(duration=probe["format"]["duration"],width=720,height=1280,fps=video["r_frame_rate"],elapsed_seconds=time.monotonic()-started)
    quality["verification"]["final_file"]={"duration":video["duration"],"frames":video.get("nb_frames"),"audio":True}
    save(folder/"quality.json",quality)
    store.update(job_id,"completed","완료",100)
