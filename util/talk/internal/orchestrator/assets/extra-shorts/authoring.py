"""Compile model-authored CSS/HTML/GSAP around verified media and speech timing."""
import hashlib
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import uuid

from spec import Artwork, Captions
import store

CONTRACT = """You author the visual template, not a fixed-theme selector.
Canvas: 720x1280 at 24fps. The renderer provides these content elements:
#shorts (root), #project-title (.project-title), #visual-N (.visual, image/video/div),
#scene-title-N (.scene-title), #caption-N-J (.caption, children .line and optional em),
#decoration-N (.decoration containing an imported Lottie or SVG), #voice-N (audio).
Every element's data-start/data-duration is already verified. Never change/remove
media, audio, clip timing, captions or narration. CSS controls all layout, typography,
colors, shadows, borders, background and visibility of optional scene headings.
For v3, #project-title is an audience-facing headline with spec.headline_rules,
not a permanent file-name label. Its text and display interval are supplied.
Style its .headline-line children with a clear hierarchy; use opacity rather
than autoAlpha/visibility so the supplied display interval remains authoritative.
You may add original HTML/SVG decorations in overlays_html; use unique IDs and
data-start/data-duration/class=clip for timed overlays. {{project_title}} is escaped
and replaced with the title. External URLs, scripts, forms and iframes are forbidden.
javascript is a GSAP timeline BODY: const spec and paused const tl already exist.
NEVER declare spec or tl again. Do not include any setup/footer or gsap.timeline().
For example: for (const c of spec.captions) tl.fromTo("#"+c.id,
{opacity:0},{opacity:1,duration:Math.min(.15,c.duration/4)},c.start);
Use tl.fromTo/to/set, loops and Math. Use spec.scenes and spec.captions for timings.
Do not create a new timeline or use timers, autonomous animation, fetch, eval,
media playback, location, storage or runtime imports. The renderer registers tl.
Lottie players are already loaded from local files with autoplay=false, loop=true.
Available offline fonts: Noto Sans CJK KR, Noto Serif CJK KR, Noto Sans, Noto Serif.
Return JSON {name,css,javascript,overlays_html,highlights,notes,limitations}.
highlights maps caption IDs to one exact word/phrase from that caption, when wanted.
Use highlights={} to retain the supplied per-caption emphasis. Do not reuse highlight
IDs from previous artwork: caption segmentation may have changed.
CSS MUST explicitly lay out .visual and .caption; there is no default visual theme.
Respect caption safe area, minimum font size, maximum lines, and the user's style.
For full-bleed photography, position photos to cover the whole canvas, add readable
contrast layers without hiding the photo. Do not add branding, fake UI or unwanted
progress bars. Follow the requested visual direction; you may use any creative layout.
Keep animations within spec.duration. Prefer gentle movement for calm narration.
Do not claim applied features in notes unless they are present in your code.
"""


def split_caption(text, width, lines):
    words = text.split()
    rows, current = [], ""
    for word in words:
        while len(word) > width:
            if current:
                rows.append(current); current = ""
            rows.append(word[:width]); word = word[width:]
        candidate = (current + " " + word).strip()
        if len(candidate) > width:
            rows.append(current); current = word
        else:
            current = candidate
    if current:
        rows.append(current)
    return ["\n".join(rows[i:i+lines]) for i in range(0, len(rows), lines)]


def caption_timeline(timeline, settings):
    settings = Captions.model_validate(settings)
    if not settings.enabled:
        return []
    result = []
    for index, scene in enumerate(timeline):
        number = 0
        for segment in scene["segments"]:
            pieces = split_caption(segment["text"], settings.max_chars_per_line, settings.max_lines)
            weights = [max(1, len(re.sub(r"\s", "", text))) for text in pieces]
            total = sum(weights)
            start = scene["voice_start"] + segment["start"]
            span = segment["end"] - segment["start"]
            cursor = 0
            for text, weight in zip(pieces, weights):
                begin = max(scene["start"], start + span * cursor / total - settings.lead_seconds)
                cursor += weight
                end = start + span * cursor / total - (settings.lead_seconds if cursor < total else 0)
                result.append({"id": f"caption-{index}-{number}", "text": text, "start": begin,
                               "duration": max(.08, end-begin), "scene": index,
                               "emphasis": next((word for word in scene.get("emphasis", []) if word in text), "")})
                number += 1
    for first, second in zip(result, result[1:]):
        first["duration"] = min(first["duration"], max(.04, second["start"]-first["start"]))
    return result


def specification(plan, timeline, duration, assets):
    return {"width": 720, "height": 1280, "fps": 24, "duration": duration,
            "title": plan["title"], "design_brief": plan["design_brief"],
            "headline_rules": plan.get("headline") if plan.get("version",2)>=3 else None,
            "caption_rules": plan["captions"], "caption_safe_pixels": {"left":720*plan["captions"]["safe_x"],"right":720*(1-plan["captions"]["safe_x"]),"top":1280*plan["captions"]["safe_top"],"bottom":1280*plan["captions"]["safe_bottom"]}, "scenes": timeline,
            "captions": caption_timeline(timeline, plan["captions"]), "assets": assets}


def _local_reference(value):
    return not re.search(r"(?:^[a-z]+:|^//|^/|\\|(?:^|/)\.\.(?:/|$))", value.strip(), re.I)


class OverlayValidator(HTMLParser):
    tags = {"div", "span", "p", "h1", "h2", "h3", "h4", "b", "strong", "em", "small", "br", "svg", "path", "rect", "circle", "ellipse", "polygon", "polyline", "line", "g", "defs", "lineargradient", "radialgradient", "stop", "mask", "clippath", "text", "tspan", "filter", "fegaussianblur"}

    def handle_starttag(self, tag, attrs):
        if tag not in self.tags:
            raise ValueError(f"Overlay tag not supported: {tag}")
        for key, value in attrs:
            if key == "id" and re.fullmatch(r"(?:shorts|project-title|visual-\d+|scene-title-\d+|caption-\d+-\d+|decoration-\d+|voice-\d+)", value or ""):
                raise ValueError("Overlay duplicates a renderer-owned ID: "+value+". Style the existing element instead of creating it again.")
            if key.startswith("on") or key in ("src", "href", "xlink:href"):
                raise ValueError("Overlay external references and handlers are not supported")


def validate_artwork(value):
    artwork = Artwork.model_validate(value)
    if re.search(r"@import|expression\s*\(|-moz-binding", artwork.css, re.I):
        raise ValueError("CSS imports and expressions are not supported")
    for ref in re.findall(r"url\(\s*['\"]?([^)'\"]+)", artwork.css, re.I):
        embedded_image = re.match(r"^data:image/(?:png|jpeg|webp|svg\+xml)(?:;base64|;utf8)?,", ref, re.I)
        if not _local_reference(ref) and not ref.startswith("#") and not embedded_image:
            raise ValueError("CSS assets must be local")
    forbidden = r"\b(?:fetch|XMLHttpRequest|WebSocket|EventSource|eval|Function|import|require|setTimeout|setInterval|requestAnimationFrame|Worker)\b|(?:window|document)\s*\.\s*(?:location|cookie|write|open)|\b(?:localStorage|sessionStorage|navigator)\b|\.\s*(?:play|pause|currentTime|volume|src)\b|</script"
    if re.search(forbidden, artwork.javascript, re.I):
        raise ValueError("Template code may animate visuals but cannot access network, storage, timers or media playback")
    if re.search(r"\b(?:const|let|var)\s+(?:spec|tl)\s*=", artwork.javascript):
        raise ValueError("spec and tl are already declared. Remove their declarations; return only the GSAP body using the supplied spec and tl.")
    if re.search(r"\brepeat\s*:\s*-\s*1\b", artwork.javascript):
        raise ValueError("Infinite repeat:-1 exceeds the video duration. Use a finite repeat count whose final end is <= spec.duration.")
    if "__timelines" in artwork.javascript or re.search(r"gsap\.timeline", artwork.javascript):
        raise ValueError("Use the provided paused tl, not a new timeline")
    OverlayValidator().feed(artwork.overlays_html)
    return artwork.model_dump()


def _caption_html(caption, highlights):
    word = highlights.get(caption["id"], caption.get("emphasis", ""))
    if word and (word not in caption["text"] or len(word) > 32):
        raise ValueError(f"Caption highlight is not in the text: {caption['id']}, highlight={word!r}, actual text={caption['text']!r}. Remove this override or choose an exact substring.")
    escaped = html.escape(caption["text"])
    if word:
        escaped = escaped.replace(html.escape(word), "<em>"+html.escape(word)+"</em>", 1)
    return "".join('<span class="line">'+line+"</span>" for line in escaped.split("\n"))


def compile_composition(folder, spec, artwork):
    artwork = validate_artwork(artwork)
    highlight_errors = []
    for caption in spec["captions"]:
        try:
            _caption_html(caption, artwork["highlights"])
        except ValueError as error:
            highlight_errors.append(str(error))
    if highlight_errors:
        raise ValueError("\n".join(highlight_errors)+"\nUse highlights={} to retain all supplied emphasis words.")
    for source, name in [("gsap/dist/gsap.min.js", "gsap.js"), ("lottie-web/build/player/lottie_light.min.js", "lottie.js")]:
        shutil.copyfile(Path("/opt/shorts/node_modules")/source, folder/name)
    clips, animations = [], []
    for index, scene in enumerate(spec["scenes"]):
        attrs = f'id="visual-{index}" class="clip visual visual-{scene["kind"]}" data-start="{scene["start"]}" data-duration="{scene["duration"]}" data-track-index="10"'
        if scene["kind"] == "image":
            clips.append(f'<img {attrs} src="{html.escape(scene["asset"], quote=True)}">')
        elif scene["kind"] == "video":
            clips.append(f'<video {attrs} src="{html.escape(scene["asset"], quote=True)}" muted playsinline></video>')
        elif scene["kind"] == "lottie":
            clips.append(f'<div {attrs}></div>')
            animations.append({"id": f"visual-{index}", "path": scene["asset"]})
        else:
            clips.append(f'<div {attrs}></div>')
        clips.append(f'<h2 id="scene-title-{index}" class="clip scene-title" data-start="{scene["start"]}" data-duration="{scene["duration"]}" data-track-index="20">{html.escape(scene["title"])}</h2>')
        if scene["voice_duration"] > 0:
            clips.append(f'<audio id="voice-{index}" src="voice-{index}.wav" data-start="{scene["voice_start"]}" data-duration="{scene["voice_duration"]}" data-track-index="{100+index}" data-volume="1"></audio>')
    for caption in spec["captions"]:
        clips.append(f'<div id="{caption["id"]}" class="clip caption" data-start="{caption["start"]}" data-duration="{caption["duration"]}" data-track-index="200">{_caption_html(caption, artwork["highlights"])}</div>')
    for index, asset in enumerate(spec["assets"]):
        if asset.get("scene_asset"):
            continue
        if asset["kind"] == "lottie":
            clips.append(f'<div id="decoration-{index}" class="clip decoration" data-start="0" data-duration="{spec["duration"]}" data-track-index="30"></div>')
            animations.append({"id": f"decoration-{index}", "path": asset["path"]})
        elif asset["kind"] == "svg":
            clips.append(f'<img id="decoration-{index}" class="clip decoration" src="{asset["path"]}" data-start="0" data-duration="{spec["duration"]}" data-track-index="30">')
    # Only canvas and timing infrastructure live here; the model owns the theme.
    reset = "*{box-sizing:border-box}html,body{margin:0;width:720px;height:1280px;overflow:hidden}#shorts{position:relative;width:720px;height:1280px;overflow:hidden}.clip{visibility:hidden}.scene-title{display:none}#project-title{margin:0}.caption .line{display:block}.caption em{font-style:normal}"
    for family in ("Noto Sans CJK KR", "Noto Serif CJK KR", "Noto Sans", "Noto Serif"):
        reset += '@font-face{font-family:"'+family+'";src:local("'+family+'");font-weight:normal}@font-face{font-family:"'+family+'";src:local("'+family+' Bold");font-weight:bold}'
    csp = "default-src 'none'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; media-src 'self' data: blob:; font-src 'self' data:; connect-src 'self'; base-uri 'none'; form-action 'none'; object-src 'none'"
    overlays = artwork["overlays_html"].replace("{{project_title}}", html.escape(spec["title"]))
    headline = spec.get("headline_rules")
    title_duration = spec["duration"]
    title_text = html.escape(spec["title"])
    if headline:
        title_duration = 0 if headline["mode"] == "none" else min(spec["duration"],headline["duration_seconds"]) if headline["mode"] == "opening" else spec["duration"]
        title_text = ''.join('<span class="headline-line">'+html.escape(line)+'</span>' for line in (headline["text"] or spec["title"]).split('\n'))
        reset += '.headline-line{display:block}'
    document = '<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="'+csp+'"><meta name="viewport" content="width=720,height=1280"><title>'+html.escape(spec["title"])+'</title><script src="gsap.js"></script><script src="lottie.js"></script><style>'+reset+"\n"+artwork["css"]+'</style></head><body><div id="shorts" data-composition-id="shorts" data-start="0" data-duration="'+str(spec["duration"])+'" data-width="720" data-height="1280" data-fps="24"><h1 id="project-title" class="clip project-title" data-start="0" data-duration="'+str(title_duration)+'" data-track-index="210">'+title_text+"</h1>"+"\n".join(clips)+overlays+'</div><script src="artwork.js"></script></body></html>'
    js_spec = json.dumps(spec, ensure_ascii=False).replace("<", "\\u003c")
    js = "const spec="+js_spec+";const tl=gsap.timeline({paused:true});window.__shorts_spec=spec;window.__hfLottie=[];window.__lottieReady=[];\n"
    js += "for(const a of "+json.dumps(animations)+"){const p=lottie.loadAnimation({container:document.getElementById(a.id),renderer:'svg',autoplay:false,loop:true,path:a.path});window.__hfLottie.push(p);window.__lottieReady.push(new Promise((resolve,reject)=>{p.addEventListener('DOMLoaded',resolve);p.addEventListener('data_failed',()=>reject(new Error('Lottie load failed')));}));}\n"
    js += "window.__hf=window.__hf||{};window.__hf.buildReady=window.__hf.buildReady||{};window.__hf.buildReady.lottie=Promise.all(window.__lottieReady);\n"
    js += artwork["javascript"]+"\ntl.to({}, {duration:spec.duration},0);window.__timelines={shorts:tl};"
    document = document.replace('<script src="artwork.js"></script>', "<script>"+js+"</script>")
    (folder/"index.html").write_text(document)
    (folder/"artwork.js").write_text(js)
    (folder/"composition-spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2))
    (folder/"artwork.json").write_text(json.dumps(artwork, ensure_ascii=False, indent=2))
    return {"template_sha256": hashlib.sha256((artwork["css"]+artwork["javascript"]+artwork["overlays_html"]).encode()).hexdigest(), "author": "local_model", "name": artwork["name"]}


def template_root(session_id):
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", session_id):
        raise ValueError("잘못된 대화 ID")
    return store.ROOT / "_templates" / session_id


def save_template(session_id, name, artwork, source_job_id=""):
    artwork = validate_artwork(artwork)
    template_id = uuid.uuid4().hex
    root = template_root(session_id); root.mkdir(parents=True, exist_ok=True)
    record = {"id": template_id, "name": name[:120], "source_job_id": source_job_id, "artwork": artwork}
    (root/(template_id+".json")).write_text(json.dumps(record, ensure_ascii=False, indent=2))
    return {key: value for key, value in record.items() if key != "artwork"}


def get_template(session_id, template_id):
    if not re.fullmatch(r"[a-f0-9]{32}", template_id):
        raise ValueError("잘못된 템플릿 ID")
    path = template_root(session_id)/(template_id+".json")
    if not path.exists():
        raise ValueError("이 대화에 템플릿이 없습니다")
    return json.loads(path.read_text())


def list_templates(session_id):
    return [{key:value for key,value in json.loads(path.read_text()).items() if key!="artwork"} for path in template_root(session_id).glob("*.json")]
