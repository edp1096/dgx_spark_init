"""The local director's editable production plan. No baked-in visual theme."""
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Captions(Strict):
    enabled: bool = True
    max_chars_per_line: int = Field(default=14, ge=6, le=32)
    max_lines: int = Field(default=2, ge=1, le=3)
    lead_seconds: float = Field(default=.15, ge=0, le=.4)
    min_font_px: int = Field(default=38, ge=24, le=80)
    safe_x: float = Field(default=.08, ge=0, le=.25)
    safe_top: float = Field(default=.15, ge=0, le=.5)
    safe_bottom: float = Field(default=.78, ge=.5, le=1)


from voice_catalog import VoiceID


class Voice(Strict):
    preset: VoiceID = "sohee"
    reference_revision: int | None = Field(default=None, ge=1)
    instructions: str = Field(default="", max_length=200)
    speed: float = Field(default=1, ge=.8, le=1.2)
    sentence_pause: float = Field(default=.25, ge=.08, le=.8)
    max_gap_seconds: float = Field(default=1.5, ge=.5, le=3)
    min_coverage: float = Field(default=.85, ge=.7, le=.98)
    delivery: str = Field(default="", max_length=400)


class Headline(Strict):
    mode: Literal["opening", "persistent", "none"] = "opening"
    text: str = Field(default="", max_length=100)
    duration_seconds: float = Field(default=3.5, ge=1, le=8)
    min_font_px: int = Field(default=64, ge=32, le=144)
    max_lines: int = Field(default=3, ge=1, le=4)
    safe_left: float = Field(default=.08, ge=0, le=.3)
    safe_right: float = Field(default=.86, ge=.7, le=1)
    safe_top: float = Field(default=.08, ge=0, le=.4)
    safe_bottom: float = Field(default=.8, ge=.5, le=1)


class Note(Strict):
    start: float = Field(ge=0, le=120)
    duration: float = Field(default=1, gt=0, le=12)
    midi: int = Field(ge=36, le=96)
    velocity: float = Field(default=.5, ge=.05, le=1)


class Music(Strict):
    kind: Literal["none", "generated", "synth_keys", "synth_pad"] = "none"
    gain: float = Field(default=.12, ge=.01, le=.4)
    notes: list[Note] = Field(default_factory=list, max_length=300)
    description: str = Field(default="", max_length=400)
    model: Literal["ace-turbo", "ace-xl"] = "ace-turbo"
    prompt: str = Field(default="", max_length=2000)
    seed: int = Field(default=42, ge=0, le=2147483647)
    bpm: int | None = Field(default=None, ge=30, le=300)

    @model_validator(mode="after")
    def generation_prompt(self):
        if self.kind == "generated" and len(self.prompt.strip()) < 5:
            raise ValueError("생성형 BGM에는 음악 프롬프트가 필요합니다")
        return self


class Scene(Strict):
    title: str = Field(min_length=1, max_length=80)
    narration: str = Field(default="", max_length=500)
    visual: Literal["diagram", "image", "video", "lottie", "svg", "typography"] = "image"
    image_prompt: str = Field(default="", max_length=2000)
    attachment_id: str = Field(default="", max_length=128)
    points: list[Annotated[str, Field(min_length=1, max_length=80)]] = Field(default_factory=list, max_length=12)
    emphasis: list[Annotated[str, Field(min_length=1, max_length=32)]] = Field(default_factory=list, max_length=8)
    reuse_scene: int | None = Field(default=None, ge=0, le=15)
    asset_id: str = Field(default="", max_length=32)
    duration_seconds: float = Field(default=0, ge=0, le=120)
    manim: dict = Field(default_factory=dict, description="Optional declarative Manim: background color, objects array (up to 50). Each object: type=text/circle/rect/bar/line/arrow/polygon; x,y (-4..4), width,height, color,fill,fill_opacity, text,font_size; start,duration seconds; animation=fade/write/draw/grow. line/arrow: from/to [x,y]; polygon: vertices [[x,y],...].")


class Plan(Strict):
    version: int = 3
    title: str = Field(min_length=1, max_length=100)
    design_brief: str = Field(min_length=1, max_length=6000)
    scenes: list[Scene] = Field(min_length=1, max_length=16)
    captions: Captions = Field(default_factory=Captions)
    headline: Headline = Field(default_factory=Headline)
    voice: Voice = Field(default_factory=Voice)
    music: Music = Field(default_factory=Music)
    asset_ids: list[str] = Field(default_factory=list, max_length=12)
    lottie_queries: list[Annotated[str, Field(max_length=60)]] = Field(default_factory=list, max_length=3)
    sources: list[dict] = Field(default_factory=list, max_length=20)
    limitations: list[str] = Field(default_factory=list, max_length=12)
    verbatim_narration: bool = False

    @model_validator(mode="after")
    def narration_budget(self):
        if sum(len(scene.narration) for scene in self.scenes) > 1800:
            raise ValueError("전체 내레이션은 1800자 이하여야 합니다")
        if not any(scene.narration.strip() for scene in self.scenes):
            raise ValueError("내레이션 대본이 필요합니다")
        if sum(not scene.narration.strip() for scene in self.scenes) > 2:
            raise ValueError("내레이션 없는 장면은 도입·마무리 두 개까지 지원합니다")
        return self


class Artwork(Strict):
    name: str = Field(default="Local model composition", max_length=120)
    css: str = Field(min_length=30, max_length=30000)
    javascript: str = Field(default="", max_length=20000)
    overlays_html: str = Field(default="", max_length=20000)
    highlights: dict[str, str] = Field(default_factory=dict)
    notes: str = Field(default="", max_length=3000)
    limitations: list[str] = Field(default_factory=list, max_length=12)


class Review(Strict):
    passed: bool = Field(strict=True)
    issues: list[str] = Field(default_factory=list, max_length=20)
    summary: str = Field(default="", max_length=3000)
