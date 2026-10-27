import json
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from spec import Scene
from voice_catalog import VoiceID, get_voice

ROOT = Path(os.environ.get("SHORTS_DATA_DIR", "/data"))


class Project(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str = Field(min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_-]+$")
    title: str = Field(min_length=1, max_length=100)
    brief: str = Field(default="", max_length=12000)
    scenes: list[Scene] = Field(default_factory=list, max_length=16)
    bgm: bool = False
    duration_seconds: int = Field(default=0, ge=0, le=120)
    source_job_id: str = Field(default="", pattern=r"^(?:[a-f0-9]{32})?$")
    template_id: str = Field(default="", pattern=r"^(?:[a-f0-9]{32})?$")
    design_mode: Literal["redesign", "preserve"] = "redesign"
    asset_ids: list[str] = Field(default_factory=list, max_length=12)
    preview_only: bool = False
    plan: dict | None = None
    voice_preset: VoiceID | None = None
    voice_revision: int | None = Field(default=None, ge=1)
    voice_only: bool = False

    @model_validator(mode="after")
    def bounded_narration(self):
        if sum(len(scene.narration) for scene in self.scenes) > 1800:
            raise ValueError("전체 내레이션은 1800자 이하여야 합니다")
        return self


def connect() -> sqlite3.Connection:
    ROOT.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(ROOT / "jobs.sqlite", timeout=30)
    connection.row_factory = sqlite3.Row
    return connection


def initialize() -> None:
    with connect() as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE IF NOT EXISTS voice_preferences (session_id TEXT PRIMARY KEY, voice_id TEXT NOT NULL)")
        connection.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, title TEXT NOT NULL, request TEXT NOT NULL, status TEXT NOT NULL, stage TEXT NOT NULL, progress REAL NOT NULL, error TEXT NOT NULL DEFAULT '', canceled INTEGER NOT NULL DEFAULT 0, created REAL NOT NULL, updated REAL NOT NULL)")
        connection.execute("UPDATE jobs SET status=CASE WHEN canceled=1 THEN 'canceled' ELSE 'queued' END, stage=CASE WHEN canceled=1 THEN '중지됨' ELSE '이어서 준비 중' END, updated=? WHERE status='running'", (time.time(),))


def create(project: Project) -> dict:
    if project.source_job_id and get(project.source_job_id)["session_id"] != project.session_id:
        raise ValueError("다른 대화의 제작물을 재사용할 수 없습니다")
    if project.voice_preset is None and not project.source_job_id and project.plan is None:
        project = project.model_copy(update={"voice_preset": selected_voice(project.session_id)})
    if project.plan is not None and project.voice_preset is None:
        planned = project.plan.get('voice', {})
        project = project.model_copy(update={'voice_preset':planned.get('preset', 'sohee'),'voice_revision':planned.get('reference_revision')})
    if project.voice_preset is not None:
        selected = get_voice(project.voice_preset, project.session_id, project.voice_revision)
        project = project.model_copy(update={"voice_revision": selected['revision']})
    job_id = uuid.uuid4().hex
    timestamp = time.time()
    with connect() as connection:
        connection.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?, ?,?,?,?)", (job_id, project.session_id, project.title, project.model_dump_json(), "queued", "대기 중", 0, "", 0, timestamp, timestamp))
    return get(job_id)


def get(job_id: str) -> dict:
    if len(job_id) != 32 or any(character not in "0123456789abcdef" for character in job_id):
        raise ValueError("Invalid project ID")
    with connect() as connection:
        row = connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if row is None:
        raise KeyError(job_id)
    result = dict(row)
    result["project"] = json.loads(result.pop("request"))
    result["output_ready"] = result["status"] == "completed" and (ROOT / job_id / "final.mp4").is_file()
    report = ROOT / job_id / "quality.json"
    if report.exists() and result["status"] in ("completed", "preview_ready"):
        value = json.loads(report.read_text())
        result["applied"] = value.get("applied", {})
        result["verification"] = value.get("verification", {})
        result["limitations"] = value.get("limitations", [])
    result["preview_ready"] = (ROOT / job_id / "preview.png").is_file()
    return result


def list_jobs(session_id: str) -> list[dict]:
    with connect() as connection:
        ids = [row[0] for row in connection.execute("SELECT id FROM jobs WHERE session_id=? ORDER BY created DESC LIMIT 50", (session_id,))]
    return [get(job_id) for job_id in ids]


def update(job_id: str, status: str, stage: str, progress: float, error: str = "") -> None:
    with connect() as connection:
        connection.execute("UPDATE jobs SET status=?,stage=?,progress=?,error=?,updated=? WHERE id=?", (status, stage, min(100, progress), error[:2000], time.time(), job_id))


def claim() -> str | None:
    with connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute("SELECT id FROM jobs WHERE status='queued' AND canceled=0 ORDER BY created LIMIT 1").fetchone()
        if row is None:
            return None
        connection.execute("UPDATE jobs SET status='running',stage='준비 중',updated=? WHERE id=?", (time.time(), row[0]))
    return row[0]


def action(job_id: str, operation: str) -> dict:
    current = get(job_id)
    with connect() as connection:
        if operation == "cancel" and current["status"] in ("queued", "running"):
            connection.execute("UPDATE jobs SET canceled=1,status=CASE WHEN status='queued' THEN 'canceled' ELSE status END,stage='중지 요청',updated=? WHERE id=?", (time.time(), job_id))
        elif operation == "retry" and current["status"] in ("failed", "canceled"):
            connection.execute("UPDATE jobs SET canceled=0,status='queued',stage='이어서 대기 중',error='',updated=? WHERE id=?", (time.time(), job_id))
        elif operation == "render" and current["status"] == "preview_ready":
            project = current["project"]
            project["preview_only"] = False
            connection.execute("UPDATE jobs SET request=?,status='queued',stage='검사한 장면으로 렌더링',error='',updated=? WHERE id=?", (json.dumps(project, ensure_ascii=False), time.time(), job_id))
        else:
            raise ValueError("This action is unavailable for the current state")
    return get(job_id)


def selected_voice(session_id):
    with connect() as connection:
        row = connection.execute("SELECT voice_id FROM voice_preferences WHERE session_id=?", (session_id,)).fetchone()
    return row[0] if row else None


def select_voice(session_id, voice_id):
    item = get_voice(voice_id, session_id)
    if item.get('scope') == 'temporary':
        raise ValueError('일회성 음성은 기본 선택으로 저장할 수 없습니다. 먼저 영구 등록해 주세요')
    with connect() as connection:
        connection.execute("INSERT INTO voice_preferences VALUES (?,?) ON CONFLICT(session_id) DO UPDATE SET voice_id=excluded.voice_id", (session_id, voice_id))
