"""Save and reopen a whole working session: shots, casting, transcripts, settings."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lensplit.core.casting import CastingData
from lensplit.core.model import ShotProject

PROJECT_SUFFIX = ".lensplit"
SESSION_VERSION = 1
DEFAULT_NUMBERING: dict[str, Any] = {
    "start": 1,
    "width": 3,
    "prefix": "",
    "suffix": "",
    "sequence": "first",
    "merge_mode": "combined",
}


@dataclass
class ProjectSession:
    """Everything the window needs to restore a working state."""

    video_path: str = ""
    project: ShotProject | None = None
    casting: CastingData = field(default_factory=CastingData)
    transcripts: dict[str, str] = field(default_factory=dict)
    analysis: dict[str, Any] = field(default_factory=dict)
    numbering: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the session."""

        return {
            "app": "lensplit",
            "version": SESSION_VERSION,
            "video_path": self.video_path,
            "project": self.project.snapshot() if self.project is not None else None,
            "casting": self.casting.to_dict(),
            "transcripts": dict(self.transcripts),
            "analysis": dict(self.analysis),
            "numbering": {**DEFAULT_NUMBERING, **dict(self.numbering)},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectSession:
        """Rebuild a session, dropping references to shots that no longer exist."""

        if not isinstance(data, dict) or data.get("app") != "lensplit":
            raise ValueError("这不是 Lensplit 工程文件")
        version = int(data.get("version") or 0)
        if version > SESSION_VERSION:
            raise ValueError("工程文件由更新版本的 Lensplit 创建，请升级后再打开")
        snapshot = data.get("project")
        project = ShotProject.from_snapshot(snapshot) if snapshot else None
        known = {group.uuid for group in project.groups} if project is not None else set()

        casting = CastingData.from_dict(data.get("casting") or {})
        casting.shots = {
            group_uuid: shot
            for group_uuid, shot in casting.shots.items()
            if group_uuid in known
        }
        kept = {
            role.uuid
            for role in casting.roles
            if not role.shot_uuid or role.shot_uuid in known
        }
        casting.roles = [role for role in casting.roles if role.uuid in kept]
        for shot in casting.shots.values():
            shot.role_ids = [role_id for role_id in shot.role_ids if role_id in kept]

        transcripts = {
            group_uuid: str(text)
            for group_uuid, text in (data.get("transcripts") or {}).items()
            if group_uuid in known
        }
        return cls(
            video_path=str(data.get("video_path") or ""),
            project=project,
            casting=casting,
            transcripts=transcripts,
            analysis=dict(data.get("analysis") or {}),
            numbering={**DEFAULT_NUMBERING, **dict(data.get("numbering") or {})},
        )

    def save(self, path: str | Path) -> Path:
        """Write the session as UTF-8 JSON and return the resolved path."""

        target = Path(path).expanduser()
        if target.suffix.lower() != PROJECT_SUFFIX:
            target = target.with_suffix(PROJECT_SUFFIX)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> ProjectSession:
        """Read a session file written by :meth:`save`."""

        source = Path(path).expanduser()
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except UnicodeDecodeError as error:
            raise ValueError("工程文件不是 UTF-8 编码，可能已损坏") from error
        except json.JSONDecodeError as error:
            raise ValueError(f"工程文件内容不完整：{error}") from error
        return cls.from_dict(data)


__all__ = ["DEFAULT_NUMBERING", "PROJECT_SUFFIX", "ProjectSession", "SESSION_VERSION"]
