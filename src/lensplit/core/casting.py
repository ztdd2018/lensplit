"""Actors, roles, availability windows, and per-shot casting assignments."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any
from uuid import uuid4


def _new_uuid() -> str:
    return str(uuid4())


def parse_day(value: str | None) -> date | None:
    """Parse an ISO date, returning None for empty or malformed input."""

    if not value:
        return None
    try:
        return date.fromisoformat(str(value).strip())
    except ValueError:
        return None


def _clean_day(value: str | None) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    day = parse_day(text)
    if day is None:
        raise ValueError(f"日期格式应为 YYYY-MM-DD：{text}")
    return day.isoformat()


def _short_day(value: str) -> str:
    return value[5:] if len(value) >= 10 else value


@dataclass
class Actor:
    """A performer with an optional inclusive availability window."""

    uuid: str
    name: str
    available_from: str = ""
    available_to: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.uuid:
            raise ValueError("演员 UUID 不能为空")
        self.name = str(self.name).strip()
        if not self.name:
            raise ValueError("演员姓名不能为空")
        self.available_from = _clean_day(self.available_from)
        self.available_to = _clean_day(self.available_to)
        if self.available_from and self.available_to and self.available_from > self.available_to:
            raise ValueError("可拍摄开始日期不能晚于结束日期")
        self.notes = str(self.notes or "")

    def set_name(self, name: str) -> None:
        """Rename the actor, rejecting blank names."""

        text = str(name).strip()
        if not text:
            raise ValueError("演员姓名不能为空")
        self.name = text

    def set_availability(self, start: str, end: str) -> None:
        """Replace the availability window, validating both bounds."""

        cleaned_start = _clean_day(start)
        cleaned_end = _clean_day(end)
        if cleaned_start and cleaned_end and cleaned_start > cleaned_end:
            raise ValueError("可拍摄开始日期不能晚于结束日期")
        self.available_from = cleaned_start
        self.available_to = cleaned_end

    def window_text(self) -> str:
        """Return a compact human-readable availability window."""

        if self.available_from and self.available_to:
            if self.available_from == self.available_to:
                return _short_day(self.available_from)
            return f"{_short_day(self.available_from)} ~ {_short_day(self.available_to)}"
        if self.available_from:
            return f"{_short_day(self.available_from)} 起"
        if self.available_to:
            return f"至 {_short_day(self.available_to)}"
        return "未填写档期"

    def is_available(self, day: str) -> bool:
        """Return True when the actor can work on an ISO date."""

        target = parse_day(day)
        if target is None:
            return True
        start = parse_day(self.available_from)
        end = parse_day(self.available_to)
        if start is not None and target < start:
            return False
        if end is not None and target > end:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "uuid": self.uuid,
            "name": self.name,
            "available_from": self.available_from,
            "available_to": self.available_to,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Actor:
        """Create an actor from serialized data."""

        return cls(**dict(data))


@dataclass
class Role:
    """A character played by exactly one (possibly unassigned) actor."""

    uuid: str
    name: str
    actor_uuid: str = ""
    persona: str = ""
    notes: str = ""
    shot_uuid: str = ""

    def __post_init__(self) -> None:
        if not self.uuid:
            raise ValueError("角色 UUID 不能为空")
        self.name = str(self.name).strip()
        if not self.name:
            raise ValueError("角色名称不能为空")
        self.actor_uuid = str(self.actor_uuid or "")
        self.persona = str(self.persona or "")
        self.notes = str(self.notes or "")
        self.shot_uuid = str(self.shot_uuid or "")

    @property
    def is_slot(self) -> bool:
        """True when the role was auto-created for one shot's person count."""

        return bool(self.shot_uuid)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "uuid": self.uuid,
            "name": self.name,
            "actor_uuid": self.actor_uuid,
            "persona": self.persona,
            "notes": self.notes,
            "shot_uuid": self.shot_uuid,
        }

    def set_name(self, name: str) -> None:
        """Rename the role, rejecting blank names."""

        text = str(name).strip()
        if not text:
            raise ValueError("角色名称不能为空")
        self.name = text

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Role:
        """Create a role from serialized data."""

        return cls(**dict(data))


@dataclass
class ShotCasting:
    """Casting facts for one logical shot."""

    group_uuid: str
    character_count: int = 0
    role_ids: list[str] = field(default_factory=list)
    shoot_date: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        if not self.group_uuid:
            raise ValueError("镜头 UUID 不能为空")
        self.character_count = int(self.character_count)
        if self.character_count < 0:
            raise ValueError("画面人数不能为负数")
        self.role_ids = list(dict.fromkeys(self.role_ids))
        self.shoot_date = _clean_day(self.shoot_date)
        self.note = str(self.note or "")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "group_uuid": self.group_uuid,
            "character_count": self.character_count,
            "role_ids": list(self.role_ids),
            "shoot_date": self.shoot_date,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ShotCasting:
        """Create a shot casting record from serialized data."""

        return cls(**dict(data))


@dataclass(frozen=True)
class CastingConflict:
    """A scheduling or casting problem worth surfacing to the user."""

    level: str
    message: str
    group_uuid: str = ""
    actor_uuid: str = ""

    @property
    def is_error(self) -> bool:
        return self.level == "error"


@dataclass(frozen=True)
class AvailabilityIssue:
    """A shoot day that is missing from, or outside, an actor's availability."""

    group_uuid: str
    actor_uuid: str
    day: str
    window: str
    kind: str
    actor_name: str = ""

    @property
    def message(self) -> str:
        """Explain the problem in call-sheet language."""

        name = self.actor_name or "该演员"
        if self.kind == "empty":
            return f"{name} 的档期未填写，镜头拍摄日是 {self.day}"
        return f"{name} 的档期 {self.window} 不含拍摄日 {self.day}"


ACTOR_CSV_HEADER = "姓名,可拍摄开始,可拍摄结束,备注"
ACTOR_CSV_RULES = (
    "CSV 第一行必须是表头：姓名,可拍摄开始,可拍摄结束,备注\n"
    "－ 姓名必填，同名演员视为更新（补填档期 / 备注）\n"
    "－ 日期格式 YYYY-MM-DD，可留空表示不限\n"
    "－ 可拍摄开始 不能晚于 可拍摄结束\n"
    "－ 备注可写演员特点、造型要求等\n"
    "－ 文件编码 UTF-8（带不带 BOM 都可以）或 GBK"
)

_COLUMN_ALIASES = {
    "姓名": "name",
    "演员": "name",
    "演员姓名": "name",
    "name": "name",
    "可拍摄开始": "available_from",
    "开始": "available_from",
    "档期开始": "available_from",
    "start": "available_from",
    "可拍摄结束": "available_to",
    "结束": "available_to",
    "档期结束": "available_to",
    "end": "available_to",
    "备注": "notes",
    "特点": "notes",
    "演员特点": "notes",
    "notes": "notes",
}


@dataclass(frozen=True)
class ActorImportResult:
    """Outcome of one batch import of the actor library."""

    added: int
    updated: int
    errors: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.added + self.updated

    def summary(self) -> str:
        text = f"新增 {self.added} 人，更新 {self.updated} 人"
        if self.errors:
            text += f"，跳过 {len(self.errors)} 行"
        return text


def parse_actor_csv(text: str) -> tuple[list[dict[str, str]], list[str]]:
    """Parse an actor CSV into rows, returning ``(rows, errors)``.

    Raises ``ValueError`` when the header is missing required columns.
    """

    import csv
    import io

    reader = csv.reader(io.StringIO(text.lstrip("\ufeff")))
    rows = [row for row in reader if any(cell.strip() for cell in row)]
    if not rows:
        raise ValueError("文件是空的")
    header = [cell.strip() for cell in rows[0]]
    mapping: dict[int, str] = {}
    for index, title in enumerate(header):
        key = _COLUMN_ALIASES.get(title.lower() if title.isascii() else title)
        if key is None:
            key = _COLUMN_ALIASES.get(title)
        if key and key not in mapping.values():
            mapping[index] = key
    if "name" not in mapping.values():
        raise ValueError(f"表头缺少「姓名」列，应为：{ACTOR_CSV_HEADER}")
    parsed: list[dict[str, str]] = []
    errors: list[str] = []
    for line_no, row in enumerate(rows[1:], start=2):
        if not any(cell.strip() for cell in row):
            continue
        item = {"name": "", "available_from": "", "available_to": "", "notes": ""}
        for column, key in mapping.items():
            if column < len(row):
                item[key] = row[column].strip()
        if not item["name"]:
            errors.append(f"第 {line_no} 行：姓名为空，已跳过")
            continue
        parsed.append(item)
    return parsed, errors


class CastingData:
    """The casting sheet: actor library, role library, and per-shot assignments."""

    def __init__(
        self,
        actors: list[Actor] | None = None,
        roles: list[Role] | None = None,
        shots: dict[str, ShotCasting] | None = None,
    ) -> None:
        self.actors: list[Actor] = list(actors or [])
        self.roles: list[Role] = list(roles or [])
        self.shots: dict[str, ShotCasting] = dict(shots or {})
        self._validate()

    def _validate(self) -> None:
        actor_uuids = [actor.uuid for actor in self.actors]
        if len(set(actor_uuids)) != len(actor_uuids):
            raise ValueError("演员 UUID 必须唯一")
        role_uuids = [role.uuid for role in self.roles]
        if len(set(role_uuids)) != len(role_uuids):
            raise ValueError("角色 UUID 必须唯一")
        known_actors = set(actor_uuids)
        for role in self.roles:
            if role.actor_uuid and role.actor_uuid not in known_actors:
                raise ValueError(f"角色「{role.name}」引用了不存在的演员")
        known_roles = set(role_uuids)
        for shot in self.shots.values():
            missing = [role_id for role_id in shot.role_ids if role_id not in known_roles]
            if missing:
                raise ValueError("镜头引用了不存在的角色")

    # -- lookups --------------------------------------------------------
    def actor(self, actor_uuid: str) -> Actor | None:
        return next((actor for actor in self.actors if actor.uuid == actor_uuid), None)

    def role(self, role_uuid: str) -> Role | None:
        return next((role for role in self.roles if role.uuid == role_uuid), None)

    def shot(self, group_uuid: str) -> ShotCasting | None:
        return self.shots.get(group_uuid)

    def ensure_shot(self, group_uuid: str) -> ShotCasting:
        shot = self.shots.get(group_uuid)
        if shot is None:
            shot = ShotCasting(group_uuid)
            self.shots[group_uuid] = shot
        return shot

    def roles_for_shot(self, group_uuid: str) -> list[Role]:
        shot = self.shots.get(group_uuid)
        if shot is None:
            return []
        known = {role.uuid: role for role in self.roles}
        return [known[role_id] for role_id in shot.role_ids if role_id in known]

    def actors_for_shot(self, group_uuid: str) -> list[Actor]:
        seen: dict[str, Actor] = {}
        for role in self.roles_for_shot(group_uuid):
            actor = self.actor(role.actor_uuid)
            if actor is not None and actor.uuid not in seen:
                seen[actor.uuid] = actor
        return list(seen.values())

    # -- library edits --------------------------------------------------
    def add_actor(
        self, name: str, available_from: str = "", available_to: str = ""
    ) -> Actor:
        actor = Actor(_new_uuid(), name, available_from, available_to)
        self.actors.append(actor)
        return actor

    def add_role(self, name: str, actor_uuid: str = "") -> Role:
        if actor_uuid and self.actor(actor_uuid) is None:
            raise ValueError("所选演员不存在")
        role = Role(_new_uuid(), name, actor_uuid)
        self.roles.append(role)
        return role

    def assign_actor(self, role_uuid: str, actor_uuid: str) -> None:
        role = self.role(role_uuid)
        if role is None:
            raise ValueError("角色不存在")
        if actor_uuid and self.actor(actor_uuid) is None:
            raise ValueError("所选演员不存在")
        role.actor_uuid = actor_uuid

    def remove_actor(self, actor_uuid: str) -> None:
        self.actors = [actor for actor in self.actors if actor.uuid != actor_uuid]
        for role in self.roles:
            if role.actor_uuid == actor_uuid:
                role.actor_uuid = ""

    def remove_role(self, role_uuid: str) -> None:
        self.roles = [role for role in self.roles if role.uuid != role_uuid]
        for shot in self.shots.values():
            if role_uuid in shot.role_ids:
                shot.role_ids = [item for item in shot.role_ids if item != role_uuid]

    # -- per-shot edits -------------------------------------------------
    def set_character_count(self, group_uuid: str, count: int) -> None:
        value = int(count)
        if value < 0:
            raise ValueError("画面人数不能为负数")
        shot = self.ensure_shot(group_uuid)
        shot.character_count = value
        self._sync_slots(shot, value)

    def _sync_slots(self, shot: ShotCasting, count: int) -> None:
        """Keep the shot's auto-created role slots in step with its person count."""

        known = {role.uuid: role for role in self.roles}
        shot.role_ids = [role_id for role_id in shot.role_ids if role_id in known]
        slots = [
            known[role_id]
            for role_id in shot.role_ids
            if known[role_id].shot_uuid == shot.group_uuid
        ]
        if len(slots) > count:
            dropped = {role.uuid for role in slots[count:]}
            shot.role_ids = [
                role_id for role_id in shot.role_ids if role_id not in dropped
            ]
            self.roles = [role for role in self.roles if role.uuid not in dropped]
            slots = slots[:count]
        while len(slots) < count:
            slot = self._new_slot(shot.group_uuid, slots)
            slots.append(slot)
            shot.role_ids.append(slot.uuid)

    def _new_slot(self, group_uuid: str, existing: list[Role]) -> Role:
        names = {role.name for role in existing}
        index = len(existing) + 1
        while f"角色 {index}" in names:
            index += 1
        slot = Role(_new_uuid(), f"角色 {index}", shot_uuid=group_uuid)
        self.roles.append(slot)
        return slot

    def shot_slots(self, group_uuid: str) -> list[Role]:
        """Return this shot's roles in display order (one per person slot)."""

        return self.roles_for_shot(group_uuid)

    def rename_role(self, role_uuid: str, name: str) -> None:
        role = self.role(role_uuid)
        if role is None:
            raise ValueError("角色不存在")
        role.set_name(name)

    def import_actors(self, rows: list[dict[str, str]]) -> "ActorImportResult":
        """Add or update actors from parsed CSV rows, matching by name."""

        added = 0
        updated = 0
        errors: list[str] = []
        by_name = {actor.name: actor for actor in self.actors}
        for index, row in enumerate(rows, start=2):
            name = str(row.get("name", "")).strip()
            if not name:
                errors.append(f"第 {index} 行：姓名为空，已跳过")
                continue
            start = str(row.get("available_from", "")).strip()
            end = str(row.get("available_to", "")).strip()
            notes = str(row.get("notes", "")).strip()
            actor = by_name.get(name)
            try:
                if actor is None:
                    actor = Actor(_new_uuid(), name, start, end, notes)
                    self.actors.append(actor)
                    by_name[name] = actor
                    added += 1
                else:
                    if start or end:
                        actor.set_availability(start, end)
                    if notes:
                        actor.notes = notes
                    updated += 1
            except ValueError as error:
                errors.append(f"第 {index} 行（{name}）：{error}")
        return ActorImportResult(added, updated, errors)

    def set_shoot_date(self, group_uuid: str, value: str) -> None:
        self.ensure_shot(group_uuid).shoot_date = _clean_day(value)

    def set_roles(self, group_uuid: str, role_ids: list[str]) -> None:
        known = {role.uuid for role in self.roles}
        missing = [role_id for role_id in role_ids if role_id not in known]
        if missing:
            raise ValueError("所选角色不存在")
        self.ensure_shot(group_uuid).role_ids = list(dict.fromkeys(role_ids))

    def merge_shots(self, old_ids: list[str], new_id: str) -> None:
        """Carry assignments over when adjacent shots become one shot."""

        merged = ShotCasting(new_id)
        for group_id in old_ids:
            shot = self.shots.get(group_id)
            if shot is None:
                continue
            for role_id in shot.role_ids:
                if role_id not in merged.role_ids:
                    merged.role_ids.append(role_id)
            merged.character_count = max(merged.character_count, shot.character_count)
            if not merged.shoot_date and shot.shoot_date:
                merged.shoot_date = shot.shoot_date
        merged.character_count = max(merged.character_count, len(merged.role_ids))
        known = {role.uuid: role for role in self.roles}
        for role_id in merged.role_ids:
            role = known.get(role_id)
            if role is None:
                continue
            if role.shot_uuid in old_ids:
                # Slot roles belong to a shot, so the merged shot adopts them.
                role.shot_uuid = new_id
        for group_id in old_ids:
            self.shots.pop(group_id, None)
        self.shots[new_id] = merged

    def split_shot(self, old_id: str, new_ids: list[str]) -> None:
        """Copy assignments onto every restored piece of a split shot."""

        original = self.shots.pop(old_id, None)
        if original is None:
            return
        for group_id in new_ids:
            role_ids: list[str] = []
            for role_id in original.role_ids:
                role = self.role(role_id)
                if role is None:
                    continue
                if role.is_slot:
                    # Each piece gets its own slot so actors stay editable.
                    copy = Role(
                        _new_uuid(),
                        role.name,
                        role.actor_uuid,
                        role.persona,
                        role.notes,
                        group_id,
                    )
                    self.roles.append(copy)
                    role_ids.append(copy.uuid)
                else:
                    role_ids.append(role_id)
            self.shots[group_id] = ShotCasting(
                group_id,
                original.character_count,
                role_ids,
                original.shoot_date,
                original.note,
            )

    # -- availability reconciliation --------------------------------------
    def shots_for_actor(self, actor_uuid: str) -> list[tuple[str, str]]:
        """Return ``(group_uuid, shoot_date)`` pairs for one actor, by date."""

        role_ids = {role.uuid for role in self.roles if role.actor_uuid == actor_uuid}
        if not role_ids:
            return []
        found: list[tuple[str, str]] = []
        for group_uuid, shot in self.shots.items():
            if not shot.shoot_date:
                continue
            if role_ids.intersection(shot.role_ids):
                found.append((group_uuid, shot.shoot_date))
        return sorted(found, key=lambda item: (item[1], item[0]))

    def shoot_days(self, actor_uuid: str) -> list[str]:
        """Return the distinct shoot days already planned for one actor."""

        return sorted({day for _group, day in self.shots_for_actor(actor_uuid)})

    def check_group_schedule(self, group_uuid: str) -> list[AvailabilityIssue]:
        """Return availability problems for every actor planned in one shot."""

        shot = self.shots.get(group_uuid)
        if shot is None or not shot.shoot_date:
            return []
        issues: list[AvailabilityIssue] = []
        for actor in self.actors_for_shot(group_uuid):
            if not actor.available_from and not actor.available_to:
                kind = "empty"
            elif not actor.is_available(shot.shoot_date):
                kind = "outside"
            else:
                continue
            issues.append(
                AvailabilityIssue(
                    group_uuid,
                    actor.uuid,
                    shot.shoot_date,
                    actor.window_text(),
                    kind,
                    actor.name,
                )
            )
        return issues

    def check_actor_schedule(self, actor_uuid: str) -> list[AvailabilityIssue]:
        """Return availability problems for one actor across all shots."""

        actor = self.actor(actor_uuid)
        if actor is None:
            return []
        issues: list[AvailabilityIssue] = []
        for group_uuid, day in self.shots_for_actor(actor_uuid):
            if not actor.available_from and not actor.available_to:
                kind = "empty"
            elif not actor.is_available(day):
                kind = "outside"
            else:
                continue
            issues.append(
                AvailabilityIssue(
                    group_uuid, actor.uuid, day, actor.window_text(), kind, actor.name
                )
            )
        return issues

    def fill_missing_windows(self, group_uuid: str) -> list[str]:
        """Seed empty availability windows from the shot's planned date.

        Page 3 is the master schedule, so a missing entry adopts the date that
        pages 1 and 2 already planned. Returns the affected actor names.
        """

        shot = self.shots.get(group_uuid)
        if shot is None or not shot.shoot_date:
            return []
        updated: list[str] = []
        for actor in self.actors_for_shot(group_uuid):
            if actor.available_from or actor.available_to:
                continue
            actor.set_availability(shot.shoot_date, shot.shoot_date)
            updated.append(actor.name)
        return updated

    def extend_availability(self, actor_uuid: str, day: str) -> bool:
        """Grow an actor's window so that ``day`` becomes available."""

        actor = self.actor(actor_uuid)
        target = parse_day(day)
        if actor is None or target is None:
            return False
        if actor.is_available(day):
            return False
        start = actor.available_from or day
        end = actor.available_to or day
        if not actor.available_from or target < parse_day(start):
            start = day
        if not actor.available_to or target > parse_day(end):
            end = day
        actor.set_availability(start, end)
        return True

    def extend_for_issues(self, issues: list[AvailabilityIssue]) -> int:
        """Apply the "extend the window" decision to a batch of issues."""

        changed = 0
        seen: set[tuple[str, str]] = set()
        for issue in issues:
            key = (issue.actor_uuid, issue.day)
            if key in seen:
                continue
            seen.add(key)
            if self.extend_availability(issue.actor_uuid, issue.day):
                changed += 1
        return changed

    def day_within_window(self, actor_uuid: str, day: str) -> str:
        """Clamp a shoot day into the actor's window (used by "改为档期内")."""

        actor = self.actor(actor_uuid)
        target = parse_day(day)
        if actor is None or target is None:
            return day
        start = parse_day(actor.available_from)
        end = parse_day(actor.available_to)
        if start is not None and target < start:
            return actor.available_from
        if end is not None and target > end:
            return actor.available_to
        return day

    def clamp_for_issues(self, issues: list[AvailabilityIssue]) -> list[tuple[str, str]]:
        """Move each shot day into the actor's window; returns the new days."""

        moved: list[tuple[str, str]] = []
        for issue in issues:
            if issue.kind == "empty":
                continue
            new_day = self.day_within_window(issue.actor_uuid, issue.day)
            if new_day == issue.day:
                continue
            shot = self.ensure_shot(issue.group_uuid)
            if shot.shoot_date != new_day:
                shot.shoot_date = new_day
                moved.append((issue.group_uuid, new_day))
        return moved

    # -- conflict detection ---------------------------------------------
    def conflicts(self, labels: dict[str, str] | None = None) -> list[CastingConflict]:
        """Return hard clashes first, then schedule warnings."""

        names = labels or {}

        def label(group_uuid: str) -> str:
            return names.get(group_uuid) or f"镜头 {group_uuid[:8]}"

        issues: list[CastingConflict] = []
        roles_per_day: dict[tuple[str, str], set[str]] = {}
        shots_per_day: dict[tuple[str, str], list[str]] = {}

        for group_uuid, shot in self.shots.items():
            roles = self.roles_for_shot(group_uuid)
            if not roles and shot.character_count == 0 and not shot.shoot_date:
                continue
            by_actor: dict[str, list[Role]] = {}
            for role in roles:
                actor = self.actor(role.actor_uuid)
                if actor is None:
                    if role.is_slot:
                        # Auto slots start empty; page 1 nudges the user instead.
                        continue
                    issues.append(
                        CastingConflict(
                            "warning",
                            f"{label(group_uuid)}：角色「{role.name}」尚未分配演员",
                            group_uuid,
                            role.actor_uuid,
                        )
                    )
                    continue
                by_actor.setdefault(actor.uuid, []).append(role)
            for actor_uuid, actor_roles in by_actor.items():
                actor = self.actor(actor_uuid)
                if actor is None:
                    continue
                if len(actor_roles) > 1:
                    role_names = "、".join(role.name for role in actor_roles)
                    issues.append(
                        CastingConflict(
                            "error",
                            f"{label(group_uuid)}：同一镜头内演员「{actor.name}」被分配到 "
                            f"{len(actor_roles)} 个角色（{role_names}），请拆分镜头或换人",
                            group_uuid,
                            actor_uuid,
                        )
                    )
                if shot.shoot_date and not actor.is_available(shot.shoot_date):
                    issues.append(
                        CastingConflict(
                            "error",
                            f"{label(group_uuid)}：演员「{actor.name}」档期 "
                            f"{actor.window_text()} 不含拍摄日 {shot.shoot_date}",
                            group_uuid,
                            actor_uuid,
                        )
                    )
                if shot.shoot_date:
                    key = (actor_uuid, shot.shoot_date)
                    roles_per_day.setdefault(key, set()).update(
                        role.uuid for role in actor_roles
                    )
                    buckets = shots_per_day.setdefault(key, [])
                    if group_uuid not in buckets:
                        buckets.append(group_uuid)

        for (actor_uuid, day), role_ids in roles_per_day.items():
            if len(role_ids) < 2:
                continue
            actor = self.actor(actor_uuid)
            if actor is None:
                continue
            role_names = "、".join(
                role.name for role in self.roles if role.uuid in role_ids
            )
            shot_names = "、".join(label(group_id) for group_id in shots_per_day[(actor_uuid, day)])
            issues.append(
                CastingConflict(
                    "warning",
                    f"演员「{actor.name}」在 {day} 饰演 {len(role_ids)} 个不同角色"
                    f"（{role_names}），涉及 {shot_names}，请确认造型与排期",
                    "",
                    actor_uuid,
                )
            )
        return issues

    # -- serialization ---------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of the whole casting sheet."""

        return {
            "actors": [actor.to_dict() for actor in self.actors],
            "roles": [role.to_dict() for role in self.roles],
            "shots": [shot.to_dict() for shot in self.shots.values()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CastingData:
        """Rebuild a casting sheet from serialized data."""

        return cls(
            actors=[Actor.from_dict(item) for item in data.get("actors", [])],
            roles=[Role.from_dict(item) for item in data.get("roles", [])],
            shots={
                item["group_uuid"]: ShotCasting.from_dict(item)
                for item in data.get("shots", [])
            },
        )

    def snapshot(self) -> dict[str, Any]:
        """Return an immutable-by-convention snapshot for the undo history."""

        return self.to_dict()

    def restore(self, snapshot: dict[str, Any]) -> None:
        """Replace all casting state after validating the snapshot."""

        restored = CastingData.from_dict(snapshot)
        self.actors = restored.actors
        self.roles = restored.roles
        self.shots = restored.shots


__all__ = [
    "ACTOR_CSV_HEADER",
    "ACTOR_CSV_RULES",
    "Actor",
    "ActorImportResult",
    "AvailabilityIssue",
    "CastingConflict",
    "CastingData",
    "Role",
    "ShotCasting",
    "parse_actor_csv",
    "parse_day",
]
