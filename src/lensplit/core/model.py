"""Source-shot, logical-group, and transition state."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable
from uuid import uuid4


def _new_uuid() -> str:
    return str(uuid4())


@dataclass
class SourceShot:
    """An immutable-by-convention detected shot, using seconds for PTS bounds."""

    uuid: str
    ordinal: int
    start_pts: float
    end_pts: float
    start_frame: int | None = None
    end_frame: int | None = None
    thumbnail_path: str | None = None

    def __post_init__(self) -> None:
        if not self.uuid:
            raise ValueError("Source shot UUID cannot be empty")
        if self.ordinal < 1:
            raise ValueError("Source shot ordinals start at 1")
        self.start_pts = float(self.start_pts)
        self.end_pts = float(self.end_pts)
        if not math.isfinite(self.start_pts) or not math.isfinite(self.end_pts):
            raise ValueError("Shot PTS bounds must be finite")
        if self.end_pts <= self.start_pts:
            raise ValueError("Shot end PTS must be greater than its start PTS")
        if (
            self.start_frame is not None
            and self.end_frame is not None
            and self.end_frame <= self.start_frame
        ):
            raise ValueError("Shot end frame must be greater than its start frame")
        if self.thumbnail_path is not None:
            self.thumbnail_path = str(self.thumbnail_path)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "uuid": self.uuid,
            "ordinal": self.ordinal,
            "start_pts": self.start_pts,
            "end_pts": self.end_pts,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "thumbnail_path": self.thumbnail_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceShot:
        """Create a source shot from serialized data."""

        return cls(**dict(data))


@dataclass
class Transition:
    """A detected boundary between two adjacent source shots."""

    before_uuid: str
    after_uuid: str
    cut_pts: float
    kind: str = "cut"
    confidence: float = 1.0
    engine: str = "unknown"

    def __post_init__(self) -> None:
        if not self.before_uuid or not self.after_uuid:
            raise ValueError("Transition source UUIDs cannot be empty")
        if self.before_uuid == self.after_uuid:
            raise ValueError("A transition must join different source shots")
        self.cut_pts = float(self.cut_pts)
        self.confidence = float(self.confidence)
        if not math.isfinite(self.cut_pts):
            raise ValueError("Transition cut PTS must be finite")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Transition confidence must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "before_uuid": self.before_uuid,
            "after_uuid": self.after_uuid,
            "cut_pts": self.cut_pts,
            "kind": self.kind,
            "confidence": self.confidence,
            "engine": self.engine,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Transition:
        """Create a transition from serialized data."""

        return cls(**dict(data))


@dataclass
class LogicalGroup:
    """An ordered, contiguous collection of source-shot UUIDs."""

    uuid: str
    source_ids: list[str]

    def __post_init__(self) -> None:
        if not self.uuid:
            raise ValueError("Logical group UUID cannot be empty")
        self.source_ids = list(self.source_ids)
        if not self.source_ids:
            raise ValueError("A logical group must contain at least one source shot")
        if len(set(self.source_ids)) != len(self.source_ids):
            raise ValueError("A logical group cannot contain duplicate source shots")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {"uuid": self.uuid, "source_ids": list(self.source_ids)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LogicalGroup:
        """Create a logical group from serialized data."""

        return cls(**dict(data))


@dataclass
class ShotProject:
    """Ordered source shots, logical groups, and all original transitions."""

    source_shots: list[SourceShot]
    groups: list[LogicalGroup] = field(default_factory=list)
    transitions: list[Transition] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.source_shots = list(self.source_shots)
        self.groups = list(self.groups)
        self.transitions = list(self.transitions)
        self._validate_shots()
        if not self.groups:
            self.groups = [
                LogicalGroup(_new_uuid(), [shot.uuid]) for shot in self.source_shots
            ]
        self._validate_groups()
        if not self.transitions and len(self.source_shots) > 1:
            self.rebuild_transitions()
        else:
            self._validate_transitions()

    def _validate_shots(self) -> None:
        uuids = [shot.uuid for shot in self.source_shots]
        if len(set(uuids)) != len(uuids):
            raise ValueError("Source shot UUIDs must be unique")
        expected_ordinals = list(range(1, len(self.source_shots) + 1))
        if [shot.ordinal for shot in self.source_shots] != expected_ordinals:
            raise ValueError("Source shots must have consecutive ordinals starting at 1")

    def _validate_groups(self) -> None:
        group_uuids = [group.uuid for group in self.groups]
        if len(set(group_uuids)) != len(group_uuids):
            raise ValueError("Logical group UUIDs must be unique")
        flattened = [source_id for group in self.groups for source_id in group.source_ids]
        if flattened != [shot.uuid for shot in self.source_shots]:
            raise ValueError("Groups must cover every source shot once, in source order")

    def _validate_transitions(self) -> None:
        expected_count = max(0, len(self.source_shots) - 1)
        if len(self.transitions) != expected_count:
            raise ValueError("There must be one transition per adjacent source-shot pair")
        for index, transition in enumerate(self.transitions):
            left = self.source_shots[index]
            right = self.source_shots[index + 1]
            if (transition.before_uuid, transition.after_uuid) != (left.uuid, right.uuid):
                raise ValueError("Transitions must follow adjacent source-shot order")
            if not math.isclose(
                left.end_pts, right.start_pts, rel_tol=0.0, abs_tol=1e-9
            ):
                raise ValueError("Adjacent source shots must share a PTS boundary")
            if not math.isclose(
                transition.cut_pts, left.end_pts, rel_tol=0.0, abs_tol=1e-9
            ):
                raise ValueError("Transition cut PTS must match the source-shot boundary")
            transition.cut_pts = left.end_pts

    def rebuild_transitions(self) -> list[Transition]:
        """Rebuild all boundaries while retaining matching detection metadata."""

        existing = {
            (transition.before_uuid, transition.after_uuid): transition
            for transition in self.transitions
        }
        rebuilt: list[Transition] = []
        for left, right in zip(self.source_shots, self.source_shots[1:]):
            if not math.isclose(
                left.end_pts, right.start_pts, rel_tol=0.0, abs_tol=1e-9
            ):
                raise ValueError("Adjacent source shots must share a PTS boundary")
            previous = existing.get((left.uuid, right.uuid))
            rebuilt.append(
                Transition(
                    before_uuid=left.uuid,
                    after_uuid=right.uuid,
                    cut_pts=left.end_pts,
                    kind=previous.kind if previous else "cut",
                    confidence=previous.confidence if previous else 1.0,
                    engine=previous.engine if previous else "unknown",
                )
            )
        self.transitions = rebuilt
        return self.transitions

    def group_interval(self, group_uuid: str) -> tuple[float, float]:
        """Return the first source start and last source end for a group."""

        group = next((item for item in self.groups if item.uuid == group_uuid), None)
        if group is None:
            raise ValueError(f"Unknown logical group: {group_uuid}")
        shots = {shot.uuid: shot for shot in self.source_shots}
        return (
            shots[group.source_ids[0]].start_pts,
            shots[group.source_ids[-1]].end_pts,
        )

    def merge_groups(self, group_uuids: Iterable[str]) -> LogicalGroup:
        """Merge two or more adjacent logical groups into one group."""

        selected = list(group_uuids)
        if len(selected) < 2:
            raise ValueError("Merging requires at least two logical groups")
        if len(set(selected)) != len(selected):
            raise ValueError("Cannot merge the same logical group more than once")
        positions = {group.uuid: index for index, group in enumerate(self.groups)}
        try:
            indices = sorted(positions[group_uuid] for group_uuid in selected)
        except KeyError as error:
            raise ValueError(f"Unknown logical group: {error.args[0]}") from error
        if indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError("Only adjacent logical groups can be merged")
        first, last = indices[0], indices[-1]
        source_ids = [
            source_id
            for group in self.groups[first : last + 1]
            for source_id in group.source_ids
        ]
        merged = LogicalGroup(_new_uuid(), source_ids)
        self.groups[first : last + 1] = [merged]
        return merged

    def split_group_at(self, group_uuid: str, cut_pts: float) -> list[LogicalGroup]:
        """Split one logical group at a PTS value into two independent groups."""

        index = next(
            (index for index, group in enumerate(self.groups) if group.uuid == group_uuid),
            None,
        )
        if index is None:
            raise ValueError(f"Unknown logical group: {group_uuid}")
        group = self.groups[index]
        cut = float(cut_pts)
        if not math.isfinite(cut):
            raise ValueError("A manual cut must be a finite PTS value")
        shots = {shot.uuid: shot for shot in self.source_shots}
        span_start = shots[group.source_ids[0]].start_pts
        span_end = shots[group.source_ids[-1]].end_pts
        if not span_start < cut < span_end:
            raise ValueError("A manual cut must lie inside the group")
        boundary = next(
            (
                position
                for position, source_id in enumerate(group.source_ids)
                if position > 0
                and math.isclose(shots[source_id].start_pts, cut, abs_tol=1e-6)
            ),
            None,
        )
        if boundary is not None:
            left_ids = group.source_ids[:boundary]
            right_ids = group.source_ids[boundary:]
        else:
            target = next(
                source_id
                for source_id in group.source_ids
                if shots[source_id].start_pts < cut < shots[source_id].end_pts
            )
            _, right_shot = self.split_source_shot(target, cut)
            position = group.source_ids.index(target)
            left_ids = group.source_ids[: position + 1]
            right_ids = [right_shot.uuid] + group.source_ids[position + 1 :]
        left_group = LogicalGroup(group.uuid, left_ids)
        right_group = LogicalGroup(_new_uuid(), right_ids)
        self.groups[index : index + 1] = [left_group, right_group]
        return [left_group, right_group]

    def split_source_shot(
        self, shot_uuid: str, cut_pts: float
    ) -> tuple[SourceShot, SourceShot]:
        """Split a source shot in two while keeping ordinals and transitions valid."""

        index = next(
            (
                index
                for index, shot in enumerate(self.source_shots)
                if shot.uuid == shot_uuid
            ),
            None,
        )
        if index is None:
            raise ValueError(f"Unknown source shot: {shot_uuid}")
        shot = self.source_shots[index]
        cut = float(cut_pts)
        if not math.isfinite(cut) or not shot.start_pts < cut < shot.end_pts:
            raise ValueError("A split PTS must lie strictly inside the source shot")
        trailing = next(
            (
                transition
                for transition in self.transitions
                if transition.before_uuid == shot.uuid
            ),
            None,
        )
        left = SourceShot(
            shot.uuid,
            shot.ordinal,
            shot.start_pts,
            cut,
            shot.start_frame,
            None,
            shot.thumbnail_path,
        )
        right = SourceShot(
            _new_uuid(),
            shot.ordinal + 1,
            cut,
            shot.end_pts,
            None,
            shot.end_frame,
            None,
        )
        self.source_shots[index : index + 1] = [left, right]
        for ordinal, item in enumerate(self.source_shots, start=1):
            item.ordinal = ordinal
        self.rebuild_transitions()
        for transition in self.transitions:
            pair = (transition.before_uuid, transition.after_uuid)
            if pair == (left.uuid, right.uuid):
                transition.kind = "manual"
                transition.engine = "manual"
                transition.confidence = 1.0
            elif trailing is not None and pair == (right.uuid, trailing.after_uuid):
                transition.kind = trailing.kind
                transition.engine = trailing.engine
                transition.confidence = trailing.confidence
        return left, right

    def split_group(self, group_uuid: str) -> list[LogicalGroup]:
        """Restore a group to its original single-source groups."""

        index = next(
            (index for index, group in enumerate(self.groups) if group.uuid == group_uuid),
            None,
        )
        if index is None:
            raise ValueError(f"Unknown logical group: {group_uuid}")
        group = self.groups[index]
        if len(group.source_ids) == 1:
            return [group]
        restored = [LogicalGroup(_new_uuid(), [source_id]) for source_id in group.source_ids]
        self.groups[index : index + 1] = restored
        return restored

    def adjust_boundary(self, left_source_uuid: str, new_cut: float) -> Transition:
        """Atomically move a shared source-shot boundary and its transition."""

        index = next(
            (
                index
                for index, shot in enumerate(self.source_shots)
                if shot.uuid == left_source_uuid
            ),
            None,
        )
        if index is None:
            raise ValueError(f"Unknown source shot: {left_source_uuid}")
        if index == len(self.source_shots) - 1:
            raise ValueError("The final source shot has no right boundary")
        left = self.source_shots[index]
        right = self.source_shots[index + 1]
        cut = float(new_cut)
        if not math.isfinite(cut) or cut <= 0:
            raise ValueError("A boundary cut must be a finite, positive PTS value")
        if not left.start_pts < cut < right.end_pts:
            raise ValueError("A boundary cut must lie inside both adjacent shots")
        transition = self.transitions[index]
        if (transition.before_uuid, transition.after_uuid) != (left.uuid, right.uuid):
            raise ValueError("Transition order does not match source-shot order")
        left.end_pts = cut
        right.start_pts = cut
        transition.cut_pts = cut
        return transition

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot suitable for an undo history."""

        return {
            "source_shots": [shot.to_dict() for shot in self.source_shots],
            "groups": [group.to_dict() for group in self.groups],
            "transitions": [transition.to_dict() for transition in self.transitions],
        }

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, Any]) -> ShotProject:
        """Build a validated project from a snapshot."""

        return cls(
            source_shots=[
                SourceShot.from_dict(data) for data in snapshot["source_shots"]
            ],
            groups=[LogicalGroup.from_dict(data) for data in snapshot["groups"]],
            transitions=[Transition.from_dict(data) for data in snapshot["transitions"]],
        )

    def restore(self, snapshot: dict[str, Any]) -> None:
        """Restore a snapshot atomically after validating all of its state."""

        restored = self.from_snapshot(snapshot)
        self.source_shots = restored.source_shots
        self.groups = restored.groups
        self.transitions = restored.transitions
