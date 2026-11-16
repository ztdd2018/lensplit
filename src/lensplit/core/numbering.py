"""Display numbering and transition statistics for logical groups."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, TypedDict

from lensplit.core.model import LogicalGroup, ShotProject


class NumberingSequence(str, Enum):
    """Supported ordinal sequences."""

    DECIMAL = "decimal"
    LOWER = "lower"
    UPPER = "upper"


@dataclass
class NumberingStyle:
    """Formatting applied to display numbers without changing source ordinals."""

    start: int = 1
    width: int = 3
    prefix: str = ""
    suffix: str = ""
    sequence: NumberingSequence | str = NumberingSequence.DECIMAL

    def __post_init__(self) -> None:
        if self.start < 1:
            raise ValueError("Numbering must start at 1 or greater")
        if self.width < 0:
            raise ValueError("Numbering width cannot be negative")
        self.sequence = NumberingSequence(self.sequence)


class MergeMode(str, Enum):
    """Ways to derive display numbers after logical-group merging."""

    COMBINED = "combined"
    SEQUENTIAL = "sequential"
    FIRST = "first"


class TransitionStats(TypedDict):
    """Original, internal, and effective transition counts."""

    total: int
    internal: int
    effective: int


def _alphabetic(value: int) -> str:
    letters: list[str] = []
    # 双射二十六进制没有独立的零位，取模前减一才能正确得到 z 后的 aa。
    while value:
        value, remainder = divmod(value - 1, 26)
        letters.append(chr(ord("a") + remainder))
    return "".join(reversed(letters))


def _format_body(ordinal: int, style: NumberingStyle) -> str:
    if ordinal < 1:
        raise ValueError("Source ordinals start at 1")
    value = style.start + ordinal - 1
    if style.sequence is NumberingSequence.DECIMAL:
        return f"{value:0{style.width}d}"
    body = _alphabetic(value)
    if style.sequence is NumberingSequence.UPPER:
        body = body.upper()
    return body.rjust(style.width, "0")


def format_ordinal(ordinal: int, style: NumberingStyle | None = None) -> str:
    """Format one original ordinal with the configured affixes."""

    style = style or NumberingStyle()
    return f"{style.prefix}{_format_body(ordinal, style)}{style.suffix}"


def format_groups(
    groups: Iterable[LogicalGroup],
    style: NumberingStyle | None = None,
    mode: MergeMode | str = MergeMode.COMBINED,
) -> list[str]:
    """Compute display labels for ordered, contiguous logical groups."""

    style = style or NumberingStyle()
    mode = MergeMode(mode)
    labels: list[str] = []
    first_source_ordinal = 1
    for group_ordinal, group in enumerate(groups, start=1):
        source_ordinals = range(
            first_source_ordinal, first_source_ordinal + len(group.source_ids)
        )
        if mode is MergeMode.COMBINED:
            body = "+".join(_format_body(ordinal, style) for ordinal in source_ordinals)
            labels.append(f"{style.prefix}{body}{style.suffix}")
        elif mode is MergeMode.SEQUENTIAL:
            labels.append(format_ordinal(group_ordinal, style))
        else:
            labels.append(format_ordinal(first_source_ordinal, style))
        first_source_ordinal += len(group.source_ids)
    return labels


def transition_statistics(project: ShotProject) -> TransitionStats:
    """Count original transitions inside groups and between effective groups."""

    internal_pairs = {
        (before, after)
        for group in project.groups
        for before, after in zip(group.source_ids, group.source_ids[1:])
    }
    internal = sum(
        (transition.before_uuid, transition.after_uuid) in internal_pairs
        for transition in project.transitions
    )
    total = len(project.transitions)
    return TransitionStats(total=total, internal=internal, effective=total - internal)
