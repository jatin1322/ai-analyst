"""Deterministic column-family inference from name structure (WP4).

Names only PROPOSE. Nothing here writes a classification: a proposal becomes
one only when a tenant confirms it with a `FamilyDeclaration`, and even then
through the ordinary grant rules. An unconfirmed family member stays
unclassified and quarantined (CLAUDE.md 2.2, 22).

Structure recognised: a window suffix (`_7d`, `_lifetime`), a direction prefix
(`ALL_`, `IN_`, `OUT_`), `_label` / `_label_mask` pairs, and a per-field
`_updated_days` suffix. Columns sharing a stem form a family; a column alone
in its stem is a singleton.
"""

from __future__ import annotations

import re
from collections import defaultdict

from ai_analyst.contracts.columns import Availability, ColumnCategory, Disposition
from ai_analyst.contracts.families import FamilyProposal, ProposedClassification

_WINDOW = re.compile(r"^(?P<stem>.+)_(?P<window>\d+d|lifetime)$")
_DIRECTION = re.compile(r"^(?P<direction>ALL|IN|OUT)_(?P<rest>.+)$")
_LABEL_MASK = re.compile(r"^(?P<stem>.+)_label_mask$")
_LABEL = re.compile(r"^(?P<stem>.+)_label$")
_UPDATED_DAYS = re.compile(r"^(?P<field>.+)_updated_days$")

_SPLIT = re.compile(r"^(train|test|valid|validation|holdout|split)(_flag|_split|_set)?$")
_OUTCOME = re.compile(r"(^|_)(outcome|fate|won|win|lost|loss|closed_won|closed_lost)(_|$)")
_ETL = re.compile(
    r"^(pipeline_version|config_hash|run_id|scored_at|ingested_at|loaded_at|etl_.+)$"
)
_ACTIVITY = re.compile(r"(^|_)(count|sum|avg|total|num)$")


def _window_key(w: str) -> tuple[int, int]:
    return (1, 0) if w == "lifetime" else (0, int(w[:-1]))


def _propose(name: str, members: tuple[str, ...], windowed: bool) -> ProposedClassification:
    if any(_SPLIT.match(m) or _LABEL.match(m) or _LABEL_MASK.match(m) for m in members) or (
        _OUTCOME.search(name)
    ):
        return ProposedClassification(
            category=ColumnCategory.OUTCOME,
            availability=Availability.FUTURE_CONTAMINATED,
            disposition=Disposition.QUARANTINE,
            reason=(
                "name looks like a label, mask, split flag, or outcome field; such "
                "columns are typically built from final outcomes"
            ),
        )
    if all(_ETL.match(m) for m in members):
        return ProposedClassification(
            category=ColumnCategory.METADATA,
            availability=Availability.UNKNOWN,
            disposition=Disposition.QUARANTINE,
            reason="name looks like pipeline/ETL metadata, not analytical content",
        )
    if windowed and _ACTIVITY.search(name):
        return ProposedClassification(
            category=ColumnCategory.HISTORICAL_FEATURE,
            availability=Availability.BACKWARD_DERIVED,
            disposition=Disposition.USE_WITH_PROOF,
            reason="window assumed to end at as_of; unconfirmed",
        )
    return ProposedClassification(
        category=ColumnCategory.METADATA,
        availability=Availability.UNKNOWN,
        disposition=Disposition.QUARANTINE,
        reason="no structural pattern places this column; unknown fails closed",
    )


def infer_families(column_names: list[str]) -> list[FamilyProposal]:
    """Group columns into proposed families by name structure alone.

    Deterministic: the result depends only on the names, and is ordered by the
    first appearance of each family.
    """
    # key -> (members, windows, directions, fields)
    groups: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: {"members": [], "windows": [], "directions": [], "fields": []}
    )
    windowed_keys: set[str] = set()
    for col in dict.fromkeys(column_names):
        window = direction = field = None
        if m := _LABEL_MASK.match(col):
            key = f"{m['stem']}_label"
        elif _LABEL.match(col):
            key = col
        elif m := _UPDATED_DAYS.match(col):
            key, field = "*_updated_days", m["field"]
        else:
            rest = col
            if m := _DIRECTION.match(rest):
                direction, rest = m["direction"], m["rest"]
            if m := _WINDOW.match(rest):
                window, rest = m["window"], m["stem"]
            key = rest if (window or direction) else col
        g = groups[key]
        g["members"].append(col)
        for slot, value in (("windows", window), ("directions", direction), ("fields", field)):
            if value and value not in g[slot]:
                g[slot].append(value)
        if window:
            windowed_keys.add(key)

    proposals: list[FamilyProposal] = []
    for key, g in groups.items():
        members = tuple(g["members"])
        if len(members) == 1:
            name = members[0]
            proposals.append(
                FamilyProposal(
                    name=name, members=members, proposed=_propose(name, members, False)
                )
            )
            continue
        proposals.append(
            FamilyProposal(
                name=key,
                members=members,
                windows=tuple(sorted(g["windows"], key=_window_key)),
                directions=tuple(g["directions"]),
                fields=tuple(g["fields"]),
                proposed=_propose(key, members, key in windowed_keys),
            )
        )
    return proposals


def onboarding_report(proposals: list[FamilyProposal]) -> str:
    """A compact text report of proposed families. Names and counts only; no values."""
    n_cols = sum(len(p.members) for p in proposals)
    families = [p for p in proposals if not p.is_singleton]
    singletons = [p for p in proposals if p.is_singleton]
    lines = [
        f"{n_cols} columns -> {len(proposals)} groups "
        f"({len(families)} families, {len(singletons)} singletons)",
        "All classifications below are PROPOSALS. Nothing is confirmed or readable "
        "until a tenant declares it.",
        "",
    ]

    def describe(p: FamilyProposal) -> str:
        axes = []
        if p.windows:
            axes.append(f"{len(p.windows)} windows: {', '.join(p.windows)}")
        if p.directions:
            axes.append(f"directions: {', '.join(p.directions)}")
        if p.fields:
            axes.append(f"{len(p.fields)} fields: {', '.join(p.fields)}")
        c = p.proposed
        head = f"{p.name} ({len(p.members)} columns"
        head += f"; {'; '.join(axes)})" if axes else ")"
        return (
            f"- {head}\n"
            f"    proposed: {c.category.value} / {c.availability.value} / "
            f"{c.disposition.value}\n"
            f"    reason: {c.reason}"
        )

    if families:
        lines.append("Families:")
        lines.extend(describe(p) for p in families)
    if singletons:
        lines.append("")
        lines.append("Singletons:")
        for p in singletons:
            c = p.proposed
            lines.append(
                f"- {p.name}: {c.category.value} / {c.availability.value} / "
                f"{c.disposition.value} ({c.reason})"
            )
    return "\n".join(lines)
