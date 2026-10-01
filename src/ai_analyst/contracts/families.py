"""Column families inferred from name structure (WP4).

A family groups columns that differ only by a structural axis: a window
(`_7d`, `_lifetime`), a direction prefix (`ALL_`, `IN_`, `OUT_`), or a per-field
suffix (`_updated_days`). Grouping is a convenience for onboarding, not
evidence: a name never classifies a column (CLAUDE.md 2.2, 22). Every
`FamilyProposal` is therefore proposal-only. It becomes a classification only
when a tenant confirms it with a `FamilyDeclaration`.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict

from ai_analyst.contracts.columns import Availability, ColumnCategory, Disposition


class ProposedClassification(BaseModel):
    """A classification suggested from a name pattern. Never applied by itself."""

    model_config = ConfigDict(frozen=True)

    category: ColumnCategory
    availability: Availability
    disposition: Disposition
    reason: str
    proposal_only: Literal[True] = True


class FamilyProposal(BaseModel):
    """One proposed family, or a singleton (one member, no structural axes)."""

    model_config = ConfigDict(frozen=True)

    name: str
    members: tuple[str, ...]
    windows: tuple[str, ...] = ()
    directions: tuple[str, ...] = ()
    fields: tuple[str, ...] = ()
    proposed: ProposedClassification
    proposal_only: Literal[True] = True

    @property
    def is_singleton(self) -> bool:
        return len(self.members) == 1

    @property
    def member_pattern(self) -> str:
        """A regex matching exactly these members, for a `FamilyDeclaration`."""
        return "(?:" + "|".join(re.escape(m) for m in self.members) + ")"
