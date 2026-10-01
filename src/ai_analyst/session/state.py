"""Session state: plans and their lineage (ARCHITECTURE 13.13)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ai_analyst.contracts.investigation import InvestigationPlan
from ai_analyst.contracts.plan import AnalysisPlan, AnalysisStance
from ai_analyst.contracts.session import CarryForwardReport, PlanEdit
from ai_analyst.session.edits import apply_edit


class SessionState(BaseModel):
    """What a session remembers: plans by id, which one is active, and why."""

    session_id: str
    dataset_id: str
    stance: AnalysisStance = AnalysisStance.PROSPECTIVE
    plans: dict[str, AnalysisPlan | InvestigationPlan] = Field(default_factory=dict)
    active_plan_id: str | None = None
    # Every edit applied, in order, beside the report it produced. With the
    # plans (which carry plan_id and parent_plan_id) this is the full lineage:
    # what was asked, what changed, and what was carried forward.
    edits: list[PlanEdit] = Field(default_factory=list)
    reports: list[CarryForwardReport] = Field(default_factory=list)

    def record(self, plan: AnalysisPlan | InvestigationPlan) -> None:
        """Adopt a freshly planned question as the active plan."""
        self.plans[plan.plan_id] = plan
        self.active_plan_id = plan.plan_id

    @property
    def active(self) -> AnalysisPlan | InvestigationPlan | None:
        return self.plans.get(self.active_plan_id) if self.active_plan_id else None

    def edit(self, edit: PlanEdit) -> tuple[AnalysisPlan, CarryForwardReport]:
        """Apply a follow-up to the plan it names and make the result active."""
        base = self.plans.get(edit.base_plan_id)
        if not isinstance(base, AnalysisPlan):
            from ai_analyst.contracts.session import EditConflictCode
            from ai_analyst.session.edits import EditConflict

            raise EditConflict(
                EditConflictCode.BASE_MISMATCH,
                f"no analysis plan {edit.base_plan_id!r} in this session",
            )
        edited = apply_edit(base, edit)
        self.record(edited.plan)
        self.edits.append(edit)
        self.reports.append(edited.report)
        return edited.plan, edited.report

    def dump(self) -> str:
        """The session as JSON, for persistence. Round-trips through `load`."""
        return self.model_dump_json()

    @classmethod
    def load(cls, payload: str) -> SessionState:
        return cls.model_validate_json(payload)

    def lineage(self, plan_id: str) -> list[str]:
        """The chain of plan ids from the first question to this one."""
        chain: list[str] = []
        current = self.plans.get(plan_id)
        while current is not None:
            chain.append(current.plan_id)
            parent = getattr(current, "parent_plan_id", None)
            current = self.plans.get(parent) if parent else None
        return list(reversed(chain))
