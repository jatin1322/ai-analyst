"""Agreement tests: deterministic checks whose failure is decisive (ARCHITECTURE 12.3).

Several questions in this system have one shape: *does this column mean what we
think it means?* An agreement test answers it with a row-level SQL predicate and
reports the disagreeing row count.

Three properties, and the third is the one people get wrong:

* **Failure is decisive.** One disagreeing row disproves the reading.
* **The count is reported**, so a near-miss is visible rather than rounded away.
* **A pass is evidence, not proof.** Agreement can be coincidental, and the
  export may not contain the case that would break it.

Because of the third, an `AgreementResult` can never confirm a binding by
itself: `EvidenceKind.AGREEMENT_TEST.can_confirm` is False and this module
offers no way around it. A passing test raises confidence and a failing test
withholds a binding entirely.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai_analyst.contracts.binding import (
    BindingEvidence,
    EvidenceKind,
    ReconstructionAssessment,
    ReconstructionVerdict,
)
from ai_analyst.contracts.concepts import BusinessConcept


class AgreementRole(StrEnum):
    """What a failure of the test means (ARCHITECTURE 13.1).

    Assigned by declaration, never inferred from how a disagreement looks.

    * `VALIDITY` checks the derivation itself, or a field whose semantics are
      declared. A failure is contradictory evidence and withholds the concept.
    * `RECONCILIATION` checks agreement with an ancillary field whose semantics
      nobody has written down. A failure is a warning: the concept stays usable,
      the disagreement is disclosed, and trust is capped at B. Requiring an
      undocumented field to agree would make the reconstruction's validity
      hostage to that field's unknown convention.
    """

    VALIDITY = "validity"
    RECONCILIATION = "reconciliation"


class AgreementKind(StrEnum):
    """What an agreement test asserts."""

    # "this column is consistent with concept X"
    COLUMN_CONSISTENT_WITH_CONCEPT = "column_consistent_with_concept"
    # "these columns satisfy relationship Y"
    COLUMNS_SATISFY_RELATIONSHIP = "columns_satisfy_relationship"


class AgreementTest(BaseModel):
    """One checkable assertion about a dataset's columns.

    `predicate_sql` is written so that **true means agreement**. Rows where it
    is false are counted as disagreements; rows where it is NULL are outside
    the test's scope and are not counted either way, which keeps a column full
    of nulls from reading as universal agreement.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    kind: AgreementKind
    assertion: str
    predicate_sql: str
    # Restricts the rows the test applies to. Empty means every row.
    scope_sql: str = ""
    # Physical columns the predicate reads. Used to skip a test whose inputs
    # are absent rather than failing it, which would be a false negative.
    requires_columns: tuple[str, ...] = ()
    concept: BusinessConcept | None = None
    # Columns echoed in a disagreement sample, for a human to read.
    sample_columns: tuple[str, ...] = ()
    # What a failure means. The default is the strict one: a test nobody has
    # classified is treated as checking the derivation itself.
    role: AgreementRole = AgreementRole.VALIDITY
    # True when the test compares against fiscal quarters. Such a test cannot be
    # a VALIDITY test while the tenant's fiscal calendar is unresolved, because
    # a disagreement could be the calendar and not the date (13.1 rule 5).
    calendar_dependent: bool = False

    @model_validator(mode="after")
    def _concept_tests_name_a_concept(self) -> AgreementTest:
        if self.kind is AgreementKind.COLUMN_CONSISTENT_WITH_CONCEPT and self.concept is None:
            raise ValueError(f"{self.id}: a concept-consistency test must name its concept")
        return self


class AgreementSample(BaseModel):
    """One disagreeing row, rendered as strings."""

    model_config = ConfigDict(frozen=True)

    values: dict[str, str | None] = Field(default_factory=dict)


class AgreementResult(BaseModel):
    """The outcome of one agreement test on one dataset."""

    model_config = ConfigDict(frozen=True)

    test_id: str
    assertion: str
    kind: AgreementKind
    concept: BusinessConcept | None = None
    checked_rows: int = 0
    disagreeing_rows: int = 0
    samples: tuple[AgreementSample, ...] = ()
    role: AgreementRole = AgreementRole.VALIDITY
    calendar_dependent: bool = False
    # Aggregate diagnostics for a human deciding what a convention is, such as
    # the offset histogram of a day-count field. Never used to decide a pass.
    diagnostics: dict[str, int] = Field(default_factory=dict)
    # A test whose inputs are absent is skipped, never passed. Treating an
    # unevaluable test as a pass is how an unverified reading gets through.
    skipped: bool = False
    skip_reason: str = ""
    note: str = ""

    @property
    def passed(self) -> bool:
        """True only when the test ran, saw rows, and found no disagreement."""
        return not self.skipped and self.checked_rows > 0 and self.disagreeing_rows == 0

    @property
    def failed(self) -> bool:
        return not self.skipped and self.disagreeing_rows > 0

    @property
    def inconclusive(self) -> bool:
        """Skipped, or ran against no rows at all."""
        return self.skipped or self.checked_rows == 0

    @property
    def agreement_rate(self) -> float:
        if self.checked_rows == 0:
            return 0.0
        return (self.checked_rows - self.disagreeing_rows) / self.checked_rows

    @property
    def summary(self) -> str:
        if self.skipped:
            return f"skipped: {self.skip_reason}"
        if self.passed:
            return f"passed on {self.checked_rows} rows"
        if self.checked_rows == 0:
            return "inconclusive: no rows in scope"
        return f"failed: {self.disagreeing_rows} of {self.checked_rows} rows disagree"

    def as_evidence(self) -> BindingEvidence:
        """Express this result as binding evidence.

        Always `AGREEMENT_TEST`, which cannot confirm. A pass supports the
        binding and a failure is recorded as evidence against it rather than
        being discarded.
        """
        return BindingEvidence(
            kind=EvidenceKind.AGREEMENT_TEST,
            detail=f"{self.assertion} ({self.summary})",
            source=self.test_id,
            confidence=round(self.agreement_rate, 4),
            supports=self.passed,
        )

    @model_validator(mode="after")
    def _counts_are_coherent(self) -> AgreementResult:
        if self.disagreeing_rows > self.checked_rows:
            raise ValueError(
                f"{self.test_id}: {self.disagreeing_rows} disagreements exceed "
                f"{self.checked_rows} checked rows"
            )
        if self.skipped and not self.skip_reason:
            raise ValueError(f"{self.test_id}: a skipped test must say why")
        return self


class AgreementReport(BaseModel):
    """Every agreement test evaluated for one dataset.

    Persisted beside the dataset at ingestion, so a failure stays visible in the
    dataset's metadata rather than existing only while a process is running.
    """

    model_config = ConfigDict(frozen=True)

    dataset_id: str
    results: tuple[AgreementResult, ...] = ()
    # The fiscal calendar the quarter tests assumed. A report evaluated under a
    # different start month than the current setting is stale.
    fiscal_year_start_month: int | None = None

    def by_id(self) -> dict[str, AgreementResult]:
        return {r.test_id: r for r in self.results}

    def for_concept(self, concept: BusinessConcept) -> list[AgreementResult]:
        return [r for r in self.results if r.concept is concept]

    @property
    def failures(self) -> list[AgreementResult]:
        return [r for r in self.results if r.failed]

    @property
    def passes(self) -> list[AgreementResult]:
        return [r for r in self.results if r.passed]

    @property
    def skipped(self) -> list[AgreementResult]:
        return [r for r in self.results if r.skipped]

    def effective_role(
        self,
        result: AgreementResult,
        *,
        promoted: frozenset[str] = frozenset(),
        calendar_resolved: bool = False,
    ) -> AgreementRole:
        """The role a result plays once declarations and the calendar are known.

        A reconciliation test becomes a validity test only when a tenant has
        declared the field's semantics (`promoted`), and a calendar-dependent
        test never becomes one while the fiscal calendar is unresolved.
        """
        role = result.role
        if result.test_id in promoted:
            role = AgreementRole.VALIDITY
        if role is AgreementRole.VALIDITY and result.calendar_dependent and not calendar_resolved:
            role = AgreementRole.RECONCILIATION
        return role

    def concept_is_contradicted(
        self,
        concept: BusinessConcept,
        *,
        promoted: frozenset[str] = frozenset(),
        calendar_resolved: bool = False,
    ) -> bool:
        """Whether a validity test for this concept actively failed.

        A failed reconciliation test is a warning, not a contradiction.
        """
        return any(
            r.failed
            and self.effective_role(r, promoted=promoted, calendar_resolved=calendar_resolved)
            is AgreementRole.VALIDITY
            for r in self.for_concept(concept)
        )

    def assess(
        self,
        concept: BusinessConcept,
        *,
        declared: bool,
        promoted: frozenset[str] = frozenset(),
        calendar_resolved: bool = False,
    ) -> ReconstructionAssessment:
        """The verdict for a reconstructed concept (ARCHITECTURE 13.1).

        In order: an undeclared derivation is unavailable whatever its tests
        say; a failed validity test contradicts it; an unverifiable validity
        test or a failed reconciliation test leaves it usable with warnings;
        otherwise it is valid.

        Vacuous truth is still refused: a concept with no validity test
        evaluated at all is not `VALID`. Its evidence is the declaration alone,
        and it is reported with a warning saying so.
        """
        results = self.for_concept(concept)
        if not declared:
            return ReconstructionAssessment(
                concept=concept,
                verdict=ReconstructionVerdict.UNDECLARED,
                reasons=("the derivation is not declared by any accountable source",),
            )

        def role(r: AgreementResult) -> AgreementRole:
            return self.effective_role(r, promoted=promoted, calendar_resolved=calendar_resolved)

        validity = [r for r in results if role(r) is AgreementRole.VALIDITY]
        reconciliation = [r for r in results if role(r) is AgreementRole.RECONCILIATION]

        contradicted = [r for r in validity if r.failed]
        if contradicted:
            return ReconstructionAssessment(
                concept=concept,
                verdict=ReconstructionVerdict.CONTRADICTED,
                contradicted=tuple(r.test_id for r in contradicted),
                reasons=tuple(f"{r.test_id}: {r.summary}" for r in contradicted),
            )

        unverified = tuple(r.test_id for r in validity if r.inconclusive)
        warned = tuple(r.test_id for r in reconciliation if r.failed)
        demoted = tuple(
            r.test_id
            for r in results
            if r.role is AgreementRole.VALIDITY or r.test_id in promoted
            if role(r) is AgreementRole.RECONCILIATION
        )
        reasons: list[str] = []
        if not any(r.passed for r in validity):
            reasons.append("no validity test corroborated the reconstruction")
        reasons += [f"{t}: not verified" for t in unverified]
        reasons += [f"{t}: reconciliation disagreement" for t in warned]
        reasons += [f"{t}: treated as reconciliation, fiscal calendar unresolved" for t in demoted]
        verdict = (
            ReconstructionVerdict.VALID_WITH_WARNINGS
            if unverified or warned or not any(r.passed for r in validity)
            else ReconstructionVerdict.VALID
        )
        return ReconstructionAssessment(
            concept=concept,
            verdict=verdict,
            unverified=unverified,
            reconciliation_warnings=warned,
            demoted=demoted,
            reasons=tuple(reasons),
        )
