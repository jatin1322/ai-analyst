"""The stable analytical ontology (ARCHITECTURE 12.1).

A **concept** is a stable analytical idea with a written definition. It is the
vocabulary metrics and plans are written in, and it never varies by tenant.
A **physical column** is what one dataset actually contains. Nothing in this
module names a physical column: the connection between the two is a
`ConceptBinding` (`contracts/binding.py`), made per dataset and backed by
evidence.

Keeping the two apart is what lets one metric definition serve a tenant whose
amount column is `new_amount` and one whose amount column is `ARR`.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai_analyst.contracts.schema import DataType


class BusinessConcept(StrEnum):
    """Every analytical idea the system is allowed to reason about."""

    OPPORTUNITY_ID = "opportunity_id"
    SNAPSHOT_DATE = "snapshot_date"
    ACCOUNT_ID = "account_id"
    OWNER_ID = "owner_id"
    AMOUNT = "amount"
    EXPECTED_CLOSE_DATE = "expected_close_date"
    CREATED_DATE = "created_date"
    STAGE = "stage"
    OPPORTUNITY_STATUS = "opportunity_status"
    FORECAST_CATEGORY = "forecast_category"
    TERMINAL_OUTCOME = "terminal_outcome"
    TERMINAL_DATE = "terminal_date"
    TERMINAL_AMOUNT = "terminal_amount"
    QUARTER = "quarter"
    CUSTOMER_SEGMENT = "customer_segment"
    NARRATIVE_TEXT = "narrative_text"


class SemanticType(StrEnum):
    """What kind of thing a concept's values are.

    Distinct from `DataType`, which is storage. `MONEY` and `QUANTITY` are both
    stored as numbers; only the first may never be summed through a float.
    """

    IDENTIFIER = "identifier"
    DATE = "date"
    MONEY = "money"
    CATEGORY = "category"
    QUANTITY = "quantity"
    OUTCOME = "outcome"
    PERIOD = "period"
    TEXT = "text"


class ConceptRole(StrEnum):
    """How an analysis may use a concept."""

    GRAIN = "grain"
    MEASURE = "measure"
    DIMENSION = "dimension"
    TEMPORAL = "temporal"
    OUTCOME = "outcome"
    NARRATIVE = "narrative"


class ConceptCardinality(StrEnum):
    """How many physical columns may satisfy a concept."""

    ONE = "one"
    MANY = "many"


class AnalyticalOperation(StrEnum):
    """Named analytical operations, used only to say what a concept is needed for.

    This is not a metric registry and carries no compiler. It exists so a
    concept can state whether it is required for a known operation, which is
    what lets a later layer hide an operation instead of failing it.
    """

    SNAPSHOT_PIPELINE = "snapshot_pipeline"
    PIPELINE_BRIDGE = "pipeline_bridge"
    TRANSITION = "transition"
    COHORT_TRACE = "cohort_trace"
    WIN_RATE = "win_rate"
    RANKED_LIST = "ranked_list"
    SEGMENT_ANALYSIS = "segment_analysis"
    FORECAST_ACCURACY = "forecast_accuracy"
    TEXT_CATALOGUE = "text_catalogue"


class ConceptDefinition(BaseModel):
    """One concept's stable definition. Contains no physical column name."""

    model_config = ConfigDict(frozen=True)

    concept: BusinessConcept
    display_name: str
    definition: str
    semantic_type: SemanticType
    expected_dtype: DataType
    role: ConceptRole
    cardinality: ConceptCardinality = ConceptCardinality.ONE
    # Operations that cannot be computed at all without this concept.
    required_for: tuple[AnalyticalOperation, ...] = ()
    # True for a concept the system computes rather than binds, such as the
    # fiscal quarter, which is derived from a date by the calendar (5.6).
    computed: bool = False
    derived_from: BusinessConcept | None = None
    # A concept whose values describe what eventually happened. Reading one
    # under a prospective stance is the leakage failure of 5.8.
    retrospective: bool = False

    @property
    def is_required_for_any_operation(self) -> bool:
        return bool(self.required_for)

    @property
    def is_monetary(self) -> bool:
        return self.semantic_type is SemanticType.MONEY

    @model_validator(mode="after")
    def _computed_concepts_declare_their_source(self) -> ConceptDefinition:
        if self.computed and self.derived_from is None:
            raise ValueError(f"{self.concept}: a computed concept must name what it derives from")
        if not self.computed and self.derived_from is not None:
            raise ValueError(f"{self.concept}: only a computed concept may name a source")
        return self


def _d(
    concept: BusinessConcept,
    display_name: str,
    definition: str,
    semantic_type: SemanticType,
    expected_dtype: DataType,
    role: ConceptRole,
    **kwargs: object,
) -> ConceptDefinition:
    return ConceptDefinition(
        concept=concept,
        display_name=display_name,
        definition=definition,
        semantic_type=semantic_type,
        expected_dtype=expected_dtype,
        role=role,
        **kwargs,  # type: ignore[arg-type]
    )


_OP = AnalyticalOperation

CONCEPTS: dict[BusinessConcept, ConceptDefinition] = {
    d.concept: d
    for d in [
        _d(
            BusinessConcept.OPPORTUNITY_ID,
            "Opportunity",
            "Stable identifier for one opportunity across every snapshot. Half of "
            "the grain. Never a measure.",
            SemanticType.IDENTIFIER,
            DataType.VARCHAR,
            ConceptRole.GRAIN,
            required_for=tuple(_OP),
        ),
        _d(
            BusinessConcept.SNAPSHOT_DATE,
            "Snapshot date",
            "The date at which this row's state was recorded. Half of the grain. "
            "An opportunity appears once per snapshot.",
            SemanticType.DATE,
            DataType.DATE,
            ConceptRole.GRAIN,
            required_for=tuple(_OP),
        ),
        _d(
            BusinessConcept.ACCOUNT_ID,
            "Account",
            "The account the opportunity belongs to.",
            SemanticType.IDENTIFIER,
            DataType.VARCHAR,
            ConceptRole.DIMENSION,
        ),
        _d(
            BusinessConcept.OWNER_ID,
            "Owner",
            "The rep who owns the opportunity in this snapshot. Mutable across "
            "snapshots, so attribution must state which snapshot it used.",
            SemanticType.IDENTIFIER,
            DataType.VARCHAR,
            ConceptRole.DIMENSION,
        ),
        _d(
            BusinessConcept.AMOUNT,
            "Amount",
            "Deal value in the reporting currency as recorded in this snapshot. "
            "Mutable across snapshots. Monetary: never aggregated through a float.",
            SemanticType.MONEY,
            DataType.DECIMAL,
            ConceptRole.MEASURE,
            required_for=(
                _OP.SNAPSHOT_PIPELINE,
                _OP.PIPELINE_BRIDGE,
                _OP.RANKED_LIST,
                _OP.FORECAST_ACCURACY,
            ),
        ),
        _d(
            BusinessConcept.EXPECTED_CLOSE_DATE,
            "Expected close date",
            "The close date recorded in this snapshot, not the realized one. What "
            "the rep expected at the time. Moving it later is a slip.",
            SemanticType.DATE,
            DataType.DATE,
            ConceptRole.TEMPORAL,
            required_for=(
                _OP.SNAPSHOT_PIPELINE,
                _OP.PIPELINE_BRIDGE,
                _OP.TRANSITION,
                _OP.FORECAST_ACCURACY,
            ),
        ),
        _d(
            BusinessConcept.CREATED_DATE,
            "Created date",
            "When the opportunity was created. The default basis for 'created in "
            "period' (5.3 ambiguity 1). Not required for any operation: a dataset "
            "without it falls back to first appearance in a snapshot, which is a "
            "different question and is reported as such.",
            SemanticType.DATE,
            DataType.DATE,
            ConceptRole.TEMPORAL,
        ),
        _d(
            BusinessConcept.STAGE,
            "Stage",
            "Sales stage label in this snapshot. An attribute, never authoritative "
            "for open, won, or lost: a vocabulary can encode an outcome in a label.",
            SemanticType.CATEGORY,
            DataType.VARCHAR,
            ConceptRole.DIMENSION,
            required_for=(_OP.TRANSITION,),
        ),
        _d(
            BusinessConcept.OPPORTUNITY_STATUS,
            "Status",
            "Authoritative open, won, lost, or excluded state in this snapshot. "
            "Five-valued including unknown, because a deleted record is neither "
            "open nor closed.",
            SemanticType.CATEGORY,
            DataType.VARCHAR,
            ConceptRole.DIMENSION,
            required_for=(_OP.WIN_RATE, _OP.PIPELINE_BRIDGE, _OP.COHORT_TRACE),
        ),
        _d(
            BusinessConcept.FORECAST_CATEGORY,
            "Forecast category",
            "Commit, Best Case, Pipeline, or Omitted, as recorded in this snapshot.",
            SemanticType.CATEGORY,
            DataType.VARCHAR,
            ConceptRole.DIMENSION,
            required_for=(_OP.FORECAST_ACCURACY,),
        ),
        _d(
            BusinessConcept.TERMINAL_OUTCOME,
            "Terminal outcome",
            "What eventually happened to the opportunity. Depends on observations "
            "after the snapshot, so it is retrospective only.",
            SemanticType.OUTCOME,
            DataType.VARCHAR,
            ConceptRole.OUTCOME,
            # Not required for COHORT_TRACE. A trace resolves each cohort
            # member's fate from the *status* concept across the snapshots it
            # already has, which is why a tenant with no terminal column can
            # still be traced. A terminal outcome is the reconciliation target
            # for a trace, never an input to one (5.12): requiring it here
            # would have made the primitive unavailable on exactly the datasets
            # it was designed to serve. Forecast accuracy genuinely needs the
            # realized outcome and keeps the requirement.
            required_for=(_OP.FORECAST_ACCURACY,),
            retrospective=True,
        ),
        _d(
            BusinessConcept.TERMINAL_DATE,
            "Terminal date",
            "The date the opportunity actually reached its terminal state. "
            "Retrospective only.",
            SemanticType.DATE,
            DataType.DATE,
            ConceptRole.OUTCOME,
            retrospective=True,
        ),
        _d(
            BusinessConcept.TERMINAL_AMOUNT,
            "Terminal amount",
            "The realized value at close. Monetary and retrospective: it is the "
            "reconciliation target for a trace, never an input to a forecast.",
            SemanticType.MONEY,
            DataType.DECIMAL,
            ConceptRole.OUTCOME,
            retrospective=True,
        ),
        _d(
            BusinessConcept.QUARTER,
            "Quarter",
            "The fiscal quarter a date falls in. Computed by the fiscal calendar "
            "from a date concept, never read from a stamped label, because a "
            "stamped quarter may have been derived from the final close date.",
            SemanticType.PERIOD,
            DataType.VARCHAR,
            ConceptRole.TEMPORAL,
            computed=True,
            derived_from=BusinessConcept.SNAPSHOT_DATE,
        ),
        _d(
            BusinessConcept.CUSTOMER_SEGMENT,
            "Customer segment",
            "The customer segment recorded in this snapshot. Mutable, so segment "
            "analysis must state which snapshot's segment it attributed by.",
            SemanticType.CATEGORY,
            DataType.VARCHAR,
            ConceptRole.DIMENSION,
            required_for=(_OP.SEGMENT_ANALYSIS,),
        ),
        _d(
            BusinessConcept.NARRATIVE_TEXT,
            "Narrative text",
            "Free-text CRM fields such as manager notes and next steps. Catalogued "
            "and profiled only: no content is read, embedded, or retrieved.",
            SemanticType.TEXT,
            DataType.VARCHAR,
            ConceptRole.NARRATIVE,
            cardinality=ConceptCardinality.MANY,
            required_for=(_OP.TEXT_CATALOGUE,),
        ),
    ]
}

GRAIN_CONCEPTS: tuple[BusinessConcept, ...] = (
    BusinessConcept.OPPORTUNITY_ID,
    BusinessConcept.SNAPSHOT_DATE,
)

MONETARY_CONCEPTS: frozenset[BusinessConcept] = frozenset(
    c for c, d in CONCEPTS.items() if d.is_monetary
)

RETROSPECTIVE_CONCEPTS: frozenset[BusinessConcept] = frozenset(
    c for c, d in CONCEPTS.items() if d.retrospective
)


def definition(concept: BusinessConcept) -> ConceptDefinition:
    return CONCEPTS[concept]


def concepts_required_for(operation: AnalyticalOperation) -> list[BusinessConcept]:
    """Every concept without which an operation cannot be computed."""
    return [c for c, d in CONCEPTS.items() if operation in d.required_for]


class ConceptRequirement(BaseModel):
    """Whether one operation's concepts are all bound in a dataset."""

    model_config = ConfigDict(frozen=True)

    operation: AnalyticalOperation
    satisfied: bool
    missing: tuple[BusinessConcept, ...] = Field(default_factory=tuple)
    caveated: tuple[BusinessConcept, ...] = Field(default_factory=tuple)

    @property
    def reasons(self) -> tuple[str, ...]:
        """One reason per missing concept, each naming a single concept."""
        return tuple(f"{c.value} concept unavailable" for c in self.missing)

    @property
    def reason(self) -> str:
        """The reasons joined, so a single missing concept reads exactly as specified."""
        return "; ".join(self.reasons)
