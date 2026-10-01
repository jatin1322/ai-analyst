"""The target export acceptance pass (ARCHITECTURE 12.19). Opt-in.

The target export lives in S3 and has never been read by this project: no AWS
credentials were available through the standard chain. That is recorded as the
`target_export_unprobed` open question, not papered over.

The always-on tests pin the two things that must hold without credentials: the
probe script refuses cleanly and claims nothing, and the open question is on
record. The opt-in test runs the probe when `AI_ANALYST_TARGET_EXPORT` names the
export and credentials resolve through the standard chain; it asserts only that
the probe completes with aggregates, and records nothing as resolved. The
eoq_close_diff and CD_in_qtr conventions, the fiscal calendar, the
authoritative status and the date declarations stay open whatever it finds.

Credentials are never an argument, an environment variable read here, or a
file this test writes.
"""

from __future__ import annotations

import json
import os

import pytest

from ai_analyst.contracts.opportunity_snapshot_v1 import OPPORTUNITY_SNAPSHOT_V1
from ai_analyst.data import probe as probe_module
from ai_analyst.data.probe import ProbeAccessError

STILL_OPEN = (
    "target_export_unprobed",
    "eoq_close_diff_convention",
    "quarter_boundary_day_convention",
    "fiscal_year_start_month",
    "date_encoding_declarations",
)


def test_the_unprobed_target_export_is_an_open_question():
    open_ids = {u.id for u in OPPORTUNITY_SNAPSHOT_V1.unresolved if not u.resolved}
    assert set(STILL_OPEN) <= open_ids


def test_the_probe_script_refuses_without_credentials_and_claims_nothing(
    monkeypatch, capsys
):
    import scripts.probe_production as script

    def unreachable(uri, **kwargs):
        raise ProbeAccessError("no usable AWS credentials in the standard chain")

    monkeypatch.setattr(script, "probe_export", unreachable)
    assert script.main(["s3://bucket/prefix/snapshot.parquet/"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("probe not run:")


def test_the_probe_accepts_no_credential_argument():
    import scripts.probe_production as script

    with pytest.raises(SystemExit):
        script.main(["s3://bucket/x/", "--aws-secret-access-key", "x"])


TARGET = os.environ.get("AI_ANALYST_TARGET_EXPORT")


@pytest.mark.skipif(not TARGET, reason="set AI_ANALYST_TARGET_EXPORT to the target's s3:// URI")
def test_target_export_probe_reports_aggregates_only():
    try:
        report = probe_module.probe_export(TARGET)
    except ProbeAccessError as exc:
        pytest.skip(f"target export unreachable: {exc}")
    as_json = json.loads(report.model_dump_json())
    assert report.extent.rows > 0
    # Status fields are candidates, never a mapping.
    assert "mapping" not in json.dumps(as_json["status_candidates"]).lower()
    # Nothing the probe returns resolves an open convention.
    assert {u.id for u in OPPORTUNITY_SNAPSHOT_V1.unresolved if not u.resolved} >= set(STILL_OPEN)
