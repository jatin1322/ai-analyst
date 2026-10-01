"""Type coercion and stage classification."""

from __future__ import annotations

import pytest

from ai_analyst.data.conform import classify_stage, quote_ident, quote_literal


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        ("Closed Won", (True, True)),
        ("closed won", (True, True)),
        ("Won", (True, True)),
        ("Closed Lost", (True, False)),
        ("Lost", (True, False)),
        ("Disqualified", (True, False)),
        ("No Decision", (True, False)),
        ("Discovery", (False, False)),
        ("Qualification", (False, False)),
        ("Proposal", (False, False)),
        ("Negotiation", (False, False)),
        (None, (False, False)),
    ],
)
def test_classify_stage(stage, expected):
    assert classify_stage(stage) == expected


def test_lost_wins_over_won_for_ambiguous_labels():
    # "Closed Lost" contains neither "won" nor "win", but a label such as
    # "Won Then Lost" must not be counted as a win.
    assert classify_stage("Won Then Lost") == (True, False)


def test_classify_stage_ignores_surrounding_whitespace_and_case():
    assert classify_stage("  CLOSED WON  ") == (True, True)


def test_identifier_quoting_escapes_embedded_quotes():
    assert quote_ident("plain") == '"plain"'
    assert quote_ident('we"ird') == '"we""ird"'


def test_literal_quoting_escapes_apostrophes():
    assert quote_literal("O'Brien") == "'O''Brien'"
