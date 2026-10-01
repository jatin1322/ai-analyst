"""Compatibility shim: the pack now lives in `contracts.example_packs`.

It is an example for one export shape, not a default. Onboarding never uses it.
"""

import sys

from ai_analyst.contracts.example_packs import opportunity_snapshot_v1 as _pack

sys.modules[__name__] = _pack
