"""Guards on the pinned shared dependencies."""

from __future__ import annotations

import re
from importlib.metadata import version


def test_portfolio_common_pin_includes_the_live_writer_attach_fix() -> None:
    # v1.2.2 is the first portfolio-common whose read-only ATTACH accepts a
    # SOURCE held open by a live writer. Guard against the pin being lowered.
    # Only the leading X.Y.Z is compared, so a prerelease/local suffix
    # (1.3.0rc1, 1.2.2+local) cannot make this fail for an unrelated reason.
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", version("portfolio-common"))
    assert match is not None
    assert tuple(int(g) for g in match.groups()) >= (1, 2, 2)
