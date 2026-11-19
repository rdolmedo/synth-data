"""Every M5 Pi trace launcher passes the eval's 2 h agent wall cap (no train/test gap: #378 train/test audit,
2026-11-14; AA pi_eval readouts run with --agent-wall-cap-sec 7200)."""
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


@pytest.mark.parametrize('launcher', ['automationbench/traces.py', 'gdpval/traces.py', 'critpt/traces.py',
                                      'tbsci_lite/traces.py'])
def test_launcher_passes_the_eval_wall_cap(launcher):
    text = (HERE / launcher).read_text()
    assert "'--agent-wall-cap-sec', '7200'" in text
