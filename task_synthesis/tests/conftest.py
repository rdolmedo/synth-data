"""Tests never read the live cluster through tbshared's fill (condor_cost.condor_status_slots): it is unavailable
here, so fill_plan leaves shapes as given (node class unknown) unless a test passes its own condor_status lines."""
import pytest


@pytest.fixture(autouse=True)
def _no_live_condor_status(monkeypatch):
    from tbshared import condor_cost

    def unavailable(*a, **k):
        raise OSError('condor_status is not read in tests')
    monkeypatch.setattr(condor_cost, 'condor_status_slots', unavailable)


@pytest.fixture(autouse=True)
def _no_serving_root_from_the_shell():
    """Every test runs as from a shell without $TASKSYNTH_SERVING_ROOT (a canary shell would make a Driver refuse a
    production run), and nothing a test sets leaks into the next (Driver / pin_run_root write os.environ directly)."""
    import os
    from unittest import mock
    with mock.patch.dict(os.environ):
        os.environ.pop('TASKSYNTH_SERVING_ROOT', None)
        yield
