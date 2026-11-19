"""M3.4 trace judge (#177): flags infra failures in a model's traces. See README.md.

    from aa_evals.tracejudge import fold_summary
    summary = fold_summary(run_dir, summary)   # adds n_flagged, n_flagged_by_category, tracejudge
"""
from aa_evals.tracejudge.hook import fold_file, fold_summary  # noqa: F401
