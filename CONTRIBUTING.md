# Contributing

This repository is generated from a private development trunk by an export script: each release is one commit,
"Export of private main <sha>", and `CORE_VERSION` names the trunk commit it came from.

Pull requests are welcome here. We review them on GitHub, then import the change into the trunk (paths map 1:1),
and it comes back in the next export with you as co-author. Please keep PRs small and include a test.

Run the unit tests with:

    pip install -r requirements-test.txt
    python -m pytest -q

Tests that need our cluster (HTCondor, shared filesystems, GPUs, container images) are listed in
`CLUSTER_ONLY_TESTS` and skipped automatically.
