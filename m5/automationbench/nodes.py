"""Nodes every submit excludes: aa-serving's BAD_NODES plus the M5 shared list (m5-coord/excluded-nodes.txt),
read at submit time."""
from pathlib import Path

BAD_NODES = ['cpu203', 'cpu205', 'cpu207', 'cpu208', 'gpu110', 'gpu132', 'gpu123', 'gpu143', 'gpu166', 'gpu172', 'gpu106', 'cpu305']
SHARED = Path('/opt/aai/m5-coord/excluded-nodes.txt')


def excluded() -> list[str]:
    extra = []
    try:
        for line in SHARED.read_text().splitlines():
            name = line.split('#', 1)[0].strip()
            if name:
                extra.append(name.split()[0])
    except OSError:
        pass
    return sorted(set(BAD_NODES) | set(extra))
