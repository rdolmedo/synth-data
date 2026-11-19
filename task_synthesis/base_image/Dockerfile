# tasksynth-base: the one lean, general image every synthesized task builds FROM (BLUEPRINT.md, Execution environment).
# Standard command-line tools and a small Python stack; anything task-specific goes in the task's thin layer.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 LC_ALL=C.UTF-8 TZ=Etc/UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl wget git openssh-client rsync \
        tmux vim nano less tree file procps psmisc lsof iproute2 iputils-ping netcat-openbsd dnsutils \
        unzip zip xz-utils bzip2 zstd patch diffutils bc gawk jq sqlite3 \
        build-essential make cmake pkg-config gdb \
        python3 python3-pip python3-venv python3-dev python-is-python3 \
        time xxd bsdextrautils \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu 24.04 marks the system Python as externally managed (PEP 668). The base stack is installed past that once;
# the marker stays, so agents meet the stock behavior (use a venv or --break-system-packages).
RUN python3 -m pip install --no-cache-dir --break-system-packages \
        numpy pandas scipy requests pyyaml pytest

WORKDIR /app
