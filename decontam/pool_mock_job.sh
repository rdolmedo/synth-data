#!/bin/bash
# CPU job: decontam.pool_mock WORK SOURCE_STREAM [CHUNKS]
exec env DECONTAM_MODULE=decontam.pool_mock "$(dirname "$0")/pool_job.sh" --work "$1" --source "$2" --chunks "${3:-4}"
