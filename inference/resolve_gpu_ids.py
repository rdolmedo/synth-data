#!/usr/bin/env python3
"""Expand only Condor-assigned UUID prefixes for vLLM's NVML calls.

CUDA accepts abbreviated UUIDs; NVML nvmlDeviceGetHandleByUUID requires the
complete UUID. Never substitute an unassigned GPU when expansion fails.
"""
import sys

import pynvml


def main():
    assigned=sys.argv[1].split(',')
    pynvml.nvmlInit()
    available=[]
    try:
        for index in range(pynvml.nvmlDeviceGetCount()):
            try:
                uuid=pynvml.nvmlDeviceGetUUID(pynvml.nvmlDeviceGetHandleByIndex(index))
                if isinstance(uuid,bytes):uuid=uuid.decode()
                available.append(uuid)
            except pynvml.NVMLError:
                continue
        full=[]
        for item in assigned:
            if item.startswith('GPU-'):
                matches=[uuid for uuid in available if uuid.startswith(item)]
                if len(matches)!=1:
                    raise RuntimeError(f'Cannot uniquely expand assigned GPU {item}: {matches}')
                full.append(matches[0])
            else:
                full.append(item)
        if len(set(full))!=len(assigned):raise RuntimeError('Duplicate assigned GPU identifiers')
        print(','.join(full))
    finally:
        pynvml.nvmlShutdown()


if __name__=='__main__':main()
