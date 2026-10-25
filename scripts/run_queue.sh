#!/bin/bash
# usage: scripts/run_queue.sh <queue file> <out dir>   (resumable: finished tasks are skipped)
cd /home/claude/g4res
cat "$1" | xargs -P 2 -I{} timeout 1800 python3 -I scripts/verify_task.py data/candidates.jsonl {} --out "$2" 2>>data/verify.log
echo QUEUE_DONE >> data/verify.log
