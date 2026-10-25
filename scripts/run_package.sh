#!/bin/bash
# usage: scripts/run_package.sh <id list file>   (resumable: tasks with an offline verdict are skipped)
cd /home/claude/g4res
cat "$1" | xargs -P 2 -I{} timeout 2400 python3 -I scripts/package_task.py data/verify {} --release release 2>>data/package.log
echo PACKAGE_DONE "$1" >> data/package.log
