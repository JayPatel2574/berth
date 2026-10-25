#!/bin/bash
# Gold replay x3 per task, no network, portable Python 3.13 + the bundled uv (the Kaggle setup). Resumable.
cd /home/claude/g4res
mkdir -p /tmp/g4tools && tar -xzf kaggle_tools/python-3.13.tar.gz -C /tmp/g4tools
unshare -rn python3 -I scripts/run_gemma.py --release release --run stability --out runs --model gold --samples 3 \
  --workers 2 --python /tmp/g4tools/cpython-3.13.14-linux-x86_64-gnu/bin/python3.13 --uv kaggle_tools/uv \
  --wheels release/wheels --cache /home/claude/uvcache_stab --tasks $(cat data/stab_ids.txt) > data/stability.log 2>&1
echo STABILITY_DONE >> data/stability.log
