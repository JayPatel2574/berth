"""Write the Kaggle notebook that runs Gemma 4 (the competition checkpoint, served by vLLM) on our tasks.

Also writes kernel-metadata.json next to the notebook, for `kaggle kernels push`. The 4x L4 machine is NOT offered
in the web editor; it is provisioned only through the API with the competition attached as a source and
machine_shape "NvidiaL4" (found by github.com/nakranivaibhav/gemma-4-developer-agent, journal 2026-10-15). Inputs:
  - competition: gemma-4-developer-agent, plus the dataset metric/gemma-4-developer-agent-wheelhouse (vLLM wheels)
  - model: google/gemma-4/Other/<model>/2
  - our private dataset (kaggle_bundle/: tasks.jsonl, snapshots/, wheels/, tools/, run_gemma.py)
  - optional, to resume: an earlier version of this kernel (--resume-from owner/slug)

The vLLM install and server settings copy the official Getting Started notebook as reproduced in
github.com/nakranivaibhav/gemma-4-developer-agent (kits/swe-agent/bench/kaggle_speed_notebooks.py, Apache-2.0).

usage: python -I build_kaggle_notebook.py <out dir> --owner <kaggle user> --dataset <owner/slug> [--slug ...]
         [--model gemma-4-31b-it-qat-w4a16-ct] [--budget-min 630] [--resume-from owner/slug]
"""
import argparse
import json
from pathlib import Path


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")}


def cells(model, budget_min, run_name, workers, samples, split):
    return [
md(f"""
# Gemma 4 on harness-compatible tasks (difficulty labels + trajectories)

Serves **{model}** with vLLM on 4x L4 (internet off), then runs a small harness-like agent on our tasks and grades
each patch with the tasks' hidden tests, fully offline (task environments install from the bundled wheels).

**Before running:** Settings -> Accelerator **GPU L4 x4**, Internet **off**; inputs attached: the competition data,
the model, our dataset, and (to resume) the previous version's output. Use *Save Version -> Save & Run All*.
"""),
code(f"""
import time
T0 = time.time()
MODEL_NAME = '{model}'
RUN_NAME = '{run_name}'
BUDGET_MIN = {budget_min}       # start no new task after this many minutes of the agent run
WORKERS = {workers}              # agents in parallel (vLLM batches their requests)
SAMPLES = {samples}
SPLIT = '{split}'                # which tasks from tasks.jsonl to run

# stop early (and spend no quota) if Kaggle fell back to another machine: the 31B needs 4x L4
import subprocess
gpus = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                      capture_output=True, text=True).stdout.strip().splitlines()
print('GPUs:', gpus)
assert len(gpus) == 4 and all('L4' in g for g in gpus), f'expected 4x NVIDIA L4, got {{gpus}}'
"""),
md("## 1. Install the competition's vLLM / harness wheels (offline)"),
code("""
import glob, importlib, json, logging, os, shutil, subprocess, sys
from pathlib import Path

os.environ.update({
    'TRANSFORMERS_NO_TF': '1', 'TRANSFORMERS_NO_FLAX': '1', 'TRANSFORMERS_OFFLINE': '1',
    'VLLM_NO_USAGE_STATS': '1', 'VLLM_WORKER_MULTIPROC_METHOD': 'spawn',
    'VLLM_ENGINE_READY_TIMEOUT_S': '1200', 'VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS': '1',
    'NCCL_DEBUG': 'WARN', 'OPENAI_API_KEY': 'EMPTY', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
    'OTEL_SDK_DISABLED': 'true',
})
for name in ('httpx', 'httpcore', 'openai', 'urllib3'):
    logging.getLogger(name).setLevel(logging.WARNING)

# the wheelhouse folder: the input folder that holds the vllm wheel
vllm_whl = sorted(glob.glob('/kaggle/input/**/vllm-*.whl', recursive=True))
assert vllm_whl, 'competition wheelhouse not found: attach the competition data (gemma-4-developer-agent)'
WHEELHOUSE_DIR = Path(vllm_whl[0]).parent
print('wheelhouse:', WHEELHOUSE_DIR)
for pth in glob.glob('/usr/local/lib/python*/*-packages/*cutlass*.pth'):
    os.unlink(pth)
tmp_whl = Path('/tmp/wheelhouse'); tmp_whl.mkdir(parents=True, exist_ok=True)
for w in WHEELHOUSE_DIR.glob('*.whl'):
    if 'cutlass' in w.name.lower():
        continue
    name = w.name.replace('cu128', '+cu128') if ('cu128' in w.name and '+' not in w.name) else w.name  # Kaggle strips '+'
    if not (tmp_whl / name).exists():
        os.symlink(w, tmp_whl / name)
t = time.time()
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '--force-reinstall',
                *sorted(str(w) for w in tmp_whl.glob('*.whl'))], check=True)
importlib.invalidate_caches()
print(f'installed wheels in {time.time() - t:.0f} s')
"""),
md("## 2. Find the inputs"),
code("""
def find_one(pattern, what):
    hits = sorted(glob.glob(pattern, recursive=True))
    assert hits, f'{what} not found ({pattern})'
    return Path(hits[0])

MODEL_PATH = find_one(f'/kaggle/input/**/{MODEL_NAME}/**/config.json', 'model').parent
BUNDLE = find_one('/kaggle/input/**/g4res_bundle.json', 'our dataset').parent
WORK = Path('/kaggle/working')
print('model :', MODEL_PATH)
print('bundle:', BUNDLE, json.loads((BUNDLE / 'g4res_bundle.json').read_text()))

# resume: copy finished results from an attached earlier output of this notebook
OUT = WORK / 'runs'
prev = [Path(p).parent for p in glob.glob(f'/kaggle/input/**/runs/{RUN_NAME}/*__s*.json', recursive=True)]
for d in sorted(set(prev)):
    dst = OUT / RUN_NAME; dst.mkdir(parents=True, exist_ok=True)
    for f in d.iterdir():
        if not (dst / f.name).exists():
            shutil.copy(f, dst / f.name)
print('resumed results:', len(list((OUT / RUN_NAME).glob('*__s*.json'))) if (OUT / RUN_NAME).exists() else 0)
"""),
md("## 3. Tools for the task environments: portable Python 3.13 + uv (bundled)"),
code("""
TOOLS = Path('/tmp/g4tools'); TOOLS.mkdir(exist_ok=True)
subprocess.run(['tar', '-xzf', str(BUNDLE / 'tools/python-3.13.pyarch'), '-C', str(TOOLS)], check=True)
PY313 = next(TOOLS.glob('*/bin/python3.13'))
UV = TOOLS / 'uv'; shutil.copy(BUNDLE / 'tools/uv', UV); UV.chmod(0o755)
subprocess.run(['tar', '-xf', str(BUNDLE / 'tools/wheels.pyarch'), '-C', '/tmp'], check=True)
WHEELS = Path('/tmp/wheels'); print(len(list(WHEELS.glob('*.whl'))), 'task wheels')
print(subprocess.run([str(PY313), '--version'], capture_output=True, text=True).stdout,
      subprocess.run([str(UV), '--version'], capture_output=True, text=True).stdout)
tasks = [json.loads(l) for l in (BUNDLE / 'tasks.jsonl').read_text().splitlines() if l.strip()]
run_ids = [t['instance_id'] for t in tasks if SPLIT in ('all', t['split'])]
print(len(tasks), 'tasks in bundle;', len(run_ids), 'in split', SPLIT)
"""),
md("## 4. Start the model server (official Getting Started settings, no LoRA)"),
code("""
import requests, torch
from adk_submission import VllmConfig, VllmServer

EXTRA = {'gemma-4-12b-it-qat-w4a16-ct': ('--hf-overrides', '{"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}')}
cfg = VllmConfig(
    model=str(MODEL_PATH), port=8000, host='127.0.0.1',
    tool_call_parser='gemma4', reasoning_parser='gemma4',
    max_model_len=32768, dtype='bfloat16', enable_auto_tool_choice=True,
    scheduler_reserve_full_isl=False, tensor_parallel_size=torch.cuda.device_count(),
    startup_timeout=60 * 20, gpu_memory_utilization=0.90, enable_lora=False, max_loras=1, max_lora_rank=128,
    default_chat_template_kwargs={'enable_thinking': True},
    extra_args=(*EXTRA.get(MODEL_NAME, ()), '--max-num-seqs', str(max(8, 2 * WORKERS))),
)
t = time.time()
server = VllmServer(cfg); server.start()
SERVED = requests.get('http://127.0.0.1:8000/v1/models', timeout=30).json()['data'][0]['id']
print(f'server ready in {time.time() - t:.0f} s; serving', SERVED)
"""),
md("## 5. Run the agent on the tasks (resumable; stops starting new tasks at the budget)"),
code("""
log = WORK / f'{RUN_NAME}.log'
cmd = [sys.executable, '-u', str(BUNDLE / 'run_gemma.py'), '--release', str(BUNDLE), '--run', RUN_NAME,
       '--out', str(OUT), '--model', SERVED, '--endpoint', 'http://127.0.0.1:8000/v1/chat/completions',
       '--thinking', '--max-tokens', '4096', '--rpm', '100000', '--workers', str(WORKERS),
       '--samples', str(SAMPLES), '--python', str(PY313), '--uv', str(UV), '--wheels', str(WHEELS),
       '--cache', '/tmp/uvcache', '--deadline-min', str(BUDGET_MIN), '--tasks', *run_ids]
t = time.time()
with open(log, 'w') as fh:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        fh.write(line); fh.flush()
        if ': done ' in line or ': error' in line or 'resolved (' in line or 'Traceback' in line:
            print(f'[{(time.time() - t) / 60:6.1f} min] {line.rstrip()}', flush=True)
    p.wait()
print('agent run finished, exit', p.returncode)
"""),
md("## 6. Summary"),
code("""
import collections
res = [json.loads(f.read_text()) for f in (OUT / RUN_NAME).glob('*__s*.json')]
by = collections.defaultdict(list)
for r in res:
    by[r['instance_id']].append(bool(r.get('resolved')))
print(f'{len(res)} graded runs over {len(by)} tasks; resolved {sum(map(any, by.values()))} tasks at least once; '
      f'pass rate {sum(sum(v) for v in by.values()) / max(len(res), 1):.3f}')
server.stop()
print(f'total {(time.time() - T0) / 60:.0f} min')
"""),
    ]


STARTER_IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--owner", required=True)
    ap.add_argument("--dataset", required=True, help="owner/slug of the uploaded bundle")
    ap.add_argument("--slug", default="g4res-gemma-run")
    ap.add_argument("--title", default="g4res gemma run")
    ap.add_argument("--resume-from", default=None, help="owner/slug of an earlier run whose output to continue")
    ap.add_argument("--model", default="gemma-4-31b-it-qat-w4a16-ct")
    ap.add_argument("--budget-min", type=int, default=630)
    ap.add_argument("--run-name", default="g31_core_t07")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--split", default="core")
    a = ap.parse_args()
    nb = {"cells": cells(a.model, a.budget_min, a.run_name, a.workers, a.samples, a.split),
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "notebook.ipynb").write_text(json.dumps(nb, indent=1))
    meta = {"id": f"{a.owner}/{a.slug}", "title": a.title, "code_file": "notebook.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": False, "dataset_sources": ["metric/gemma-4-developer-agent-wheelhouse", a.dataset],
            "kernel_sources": [a.resume_from] if a.resume_from else [],
            "competition_sources": ["gemma-4-developer-agent"],
            "model_sources": [f"google/gemma-4/Other/{a.model}/2"],
            "docker_image": STARTER_IMAGE, "machine_shape": "NvidiaL4"}
    (a.out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print("wrote", a.out, meta["id"])


if __name__ == "__main__":
    main()
