# Windows PDS Inference Runbook

This runbook records how PDS request synchronization was tested on the local
Windows machine. It focuses on practical pitfalls that can block another agent
from reproducing the two-way-ready request test.

## Environment Used

Repository:

```text
H:\github\mix-sglang
```

Python environment:

```text
C:\tmp\sglang-qwen3-venv\Scripts\python.exe
```

Model path:

```text
C:\tmp\hf-cache\hub\models--Qwen--Qwen3-0.6B\snapshots\c1899de289a04d12100db370d81485cdf75e47ca
```

The tested server used the Transformers model path and pure PyTorch fallback
components:

- `--model-impl transformers`
- `--attention-backend torch_native`
- `--sampling-backend pytorch`
- `--grammar-backend none`
- `--disable-cuda-graph`
- `--disable-overlap-schedule`

These flags avoid Linux-only or custom-kernel assumptions that are fragile on
this Windows host.

## Known Local Workspace State

There are local Windows-only compatibility edits in several files outside the
PDS scheduler code. Keep them separate from PDS commits unless the user asks to
commit Windows support work:

```text
python/sglang/jit_kernel/clamp_position.py
python/sglang/jit_kernel/utils.py
python/sglang/srt/configs/cohere2_moe.py
python/sglang/srt/layers/layernorm.py
python/sglang/srt/model_executor/model_runner.py
python/sglang/srt/server_args.py
```

The PDS synchronization code itself was committed separately in:

```text
00703e622 Support PDS request sample synchronization
```

Do not use `git add .` in this workspace if the goal is a clean PDS-only
commit.

## Server Launch Pattern

Use a PowerShell command that first normalizes the duplicated `Path`/`PATH`
environment keys. Without this, `Start-Process` can fail with:

```text
Start-Process : 已添加项。字典中的关键字:“Path”所添加的关键字:“PATH”
```

The important setup is:

```powershell
$pathVal=[Environment]::GetEnvironmentVariable('Path','Process')
[Environment]::SetEnvironmentVariable('PATH',$null,'Process')
[Environment]::SetEnvironmentVariable('Path',$pathVal,'Process')

$root='H:\github\mix-sglang'
$python='C:\tmp\sglang-qwen3-venv\Scripts\python.exe'
$model='C:\tmp\hf-cache\hub\models--Qwen--Qwen3-0.6B\snapshots\c1899de289a04d12100db370d81485cdf75e47ca'
$port=29220

$env:PYTHONPATH=(Join-Path $root 'python') + ';C:\tmp\sp'
$env:SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK='1'
$env:FLASHINFER_WORKSPACE_BASE=$root
$env:TVM_FFI_CACHE_DIR=(Join-Path $root '.tmp_pds_run\tvm_cache')
```

Then launch:

```powershell
$p=Start-Process -FilePath $python -ArgumentList @(
  '-m','sglang.launch_server',
  '--model-path',$model,
  '--served-model-name','qwen3-0.6b',
  '--host','127.0.0.1',
  '--port',[string]$port,
  '--model-impl','transformers',
  '--attention-backend','torch_native',
  '--sampling-backend','pytorch',
  '--grammar-backend','none',
  '--disable-cuda-graph',
  '--disable-overlap-schedule',
  '--reasoning-parser','qwen3',
  '--dtype','float16',
  '--mem-fraction-static','0.60',
  '--max-running-requests','4',
  '--skip-server-warmup'
) -WorkingDirectory $root `
  -WindowStyle Hidden `
  -RedirectStandardOutput (Join-Path $root '.tmp_pds_run\pds-sync-final-server.out.log') `
  -RedirectStandardError (Join-Path $root '.tmp_pds_run\pds-sync-final-server.err.log') `
  -PassThru
```

Wait for readiness by polling:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:$port/v1/models" -TimeoutSec 2
```

Always stop the server and child processes after the test:

```powershell
$children=Get-CimInstance Win32_Process -Filter "ParentProcessId=$($p.Id)" -ErrorAction SilentlyContinue
Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
$children | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
```

## Request Client

Prefer Python or `Invoke-WebRequest -UseBasicParsing`. Plain PowerShell
`Invoke-WebRequest` on Windows PowerShell 5 can fail after a successful HTTP
200 with:

```text
无法分析响应内容，因为 Internet Explorer 引擎不可用
```

Python also avoids mojibake when saving the response. This client sends the
two-way synchronized request and saves raw UTF-8 bytes:

```powershell
$env:PDS_TEST_PORT=[string]$port
$env:PDS_TEST_ROOT=$root
$env:PDS_TEST_MODEL=$model
$env:PYTHONIOENCODING='utf-8'

& $python -c @'
import json
import os
import pathlib
import urllib.request

from transformers import AutoTokenizer

port = os.environ["PDS_TEST_PORT"]
root = pathlib.Path(os.environ["PDS_TEST_ROOT"])
model = os.environ["PDS_TEST_MODEL"]
tok = AutoTokenizer.from_pretrained(model)

body = {
    "rid": ["reqA", "reqB"],
    "text": ["1+1等于几？只回答数字。", "请写一首五言古诗，四句即可。"],
    "sampling_params": [
        {
            "max_new_tokens": 8,
            "temperature": 0,
            "ignore_eos": True,
            "skip_special_tokens": True,
            "custom_params": {"__pds_wait_for_rids": ["reqB"]},
        },
        {
            "max_new_tokens": 32,
            "temperature": 0,
            "ignore_eos": True,
            "skip_special_tokens": True,
            "custom_params": {"__pds_wait_for_rids": ["reqA"]},
        },
    ],
}

data = json.dumps(body, ensure_ascii=False).encode("utf-8")
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/generate",
    data=data,
    headers={"Content-Type": "application/json; charset=utf-8"},
    method="POST",
)
raw = urllib.request.urlopen(req, timeout=240).read()
path = root / ".tmp_pds_run" / "pds_sync_two_req_results.json"
path.write_bytes(raw)
parsed = json.loads(raw.decode("utf-8"))
print("saved", path)
for item in parsed:
    print(
        item["meta_info"]["id"],
        item["text"],
        "| decode=",
        tok.decode(item["output_ids"], skip_special_tokens=True),
        "| finish=",
        item["meta_info"]["finish_reason"],
    )
'@
```

Expected result:

- HTTP request returns successfully.
- Both `reqA` and `reqB` appear in the JSON response.
- `meta_info.id` preserves `reqA` and `reqB`.
- The server log has no `Scheduler hit an exception`.
- The server log has no `req_to_token_pool memory leak detected`.

The final checked output path was:

```text
H:\github\mix-sglang\.tmp_pds_run\pds_sync_two_req_results.json
```

## Windows-Specific Pitfalls

### PowerShell Response Mojibake

If `Invoke-WebRequest` output is printed directly, Chinese text may appear as
mojibake such as:

```text
ä¸å¸¦...
```

This can be a client decoding problem, not a server problem. Save raw bytes with
Python, then decode as UTF-8. Also decode `output_ids` with the tokenizer to
separate encoding issues from model quality issues.

### JIT KV-Cache Kernel Failure

The logs can show:

```text
Failed to load JIT KV-Cache kernel
nvcc fatal : A single input file is required for a non-link phase when an outputfile is specified
```

For this fallback run, the server continued and the test still completed. Do
not chase this first if the HTTP request succeeds and the scheduler has no
exception.

### NCCL Is Not Available

On this Windows PyTorch build, logs include:

```text
NCCL is not available in this PyTorch build. Falling back to gloo
```

This is expected for the local single-GPU fallback test.

### `/dev/shm` Snapshot Warning

Logs may include:

```text
load snapshot writer init failed: [Errno 2] No such file or directory: '/dev/shm/...'
```

This is Linux shared-memory infrastructure and was not blocking for the local
test.

### Model Quality Versus Scheduler Correctness

Qwen3-0.6B on this Windows Transformers fallback can return weak or irrelevant
answers for short prompts. For this PDS test, the correctness signal is that the
two synchronized requests both finish without deadlock, shape mismatch, or pool
leak. Do not treat "reqA did not answer exactly 2" as proof that PDS scheduling
failed.

## Debugging Checklist

If the two-way-ready request hangs:

1. Check whether one request is waiting for a rid that is still active but never
   reaches `pending_sample_reqs`.
2. Check for `Scheduler hit an exception` in the stderr log.
3. If a shape mismatch mentions prompt length versus batch size, verify live
   batch input staging was cleared when parking PDS.
4. If `req_to_token_pool memory leak detected` appears, verify pending sample
   requests are counted as non-idle and that re-admission uses a stable pending
   batch snapshot.
5. Save raw response bytes with Python before diagnosing mojibake.
