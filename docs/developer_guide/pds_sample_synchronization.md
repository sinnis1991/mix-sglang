# PDS Sample Synchronization And Distribution Fusion

This note explains the local PDS implementation in the scheduler. It is meant
for agents that need to understand or extend the code without rediscovering the
same state-machine bugs.

## Goal

PDS means "prefill/decode/sample separation" in this branch. The model forward
stage computes logits but does not sample immediately. The scheduler stores the
logits on the corresponding request, then runs an explicit sample stage before
the next prefill/decode scheduling decision.

The first PDS use case is sample synchronization: one request can wait until
another request also has logits before sampling. The newer use case is
distribution fusion: multiple requests in the same sample group each compute an
independent next-token distribution, the scheduler fuses those distributions,
samples one token from the fused distribution, and writes that same token back to
every request in the group.

For a two-request group with logits `logits_a` and `logits_b`, the current
fusion method is:

```python
p = softmax(filter(logits_a, sampling_params_a))
q = softmax(filter(logits_b, sampling_params_b))
fused_probs = (p + q) / 2
next_token = sample(fused_probs)
```

The same `next_token` is then assigned to both requests. From that point on,
their `output_ids`, `seq_lens`, FutureMap relay token, and KV-cache progression
advance on the same token.

## Request API

PDS uses temporary `sampling_params.custom_params` keys. These are not intended
to be a stable public API yet.

### Wait For Other Requests

```json
{
  "custom_params": {
    "__pds_wait_for_rids": ["reqB"]
  }
}
```

With this parameter, `reqA` can compute logits and enter the pending sample
queue, but its sample step is blocked until `reqB` has also computed logits and
entered the pending sample queue.

### Fused Sample Group

```json
{
  "custom_params": {
    "__pds_sample_group": "group_1",
    "__pds_fuse_method": "avg_probs"
  }
}
```

Requests with the same `__pds_sample_group` and `__pds_fuse_method="avg_probs"`
wait until every live request in the group has pending logits. The scheduler
then fuses their distributions and broadcasts one sampled token to the whole
group.

The optional `__pds_fuse_weight` key can assign a scalar weight to a request.
When no weight is provided, the default is `1.0`, so a two-request group uses a
plain average.

To return selected-token probability trajectories on one request, opt that
request in with `__pds_return_prob_trajectory=true`:

```json
{
  "custom_params": {
    "__pds_sample_group": "group_1",
    "__pds_fuse_method": "avg_probs",
    "__pds_return_prob_trajectory": true
  }
}
```

The response `meta_info` then contains two arrays aligned one-to-one with
`output_ids`:

- `pds_source_token_probs`: the probability of each emitted token under this
  request's filtered source distribution.
- `pds_fused_token_probs`: the probability of the same token under the fused
  distribution used for sampling.

Requests in the same group that do not opt in carry no probability values. In a
mixed output batch, their corresponding customized-info entries are empty so
the scheduler-to-tokenizer batch indices remain aligned.

## Relevant Files

- `python/sglang/srt/managers/scheduler.py`
- `python/sglang/srt/managers/schedule_batch.py`

The `PendingSample` dataclass is in `schedule_batch.py`. It stores:

- `batch`: the batch snapshot used to call `process_batch_result`.
- `active_batch`: the batch that is admitted back into decode scheduling.
- `result`: the `GenerationBatchResult` that still owns logits.
- `forward_batch`: the forward metadata needed for logit slicing.
- `wait_for_rids`: the request IDs whose logits must also be ready.
- `sample_group`: the optional fused-sampling group name.
- `fuse_method`: the optional fusion method, currently `avg_probs`.
- `fuse_weight`: a scalar distribution-fusion weight, defaulting to `1.0`.
- `external_params_required` and `external_params`: a placeholder for future
  externally supplied sample parameters.

Most of the runtime logic is in `scheduler.py`:

- `_park_pending_sample_result`
- `_pending_sample_ready`
- `_run_ready_fused_sample_groups`
- `_pds_distribution_for_req`
- `_sample_from_fused_probs`
- `_sample_pending_result_in_place`
- `_finalize_pending_sample_result`
- `run_pending_sample_stage`
- `_maybe_admit_sampled_batch`

## Control Flow

1. `run_batch` calls model forward with `skip_sample=True`.
2. The model returns `GenerationBatchResult` with `pending_sample=True`.
3. `process_batch_result` calls `_park_pending_sample_result` and returns early.
4. `_park_pending_sample_result` copies enough batch state for deferred result
   processing and later decode admission, then stores `PendingSample` on each
   request.
5. At the beginning of each scheduler event-loop iteration,
   `run_pending_sample_stage` scans `pending_sample_reqs`.
6. `_pending_sample_ready` checks:
   - the request has pending logits,
   - external sample parameters are present if required,
   - every live dependency in `wait_for_rids` has also entered pending sample,
   - every live request in `sample_group` has also entered pending sample.
7. `run_pending_sample_stage` first tries `_run_ready_fused_sample_groups`.
8. Fused groups sample once per group and fill the sampled token into the
   corresponding `GenerationBatchResult.next_token_ids` positions.
9. `_finalize_pending_sample_result` relays the sampled tokens through
   FutureMap, calls `process_batch_result`, and admits unfinished requests back
   into `running_batch`.
10. Any ready pending result that is not handled by the fused path falls back to
    `_sample_pending_result_in_place`, which calls the normal model-runner
    sample path.

## Wait Semantics

The important semantic is "wait for live dependencies only."

If `reqA.wait_for_rids = {"reqB"}`:

- If `reqB` is still active in the scheduler, `reqA` waits until `reqB` is also
  in `pending_sample_reqs`.
- If `reqB` has already finished or left scheduler ownership, `reqA` does not
  wait forever.

The same live-request idea is used for fused sample groups. A request with
`sample_group="group_1"` waits for every live request that still has
`sample_group="group_1"`. If one member finishes and is filtered out of
scheduler-owned batches, later steps do not wait for it forever.

This is implemented by taking two snapshots in `run_pending_sample_stage`:

- `pending_rids_snapshot`: requests that currently have pending logits.
- `active_rids_snapshot`: requests still owned by the scheduler, including
  `pending_sample_reqs`, `waiting_queue`, `cur_batch`, `running_batch`, and
  `last_batch`.

Then `_pending_sample_ready` only requires:

```python
pending.wait_for_rids.intersection(active_rids_snapshot) <= pending_rids_snapshot
```

For sample groups, `_collect_live_sample_group_rids` recomputes the live members
from the same scheduler-owned request sets and requires those rids to be pending.

## Fused Distribution Semantics

The fused path does not average logits. It averages probability distributions.
For each request:

1. `_normalize_pending_sample_logits` reduces extend/prefill logits to the
   per-request last-token logits when the forward output is token-major.
2. `_pds_req_index` maps the request to its row in the saved active batch.
3. `_pds_distribution_for_req` applies temperature, softmax, and the current
   top-k/top-p/min-p filtering behavior to produce a normalized probability
   vector.
4. `_run_ready_fused_sample_groups` stacks those per-request probability vectors
   and computes a weighted average.
5. `_sample_from_fused_probs` samples one token from the fused probability
   vector. If every request is greedy (`top_k == 1`), it returns `argmax`.

The implementation currently assumes the group is compatible enough for this
Python-level fusion path. There is a TODO in `_sample_from_fused_probs` for the
future kernel path where requests in one group intentionally use different
sampling parameters or deterministic seeds.

## Token Broadcast And State Alignment

After sampling, the scheduler writes the same token into every request position
owned by that fused group:

```python
result.next_token_ids[self._pds_req_index(req)] = sampled_token[0]
```

That write is the key alignment point. From there, the rest of SGLang consumes
the token through normal paths:

- `_relay_forward_payload` stashes `GenerationBatchResult.next_token_ids` into
  FutureMap. Therefore the next decode input sees the fused token.
- `process_batch_result_prefill` reads `result.next_token_ids` and appends the
  token to `req.output_ids`.
- `process_batch_result_decode` normalizes `result.next_token_ids` into a
  per-request accepted-token list and extends `req.output_ids`.
- `ScheduleBatch.prepare_for_decode` advances `seq_lens`, `seq_lens_cpu`,
  `orig_seq_lens`, and decode KV allocation after the result has been admitted
  back into decode scheduling.

This means `FutureMap`, `process_batch_result_prefill/decode`,
`req.output_ids`, `seq_lens`, and KV-cache progression are all driven by the
same fused token.

## Mixed-Group Result Handling

A subtle scheduler behavior is that concurrently submitted fused groups can be
batched into the same `GenerationBatchResult`. For example, four concurrent
requests can be scheduled as:

- result 1: one request from group A
- result 2: the other request from group A plus two requests from group B

The early fused implementation could only process a result when every pending
request in that result belonged to the same sample group. In a mixed-result
case, the fused path would refuse to process the result and the normal sampling
fallback could run, causing group members to receive different tokens.

The current implementation fixes that by making `_run_ready_fused_sample_groups`
work across all ready fused groups in a stage:

1. Collect every ready `avg_probs` sample group.
2. Build a map from `GenerationBatchResult` to every pending request stored in
   that result.
3. Refuse to process if a touched result contains an unready fused group. This
   prevents partially initialized `next_token_ids`.
4. Sample one token per ready group.
5. Allocate or reuse `result.next_token_ids` for each touched result.
6. Fill each request row with the sampled token for its own group.
7. Finalize each touched result exactly once.

This lets multiple fused groups be served concurrently while preserving the
"same token inside a group" guarantee.

## Why There Is A Separate Admit Snapshot

A normal `ScheduleBatch.copy()` was originally only intended for
`process_batch_result`. It does not carry all state needed to put the request
back into decode scheduling.

For PDS, a request may wait in `pending_sample_reqs` while the normal scheduler
continues. During that time, the live `last_batch` can be filtered in place.
`filter_batch()` removes requests whose `req.pending_sample is not None`. If PDS
later reused that live batch for decode admission, it could already be empty or
partially mutated.

To avoid that, `_copy_pending_sample_admit_batch` creates an independent
admission snapshot. It starts with `ScheduleBatch.copy()` and then restores the
fields decode scheduling needs, including pool references, `seq_lens`,
`orig_seq_lens`, `req_pool_indices_cpu`, `sampling_info`, and logprob metadata.

Both `PendingSample.batch` and `PendingSample.active_batch` point at this
snapshot. `process_batch_result` uses it to update request state, and
`_maybe_admit_sampled_batch` uses it to put unfinished requests back into
`running_batch`.

## Important Pitfalls Found During Implementation

### 1. Deadlock From Checking Only Current Pending RIDs

The first wait implementation used:

```python
pending.wait_for_rids.issubset(pending_rids)
```

This works only while both requests are still generating. If one request
finishes earlier, the other request can wait forever for a rid that will never
enter pending sample again. The fix is the "live dependency" check described
above.

### 2. Stale Live Batch After Deferred Sampling

When the model forward runs with `skip_sample=True`, `run_batch` does not go
through the normal "sampled token is ready" branch. That means the live prefill
batch can still carry prompt-shaped input tensors such as `input_ids` or
`prefill_input_ids_cpu`.

If that live batch later enters decode scheduling, `resolve_forward_inputs` can
produce an input tensor with prompt length instead of batch size. The observed
failure looked like:

```text
RuntimeError: The size of tensor a (10) must match the size of tensor b (2)
```

The fix is to clear these live-batch input staging fields when parking PDS:

```python
batch.input_ids = None
batch.prefill_input_ids_cpu = None
batch.mix_running_indices = None
```

After the delayed sample runs, the sampled token is relayed through FutureMap,
and the next decode forward resolves a one-token-per-request input.

### 3. Re-admitted Batches Must Become Decode-Running Batches

The pending admission snapshot originates from a prefill/extend batch, but after
sampling it is no longer an extend batch. Before re-admission:

```python
active_batch.is_extend_in_batch = False
active_batch.all_extend_in_batch = False
```

This keeps downstream forward-batch metadata consistent with decode scheduling.

### 4. Idle Checks Must Account For Pending Samples

Pending samples are active scheduler work. If `is_fully_idle()` ignores
`pending_sample_reqs`, idle-time invariant checks can fire while requests are
still parked for sample. The implementation includes:

```python
idle &= len(self.pending_sample_reqs) == 0
```

### 5. Concurrent Fused Groups Can Share One Result

Concurrent HTTP requests are not guaranteed to form one
`GenerationBatchResult` per fused group. The scheduler can mix requests from
different sample groups into the same forward result. The fused path must
therefore fill `result.next_token_ids` by request index, not by assuming one
result equals one group.

The fixed path processes all ready fused groups together and finalizes each
touched result once. This is the reason `_run_ready_fused_sample_groups` owns
the multi-group orchestration instead of calling a simple per-group helper in a
loop.

## Test Request Shapes

### Two-Way Synchronized Waiting

A simple two-way synchronized request uses request IDs and
`__pds_wait_for_rids`:

```json
[
  {
    "rid": "reqA",
    "text": "<chat-template prompt A>",
    "sampling_params": {
      "max_new_tokens": 32,
      "temperature": 0.6,
      "top_p": 0.95,
      "top_k": 20,
      "custom_params": {
        "__pds_wait_for_rids": ["reqB"]
      }
    }
  },
  {
    "rid": "reqB",
    "text": "<chat-template prompt B>",
    "sampling_params": {
      "max_new_tokens": 32,
      "temperature": 0.6,
      "top_p": 0.95,
      "top_k": 20,
      "custom_params": {
        "__pds_wait_for_rids": ["reqA"]
      }
    }
  }
]
```

Successful scheduling is more important than model answer quality for this
test. The scheduler should return both requests without deadlock, shape
mismatch, or req-pool leak.

### Two-Request Distribution Fusion

```json
[
  {
    "rid": "reqA_math",
    "text": "<chat-template: 1+1 equals what>",
    "sampling_params": {
      "temperature": 0.6,
      "top_p": 0.95,
      "top_k": 20,
      "max_new_tokens": 256,
      "custom_params": {
        "__pds_sample_group": "group_math_poem",
        "__pds_fuse_method": "avg_probs"
      }
    }
  },
  {
    "rid": "reqB_poem",
    "text": "<chat-template: write an ancient poem>",
    "sampling_params": {
      "temperature": 0.6,
      "top_p": 0.95,
      "top_k": 20,
      "max_new_tokens": 256,
      "custom_params": {
        "__pds_sample_group": "group_math_poem",
        "__pds_fuse_method": "avg_probs"
      }
    }
  }
]
```

The expected scheduler-level property is:

```python
reqA.output_ids == reqB.output_ids
```

The text may look like a semantic blend of the prompts. For example, a math
prompt and a poem prompt can produce a "math poem."

### Concurrent Two-Group Fusion

A stronger concurrency test sends four HTTP requests at the same time:

- `group_math_poem`: math question plus poem request.
- `group_code_story`: Fibonacci Python-code request plus AI-future sci-fi
  outline request.

Expected scheduler-level properties:

```python
group_math_poem.reqA.output_ids == group_math_poem.reqB.output_ids
group_code_story.reqA.output_ids == group_code_story.reqB.output_ids
group_math_poem.output_ids may differ from group_code_story.output_ids
```

This specifically exercises mixed `GenerationBatchResult` handling. On the
local Windows fallback run, the server scheduled the four requests as a
three-request prefill batch plus a one-request prefill batch, and the fixed
multi-group fused path still preserved same-token output inside each group.

## Verification Used Locally

The fused-sample implementation was checked with:

```text
python -m py_compile python/sglang/srt/managers/scheduler.py python/sglang/srt/managers/schedule_batch.py
git diff --check -- python/sglang/srt/managers/scheduler.py python/sglang/srt/managers/schedule_batch.py
```

The local Windows inference run used Qwen3-0.6B with:

```text
temperature=0.6
top_p=0.95
top_k=20
max_new_tokens=256
enable_thinking=False
```

The important result was that every fused group had identical `output_ids` for
its member requests and no `<think>` output when chat templates were created
with `enable_thinking=False`.
