# PDS Sample Synchronization Notes

This note explains the local PDS implementation in the scheduler. It is meant
for agents that need to understand or extend the code without rediscovering the
same state-machine bugs.

## Goal

PDS means "prefill/decode/sample separation" in this branch. The model forward
stage computes logits but does not sample immediately. The scheduler stores the
logits on the corresponding request, then runs an explicit sample stage before
the next prefill/decode scheduling decision.

The extra synchronization feature lets a request wait for other requests'
logits before sampling. For example:

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

## Relevant Files

- `python/sglang/srt/managers/scheduler.py`
- `python/sglang/srt/managers/schedule_batch.py`

The `PendingSample` dataclass is in `schedule_batch.py`. It stores:

- `batch`: the batch snapshot used to call `process_batch_result`.
- `active_batch`: the batch that is admitted back into decode scheduling.
- `result`: the `GenerationBatchResult` that still owns logits.
- `forward_batch`: the forward metadata needed for logit slicing.
- `wait_for_rids`: the request IDs whose logits must also be ready.
- `external_params_required` and `external_params`: a placeholder for future
  externally supplied sample parameters.

Most of the runtime logic is in `scheduler.py`:

- `_park_pending_sample_result`
- `_pending_sample_ready`
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
   - every live dependency in `wait_for_rids` has also entered pending sample.
7. If ready, `run_pending_sample_stage` runs the delayed sample function,
   relays sampled tokens through the `FutureMap`, processes the batch result,
   and admits unfinished requests back into `running_batch`.

## Wait Semantics

The important semantic is "wait for live dependencies only."

If `reqA.wait_for_rids = {"reqB"}`:

- If `reqB` is still active in the scheduler, `reqA` waits until `reqB` is also
  in `pending_sample_reqs`.
- If `reqB` has already finished or left scheduler ownership, `reqA` does not
  wait forever.

This is implemented by taking two snapshots in `run_pending_sample_stage`:

- `pending_rids_snapshot`: requests that currently have pending logits.
- `active_rids_snapshot`: requests still owned by the scheduler, including
  `pending_sample_reqs`, `waiting_queue`, `cur_batch`, `running_batch`, and
  `last_batch`.

Then `_pending_sample_ready` only requires:

```python
pending.wait_for_rids.intersection(active_rids_snapshot) <= pending_rids_snapshot
```

This avoids a deadlock when one side of a synchronized pair finishes earlier.

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

After the delayed sample runs, the sampled token is relayed through
`FutureMap`, and the next decode forward resolves a one-token-per-request input.

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
still parked for sample. The implementation now includes:

```python
idle &= len(self.pending_sample_reqs) == 0
```

## Test Request Shape

A simple two-way synchronized batch request uses request IDs and
`__pds_wait_for_rids`:

```json
{
  "rid": ["reqA", "reqB"],
  "text": ["1+1等于几？只回答数字。", "请写一首五言古诗，四句即可。"],
  "sampling_params": [
    {
      "max_new_tokens": 8,
      "temperature": 0,
      "ignore_eos": true,
      "skip_special_tokens": true,
      "custom_params": {
        "__pds_wait_for_rids": ["reqB"]
      }
    },
    {
      "max_new_tokens": 32,
      "temperature": 0,
      "ignore_eos": true,
      "skip_special_tokens": true,
      "custom_params": {
        "__pds_wait_for_rids": ["reqA"]
      }
    }
  ]
}
```

Successful scheduling is more important than model answer quality for this
test. On the local Windows fallback path, Qwen3-0.6B can produce weak answers,
but the scheduler should return both requests without deadlock, shape mismatch,
or req-pool leak.
