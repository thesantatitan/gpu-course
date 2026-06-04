# Progress Tracking

`run_modal.py` updates this folder after each Modal assignment run unless you pass `--no-progress`.

## Files

- `runs.jsonl`: append-only raw run history. One JSON object per run.
- `summary.json`: machine-readable rollup regenerated from `runs.jsonl`.
- `summary.md`: human-readable progress table regenerated from `runs.jsonl`.

## What Gets Recorded

Each run record includes:

- timestamp and run id,
- assignment id/title/source,
- program arguments,
- requested Modal GPU and allocated GPU details,
- CUDA image,
- remote compile/run command,
- assignment exit code and pass/fail status,
- benchmark metrics emitted by the CUDA starter, such as `gpu_ms`, `block_size`, `n`, and error values,
- all timing fields ending in `_ms`,
- Modal run URL,
- local git branch, commit, and whether the working tree was dirty.

## Notes

- A run is marked done only when the CUDA program exits `0` and its JSON metrics include `"ok": true`.
- Failed correctness runs are still recorded, which is useful for seeing attempts and error values.
- Best timings are computed only from passed runs.
- Rebuild summaries at any time with:

```bash
uv run python -m common.progress
```

