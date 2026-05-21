# Rollout Generation Guide

The `generate_env_rollout.py` script now supports **continuing generation** after some iterations and **avoiding repeated generations**.

## New Features

### 1. Continue from a Specific Iteration (`--start-iter`)
Resume generation from a specific iteration number instead of always starting from 0.

**Example:**
```bash
# First run: Generate iterations 0-4 (5 total)
python scripts/generate_env_rollout.py --num-iters 5 --output-dir outputs/my_rollouts

# Later: Continue from iteration 5 to 9 (5 more)
python scripts/generate_env_rollout.py --num-iters 10 --start-iter 5 --output-dir outputs/my_rollouts
```

### 2. Skip Existing Files (`--skip-existing`)
Automatically detect and skip iterations that already have output files. Perfect for resuming interrupted jobs or adding more iterations without duplicating work.

**Example:**
```bash
# Initial run (interrupted at iteration 7)
python scripts/generate_env_rollout.py --num-iters 20 --output-dir outputs/my_rollouts
# Creates: rollouts_00000.jsonl through rollouts_00006.jsonl

# Resume with automatic skip
python scripts/generate_env_rollout.py --num-iters 20 --skip-existing --output-dir outputs/my_rollouts
# Automatically skips 0-6, continues from 7-19
```

### 3. Combined Usage
Use both flags together for maximum flexibility:

```bash
# Generate more iterations starting from 20, skip any that might exist
python scripts/generate_env_rollout.py \
    --num-iters 50 \
    --start-iter 20 \
    --skip-existing \
    --output-dir outputs/my_rollouts
```

## Usage Scenarios

### Scenario 1: Incremental Generation
Generate data in batches over time:

```bash
# Day 1: Generate first 10 iterations
python scripts/generate_env_rollout.py --num-iters 10 --output-dir outputs/exp1

# Day 2: Generate 10 more
python scripts/generate_env_rollout.py --num-iters 20 --start-iter 10 --output-dir outputs/exp1

# Day 3: Generate 30 more
python scripts/generate_env_rollout.py --num-iters 50 --start-iter 20 --output-dir outputs/exp1
```

### Scenario 2: Recovering from Interruptions
If your job gets interrupted (cluster timeout, OOM, etc.):

```bash
# Original command that got interrupted
python scripts/generate_env_rollout.py --num-iters 100 --output-dir outputs/exp2

# Resume without re-generating existing files
python scripts/generate_env_rollout.py --num-iters 100 --skip-existing --output-dir outputs/exp2
```

### Scenario 3: Expanding Dataset
Add more data to an existing dataset:

```bash
# Original dataset: 50 iterations
python scripts/generate_env_rollout.py --num-iters 50 --output-dir outputs/dataset_v1

# Expand to 100 iterations, skip the first 50
python scripts/generate_env_rollout.py --num-iters 100 --skip-existing --output-dir outputs/dataset_v1
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | str | `verl/trainer/config/ppo_memory_trainer.yaml` | Path to base YAML config |
| `--overrides` | list | `[]` | OmegaConf key=value overrides |
| `--num-iters` | int | `1` | Total number of iterations to run |
| `--start-iter` | int | `0` | Starting iteration number (for continuation) |
| `--output-dir` | str | `outputs/rollouts` | Directory to save JSONL files |
| `--save-dataproto` | flag | `False` | Also save raw DataProto binaries (.dp) |
| `--skip-existing` | flag | `False` | Skip iterations with existing output files |

## Output Files

Files are saved with zero-padded iteration numbers:
- `rollouts_00000.jsonl` - Iteration 0
- `rollouts_00001.jsonl` - Iteration 1
- `rollouts_00042.jsonl` - Iteration 42
- etc.

## How It Works

### Skip Detection
When `--skip-existing` is enabled:
1. The script scans the output directory for files matching `rollouts_*.jsonl`
2. Extracts iteration numbers from filenames
3. Prints found iterations: `[Rollout] Found existing iterations: [0, 1, 2, 5, 7]`
4. Skips those iterations during the generation loop

### Iteration Range
The loop runs from `start_iter` to `num_iters`:
```python
for it in range(start_iter, num_iters):
    # Generate rollout for iteration 'it'
```

So:
- `--num-iters 10` generates iterations 0-9
- `--num-iters 10 --start-iter 5` generates iterations 5-9
- `--num-iters 10 --start-iter 5 --skip-existing` generates 5-9, skipping any that exist

## Tips

1. **Always use the same output directory** when continuing generation
2. **Check existing files** before running: `ls outputs/my_rollouts/rollouts_*.jsonl | wc -l`
3. **Use `--skip-existing`** as a safety measure when resuming
4. **Backup your data** before running with large `--num-iters` values
5. **Monitor disk space** - each iteration generates a JSONL file (and optionally a .dp file)

## Example Workflow

```bash
# Step 1: Initial generation
python scripts/generate_env_rollout.py \
    --config configs/my_config.yaml \
    --num-iters 50 \
    --output-dir outputs/2025_exp \
    --skip-existing

# Step 2: Check progress
ls -lh outputs/2025_exp/rollouts_*.jsonl | tail -5

# Step 3: Add more data
python scripts/generate_env_rollout.py \
    --config configs/my_config.yaml \
    --num-iters 100 \
    --output-dir outputs/2025_exp \
    --skip-existing

# Step 4: Verify all files generated
ls outputs/2025_exp/rollouts_*.jsonl | wc -l  # Should show 100
```
