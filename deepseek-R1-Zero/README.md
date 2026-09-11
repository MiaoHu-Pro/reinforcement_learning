# Qwen2.5-3B R1-Zero-style training

This folder implements a small, educational version of the central DeepSeek-R1-Zero training recipe using **Qwen2.5-3B Base**.

It is not the DeepSeek-R1-Zero model and does not reproduce DeepSeek's architecture, private training data, compute scale, or final capability. It demonstrates the algorithmic workflow on one verifiable arithmetic task and one A100 80 GB GPU.

## Why the Base model matters

R1-Zero-style training starts directly from a pretrained base model without a reasoning-SFT warm-up. The default checkpoint is therefore:

```text
~/scratch/llms_model/GRPO-Zero/Qwen2.5-3B
```

The training script rejects a path containing `Instruct`. Using Qwen2.5-3B-Instruct can still be a useful RL experiment, but it is not a strict "Zero" initialization because that model was already instruction-tuned.

## Files

| File | Purpose |
|---|---|
| `train.py` | Configuration, model loading, collect/update loop, evaluation, logging, and checkpoints |
| `grpo.py` | Group sampling, old log probabilities, group advantages, clipped GRPO, and reference KL |
| `countdown_task.py` | Plain-text problem prompts and safe rule-based rewards |
| `data_types.py` | `Problem` and `Episode` trajectory containers |
| `submit-R1-Zero-qwen253b.sh` | Slurm job for one A100 80 GB |

The existing `../rl_learning_demo/day07_GRPO/dapo` folder is not modified. This implementation reuses its local Countdown parquet data by default.

## Training workflow

For every optimizer step:

1. Load four different Countdown problems.
2. Sample eight responses for each problem, producing 32 trajectories.
3. Calculate answer-correctness and output-format rewards with deterministic rules.
4. Normalize rewards separately inside each eight-response group.
5. Recompute and store the rollout policy's token log probabilities before updating.
6. Reuse that fixed rollout for two clipped GRPO epochs.
7. Penalize divergence from a frozen Qwen2.5-3B Base reference policy.

There is no reward model and no critic/value network.

## Group-relative advantage

For response $i$ in a group of $G$ answers to the same problem:

$$
\widehat A_i
=
\frac{
R_i-\operatorname{mean}(R_1,\ldots,R_G)
}{
\operatorname{std}(R_1,\ldots,R_G)+\epsilon
}.
$$

Every response token receives its response's scalar group advantage. When all answers in a group receive the same reward, the standard deviation is zero and the implementation assigns zero advantages to that group.

## Clipped GRPO objective

The rollout policy is fixed as $\pi_{\theta_{\mathrm{old}}}$. For a sampled response token:

$$
\rho_{i,t}(\theta)
=
\frac{
\pi_\theta(o_{i,t}\mid q_i,o_{i,<t})
}{
\pi_{\theta_{\mathrm{old}}}(o_{i,t}\mid q_i,o_{i,<t})
}.
$$

The clipped policy objective is:

$$
J_{i,t}(\theta)
=
\min\left(
\rho_{i,t}(\theta)\widehat A_i,
\operatorname{clip}
\left(
\rho_{i,t}(\theta),1-\varepsilon,1+\varepsilon
\right)
\widehat A_i
\right).
$$

The script minimizes the negative objective plus a sampled KL estimator against the frozen base reference policy:

$$
L(\theta)
=
-J(\theta)
+
\beta D_{\mathrm{KL}}
\left(
\pi_\theta\parallel\pi_{\mathrm{ref}}
\right).
$$

Only generated response tokens participate. Prompt and padding tokens are masked.

## Rule rewards

The model is requested to produce:

```text
<think>
reasoning
</think>
<answer>
arithmetic expression
</answer>
```

The reward is:

$$
R
=
R_{\mathrm{accuracy}}
+
0.1R_{\mathrm{format}}.
$$

The answer checker:

- accepts only integer constants, `+`, `-`, `*`, `/`, and parentheses;
- verifies that every supplied number is used exactly once;
- evaluates with exact rational arithmetic;
- does not execute arbitrary model-generated Python code.

The reasoning text itself is not judged. This avoids supervising a preferred chain of thought and lets RL explore reasoning strategies.

## Memory choices

The defaults are intentionally smaller than the earlier DAPO demo:

```text
rollout trajectories:  4 questions × 8 responses = 32
generation batch:      32
update microbatch:      1
maximum response:       512 tokens
dtype:                  BF16 when available
```

Gradients accumulate over all 32 trajectories before an optimizer step. Therefore, `micro_batch_size=1` lowers peak memory without changing the GRPO rollout-batch objective.

The implementation also:

- never calls `.float()` on the complete `[batch, sequence, vocabulary]` logits tensor;
- computes the frozen-reference forward pass before the policy forward pass and deletes its logits;
- uses sampled surprisal for monitoring instead of allocating exact full-vocabulary entropy tensors;
- enables gradient checkpointing;
- releases generation cache blocks before backward.

## Preprocessing check

This loads the tokenizer and dataset but not the 3B model weights:

```bash
python train.py --preprocess-only
```

## Local GPU launch

```bash
conda activate rl_post_training_env
cd ~/scratch/dips_project/reinforcement_learning/deepseek-R1-Zero
python train.py
```

To make a short smoke test:

```bash
python train.py \
    --max-steps 2 \
    --save-every 2 \
    --eval-every 2 \
    --max-new-tokens 128
```

## Slurm launch

Submit from this directory so Slurm can open the relative log path:

```bash
cd ~/scratch/dips_project/reinforcement_learning/deepseek-R1-Zero
sbatch submit-R1-Zero-qwen253b.sh
```

Arguments placed after the Slurm filename override the script defaults. A
recommended first server run is a two-step smoke test:

```bash
sbatch submit-R1-Zero-qwen253b.sh \
    --max-steps 2 \
    --save-every 2 \
    --eval-every 2 \
    --max-new-tokens 128
```

## Outputs

The default output is:

```text
~/scratch/llms_model/GRPO-Zero/Qwen2.5-3B-R1-Zero
```

It contains:

- the final Hugging Face policy and tokenizer;
- `trainer_state.pt` with optimizer state and completed step;
- `training_config.json`;
- `training_log.jsonl`;
- intermediate `checkpoint-XXXXXX` directories.

Resume from an intermediate checkpoint with:

```bash
python train.py \
    --resume-from ~/scratch/llms_model/GRPO-Zero/Qwen2.5-3B-R1-Zero/checkpoint-000050
```

The data-loader position is not restored exactly; the policy, optimizer, and global step are restored.

## What this intentionally does not claim

This code is a compact study implementation. A production-scale reproduction would additionally need distributed inference/training, much larger and more diverse verifiable data, failure recovery, high-throughput rollout engines, stricter evaluation, and extensive hyperparameter experiments.
