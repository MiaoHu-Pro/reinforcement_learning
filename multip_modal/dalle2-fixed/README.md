# Educational DALL-E 2 pipeline

All dataset, architecture, and training defaults live in `config.py`.
`CLIPConfig`, `PriorConfig`, and `DecoderConfig` can be edited there, while the
most useful values can be overridden from the command line. Command-line
values are applied after presets, so an explicit option always wins.

Flickr30k is the default dataset. `--dataset all` combines Flickr8k and
Flickr30k; it does not include FashionMNIST. The old
`data/FMNISTConfig.py` remains as a compatibility import for notebooks.

## Recommended server workflow

```bash
cd ~/scratch/dips_project/reinforcement_learning/multip_modal/dalle2-fixed

TRAIN_JOB_ID=$(sbatch --parsable submit-dalle2-train.sh \
  --dataset all \
  --using-pre-CLIP \
  --large-UNet \
  --decoder-lr 2e-4)

sbatch --dependency="afterok:${TRAIN_JOB_ID}" \
  submit-dalle2-infer.sh \
  --run-name all-preclip-large-lr2e-4 \
  --prior-candidates 8 \
  --prompt "two children are playing outside" \
  --num-images 4 \
  --output generated_images/two-children.png
```

The automatic run name records the dataset, CLIP mode, U-Net size, and decoder
learning rate. Supply `--run-name my-experiment` for a more specific name,
particularly when overriding other hyperparameters.

Each experiment is isolated:

```text
trained_models/all-preclip-large-lr2e-4/
├── effective_config.json
├── prior.pt
├── prior.pt.complete
├── decoder.pt
└── decoder.pt.complete
```

A custom-CLIP run also contains `clip.pt`. Existing old flat checkpoints in
`trained_models/` are not renamed, overwritten, or deleted.

Inference with `--run-name` loads `effective_config.json`, so its training
dataset and architecture flags need not be repeated. `--prior-candidates N`
reranks more prior samples without retraining.

## Checkpoint safety

- Default behavior refuses to overwrite an existing stage checkpoint.
- `--resume` skips completed stages and starts at the first missing stage.
- `--overwrite` explicitly retrains and replaces stage checkpoints.

Resume is stage-level; it does not resume from the middle of an epoch. The
small `.complete` files distinguish a fully finished stage from a best-model
checkpoint written before an interrupted job.

## Useful configuration overrides

```bash
python config.py --help

# Preview the automatically selected directory without training.
python config.py \
  --dataset all --using-pre-CLIP --large-UNet \
  --decoder-lr 2e-4 --print-run-dir

# Examples accepted by every stage and submit-dalle2-train.sh:
--prior-lr 2e-4
--prior-epochs 100
--prior-batch-size 64
--decoder-lr 2e-4
--decoder-lr-min 1e-6
--decoder-epochs 150
--decoder-batch-size 32
--decoder-num-workers 8
```

See [understand_dalle2.md](understand_dalle2.md) for model theory,
implementation details, dataset handling, and limitations.
