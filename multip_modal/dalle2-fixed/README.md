# Educational DALL-E 2 pipeline

Train the three dependent stages in order:

```bash
python train_clip.py
python train_prior.py
python train_decoder.py
python infer.py --prompt "An image of a sneaker" --output sneaker.png
```

Use the local Flickr8k parquet dataset by adding `--dataset flickr8k` to every
command. Do not mix FashionMNIST and Flickr8k checkpoints; the scripts select
separate checkpoint filenames automatically.

To use a frozen local `openai/clip-vit-base-patch32` instead of training CLIP
from scratch, add `--using-pre-CLIP` to both training and inference. This skips
stage 1 and creates separate `_preclip.pt` prior/decoder checkpoints.

After the pretrained-CLIP prior exists, train the wider 63M-parameter decoder
on an 80 GB A100 with `sbatch submit-dalle2-large-unet-train.sh`. Select it at
inference with both `--using-pre-CLIP --large-UNet`.

See [understand_dalle2.md](understand_dalle2.md) for the model theory,
implementation workflow, dataset details, limitations, and complete commands.
