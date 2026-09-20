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

See [understand_dalle2.md](understand_dalle2.md) for the model theory,
implementation workflow, dataset details, limitations, and complete commands.
