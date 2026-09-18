# ClipCap project notes

## 1. What ClipCap does

ClipCap generates an image caption by connecting a pretrained image encoder to
a pretrained causal language model. In this project:

- Chinese CLIP converts an image into a 512-dimensional feature vector.
- An MLP converts that vector into 10 learned prefix-token embeddings.
- Chinese GPT-2 receives the image prefix followed by caption-token embeddings.
- GPT-2 predicts the caption autoregressively, one token at a time.

The image is therefore represented as a short sequence in the same
768-dimensional embedding space used by GPT-2 words:

```text
image -> Chinese CLIP -> [1, 512]
      -> projection MLP -> [10, 768]
      -> GPT-2 prefix -> generated caption
```

## 2. Main files

- `process_data.py`: extracts normalized Chinese-CLIP image features and writes
  `caption_image.pkl`.
- `clipcap_dataset.py`: tokenizes captions, adds `[SEP]`, pads them, and creates
  the attention mask.
- `model.py`: defines the projection MLP and joins image-prefix embeddings with
  caption embeddings before GPT-2.
- `train.py`: trains the projection and GPT-2 parameters and saves `model.pt`.
- `infer.py`: extracts features from `trump.jpeg` and `pokemon.jpeg`, then
  samples captions from the trained model.

## 3. Training objective

For image feature $v$, the projection network produces $K=10$ prefix vectors:

$$
P(v) = (p_1, p_2, \ldots, p_K).
$$

If the caption tokens are $(y_1,\ldots,y_T)$, GPT-2 receives:

$$
(p_1,\ldots,p_K,e(y_1),\ldots,e(y_T)).
$$

The last image-prefix position predicts $y_1$; the position containing $y_t$
predicts $y_{t+1}$. Training minimizes token cross-entropy:

$$
  L= - \sum_{t=1}^{T}\log p_\theta(y_t\mid v,y_{<t}).
$$

Padding positions are excluded from the loss. This is important because most
captions are shorter than the fixed sequence length; otherwise PAD tokens can
dominate the objective.

In this implementation all GPT-2 and projection parameters are trainable. A
common ClipCap alternative freezes GPT-2 and trains only the projection, which
uses less memory but may adapt less strongly to the caption data.

## 4. Inference

Inference first constructs the 10 image-prefix embeddings. GPT-2 then predicts
the next-token distribution from its final position. A token is sampled with
temperature 0.7, appended to the context, and fed back into GPT-2. Generation
stops when `[SEP]` is produced or the caption reaches 40 tokens:

```text
10 image tokens + at most 40 caption tokens = MAX_LENGTH 50
```

The sampling process is stochastic, so `infer.py` generates five potentially
different caption batches for the same two images.

## 5. Paths and execution order

`config.py` expects:

```text
~/scratch/llms_model/gpt2-chinese-cluecorpussmall
~/scratch/llms_model/chinese-clip-vit-base-patch16
```

Training saves `model.pt` inside this ClipCap directory, and inference loads
that same file. Submit inference with an `afterok` dependency so it starts only
if training succeeds:

```bash
cd ~/scratch/dips_project/reinforcement_learning/multip_modal/clipcap

TRAIN_JOB_ID=$(sbatch --parsable submit-clipcap-train.sh)
sbatch --dependency="afterok:${TRAIN_JOB_ID}" submit-clipcap-infer.sh
```

Logs are written under `result_out/`.

## 6. Selecting the training dataset

Without `--dataset`, training keeps the original two-image demonstration:

```bash
python train.py
```

The Flickr8k option loads the local training Parquet shards from
`datasets/flickr8k/data`. Its 6,000 training images have five captions each,
giving approximately 30,000 supervised image-caption pairs:

```bash
python train.py --dataset flickr8k
```

The first Flickr8k run uses Chinese CLIP to encode each unique image once and
saves the normalized features under `clipcap/cache/`. Later runs reuse this
cache. The cache is excluded from Git. Flickr8k captions are English while the
current decoder is Chinese GPT-2, so this configuration demonstrates the SFT
workflow but an English or multilingual causal LM should give better captions.

## 7. Scope of this demo

The current training data contains only two images and 38 captions. It is
useful for understanding the connection between CLIP and GPT-2, but the model
will mostly memorize these examples. General-purpose captioning requires a
much larger, more varied image-caption dataset and a validation split.
