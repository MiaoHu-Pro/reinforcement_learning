# Understanding DALL·E 2 and this project

## 1. The central idea

DALL·E 2 separates text-to-image generation into three learned stages:

```text
caption
   │
   ▼
CLIP text encoder ──► text embedding
                           │
                           ▼
                    diffusion prior
                           │
                           ▼
                 predicted image embedding
                           │
                           ▼
                  diffusion decoder ──► image
```

CLIP first learns a shared semantic space for matching images and captions.
The prior then learns what an image embedding could look like for a given text
embedding. Finally, the decoder turns that semantic image embedding into
pixels. The prior is necessary because a caption can describe many valid
images: text does not uniquely determine layout, colour, pose, or background.

This repository is a compact teaching implementation inspired by DALL·E 2. It
is not the production OpenAI model. In particular, it trains a small CLIP from
scratch, uses one low-resolution decoder, has no super-resolution cascade, and
does not implement classifier-free guidance.

## 2. Stage one: CLIP

The image encoder and text encoder map paired inputs to vectors in the same
latent space:

$$
v_i = \operatorname{normalize}(f_{\mathrm{image}}(I_i)),
\qquad
u_i = \operatorname{normalize}(f_{\mathrm{text}}(C_i)).
$$

For a batch of $B$ pairs, their scaled cosine-similarity matrix is

$$
s_{ij}=\exp(\tau)\,v_i^\top u_j.
$$

The diagonal pair $(I_i,C_i)$ is positive and the other batch combinations are
negatives. The symmetric contrastive loss is

$$
L_{\mathrm{CLIP}}
=
\frac{1}{2}
\left[
\operatorname{CE}(s, y)
+
\operatorname{CE}(s^\top, y)
\right],
\qquad y_i=i.
$$

In this project, `model/clip.py` contains a Vision Transformer image encoder
and a Transformer text encoder. The learned log-temperature is capped before
exponentiation to avoid unstable similarity logits.

## 3. Stage two: diffusion prior

Let $z_I$ be the frozen CLIP image embedding. The forward process adds noise at
a randomly selected time $t$:

$$
z_t
=
\sqrt{\bar\alpha_t}\,z_I
+
\sqrt{1-\bar\alpha_t}\,\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I).
$$

The prior Transformer receives the caption bytes, CLIP text embedding, time
embedding, and noisy image embedding. This implementation predicts the clean
image embedding directly:

$$
L_{\mathrm{prior}}
=
\left\|
\widehat z_I(z_t,t,C)-z_I
\right\|_2^2.
$$

At inference, `model/prior.py` starts from Gaussian noise and applies all
reverse-diffusion steps. It generates two candidates by default and retains the
one with greater cosine similarity to the CLIP text embedding. This reranking
is a simplified form of prior sampling and selection.

## 4. Stage three: diffusion decoder

The decoder is a conditional U-Net. Its forward image diffusion process is

$$
x_t
=
\sqrt{\bar\alpha_t}\,x_0
+
\sqrt{1-\bar\alpha_t}\,\epsilon.
$$

The U-Net predicts the noise that produced $x_t$:

$$
L_{\mathrm{decoder}}
=
\mathbb E_{x_0,t,\epsilon}
\left[
\left\|
\epsilon-\epsilon_\theta(x_t,t,z_I,C)
\right\|_2^2
\right].
$$

There is an important distinction between training and inference:

- During decoder training, $z_I$ is the ground-truth image's embedding from
  the frozen CLIP image encoder. The decoder can therefore learn against a
  stable and meaningful condition.
- During inference, no source image exists. The diffusion prior samples one
  predicted image embedding from the prompt. That one embedding remains fixed
  for the entire pixel reverse-diffusion trajectory.

The image embedding is injected into residual blocks, while text tokens and
image-embedding tokens condition attention blocks. The corrected implementation
does not resample the prior at every pixel-denoising step.

## 5. Dataset choices

### FashionMNIST (default)

FashionMNIST has class labels rather than natural captions, so the loader maps
each class to a template such as `An image of a sneaker`. Images are grayscale
and resized to $32\times32$. This is the fastest mode for understanding the
pipeline.

### Flickr8k

Use `--dataset flickr8k` to read the local parquet release from:

```text
datasets/flickr8k/data/
```

The expected columns are `image` and `caption_0` through `caption_4`. Flickr8k
contains 6,000 training images, 1,000 validation images, and 1,000 test images,
with five captions per image. The training loader selects one of the five
captions for each image on every access. This avoids placing duplicate images
in the same CLIP epoch while exposing different descriptions across epochs.

Flickr8k mode converts images to RGB and resizes them to $64\times64$. It uses a
small UTF-8 byte tokenizer with a 256-entry vocabulary. This removes any need
to download a tokenizer, but a learned BPE tokenizer would model natural
captions more efficiently.

For training augmentation, `RandomResizedCrop` retains 75%-100% of each source
image before resizing to $64\times64$. A fixed $64\times64$ crop must not be
applied directly to the original Flickr8k image: it would often remove the
captioned subject and corrupt the contrastive image-text pair.

Flickr8k is suitable for demonstrating the workflow, not for learning a
general-purpose text-to-image model. Six thousand training images are far too
few for broad visual knowledge, and training CLIP from scratch compounds that
limitation. Generated results should be interpreted as an educational sanity
check.

## 6. Training order and commands

The stages must be trained in order because later stages load and freeze earlier
checkpoints.

FashionMNIST:

```bash
python train_clip.py
python train_prior.py
python train_decoder.py
python infer.py --prompt "An image of a sneaker" --output sneaker.png
```

Flickr8k:

```bash
python train_clip.py --dataset flickr8k
python train_prior.py --dataset flickr8k
python train_decoder.py --dataset flickr8k
python infer.py \
  --dataset flickr8k \
  --prompt "a dog running through green grass" \
  --num-images 4 \
  --output flickr8k-generation.png
```

To override the automatic project-relative data directory, append:

```bash
--data-dir ~/scratch/dips_project/reinforcement_learning/datasets/flickr8k/data
```

Dataset-specific checkpoints are stored separately:

```text
trained_models/clip_fmnist_fixed.pt
trained_models/prior_fmnist_fixed.pt
trained_models/decoder_fmnist_fixed.pt

trained_models/clip_flickr8k.pt
trained_models/prior_flickr8k.pt
trained_models/decoder_flickr8k.pt
```

`train_prior.py` automatically trains CLIP first when its selected CLIP
checkpoint is absent. Similarly, `train_decoder.py` trains missing prerequisite
stages. On a server, explicitly submitting the three stages in order usually
makes failures and resource usage easier to diagnose.

The supplied Slurm training script runs all three stages sequentially in one
GPU allocation. Inference can be submitted with an `afterok` dependency:

```bash
cd ~/scratch/dips_project/reinforcement_learning/multip_modal/dalle2-fixed

TRAIN_JOB_ID=$(sbatch --parsable submit-dalle2-train.sh)

sbatch --dependency="afterok:${TRAIN_JOB_ID}" \
  submit-dalle2-infer.sh \
  --prompt "a dog running through green grass" \
  --num-images 4
```

The default Slurm dataset is Flickr8k. Extra arguments are forwarded to the
Python entry points. The tracked `result_out` directory must exist before
submission because Slurm opens its log file before the job begins.

## 7. Corrections made in `dalle2-fixed`

- CUDA timestep and output tensors are created on the correct device.
- Text is always truncated, terminated, and padded to a fixed token length.
- Unknown beta schedules now raise an error instead of failing later.
- Reverse diffusion multiplies noise by posterior standard deviation, not
  posterior variance.
- Pixel reverse diffusion reconstructs and clips the predicted clean image
  $\widehat x_0$ before applying the posterior mean. It does not clip Gaussian
  intermediate states $x_t$; this prevents late-time prediction errors from
  exploding into saturated red/blue/white outputs.
- The prior performs a complete reverse-diffusion trajectory rather than one
  prediction at the noisiest timestep.
- Decoder training uses the real frozen CLIP image embedding.
- Decoder inference samples the prior once and reuses that condition.
- Frozen CLIP/prior modules stay in evaluation mode, so dropout does not alter
  their targets.
- Validation runs without gradient tracking.
- Skip connections are local to each U-Net forward call.
- Checkpoint and dataset paths no longer depend on the shell's working
  directory.
- Corrected FashionMNIST checkpoints use `_fmnist_fixed.pt`, leaving the copied
  old checkpoints untouched because their prior/decoder semantics differ.
- Inference saves an image grid rather than requiring a graphical display.

## 8. Optional pretrained CLIP mode

The recommended checkpoint for this compact experiment is
`openai/clip-vit-base-patch32`. It is an English ViT-B/32 CLIP with a
512-dimensional shared image-text embedding. Download it once on the server:

```bash
mkdir -p ~/scratch/llms_model/clip-vit-base-patch32

hf download openai/clip-vit-base-patch32 \
  --local-dir ~/scratch/llms_model/clip-vit-base-patch32
```

Then submit training with:

```bash
TRAIN_JOB_ID=$(sbatch --parsable submit-dalle2-train.sh \
  --using-pre-CLIP)

sbatch --dependency="afterok:${TRAIN_JOB_ID}" \
  submit-dalle2-infer.sh \
  --using-pre-CLIP \
  --prompt "an astronaut standing on the moon" \
  --num-images 4
```

The flag must be present in both commands. Pretrained mode skips custom CLIP
training, freezes the downloaded CLIP, changes the pipeline latent width from
256 to 512, and writes:

```text
trained_models/prior_flickr8k_preclip.pt
trained_models/decoder_flickr8k_preclip.pt
```

Without `--using-pre-CLIP`, the original workflow remains active: train the
custom CLIP first, then train the prior and decoder using 256-dimensional
embeddings. These modes are intentionally checkpoint-incompatible.

The adapter reverses the dataset normalization, converts grayscale to RGB when
needed, resizes to CLIP's 224x224 input, applies the official CLIP channel
normalization, and retokenizes captions using CLIP's own tokenizer. CLIP stays
frozen, but the prior and decoder must be retrained because their conditioning
space and dimensionality changed.

Pretrained CLIP improves prompt semantics; it does not teach the pixel decoder
visual concepts absent from Flickr8k. High-quality open-domain generation still
requires a much larger decoder dataset or a pretrained generative decoder.

## 9. Larger U-Net decoder

The default decoder has 32 base channels and approximately 25.9 million
trainable parameters in pretrained-CLIP mode. `--large-UNet` doubles the
convolutional width while preserving the existing 18-residual-block depth:

```text
default: 32 -> 64  -> 128 -> 256
large:   64 -> 128 -> 256 -> 512
```

The large decoder has approximately 63.2 million trainable parameters; its
pixel U-Net grows from 13.1 million to 50.1 million parameters. Conditioning
width increases from 128 to 256, and the CLIP image embedding is expanded to
eight conditioning tokens instead of four.

It reuses the already-trained frozen CLIP and diffusion prior. Submit only the
new decoder to an 80 GB A100:

```bash
cd ~/scratch/dips_project/reinforcement_learning/multip_modal/dalle2-fixed

LARGE_UNET_JOB_ID=$(sbatch --parsable \
  submit-dalle2-large-unet-train.sh)
```

The job checks that the allocated GPU has approximately 80 GB VRAM and reads:

```text
trained_models/prior_flickr8k_preclip.pt
```

It writes the best validation checkpoint to:

```text
trained_models/decoder_flickr8k_preclip_largeunet.pt
```

Run dependent inference with both architecture-selection flags:

```bash
sbatch --dependency="afterok:${LARGE_UNET_JOB_ID}" \
  submit-dalle2-infer.sh \
  --using-pre-CLIP \
  --large-UNet \
  --prompt "a dog running through green grass" \
  --num-images 4 \
  --output generated_images/dog-large-unet.png
```

Omitting `--large-UNet` loads the original smaller decoder. The two checkpoint
architectures are intentionally separate and cannot load one another.
