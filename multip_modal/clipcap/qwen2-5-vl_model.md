# Understanding Qwen2.5-VL and comparing it with ClipCap

## 1. What Qwen2.5-VL is

Qwen2.5-VL is a family of large vision-language models that accepts text,
images, multiple images, and video, and then generates text autoregressively.
The original family contains 3B, 7B, and 72B variants. Unlike an image
captioner limited to producing a short description, it can perform visual
question answering, OCR, document parsing, object grounding, chart analysis,
video understanding, and visual-agent tasks.

The high-level architecture is:

```text
image or video
    -> dynamic-resolution Vision Transformer
    -> vision-language merger
    -> variable-length visual-token sequence
                                      \
text prompt -> text-token embeddings -> Qwen2.5 language-model decoder
                                      /
                              autoregressive answer
```

The three principal components are:

1. A vision encoder that converts image patches or video patches into visual
   features.
2. An MLP-based vision-language merger that compresses and projects visual
   features into the language model's embedding dimension.
3. A Qwen2.5 causal language-model decoder that jointly processes visual and
   textual tokens and predicts the answer.

## 2. Vision encoder

Qwen2.5-VL uses a Vision Transformer trained for native dynamic resolution.
An image is not forced into one fixed square representation. Its dimensions
are adjusted to multiples of 28 while approximately preserving its aspect
ratio, so larger images produce more visual tokens and retain more detail.

The original 3B, 7B, and 72B models share a 32-layer ViT with hidden size 1280,
14-by-14 patches, and 16 attention heads. Most layers use window attention;
only layers 7, 15, 23, and 31 use full attention. This reduces computation
while occasionally allowing global information exchange.

The merger groups each spatially adjacent set of four patch features,
concatenates them, and applies a two-layer MLP. Conceptually:

$$
(z_1,z_2,z_3,z_4)
\xrightarrow{\text{concatenate}}
z_{1:4}
\xrightarrow{\text{MLP}}
v_i,
$$

where $v_i$ has the same width as a Qwen text-token embedding. The resulting
visual tokens can therefore be inserted into the causal language-model input.

For position information, Qwen2.5-VL uses multimodal rotary position
embeddings (MRoPE). Positions are decomposed into temporal, height, and width
components. Text behaves like ordinary one-dimensional RoPE; images use height
and width; videos additionally use time aligned with the actual timestamps.

## 3. From visual tokens to generated text

Suppose the processor creates visual tokens $v_1,\ldots,v_M$ and prompt tokens
$x_1,\ldots,x_N$. The language model receives one combined sequence containing
both modalities:

$$
(v_1,\ldots,v_M,x_1,\ldots,x_N).
$$

It then generates answer tokens autoregressively:

$$
p_\theta(y_1,\ldots,y_T\mid I,x)
=
\prod_{t=1}^{T}
p_\theta(y_t\mid I,x,y_{<t}).
$$

For instruction SFT, the usual loss is answer-token cross-entropy:

$$
L_{\mathrm{SFT}}
=
-\sum_{t=1}^{T}
\log p_\theta(y_t\mid I,x,y_{<t}).
$$

Visual tokens, padding, and normally the user prompt are assigned label `-100`
so they provide context but do not become prediction targets. The assistant's
answer tokens contribute to the loss.

## 4. How Qwen2.5-VL was trained

The official training recipe is much larger than ordinary task fine-tuning:

1. **Visual pre-training:** train the ViT using caption, visual-knowledge, and
   OCR data while starting from a pretrained Qwen2.5 language model.
2. **Multimodal pre-training:** unfreeze the model and train on image-text,
   interleaved text/image, VQA, mathematics, video, grounding, agent, and
   text-only data.
3. **Long-context pre-training:** increase the sequence length and add long
   video, document, and agent trajectories.
4. **Post-training:** perform multimodal supervised fine-tuning followed by
   Direct Preference Optimization (DPO).

The report describes about 4.1 trillion pre-training tokens. This scale is why
Qwen2.5-VL can generalize across tasks while the current ClipCap exercise is a
small task-specific demonstration.

## 5. Qwen2.5-VL versus this ClipCap project

| Property | Current ClipCap project | Qwen2.5-VL |
|---|---|---|
| Primary purpose | Image captioning demonstration | General multimodal assistant |
| Vision model | Pretrained Chinese CLIP | Native dynamic-resolution ViT |
| Image representation | One 512-dimensional global vector | Variable-length spatial token sequence |
| Connector | MLP expands one vector into 10 prefix tokens | MLP merger compresses adjacent patch features |
| Language model | Chinese GPT-2 | Qwen2.5 decoder |
| Spatial detail | Mostly lost in one global vector | Retained across spatial visual tokens |
| Image resolution | Chinese-CLIP processor uses a fixed model input | Dynamic resolution and aspect ratio |
| Inputs | One image | Text, one/multiple images, or video |
| Output tasks | Caption generation | Captioning, VQA, OCR, grounding, reasoning, agents, etc. |
| Position model | GPT-2 one-dimensional positions | Temporal-height-width MRoPE |
| Training data here | 2 images/38 pairs or Flickr8k | Massive multimodal pre-training plus SFT and DPO |
| Training cost | Small educational experiment | Large pretrained model; adaptation is usually preferred |

### The most important representational difference

ClipCap does this:

```text
entire image -> one global vector -> exactly 10 learned prefix embeddings
```

Qwen2.5-VL does this:

```text
image patches -> many spatially positioned features
              -> merged variable-length visual tokens
              -> language-model sequence
```

ClipCap's global vector is suitable for broad semantic content such as "a dog
running on grass," but it discards much fine-grained location and text detail.
Qwen2.5-VL retains multiple visual tokens, making OCR, layouts, coordinates,
small objects, and relationships much more tractable.

## 6. Comparing the projection layers

The current ClipCap projection is:

$$
P_{\mathrm{ClipCap}}:
\mathbb{R}^{512}
\rightarrow
\mathbb{R}^{10\times768}.
$$

Every image therefore consumes exactly ten GPT-2 prefix positions.

In Qwen2.5-VL, the ViT returns a sequence rather than one vector. After grouping
neighboring patches, the merger applies a projection to every group:

$$
P_{\mathrm{QwenVL}}:
\mathbb{R}^{4d_v}
\rightarrow
\mathbb{R}^{d_{\mathrm{LM}}}.
$$

The number of resulting tokens depends on image resolution. More tokens can
preserve more information, but they also increase attention time and memory.
The processor's `min_pixels` and `max_pixels` settings control this trade-off.

## 7. Using Qwen2.5-VL for Flickr8k

For a fair learning experiment, start with `Qwen2.5-VL-3B-Instruct`. Format
each Flickr8k caption as a multimodal conversation:

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe this image in one sentence."},
        ],
    },
    {
        "role": "assistant",
        "content": caption,
    },
]
```

Flickr8k supplies five reference captions for each image. During training, one
can either create five SFT records per image or sample one caption each epoch.
Validation and testing should retain all five references.

A sensible progression is:

1. Evaluate the pretrained Qwen2.5-VL-3B-Instruct model without fine-tuning.
2. Train the existing ClipCap implementation on Flickr8k.
3. Apply LoRA to Qwen2.5-VL using the same Flickr8k training split.
4. Compare on the same validation/test images with caption metrics such as
   BLEU, METEOR, ROUGE-L, CIDEr, and SPICE, plus manual inspection.

LoRA is a more practical first experiment than full fine-tuning: it preserves
the pretrained multimodal model, reduces optimizer memory, and makes accidental
catastrophic forgetting less likely. On an A100 80 GB, Qwen2.5-VL-3B is the
most comfortable model in this family for experimentation; memory still varies
substantially with visual-token count, batch size, and whether the ViT is
frozen.

## 8. Minimal inference concept

Modern Transformers versions expose Qwen2.5-VL through an automatic multimodal
model class. The processor applies the chat template and prepares both visual
and textual inputs:

```python
from transformers import AutoModelForMultimodalLM, AutoProcessor

model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForMultimodalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "file:///path/to/image.jpg"},
            {"type": "text", "text": "Describe this image in one sentence."},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=64)
new_tokens = generated_ids[:, inputs.input_ids.shape[1]:]
print(processor.batch_decode(new_tokens, skip_special_tokens=True))
```

The exact class name and accepted image field can differ across Transformers
versions, so the model card matching the installed version should be checked.

## 9. When to use each model

Use this ClipCap implementation when the goal is to understand:

- how a visual embedding conditions a causal language model;
- token shifting, padding masks, and autoregressive caption loss;
- how a small projection network connects two pretrained models;
- the complete training loop in code that is easy to inspect.

Use Qwen2.5-VL when the goal is:

- strong performance on previously unseen images;
- OCR, document, chart, or spatial understanding;
- multi-image or video interaction;
- instruction following and multi-turn multimodal conversations;
- parameter-efficient adaptation to a real application domain.

ClipCap is the clearer teaching model. Qwen2.5-VL is the much stronger general
system, but more of its ability comes from large-scale pre-training rather than
the small downstream dataset.

## 10. Primary references

- [Qwen2.5-VL Technical Report](https://arxiv.org/html/2502.13923v1)
- [Official Qwen2.5-VL repository](https://github.com/QwenLM/Qwen2.5-VL)
- [Official Qwen2.5-VL announcement](https://qwenlm.github.io/blog/qwen2.5-vl/)
- [Qwen2.5-VL-3B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- [Transformers Qwen2.5-VL documentation](https://huggingface.co/docs/transformers/model_doc/qwen2_5_vl)
