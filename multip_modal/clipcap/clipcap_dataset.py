import pickle
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from config import (
    CLIP_EMBED_BATCH_SIZE,
    CLIP_MODEL_PATH,
    DEFAULT_DATASET,
    DEMO_DATA_PATH,
    FLICKR8K_CAPTION_COLUMNS,
    FLICKR8K_DATA_PATH,
    FLICKR8K_EMBED_CACHE_PATH,
    IMAGE_EMBD_DIM,
    IMAGE_TOKEN_LENGTH,
    MAX_LENGTH,
    SUPPORTED_DATASETS,
    device,
)
from model import extract_chinese_clip_image_features


class ClipCapDataset(Dataset):
    """为demo或Flickr8k构造ClipCap监督微调样本。

    Flickr8k的一张图片有5条caption。我们只编码每张图片一次，但创建5个
    训练pair；这些pair通过image_index共享图片特征，避免重复存储。
    """

    CACHE_VERSION = 1

    def __init__(self, tokenizer, dataset_name: str = DEFAULT_DATASET):
        if dataset_name not in SUPPORTED_DATASETS:
            raise ValueError(
                f"不支持的数据集{dataset_name!r}，可选值: {SUPPORTED_DATASETS}"
            )
        if tokenizer.pad_token_id is None:
            raise ValueError("tokenizer必须定义pad_token_id")
        if tokenizer.sep_token_id is None:
            raise ValueError("tokenizer必须定义sep_token_id")

        self.dataset_name = dataset_name
        if dataset_name == "demo":
            image_embeddings, caption_pairs = self._load_demo_data()
        else:
            image_embeddings, caption_pairs = self._load_flickr8k_data()

        self.image_embeddings = image_embeddings.contiguous().float()
        self.image_indices: list[int] = []
        self.caption_ids_list: list[torch.Tensor] = []
        self.mask_list: list[torch.Tensor] = []

        for image_index, caption in caption_pairs:
            caption_ids, mask = self._encode_caption(tokenizer, caption)
            self.image_indices.append(image_index)
            self.caption_ids_list.append(caption_ids)
            self.mask_list.append(mask)

        print(f"数据集: {dataset_name}")
        print(f"唯一图片数: {len(self.image_embeddings):,}")
        print(f"图片-caption训练pair数: {len(self.caption_ids_list):,}")

        # 原脚本每次构造dataset都会重写train_data.pkl，但训练从未读取该文件。
        # 这里不再产生这个冗余副作用；demo数据仍来自caption_image.pkl，
        # Flickr8k则只保存可复用的唯一图片embedding cache。

    @staticmethod
    def _load_demo_data() -> tuple[
        torch.Tensor,
        list[tuple[int, str]],
    ]:
        """读取原来的2张图片和38条手工caption。"""
        if not DEMO_DATA_PATH.is_file():
            raise FileNotFoundError(f"demo数据不存在: {DEMO_DATA_PATH}")
        with DEMO_DATA_PATH.open("rb") as input_file:
            caption_list, image_id_to_embed = pickle.load(input_file)

        image_ids = list(image_id_to_embed)
        image_id_to_index = {
            image_id: index for index, image_id in enumerate(image_ids)
        }
        embeddings = torch.stack(
            [
                torch.as_tensor(image_id_to_embed[image_id]).reshape(-1)
                for image_id in image_ids
            ]
        )
        caption_pairs = [
            (image_id_to_index[image_id], str(caption).strip())
            for image_id, caption in caption_list
        ]
        ClipCapDataset._validate_embeddings(embeddings, len(image_ids))
        return embeddings, caption_pairs

    @staticmethod
    def _find_flickr8k_train_files() -> list[Path]:
        files = sorted(FLICKR8K_DATA_PATH.glob("train-*.parquet"))
        if not files:
            raise FileNotFoundError(
                "没有找到Flickr8k训练文件，期望路径为: "
                f"{FLICKR8K_DATA_PATH}/train-*.parquet"
            )
        return files

    def _load_flickr8k_data(self) -> tuple[
        torch.Tensor,
        list[tuple[int, str]],
    ]:
        """读取Flickr8k train split并展开每张图片的5条caption。"""
        # 延迟导入：默认demo模式不依赖Hugging Face datasets包。
        from datasets import load_dataset

        train_files = self._find_flickr8k_train_files()
        flickr8k = load_dataset(
            "parquet",
            data_files={"train": [str(path) for path in train_files]},
            split="train",
        )
        required_columns = {"image", *FLICKR8K_CAPTION_COLUMNS}
        missing_columns = required_columns.difference(flickr8k.column_names)
        if missing_columns:
            raise ValueError(
                f"Flickr8k训练数据缺少列: {sorted(missing_columns)}"
            )

        embeddings = self._load_or_create_flickr8k_embeddings(flickr8k)

        # 直接读取caption列，不触发parquet中图片列的解码。
        caption_columns = {
            column: flickr8k[column]
            for column in FLICKR8K_CAPTION_COLUMNS
        }
        caption_pairs: list[tuple[int, str]] = []
        for image_index in range(len(flickr8k)):
            for column in FLICKR8K_CAPTION_COLUMNS:
                caption = caption_columns[column][image_index]
                if caption is not None and str(caption).strip():
                    # 当前GPT-2 tokenizer对小写英文的覆盖好于大写英文。
                    caption_pairs.append(
                        (image_index, str(caption).strip().lower())
                    )

        if not caption_pairs:
            raise ValueError("Flickr8k训练集没有有效caption")
        return embeddings, caption_pairs

    def _load_or_create_flickr8k_embeddings(self, flickr8k) -> torch.Tensor:
        """读取缓存；缓存不存在或不匹配时，用Chinese CLIP重新计算。"""
        dataset_fingerprint = str(flickr8k._fingerprint)
        if FLICKR8K_EMBED_CACHE_PATH.is_file():
            cache = torch.load(
                FLICKR8K_EMBED_CACHE_PATH,
                map_location="cpu",
                weights_only=True,
            )
            cache_matches = (
                cache.get("version") == self.CACHE_VERSION
                and cache.get("dataset_fingerprint") == dataset_fingerprint
                and cache.get("clip_model_path") == str(CLIP_MODEL_PATH)
                and cache.get("num_images") == len(flickr8k)
            )
            if cache_matches:
                embeddings = cache.get("image_embeddings")
                self._validate_embeddings(embeddings, len(flickr8k))
                print(
                    "读取Flickr8k图片特征缓存: "
                    f"{FLICKR8K_EMBED_CACHE_PATH}"
                )
                return embeddings
            print("Flickr8k图片特征缓存不匹配，将重新计算")

        embeddings = self._encode_flickr8k_images(flickr8k)
        FLICKR8K_EMBED_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = FLICKR8K_EMBED_CACHE_PATH.with_suffix(".tmp")
        torch.save(
            {
                "version": self.CACHE_VERSION,
                "dataset_fingerprint": dataset_fingerprint,
                "clip_model_path": str(CLIP_MODEL_PATH),
                "num_images": len(flickr8k),
                "image_embeddings": embeddings,
            },
            temporary_path,
        )
        temporary_path.replace(FLICKR8K_EMBED_CACHE_PATH)
        print(f"已保存图片特征缓存: {FLICKR8K_EMBED_CACHE_PATH}")
        return embeddings

    @staticmethod
    @torch.inference_mode()
    def _encode_flickr8k_images(flickr8k) -> torch.Tensor:
        """每张图片只运行一次Chinese CLIP，返回[N, 512] CPU tensor。"""
        from transformers import ChineseCLIPModel, ChineseCLIPProcessor

        if not CLIP_MODEL_PATH.is_dir():
            raise FileNotFoundError(
                f"Chinese CLIP模型目录不存在: {CLIP_MODEL_PATH}"
            )
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("config.py要求CUDA，但当前没有可用GPU")

        print(f"第一次使用Flickr8k，开始编码{len(flickr8k):,}张图片")
        clip_model = ChineseCLIPModel.from_pretrained(
            CLIP_MODEL_PATH
        ).to(device)
        clip_model.eval()
        processor = ChineseCLIPProcessor.from_pretrained(CLIP_MODEL_PATH)

        embedding_batches: list[torch.Tensor] = []
        starts = range(0, len(flickr8k), CLIP_EMBED_BATCH_SIZE)
        for start in tqdm(starts, desc="编码Flickr8k图片"):
            end = min(start + CLIP_EMBED_BATCH_SIZE, len(flickr8k))
            rows = flickr8k[start:end]
            images = [image.convert("RGB") for image in rows["image"]]
            inputs = processor(images=images, return_tensors="pt").to(device)
            # 兼容返回Tensor的旧版Transformers和返回结构化对象的新版。
            features = extract_chinese_clip_image_features(
                clip_model,
                **inputs,
            )
            features = F.normalize(features.float(), p=2, dim=-1)
            embedding_batches.append(features.cpu())

        embeddings = torch.cat(embedding_batches, dim=0)
        ClipCapDataset._validate_embeddings(embeddings, len(flickr8k))

        # 在创建GPT-2之前释放Chinese CLIP显存，降低训练峰值显存。
        del clip_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return embeddings

    @staticmethod
    def _validate_embeddings(
        embeddings: torch.Tensor,
        expected_images: int,
    ) -> None:
        if not isinstance(embeddings, torch.Tensor):
            raise TypeError("image_embeddings不是Tensor")
        if embeddings.shape != (expected_images, IMAGE_EMBD_DIM):
            raise ValueError(
                "图片特征形状错误，期望"
                f"[{expected_images}, {IMAGE_EMBD_DIM}]，实际{list(embeddings.shape)}"
            )
        if not torch.isfinite(embeddings).all():
            raise ValueError("图片特征包含NaN或Inf")

    @staticmethod
    def _encode_caption(tokenizer, caption: str) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """创建caption labels和图片-prefix + caption attention mask。"""
        maximum_caption_length = MAX_LENGTH - IMAGE_TOKEN_LENGTH
        # 给结尾[SEP]预留一个位置，使长caption被截断后仍有停止目标。
        caption_ids = tokenizer.encode(
            caption,
            add_special_tokens=False,
        )[: maximum_caption_length - 1]
        caption_ids.append(tokenizer.sep_token_id)

        mask = [1] * (IMAGE_TOKEN_LENGTH + len(caption_ids))
        padding_length = maximum_caption_length - len(caption_ids)
        caption_ids += [tokenizer.pad_token_id] * padding_length
        mask += [0] * padding_length
        return (
            torch.tensor(caption_ids, dtype=torch.long),
            torch.tensor(mask, dtype=torch.long),
        )

    def __len__(self) -> int:
        return len(self.caption_ids_list)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        image_embed = self.image_embeddings[self.image_indices[index]]
        caption_ids = self.caption_ids_list[index]
        mask = self.mask_list[index]
        return image_embed, caption_ids, mask
