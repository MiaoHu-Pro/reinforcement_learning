import torch
from torch.utils.data import Dataset
import pickle
from typing import Tuple
from config import IMAGE_TOKEN_LENGTH, MAX_LENGTH


class ClipCapDataset(Dataset):
    def __init__(self, tokenizer):
        # 填充符
        pad_id = tokenizer.pad_token_id
        # 取出图片的文本和图片的嵌入
        with open("caption_image.pkl", 'rb') as f:
            caption_list, image_id2embed = pickle.load(f)
        print('图片嵌入的总数:{}'.format(len(image_id2embed)))
        print('图片描述的总数:{}'.format(len(caption_list)))

        image_embed_list = []
        caption_ids_list = []
        mask_list = []
        for image_id, caption in caption_list:
            # 使用图像id获取图像的特征（clip.image_encoder输出的）
            image_embed = image_id2embed[image_id]
            # 只对文本进行分词，不添加任何特殊token
            caption_ids = tokenizer.encode(
                caption,
                add_special_tokens=False
            )
            # 在文本的token列表后面添加一个分隔符token
            caption_ids.append(tokenizer.sep_token_id)

            # 截断
            # 只能留下前90个token，因为图像对应的token需要占用10个token的位置
            # 最终的数据是：图像的token列表 + 文本的token列表
            caption_ids = caption_ids[:MAX_LENGTH - IMAGE_TOKEN_LENGTH]
            # 图像部分和文本部分的token都要掩码
            mask = [1] * (IMAGE_TOKEN_LENGTH + len(caption_ids))

            # 填充pad
            padding_len = MAX_LENGTH         \
                - IMAGE_TOKEN_LENGTH \
                - len(caption_ids)
            caption_ids += [pad_id] * padding_len
            # 将填充符掩码为0
            mask += [0] * padding_len

            caption_ids = torch.tensor(caption_ids).long()
            mask = torch.tensor(mask).long()

            image_embed_list.append(image_embed)
            caption_ids_list.append(caption_ids)
            mask_list.append(mask)
        # 保存训练数据
        with open("train_data.pkl", 'wb') as f:
            pickle.dump([
                image_embed_list,  # clip输出的图片特征的列表
                caption_ids_list,  # 图片文本的input_ids的列表
                mask_list  # 掩码的列表
            ], f)
        self.image_embed_list = image_embed_list
        self.caption_ids_list = caption_ids_list
        self.mask_list = mask_list
        print(f'训练数据总数：{len(self.image_embed_list)}')

    def __len__(self) -> int:
        return len(self.caption_ids_list)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        image_embed = self.image_embed_list[index]
        caption_ids = self.caption_ids_list[index]
        mask = self.mask_list[index]
        return image_embed, caption_ids, mask
