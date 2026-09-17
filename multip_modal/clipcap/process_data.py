from PIL import Image
import pickle
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from config import CLIP_MODEL_PATH


def main():
    # 加载clip模型
    # clip模型只用来生成图片的嵌入，不进行微调。
    clip_model = ChineseCLIPModel.from_pretrained(CLIP_MODEL_PATH)
    # 加载clip处理器
    processor = ChineseCLIPProcessor.from_pretrained(CLIP_MODEL_PATH)
    # 将2张图片进行处理，处理完之后交给clip抽取特征
    inputs_1 = processor(images=Image.open("trump.jpeg"), return_tensors="pt")
    inputs_2 = processor(images=Image.open(
        "pokemon.jpeg"), return_tensors="pt")
    # 获取第一张图片的嵌入（dim: 512）
    image_1_features = clip_model.get_image_features(**inputs_1)
    image_2_features = clip_model.get_image_features(**inputs_2)
    # 除以模长，归一化
    image_1_features = image_1_features / \
        image_1_features.norm(p=2, dim=-1, keepdim=True)  # normalize
    image_2_features = image_2_features / \
        image_2_features.norm(p=2, dim=-1, keepdim=True)  # normalize
    # key：图片id
    # value: 图片的嵌入
    # 下面的字典也可以放在chroma这样的向量数据库
    image_id2embed = {
        1: image_1_features,
        2: image_2_features,
    }
    # 图片的id和图片标题对
    caption_list = [
        (1, "特朗普在哭"),
        (1, "特朗普在哈哈哭"),
        (1, "特朗普在哭笑不得"),
        (1, "特朗普在哭笑不得"),
        (1, "特朗普在哭笑不得"),
        (1, "特朗普在哭笑不得"),
        (1, "特朗普在哭笑不得"),
        (1, "特朗普在哭笑不得"),
        (1, "特朗普在哭笑不得"),
        (1, "特朗普在哭笑不得"),
        (1, "特朗普在哭笑不得"),
        (1, "特朗普在哭笑不得"),
        (1, "特朗普在哭笑不得"),
        (1, "特朗普在哭笑不得"),
        (1, "特朗普在哭"),
        (1, "特朗普在哭"),
        (1, "特朗普在哭"),
        (1, "特朗普在哭"),
        (1, "特朗普在哭"),
        (1, "特朗普在哭"),
        (1, "特朗普在哭"),
        (1, "特朗普在哭"),
        (1, "特朗普在哭"),
        (2, "一只皮卡丘站着"),
        (2, "一只皮卡丘在玩耍"),
        (2, "一只皮卡丘在玩耍"),
        (2, "一只皮卡丘在玩耍"),
        (2, "一只皮卡丘在玩耍"),
        (2, "一只皮卡丘在玩耍"),
        (2, "一只皮卡丘在玩耍"),
        (2, "一只皮卡丘在玩耍"),
        (2, "一只皮卡丘在玩耍"),
        (2, "一只皮卡丘"),
        (2, "一只皮卡丘"),
        (2, "一只皮卡丘"),
        (2, "一只皮卡丘"),
        (2, "一只皮卡丘"),
        (2, "一只皮卡丘"),
    ]

    with open("caption_image.pkl", 'wb') as f:
        pickle.dump([caption_list, image_id2embed], f)

    print(f'图像嵌入的数量:{len(image_id2embed)}')
    print(f'图像文本的数量:{len(caption_list)}')


if __name__ == '__main__':
    main()
