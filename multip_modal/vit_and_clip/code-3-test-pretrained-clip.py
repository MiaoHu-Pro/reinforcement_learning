from PIL import Image
import requests
from transformers import ChineseCLIPProcessor, ChineseCLIPModel

model = ChineseCLIPModel.from_pretrained("./chinese-clip-vit-base-patch16")
processor = ChineseCLIPProcessor.from_pretrained("./chinese-clip-vit-base-patch16")

image = Image.open("./trump.jpeg")
# Squirtle, Bulbasaur, Charmander, Pikachu in English
texts = ["懂王", "拜登", "小火龙", "皮卡丘"]

# compute image feature
# 计算的是图片的特征向量
inputs = processor(images=image, return_tensors="pt")
image_features = model.get_image_features(**inputs)
# 转化成模长为1的特征向量
image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)  # normalize

# compute text features
inputs = processor(text=texts, padding=True, return_tensors="pt")
# 计算文本的特征向量
text_features = model.get_text_features(**inputs)
# 转化成模长为1的特征向量
text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)  # normalize

# compute image-text similarity scores
inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
# 4条文本和1张图片的余弦相似度
probs = logits_per_image.softmax(dim=1)  # probs: [[1.2686e-03, 5.4499e-02, 6.7968e-04, 9.4355e-01]]
print(probs)
