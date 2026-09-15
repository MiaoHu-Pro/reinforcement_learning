from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

model_path = "./gpt2-chinese-cluecorpussmall"
tokenizer = AutoTokenizer.from_pretrained(model_path)

tokenizer.eos_token = tokenizer.pad_token
REWARD_TOKEN_ID = tokenizer.eos_token_id

ds = load_dataset("csv", data_files="online_shopping_10_cats.csv")
ds_train = ds['train']

ds_train = ds_train.filter(lambda x: x["review"] != None and len(
    x["review"]) > 20 and len(x["review"]) < 1024)

print("数据集的数量：", len(ds_train))


def tokenize(batch):
    # 提取出文本内容
    outputs = tokenizer(batch["review"])
    # 每条数据一个评分，初始化为 0 。
    outputs["score"] = [0] * len(outputs["input_ids"])
    # 对每条数据的最后的reward token进行评分
    outputs["score_index"] = [0] * len(outputs["input_ids"])
    for i in range(len(outputs["input_ids"])):
        # 第 i 条数据的末尾添加一个 eos token，作为reward token
        outputs["input_ids"][i].append(REWARD_TOKEN_ID)
        # reward token的掩码设置为 1 。
        outputs["attention_mask"][i].append(1)
        # 正向情感的文本评分为 1 。负向情感的评分为 0 。
        outputs["score"][i] = float(batch["label"][i])
        # 对 reward token 进行评分，也就是评分的索引为 reward token 的索引。
        outputs["score_index"][i] = len(outputs["input_ids"][i]) - 1
    return outputs


map_kwargs = {
    "batched": True,
    "batch_size": 512,
    "remove_columns": ["cat", "label", "review"]
}

tokenized_dataset_train = ds_train.map(tokenize, **map_kwargs)

tokenized_dataset_train.set_format(type="torch")


class RewardModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.llm = AutoModelForCausalLM.from_pretrained(model_name)
        self.reward_head = nn.Linear(self.llm.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        # gpt2的前向传播，但是还要输出隐藏层
        transformer_outputs = self.llm.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        last_hidden_state = transformer_outputs.hidden_states[-1]
        # 给出奖励
        reward = self.reward_head(last_hidden_state).squeeze(-1)
        # sigmoid用来将奖励搞到(0,1)范围内
        return torch.sigmoid(reward)


model = RewardModel(model_path)

data_collator = DataCollatorWithPadding(tokenizer)

dataloader_params = {
    "batch_size": 2,  # 还是使用6G显存
    "shuffle": True,
    "collate_fn": data_collator
}

train_dataloader = DataLoader(
    tokenized_dataset_train,
    **dataloader_params
)

device = torch.device("cuda")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
# 二分类交叉熵损失
criterion = nn.BCELoss()
num_epochs = 1  # N+ Implementation Detail paper

model.to(device)

for epoch in range(num_epochs):
    model.train()
    for i, batch in enumerate(train_dataloader):
        inputs = batch.to(device)
        model_inputs = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask']
        }
        # 模型针对训练数据的打分
        scores = model(**model_inputs)
        batch_indices = torch.arange(scores.shape[0])
        # 模型对reward token的打分
        score = scores[batch_indices, inputs['score_index']]
        # 真实分数：0或者1
        target = inputs["score"]
        loss = criterion(score, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print("Step-", i, ", Loss: ", loss.item())

torch.save(model.state_dict(), "reward_model.pt")

model.eval()

all_predictions = []
all_labels = []

for i, batch in enumerate(train_dataloader):
    inputs = batch.to(device)
    model_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    }
    with torch.no_grad():
        scores = model(**model_inputs)
        batch_indices = torch.arange(scores.shape[0])
        score = scores[batch_indices, inputs["score_index"]]
        target = inputs["score"]
    predictions = (score > 0.5).int()

    all_predictions.extend(predictions.cpu().numpy())
    all_labels.extend(target.cpu().numpy())

print(confusion_matrix(all_labels, all_predictions))
