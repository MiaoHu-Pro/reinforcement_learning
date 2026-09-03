import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, pipeline, set_seed
from torch.utils.data import DataLoader
from datasets import load_dataset
from pprint import pprint

pretrained_model_path = "./gpt2-chinese-cluecorpussmall"

dataset = load_dataset("csv", data_files="online_shopping_10_cats.csv")

ds_train = dataset["train"]
# 将评论少于1024个字的过滤出来
ds_train = ds_train.filter(lambda x: x["review"] != None and len(
    x["review"]) > 20 and len(x["review"]) < 1024)

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
model = AutoModelForCausalLM.from_pretrained(pretrained_model_path)


def tokenize(batch):
    return tokenizer(batch["review"])


map_kwargs = {
    "batched": True,
    "batch_size": 512,
    "remove_columns": ["cat", "label", "review"]
}

tokenized_dataset_train = ds_train.map(tokenize, **map_kwargs)
# 转换成torch张量格式
tokenized_dataset_train.set_format(type="torch")
# 将pad_token作为eos_token
tokenizer.eos_token = tokenizer.pad_token
# 将数据整理成预测下一个token的任务的数据格式
data_collator = DataCollatorForLanguageModeling(
    tokenizer,
    mlm=False  # 将数据整理成预测下一个token的格式
)

dataloader_params = {
    "batch_size": 2,
    "collate_fn": data_collator
}

train_dataloader = DataLoader(
    tokenized_dataset_train,
    **dataloader_params
)

# 要更新的是model的参数
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
# 一般sft会训练1个epoch，也就是把训练数据看一遍就可以了
# 否则容易过拟合，造成灾难性遗忘
num_epochs = 1

device = torch.device("cuda")
model.to(device)
for epoch in range(num_epochs):
    model.train()
    for i, batch in enumerate(train_dataloader):
        batch = batch.to(device)
        outputs = model(**batch)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f"Step: {i}, Loss: {loss.item()}")

model.save_pretrained("./gpt2-sft")
tokenizer.save_pretrained("./gpt2-sft")


# 测试微调后的模型
g = pipeline("text-generation", model="./gpt2-sft")
set_seed(42)
pprint(g("这本书真是", max_length=300, num_return_sequences=1))
