from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding, set_seed
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# InstructGPT的reward model通常从SFT模型初始化，而不是重新从base模型开始。
model_path = Path(
    "~/scratch/llms_model/post_trained_models/"
    "gpt2-chinese-cluecorpussmall-sft"
).expanduser()
data_path = PROJECT_ROOT / "data" / "online_shopping_10_cats.csv"
output_model_dir = Path(
    "~/scratch/llms_model/post_trained_models/"
    "gpt2-chinese-cluecorpussmall-reward-model"
).expanduser()

if not model_path.is_dir():
    raise FileNotFoundError(
        f"SFT模型目录不存在: {model_path}。请先运行1-SFT-fixed.py"
    )
if not data_path.is_file():
    raise FileNotFoundError(f"训练数据不存在: {data_path}")

tokenizer = AutoTokenizer.from_pretrained(
    str(model_path),
    local_files_only=True
)

# 该中文GPT-2使用BERT词表。[SEP]是合法的序列结束/reward token；不能把
# [PAD]当作EOS，因为padding和真实reward位置必须具有不同语义。
REWARD_TOKEN_ID = tokenizer.sep_token_id
if REWARD_TOKEN_ID is None or tokenizer.pad_token_id is None:
    raise RuntimeError("Tokenizer必须同时提供[SEP]和[PAD] token")

# BertTokenizer默认产生全0的token_type_ids；GPT-2没有BERT句子分段任务，
# 因此只保留input_ids和attention_mask。
tokenizer.model_input_names = ["input_ids", "attention_mask"]

ds = load_dataset("csv", data_files=str(data_path))
ds_train = ds['train']

ds_train = ds_train.filter(lambda x: x["review"] != None and len(
    x["review"]) > 20 and len(x["review"]) < 1024)

print("数据集的数量：", len(ds_train))

# 保留10%从未参与训练的数据进行评估，避免在训练集上报告过于乐观的结果。
dataset_splits = ds_train.train_test_split(test_size=0.1, seed=42)


def tokenize(batch):
    # 提取出文本内容
    # 不自动加入[CLS]/[SEP]，由下面的代码只追加一个明确的reward token。
    # 为reward token预留一个位置，保证总长度不超过GPT-2的1024位置上限。
    outputs = tokenizer(
        batch["review"],
        add_special_tokens=False,
        truncation=True,
        max_length=1023,
        return_token_type_ids=False
    )
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

tokenized_dataset_train = dataset_splits["train"].map(tokenize, **map_kwargs)
tokenized_dataset_eval = dataset_splits["test"].map(tokenize, **map_kwargs)

tokenized_dataset_train.set_format(type="torch")
tokenized_dataset_eval.set_format(type="torch")


class RewardModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        # Reward model只需要backbone hidden states，不需要计算21128维LM logits。
        self.llm = AutoModel.from_pretrained(
            str(model_name),
            local_files_only=True
        )
        self.reward_head = nn.Linear(self.llm.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        # gpt2 backbone的前向传播直接返回最后一层隐藏状态
        transformer_outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        last_hidden_state = transformer_outputs.last_hidden_state
        # 给出奖励,
        reward_logits = self.reward_head(last_hidden_state).squeeze(-1)
        # 返回raw logits；BCEWithLogitsLoss会稳定地完成sigmoid和交叉熵。
        return reward_logits


model = RewardModel(model_path)

data_collator = DataCollatorWithPadding(tokenizer)

dataloader_params = {
    "batch_size": 2,  # 还是使用6G显存
    "shuffle": True,
    "collate_fn": data_collator,
    "pin_memory": torch.cuda.is_available()
}

train_dataloader = DataLoader(
    tokenized_dataset_train,
    **dataloader_params
)

# 评估使用独立的10%数据且不shuffle，不能在训练集上报告训练准确率。
eval_dataloader = DataLoader(
    tokenized_dataset_eval,
    batch_size=2,
    shuffle=False,
    collate_fn=data_collator,
    pin_memory=torch.cuda.is_available()
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"训练设备: {device}")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
# 合并sigmoid与二分类交叉熵，避免概率接近0或1时的数值不稳定。
criterion = nn.BCEWithLogitsLoss()
num_epochs = 1  # N+ Implementation Detail paper

set_seed(42)
model.to(device)

for epoch in range(num_epochs):
    model.train()
    for i, batch in enumerate(train_dataloader):
        inputs = {
            key: value.to(device, non_blocking=device.type == "cuda")
            for key, value in batch.items()
        }
        model_inputs = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask']
        }
        # 模型针对训练数据的打分
        scores = model(**model_inputs)
        batch_indices = torch.arange(scores.shape[0], device=scores.device)
        # 模型对reward token的打分
        score = scores[batch_indices, inputs['score_index']]
        # 真实分数：0或者1
        target = inputs["score"]
        loss = criterion(score, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if i % 100 == 0:
            print("Step-", i, ", Loss: ", loss.item())

# checkpoint包含完整backbone和reward head；同时保存config和tokenizer，方便重载。
output_model_dir.mkdir(parents=True, exist_ok=True)
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "base_model_path": str(model_path),
        "reward_token_id": REWARD_TOKEN_ID,
    },
    output_model_dir / "reward_model.pt"
)
model.llm.config.save_pretrained(output_model_dir)
tokenizer.save_pretrained(output_model_dir)
print(f"Reward model已保存到: {output_model_dir}")

model.eval()

all_predictions = []
all_labels = []

for i, batch in enumerate(eval_dataloader):
    inputs = {
        key: value.to(device, non_blocking=device.type == "cuda")
        for key, value in batch.items()
    }
    model_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    }
    with torch.no_grad():
        scores = model(**model_inputs)
        batch_indices = torch.arange(scores.shape[0], device=scores.device)
        score = scores[batch_indices, inputs["score_index"]]
        target = inputs["score"]
    # raw reward logit的0阈值等价于sigmoid(reward)>0.5。
    predictions = (score > 0).int()

    all_predictions.extend(predictions.cpu().numpy())
    all_labels.extend(target.cpu().numpy())

# 直接计算二分类confusion matrix，避免额外依赖scikit-learn。
# 行是真实标签[0,1]，列是预测标签[0,1]。
confusion = np.zeros((2, 2), dtype=np.int64)
for label, prediction in zip(all_labels, all_predictions):
    confusion[int(label), int(prediction)] += 1

print("评估集 confusion matrix（行=真实，列=预测）:")
print(confusion)
print("评估集 accuracy:", np.trace(confusion) / confusion.sum())
