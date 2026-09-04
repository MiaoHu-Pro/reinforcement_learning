import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, GenerationConfig, pipeline, set_seed
from torch.utils.data import DataLoader
from datasets import load_dataset
from pprint import pprint
from pathlib import Path

"""

  - Reliable project-relative CSV path
  - ~ expansion for model/output paths
  - File and directory validation
  - Valid Chinese GPT-2 special-token IDs:
      - [CLS] = 101 for BOS
      - [SEP] = 102 for EOS
      - [PAD] = 0 for padding

  - Removal of inappropriate BERT token_type_ids
  - Explicit truncation to the model’s 1,024-position limit
  - Shuffled, reproducible training data
  - Automatic CUDA/CPU selection
  - Explicit batch-device transfer
  - Gradient clipping
  - Automatic output-directory creation
  - Reuse of the trained model for testing
  - Transformers 5-compatible generation configuration
"""

# 使用__file__定位项目根目录，因此无论从项目根目录还是day06目录运行，
# 都能找到同一份本地CSV数据。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
pretrained_model_path = Path(
    "~/scratch/llms_model/gpt2-chinese-cluecorpussmall"
).expanduser()
data_path = PROJECT_ROOT / "data" / "online_shopping_10_cats.csv"
output_model_path = Path(
    "~/scratch/llms_model/post_trained_models/"
    "gpt2-chinese-cluecorpussmall-sft"
).expanduser()

if not pretrained_model_path.is_dir():
    raise FileNotFoundError(f"预训练模型目录不存在: {pretrained_model_path}")
if not data_path.is_file():
    raise FileNotFoundError(f"训练数据不存在: {data_path}")

dataset = load_dataset("csv", data_files=str(data_path))

ds_train = dataset["train"]
# 将评论少于1024个字的过滤出来
ds_train = ds_train.filter(lambda x: x["review"] != None and len(
    x["review"]) > 20 and len(x["review"]) < 1024)

tokenizer = AutoTokenizer.from_pretrained(
    str(pretrained_model_path),
    local_files_only=True
)
# 这里实际加载的是BertTokenizer，它默认返回全0的token_type_ids。GPT-2没有
# BERT的句子A/B分段任务；若把这些ID传入GPT-2，它们会被当作额外词嵌入相加。
# 仅保留GPT-2真正需要的input_ids和attention_mask。
tokenizer.model_input_names = ["input_ids", "attention_mask"]
model = AutoModelForCausalLM.from_pretrained(
    str(pretrained_model_path),
    local_files_only=True
)

# 这个中文GPT-2使用BERT词表（大小21128），但原始GPT-2配置中的
# bos/eos_token_id=50256已超出词表范围。使用词表中真实存在的特殊token：
# [CLS]=101作为序列开始，[SEP]=102作为序列结束，[PAD]=0只用于padding。
# 不能把EOS设置为PAD，否则生成可能在padding处错误停止。
model.config.bos_token_id = tokenizer.cls_token_id
model.config.eos_token_id = tokenizer.sep_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.generation_config.bos_token_id = tokenizer.cls_token_id
model.generation_config.eos_token_id = tokenizer.sep_token_id
model.generation_config.pad_token_id = tokenizer.pad_token_id


def tokenize(batch):
    # 字符数小于1024并不保证WordPiece token也小于1024；显式截断可避免
    # 超过GPT-2的n_positions=1024而产生位置编码越界。
    return tokenizer(
        batch["review"],
        truncation=True,
        max_length=model.config.n_positions,
        return_token_type_ids=False
    )


map_kwargs = {
    "batched": True,
    "batch_size": 512,
    "remove_columns": ["cat", "label", "review"]
}

tokenized_dataset_train = ds_train.map(tokenize, **map_kwargs)
# 转换成torch张量格式
tokenized_dataset_train.set_format(type="torch")
# 将数据整理成预测下一个token的任务的数据格式
# tokenizer已有合法的[PAD] token。collator会动态padding，并把padding位置的
# label设为-100，使它们不参与causal language-model loss。
data_collator = DataCollatorForLanguageModeling(
    tokenizer,
    mlm=False  # 将数据整理成预测下一个token的格式
)

# 固定训练shuffle与其他PyTorch随机操作，便于复现实验。
set_seed(42)

dataloader_params = {
    "batch_size": 2,
    "collate_fn": data_collator,
    # 训练时必须shuffle，避免模型按CSV中的固定类别/标签顺序学习。
    "shuffle": True,
    "pin_memory": torch.cuda.is_available()
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

# GPU可用时使用GPU；在登录节点或本地无GPU环境中仍可运行（但训练较慢）。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"训练设备: {device}")
model.to(device)
for epoch in range(num_epochs):
    model.train()
    for i, batch in enumerate(train_dataloader):
        # 显式移动每个tensor，兼容普通dict和Transformers BatchEncoding。
        batch = {
            key: value.to(device, non_blocking=device.type == "cuda")
            for key, value in batch.items()
        }
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        # 限制异常大的梯度，降低小batch训练不稳定的风险。
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if i % 100 == 0:
            print(f"Step: {i}, Loss: {loss.item()}")

output_model_path.mkdir(parents=True, exist_ok=True)
model.save_pretrained(output_model_path)
tokenizer.save_pretrained(output_model_path)


# 测试微调后的模型。直接复用当前模型和tokenizer，避免再次从磁盘加载，
# 并确保pipeline在与训练相同的设备上运行。
model.eval()
g = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device
)
set_seed(42)
test_generation_config = GenerationConfig.from_model_config(model.config)
test_generation_config.max_new_tokens = 100
test_generation_config.do_sample = True
test_generation_config.num_return_sequences = 1
pprint(g("这本书真是", generation_config=test_generation_config))
