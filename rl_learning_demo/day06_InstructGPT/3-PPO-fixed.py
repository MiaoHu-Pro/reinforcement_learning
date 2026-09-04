"""
使用PPO对SFT语言模型进行RLHF训练。

本脚本与day06中的前两个fixed脚本衔接：
1. 从1-SFT-fixed.py保存的SFT模型初始化actor和reference model；
2. 从2-RM-fixed.py保存的checkpoint恢复reward model；
3. actor生成response，reward model提供序列级奖励；
4. PPO使用GAE、概率比率裁剪和value loss更新actor-critic；
5. 最后只把训练后的语言模型（actor）保存为可直接加载的HF模型，另外保存value head。

Key fixes:

  - Uses the correct SFT and reward-model server paths.
  - Restores the reward-model checkpoint dictionary correctly.
  - Matches the AutoModel reward-model architecture from 2-RM-fixed.py.
  - Converts raw reward logits using sigmoid.
  - Correctly handles [PAD], [SEP], response masks, and token alignment.
  - Prevents GAE from propagating through prompt/padding positions.
  - Uses unwhitened GAE for critic targets and whitened advantages for the actor.
  - Supports incomplete final batches.
  - Shuffles PPO mini-batches and clips gradients.
  - Freezes the reference and reward models.
  - Adds comparable SFT-versus-PPO validation.
  - Saves results to:
    ~/scratch/llms_model/post_trained_models/gpt2-chinese-cluecorpussmall-ppo

"""

from copy import deepcopy
from pathlib import Path
import random

from datasets import load_dataset
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    set_seed,
)


# -----------------------------------------------------------------------------
# 路径和可复现实验设置
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SFT_MODEL_PATH = Path(
    "~/scratch/llms_model/post_trained_models/"
    "gpt2-chinese-cluecorpussmall-sft"
).expanduser()
REWARD_MODEL_DIR = Path(
    "~/scratch/llms_model/post_trained_models/"
    "gpt2-chinese-cluecorpussmall-reward-model"
).expanduser()
REWARD_MODEL_CHECKPOINT = REWARD_MODEL_DIR / "reward_model.pt"
PPO_OUTPUT_DIR = Path(
    "~/scratch/llms_model/post_trained_models/"
    "gpt2-chinese-cluecorpussmall-ppo"
).expanduser()
DATA_PATH = PROJECT_ROOT / "data" / "online_shopping_10_cats.csv"

for required_path, description in (
    (SFT_MODEL_PATH, "SFT模型目录"),
    (REWARD_MODEL_CHECKPOINT, "reward model checkpoint"),
    (DATA_PATH, "训练数据"),
):
    if not required_path.exists():
        raise FileNotFoundError(f"{description}不存在: {required_path}")

seed = 42
set_seed(seed)
random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"训练设备: {device}")


# -----------------------------------------------------------------------------
# Reward model：结构必须与2-RM-fixed.py完全一致
# -----------------------------------------------------------------------------
class RewardModel(nn.Module):
    """GPT-2 backbone加一个逐token的线性reward head。"""

    def __init__(self, model_name):
        super().__init__()
        # 2-RM-fixed.py使用AutoModel，而不是AutoModelForCausalLM。
        # 如果这里换成CausalLM，保存的state_dict键和网络结构都会不匹配。
        self.llm = AutoModel.from_pretrained(
            str(model_name),
            local_files_only=True,
        )
        self.reward_head = nn.Linear(self.llm.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        transformer_outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        last_hidden_state = transformer_outputs.last_hidden_state
        # 与训练reward model时一致：返回raw logits，不在模型内部做sigmoid。
        return self.reward_head(last_hidden_state).squeeze(-1)


# 2-RM-fixed.py保存的是包含model_state_dict等元数据的checkpoint字典，
# 不能把整个字典直接传给load_state_dict。
reward_checkpoint = torch.load(REWARD_MODEL_CHECKPOINT, map_location="cpu")
if "model_state_dict" not in reward_checkpoint:
    raise KeyError(
        f"{REWARD_MODEL_CHECKPOINT}缺少model_state_dict；"
        "请使用2-RM-fixed.py生成reward model"
    )

reward_model = RewardModel(SFT_MODEL_PATH)
reward_model.load_state_dict(reward_checkpoint["model_state_dict"])
reward_model.to(device)
reward_model.eval()
reward_model.requires_grad_(False)


# -----------------------------------------------------------------------------
# Actor-Critic：actor是SFT语言模型，critic是共享backbone上的value head
# -----------------------------------------------------------------------------
class ActorCriticModel(nn.Module):
    """GPT-2 actor加一个逐token的value head。"""

    def __init__(self, model_path):
        super().__init__()
        self.llm = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            local_files_only=True,
        )
        self.v_head = nn.Linear(self.llm.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        transformer_outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        lm_logits = transformer_outputs.logits
        last_hidden_state = transformer_outputs.hidden_states[-1]
        values = self.v_head(last_hidden_state).squeeze(-1)
        return lm_logits, values

    def generate(self, *args, **kwargs):
        return self.llm.generate(*args, **kwargs)


tokenizer = AutoTokenizer.from_pretrained(
    str(SFT_MODEL_PATH),
    local_files_only=True,
)
# 中文GPT-2使用BERT词表：[PAD]用于padding，[SEP]用于终止/reward token。
# 不能用tokenizer.pad_token = tokenizer.eos_token覆盖已有的[PAD]。
if tokenizer.pad_token_id is None or tokenizer.sep_token_id is None:
    raise RuntimeError("Tokenizer必须同时提供[PAD]和[SEP] token")
tokenizer.model_input_names = ["input_ids", "attention_mask"]

REWARD_TOKEN_ID = reward_checkpoint.get(
    "reward_token_id",
    tokenizer.sep_token_id,
)
if REWARD_TOKEN_ID != tokenizer.sep_token_id:
    raise RuntimeError(
        "Reward model使用的reward token与SFT tokenizer的[SEP]不一致"
    )

model = ActorCriticModel(SFT_MODEL_PATH).to(device)

# reference model是训练开始时actor的冻结副本，用来计算KL惩罚。
# 它在整个PPO训练中保持不变，不能加入optimizer。
ref_model = deepcopy(model).to(device)
ref_model.eval()
ref_model.requires_grad_(False)


# -----------------------------------------------------------------------------
# 准备prompt数据
# -----------------------------------------------------------------------------
ds = load_dataset("csv", data_files=str(DATA_PATH))
ds_train = ds["train"]
ds_train = ds_train.filter(
    lambda x: x["review"] is not None
    and 20 < len(x["review"]) < 1024
)

# 从每条评论开头随机截取2～8个token作为prompt。
input_min_token_length = 2
input_max_token_length = 8
input_token_length_range = list(range(
    input_min_token_length,
    input_max_token_length + 1,
))

# actor每次生成10～30个新token。
output_min_length = 10
output_max_length = 30
output_token_length_range = list(range(
    output_min_length,
    output_max_length + 1,
))


def tokenize(sample):
    input_size = random.choice(input_token_length_range)
    # 不自动加入[CLS]/[SEP]；否则很短的prompt会主要由特殊token构成。
    token_ids = tokenizer.encode(
        sample["review"],
        add_special_tokens=False,
    )[:input_size]
    sample["input_ids"] = token_ids
    sample["attention_mask"] = [1] * len(token_ids)
    sample["query"] = tokenizer.decode(token_ids)
    return sample


tokenized_dataset_train = ds_train.map(
    tokenize,
    batched=False,
    remove_columns=["cat", "review", "label"],
)
tokenized_dataset_train.set_format(type="torch")

batch_size = 32


def collator(batch):
    """保留变长prompt列表；生成阶段会逐条处理，所以此处不padding。"""
    return {key: [sample[key] for sample in batch] for key in batch[0]}


train_dataloader = DataLoader(
    tokenized_dataset_train,
    batch_size=batch_size,
    collate_fn=collator,
    shuffle=True,
)
validation_dataloader = DataLoader(
    tokenized_dataset_train,
    batch_size=batch_size,
    collate_fn=collator,
    shuffle=False,
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
generation_kwargs = {
    "top_k": 0,             # 0表示不使用top-k截断；必须是int而不是0.0
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.sep_token_id,
}


def score_query_response(query_response):
    """使用冻结reward model给完整的prompt+response打一个序列级分数。"""
    # RM训练时文本末尾恰好有一个[SEP] reward token。若生成结果已经以
    # [SEP]结束则不重复添加，否则临时追加一个[SEP]用于评分。
    if query_response[-1].item() == REWARD_TOKEN_ID:
        score_input_ids = query_response
    else:
        reward_token = torch.tensor(
            [REWARD_TOKEN_ID],
            dtype=query_response.dtype,
            device=query_response.device,
        )
        score_input_ids = torch.cat([query_response, reward_token])

    attention_mask = torch.ones_like(score_input_ids, dtype=torch.long)
    with torch.no_grad():
        reward_logits = reward_model(
            score_input_ids.unsqueeze(0),
            attention_mask.unsqueeze(0),
        )
        final_logit = reward_logits[0, -1]
        # RM由BCEWithLogitsLoss训练，因此先sigmoid得到正向概率，再映射到[-1, 1]。
        return 2.0 * torch.sigmoid(final_logit) - 1.0


# -----------------------------------------------------------------------------
# 轨迹奖励：reward = reward-model score - beta * sampled KL
# -----------------------------------------------------------------------------
def compute_rewards(
    input_data,
    query_tensors,
    response_tensors,
    score_tensors,
):
    with torch.no_grad():
        logits, all_values = model(**input_data)
        ref_logits, _ = ref_model(**input_data)

        # 对于输入[x0,x1,...,xT]，位置t的logits预测x(t+1)，所以：
        # logits去掉最后一项，labels去掉第一项，再用gather提取实际动作概率。
        labels = input_data["input_ids"][:, 1:]
        logprobs = torch.gather(
            F.log_softmax(logits[:, :-1, :], dim=-1),
            2,
            labels.unsqueeze(-1),
        ).squeeze(-1)
        ref_logprobs = torch.gather(
            F.log_softmax(ref_logits[:, :-1, :], dim=-1),
            2,
            labels.unsqueeze(-1),
        ).squeeze(-1)

        # 对采样动作的log-ratio是KL的无偏Monte-Carlo估计项。
        beta = 0.2
        rewards = -beta * (logprobs - ref_logprobs)

        masks = input_data["attention_mask"][:, 1:].clone().float()
        values = all_values[:, :-1].clone()

        for j in range(len(query_tensors)):
            # 第一枚response token由query最后一个位置预测，所以start=q_len-1。
            start = len(query_tensors[j]) - 1
            end = start + len(response_tensors[j])
            if end <= start:
                raise RuntimeError("生成了空response，无法构造PPO轨迹")

            masks[j, :start] = 0
            masks[j, end:] = 0
            # 序列级RM分数只加到response的最后一步。
            rewards[j, end - 1] += score_tensors[j]

        rewards = rewards * masks
        values = values * masks

    # 这些量描述旧策略采样得到的固定轨迹，后续PPO epoch不能反传到这里。
    return logprobs.detach(), rewards.detach(), values.detach(), masks


def masked_mean(values, mask):
    denominator = mask.sum()
    if denominator.item() == 0:
        raise RuntimeError("mask中没有有效的response token")
    return (values * mask).sum() / denominator


def masked_var(values, mask):
    mean = masked_mean(values, mask)
    return masked_mean((values - mean) ** 2, mask)


def masked_whiten(values, mask):
    """只根据有效response token计算均值/方差，并变换为零均值、单位方差。"""
    mean = masked_mean(values, mask)
    variance = masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(variance + 1e-8)
    return whitened * mask


def compute_advantage(rewards, values, masks):
    """
    计算GAE：
      delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
      A_t = delta_t + gamma * lambda * A_{t+1}

    critic target必须由未白化的GAE构造；白化只用于稳定actor更新。
    """
    last_gae = torch.zeros(rewards.shape[0], device=rewards.device)
    reversed_advantages = []
    gamma, gae_lambda = 1.0, 0.95

    for t in reversed(range(rewards.shape[1])):
        if t < rewards.shape[1] - 1:
            next_values = values[:, t + 1]
        else:
            next_values = torch.zeros_like(values[:, t])

        delta = rewards[:, t] + gamma * next_values - values[:, t]
        # 乘mask防止GAE越过response边界传播到prompt/padding位置。
        last_gae = (
            delta + gamma * gae_lambda * last_gae
        ) * masks[:, t]
        reversed_advantages.append(last_gae)

    raw_advantages = torch.stack(reversed_advantages[::-1], dim=1)
    returns = (raw_advantages + values).detach()
    actor_advantages = masked_whiten(raw_advantages, masks).detach()
    return actor_advantages, returns


# -----------------------------------------------------------------------------
# PPO loss和更新
# -----------------------------------------------------------------------------
learning_rate = 1e-5
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
ppo_epochs = 4
mini_batch_size = 4


def compute_loss(
    old_logprobs,
    logprobs,
    vpreds,
    masks,
    advantages,
    returns,
):
    # ratio_t(theta) = pi_theta(a_t|s_t) / pi_old(a_t|s_t)
    ratio = torch.exp(logprobs - old_logprobs)
    pg_loss1 = -ratio * advantages
    pg_loss2 = -torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * advantages
    pg_loss = masked_mean(torch.maximum(pg_loss1, pg_loss2), masks)

    # returns是未白化GAE + old V，作为critic回归目标。
    value_loss = masked_mean((vpreds - returns) ** 2, masks)
    return pg_loss + 0.1 * value_loss


def ppo_update(input_data, old_logprobs, masks, advantages, returns):
    model.train()
    rollout_batch_size = input_data["input_ids"].shape[0]

    for ppo_epoch in range(ppo_epochs):
        # 每个PPO epoch重新打乱同一批轨迹，避免固定mini-batch顺序。
        batch_indices = torch.randperm(rollout_batch_size).tolist()
        for start in range(0, rollout_batch_size, mini_batch_size):
            mini_batch_indices = batch_indices[start:start + mini_batch_size]
            model_inputs = {
                "input_ids": input_data["input_ids"][mini_batch_indices],
                "attention_mask": input_data["attention_mask"][mini_batch_indices],
            }

            logits, value_predictions = model(**model_inputs)
            labels = model_inputs["input_ids"][:, 1:]
            new_logprobs = torch.gather(
                F.log_softmax(logits[:, :-1, :], dim=-1),
                2,
                labels.unsqueeze(-1),
            ).squeeze(-1)

            loss = compute_loss(
                old_logprobs[mini_batch_indices],
                new_logprobs,
                value_predictions[:, :-1],
                masks[mini_batch_indices],
                advantages[mini_batch_indices],
                returns[mini_batch_indices],
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        print(
            f"PPO epoch {ppo_epoch + 1}/{ppo_epochs}, "
            f"last mini-batch loss: {loss.item():.4f}"
        )

    model.eval()


# -----------------------------------------------------------------------------
# 收集on-policy轨迹并立即进行PPO更新
# -----------------------------------------------------------------------------
num_epochs = 1
# max_rollout_batches = 100  # 教学示例：最多使用100批轨迹；设为None可遍历整轮。
max_rollout_batches = None  # 教学示例：最多使用100批轨迹；设为None可遍历整轮。
rollout_count = 0
model.eval()

for epoch in range(num_epochs):
    for batch in train_dataloader:
        if max_rollout_batches is not None and rollout_count >= max_rollout_batches:
            break
        rollout_count += 1

        query_tensors = batch["input_ids"]
        query_attention_masks = batch["attention_mask"]
        response_tensors = []
        query_response_tensors = []
        score_tensors = []

        # 生成动作属于采样/环境交互阶段，不构建梯度图。
        with torch.no_grad():
            for i, query in enumerate(query_tensors):
                query = query.to(device)
                query_attention_mask = query_attention_masks[i].to(device)
                new_tokens = random.choice(output_token_length_range)

                query_response = model.generate(
                    input_ids=query.unsqueeze(0),
                    attention_mask=query_attention_mask.unsqueeze(0),
                    min_new_tokens=output_min_length,
                    max_new_tokens=new_tokens,
                    **generation_kwargs,
                ).squeeze(0)
                response = query_response[len(query):]
                if response.numel() == 0:
                    raise RuntimeError("actor生成了空response")

                response_tensors.append(response)
                query_response_tensors.append(query_response)
                score_tensors.append(score_query_response(query_response))

        # 对完整序列padding，供actor/reference并行前向传播。
        input_data = data_collator([
            {
                "input_ids": ids,
                "attention_mask": torch.ones_like(ids),
            }
            for ids in query_response_tensors
        ]).to(device)

        old_logprobs, rewards, values, masks = compute_rewards(
            input_data,
            query_tensors,
            response_tensors,
            score_tensors,
        )
        advantages, returns = compute_advantage(rewards, values, masks)
        # ppo_update按实际batch大小工作，因此最后一个不足32的batch也能更新。
        ppo_update(
            input_data,
            old_logprobs,
            masks,
            advantages,
            returns,
        )
        print(f"rollout batch {rollout_count} PPO update完成")


# -----------------------------------------------------------------------------
# 用相同prompt、生成长度和随机种子公平比较SFT reference与PPO actor
# -----------------------------------------------------------------------------
def validate(policy_model, model_name, max_batches=10):
    scores = []
    policy_model.eval()

    # 两次validate都重置同样的随机数，使prompt、长度和采样随机流可比较。
    random.seed(2026)
    torch.manual_seed(2026)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2026)

    with torch.no_grad():
        for batch_index, batch in enumerate(validation_dataloader):
            if batch_index >= max_batches:
                break

            for i, query in enumerate(batch["input_ids"]):
                query = query.to(device)
                query_attention_mask = batch["attention_mask"][i].to(device)
                new_tokens = random.choice(output_token_length_range)
                query_response = policy_model.generate(
                    input_ids=query.unsqueeze(0),
                    attention_mask=query_attention_mask.unsqueeze(0),
                    min_new_tokens=output_min_length,
                    max_new_tokens=new_tokens,
                    **generation_kwargs,
                ).squeeze(0)
                scores.append(score_query_response(query_response).item())

    if not scores:
        raise RuntimeError("验证集没有产生任何reward score")
    mean_score = sum(scores) / len(scores)
    print(f"{model_name}平均reward score: {mean_score:.4f}")
    return mean_score


validate(ref_model, "SFT reference")
validate(model, "PPO actor")


# -----------------------------------------------------------------------------
# 保存训练后的actor和value head
# -----------------------------------------------------------------------------
PPO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
model.llm.save_pretrained(PPO_OUTPUT_DIR)
tokenizer.save_pretrained(PPO_OUTPUT_DIR)
torch.save(model.v_head.state_dict(), PPO_OUTPUT_DIR / "value_head.pt")
print(f"PPO actor已保存到: {PPO_OUTPUT_DIR}")
print(f"Value head已保存到: {PPO_OUTPUT_DIR / 'value_head.pt'}")
