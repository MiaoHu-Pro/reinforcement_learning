from transformers import DataCollatorWithPadding
from copy import deepcopy
from torch.utils.data import DataLoader
import random
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from torch import nn
import numpy as np
from transformers import AutoModelForCausalLM
import torch.nn.functional as F

device = torch.device("cuda")


class RewardModel(nn.Module):
    """
    GPT2模型加上一个"奖励头"
    """

    def __init__(self, model_name):
        super().__init__()
        self.llm = AutoModelForCausalLM.from_pretrained(model_name)
        # 添加奖励头
        self.reward_head = nn.Linear(self.llm.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        # GPT2的输出
        transformer_outputs = self.llm.forward(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # 获取最后一层隐藏层
        last_hidden_state = transformer_outputs.hidden_states[-1]

        # 对隐藏层给出奖励
        rewards = self.reward_head(last_hidden_state).squeeze(-1)
        # 归一化
        return torch.sigmoid(rewards)


# 将奖励模型加载
model_path = "gpt2-sft"
reward_model = RewardModel(model_path)
reward_model.load_state_dict(torch.load(
    "reward_model.pt",
    map_location='cpu'))


class ActorCriticModel(nn.Module):
    """
    GPT2模型+一个价值头
    """

    def __init__(self, model_path):
        super().__init__()
        # 这个要初始化为我们微调出来的gpt2-sft模型
        # actor演员模型：策略模型
        self.llm = AutoModelForCausalLM.from_pretrained(model_path)
        # 添加价值头
        # critic评论家模型：价值函数模型，价值头，线性层
        self.v_head = nn.Linear(self.llm.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        # gpt2-sft模型的输出
        transformer_outputs = self.llm.forward(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # 输出的token的logits，维度为 `vocab_size`
        lm_logits = transformer_outputs.logits
        # 获取最后一层隐藏层
        last_hidden_state = transformer_outputs.hidden_states[-1]

        # 评估token的价值，评估的是最后一个隐藏层的价值
        value = self.v_head(last_hidden_state).squeeze(-1)
        # 返回输出的token的logits和token的价值
        return lm_logits, value

    def generate(self, *args, **kwargs):
        return self.llm.generate(*args, **kwargs)


# 初始化：gpt2-sft + v_head
model = ActorCriticModel(model_path)
model = model.to(device)
reward_model = reward_model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

# 准备提示词的方式是从数据集中随机截取一段开头作为提示词
ds = load_dataset("csv", data_files="online_shopping_10_cats.csv")
ds_train = ds['train']

ds_train = ds_train.filter(lambda x: x["review"] != None and len(
    x["review"]) > 20 and len(x["review"]) < 1024)

# 截取评论数据的前2～8个字作为提示词
input_min_token_length = 2
input_max_token_length = 8
input_token_length_range = list(range(
    input_min_token_length,
    input_max_token_length))
# 输出的长度5～16个token
output_min_length = 10
output_max_length = 30


def tokenize(sample):
    # 提示词token的数量随机选择一个
    input_size = random.choice(input_token_length_range)
    # 如果input_size=3，截取sentence字段文本的前3个token出来
    sample['input_ids'] = tokenizer.encode(sample['review'])[:input_size]
    # 前3个token掩码为1
    sample['attention_mask'] = [1] * len(sample['input_ids'])
    # 前3个token对应的文本
    sample['query'] = tokenizer.decode(sample['input_ids'])
    return sample


map_kwargs = {
    "batched": False,
    "remove_columns": ['cat', 'review', 'label']
}

tokenized_dataset_train = ds_train.map(tokenize, **map_kwargs)

tokenized_dataset_train.set_format(type='torch')

REWARD_TOKEN_ID = tokenizer.eos_token_id


batch_size = 32


def collator(batch):
    return dict((key, [d[key] for d in batch]) for key in batch[0])


# 提示词组成的数据集
train_dataloader = DataLoader(
    tokenized_dataset_train,
    batch_size=batch_size,
    collate_fn=collator,
    shuffle=True
)

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,  # 所有词汇表中的词都可能被选中
    "top_p": 1.0,  # 包含整个概率分布
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id
}

# 冻结的参考模型，只用来计算奖励R_t
ref_model = deepcopy(model)

# 目的是计算每个token的R_t


def compute_rewards(
    input_data,  # 输入数据：提示词+补全，一条完整的数据
    query_tensors,  # 提示词张量
    response_tensors,  # 补全的张量
    score_tensors  # 奖励模型给出的分数的张量
):
    with torch.no_grad():
        # 正在微调的模型所输出的token的logits和token的价值
        logits, values = model(**input_data)  # b, seq, vocab_size
        # 冻结的模型的输出
        ref_logits, _ = ref_model(**input_data)
        # 正在微调的模型的输出的对数概率 `log_softmax`
        # 去掉最后一个token，因为是预测下一个token的任务
        # input_data如果是："abcde"，那么建立的数据对为：
        # abcd --> bcde
        logp = F.log_softmax(
            logits[:, :-1, :],
            dim=-1
        )
        # 冻结的模型的输出的对数概率
        ref_logp = F.log_softmax(
            ref_logits[:, :-1, :],
            dim=-1
        )
        # 实际生成的token序列
        # 自回归模型是预测下一个token，所以去掉第一个token
        # 真实标签为：bcde，需要去掉a
        labels = input_data['input_ids'][:, 1:]  # b, seq
        # 使用gather提取实际token的概率
        # logp 是 vocab_size 大小的张量
        # 假设真实的label是 `hello`
        # 那么要取出 `hello` 在 logp 张量中的概率
        logp = torch.gather(
            logp,
            2,
            labels.unsqueeze(-1)
        ).squeeze(-1)  # batch, seq
        ref_logp = torch.gather(
            ref_logp,
            2,
            labels.unsqueeze(-1)
        ).squeeze(-1)  # batch, seq
        # kl散度
        kl = logp - ref_logp
        # kl散度的权重
        beta = 0.2
        # 最终奖励的计算
        rewards = - beta * kl
        attention_mask = input_data['attention_mask']
        # 预测下一个token，所以去掉第一个mask
        masks = torch.zeros_like(attention_mask[:, 1:])
        masks[:, :] = attention_mask[:, 1:]
        # 遍历批次中的每一个提示词张量
        for j in range(len(query_tensors)):
            # 补全开始的索引
            start = len(query_tensors[j]) - 1
            # 补全结束的索引
            end = start + len(response_tensors[j])
            # 提示词部分掩码为0
            masks[j, :start] = 0
            # 补全后面的填充token掩码为0
            masks[j, end:] = 0
            # 将奖励模型给出的分数加到补全的最后一个token的奖励上面
            rewards[j, end - 1] += score_tensors[j]
            # 只留下掩码为1的部分的奖励
            rewards[j, :] *= masks[j, :]
            # 只留下掩码为1的部分的价值
            values[j, :-1] *= masks[j, :]

    return logp, rewards, values[:, :-1], masks


def masked_mean(values, mask):
    # 计算带掩码的平均值
    return (values * mask).sum() / mask.sum()


def masked_var(values, mask):
    # 计算带掩码的方差
    mean = masked_mean(values, mask)
    centred_values = values - mean
    return masked_mean(centred_values ** 2, mask)


def masked_whiten(values, mask):
    '''
    对数据进行带掩码的白化处理，
    让有效数据的方差变为1，但均值保持不变
    '''
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    whitened += mean
    return whitened


def compute_advantage(rewards, values, masks):
    '''
    广义优势估计（GAE）
    '''
    lastgae = 0.0
    advantage_reversed = []
    seq_length = rewards.shape[-1]
    gamma, lam = 1.0, 0.95

    for t in reversed(range(seq_length)):
        nextvalues = values[:, t + 1] if t < seq_length - 1 else 0.0
        delta = rewards[:, t] + gamma * nextvalues - values[:, t]
        lastgae = delta + gamma * lam * lastgae
        advantage_reversed.append(lastgae)
    advantages = torch.stack(advantage_reversed[::-1], dim=1)
    # 对广义优势估计进行了白化处理
    advantages = masked_whiten(advantages, masks)

    returns = advantages + values
    return advantages, returns


learning_rate = 1e-5
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# 使用旧策略产生的轨迹（补全）更新4次模型
ppo_epochs = 4

def compute_loss(
    old_logprobs,  # 冻结的一份概率 π_old 的轨迹数据
    logprobs,  # 正在微调的模型输出的对数概率 π_theta
    vpreds,  # 价值由v_head计算
    masks,  # 掩码
    advantages,  # 广义优势估计
    returns  # 回报：GAE Target = A_GAE + V(S_t)
):
    # 比率
    ratio = torch.exp(logprobs - old_logprobs)
    # 比率 * 广义优势估计
    pg_loss1 = - ratio * advantages
    # clip(比率，1-ϵ,1+ϵ) * 广义优势估计
    pg_loss2 = - torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * advantages
    # 策略（gpt2-sft）的损失
    pg_loss = masked_mean(torch.max(pg_loss1, pg_loss2), masks)
    # 价值网络（价值头）的损失，mse
    v_loss = masked_mean((vpreds - returns) ** 2, masks)
    # 由于 正在微调的模型 = gpt2-sft + value_head
    # 总的损失 = 策略网络的损失 + 0.1 * 价值网络的损失
    loss = pg_loss + 0.1 * v_loss

    return loss


num_epochs = 1

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
mini_batch_size = 4


def ppo_update(input_data, logprobs, masks, advantages, returns):
    for ep in range(ppo_epochs):
        # range(0, 32, 4)
        batch_inds = list(range(batch_size))
        for start in range(0, batch_size, mini_batch_size):
            mini_batch_inds = batch_inds[start:start+mini_batch_size]

            mb_model_inputs = {
                'input_ids': input_data
                ['input_ids']
                [mini_batch_inds],
                'attention_mask': input_data
                ['attention_mask']
                [mini_batch_inds]
            }
            # 模型的输出是token的logits和value
            mb_logits, mb_vpreds = model(**mb_model_inputs)
            # 去掉最后一个token
            mb_logits = F.log_softmax(
                mb_logits[:, :-1, :],
                dim=-1
            )
            # 取出真实标签对应的概率
            mb_logprobs = torch.gather(
                mb_logits,
                2,
                mb_model_inputs['input_ids'][:, 1:].unsqueeze(-1)
            ).squeeze(-1)

            loss= compute_loss(
                logprobs[mini_batch_inds],
                mb_logprobs,
                mb_vpreds[:, :-1],
                masks[mini_batch_inds],
                advantages[mini_batch_inds],
                returns[mini_batch_inds]
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('loss/total', loss.item())
    print('ppo update finished')


count = 0
for epoch in range(num_epochs):
    for batch in train_dataloader:
        if count == 100:
            break
        count += 1
        # 生成补全内容（回复）
        query_tensors = batch['input_ids']  # 提示词的张量
        query_attention_masks = batch['attention_mask']

        response_tensors = []  # 补全的张量
        query_response_tensors = []  # 提示词+补全的张量
        score_tensors = []  # 分数的张量

        for i, query in enumerate(query_tensors):
            query = query.to(device)
            query_attention_mask = query_attention_masks[i].to(device)
            # 随机挑一个补全的长度
            new_tokens = random.choice(list(range(
                output_min_length,
                output_max_length)))
            # 设置补全长度属性
            generation_kwargs["max_new_tokens"] = new_tokens
            # 提示词 + 补全
            query_response = model.generate(
                input_ids=query.unsqueeze(0),
                attention_mask=query_attention_mask.unsqueeze(0),
                **generation_kwargs
            ).squeeze(0)
            # 补全的长度
            response_len = len(query_response) - len(query)
            # 补全的张量
            response_tensors.append(query_response[-response_len:])
            query_response_tensors.append(query_response)
            # 从奖励模型拿分数
            with torch.no_grad():
                # 提示词 + 补全 + reward_token
                query_response_score = torch.cat([
                    query_response,
                    torch.tensor([REWARD_TOKEN_ID]).to(device)])
                attention_mask = torch.ones_like(
                    query_response_score,
                    dtype=torch.long)
                # 奖励模型的评分
                score = reward_model(
                    query_response_score.unsqueeze(0),
                    attention_mask.unsqueeze(0)
                ).squeeze(0)[-1]
                # 将奖励模型的评分从(0,1)缩放到(-1,1)
                score = 2 * (score - 0.5)
            score_tensors.append(score)

        input_data = data_collator([
            {
                'input_ids': ids,
                'attention_mask': torch.ones_like(ids)
            }
            for ids in query_response_tensors
        ]).to(device)

        # 奖励和优势
        logprobs, rewards, values, masks = compute_rewards(
            input_data,
            query_tensors,
            response_tensors,
            score_tensors
        )
        # 计算GAE和GAE Target
        advantages, returns = compute_advantage(rewards, values, masks)

        # 小批次训练
        if input_data["input_ids"].shape[0] != 32:
            break
        ppo_update(input_data, logprobs, masks, advantages, returns)

print(len(tokenized_dataset_train))
train_gen_lengths = [0] * len(tokenized_dataset_train)
for i in range(len(tokenized_dataset_train)):
    train_gen_lengths[i] = random.choice(list(range(
        output_min_length,
        output_max_length)))


def validate():
    scores = []
    count = 0
    for b, batch in enumerate(train_dataloader):
        if count == 100:
            break
        count += 1
        # 生成补全内容
        query_tensors = batch['input_ids']
        query_attention_masks = batch['attention_mask']
        for i, query in enumerate(query_tensors):
            query = query.to(device)
            query_attention_mask = query_attention_masks[i].to(device)
            new_tokens = train_gen_lengths[b * len(query_tensors) + i]
            generation_kwargs["max_new_tokens"] = new_tokens
            query_response = model.generate(
                input_ids=query.unsqueeze(0),
                attention_mask=query_attention_mask.unsqueeze(0),
                **generation_kwargs
            ).squeeze(0)
            query_response_score = torch.cat([
                query_response,
                torch.tensor([REWARD_TOKEN_ID]).to(device)])
            attention_mask = torch.ones_like(
                query_response_score, dtype=torch.long)
            score = reward_model(
                query_response_score.unsqueeze(0),
                attention_mask.unsqueeze(0)
            ).squeeze(0)[-1]
            score = 2 * (score - 0.5)
            scores.append(score.item())
    print('平均分数:', sum(scores) / len(scores))


validate()

model_path = './gpt2-sft'
model = ActorCriticModel(model_path).to(device)
validate()
