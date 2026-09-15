import dataclasses
import gc
import math
from collections import defaultdict
from typing import Callable, List

import numpy as np
import torch

from data_types import Episode, MiniBatch
from qwen2_model import Transformer
from tokenizer import Tokenizer

# 采集轨迹，也就是回答
# 根据一个问题，采样多条回答
# 假设5个问题，每个问题8个回答，那么要并行采样40条轨迹
# 可能碰到的实现的坑，都来自并行推理
# 1. 问题的长度不一样
# 2. 有的补全结束的早，有的结束的晚。
@torch.no_grad()
def rollout(
    model: Transformer, # 生成回答的llm模型
    batch: MiniBatch, # N个问题
    tokenizer: Tokenizer,
    max_gen_len: int, # 最大生成长度
    num_answer_per_question: int, # 每个问题产生多少个回答
    reward_function: Callable, # 奖励函数
    device: torch.device,
    dtype: torch.dtype,
) -> List[Episode]:
    end_token = tokenizer.eos_token
    end_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    # 问题：List[input_ids]
    prefix_token_ids = batch.prefix_token_ids
    # 批次中的问题数量 x 每个问题生成的回答数量 = 批次中的数据量
    bsz = len(batch.prefix) * num_answer_per_question
    # 最短问题长度
    min_prompt_len = min(len(t) for t in prefix_token_ids)
    # 最长问题长度
    max_prompt_len = max(len(t) for t in prefix_token_ids)
    # 总长度 = 最大生成长度 + 最大问题长度
    total_len = max_gen_len + max_prompt_len
    # 开启KV Cache，加速生成回答的速度
    model.init_kv_cache(
        max_batch_size=bsz,
        max_seq_len=total_len,
        device=device,
        dtype=dtype,
    )
    # 将所有token先初始化为填充符pad_token_id：批次中数据量 x 每条数据的总长度
    tokens = torch.full(
        (bsz, total_len),
        pad_token_id,
        dtype=torch.long,
        device=device
    )
    # 将问题部分填入
    # 第 k 个问题假设生成 num_answer_per_question 条回答
    # 那么 num_answer_per_question 条训练数据的前缀都是第k个问题
    for k, t in enumerate(prefix_token_ids):
        # 第k个问题的数据在批次中的偏移量
        offset = k * num_answer_per_question
        for i in range(num_answer_per_question):
            # 第k个问题的第i条完整数据的问题部分
            tokens[offset + i, : len(t)] = torch.tensor(
                t, dtype=torch.long, device=device
            )

    prev_pos = 0
    # 文本的掩码，填充符置为False
    input_text_mask = tokens != pad_token_id
    # 确保最小的问题长度小于总长度
    assert min_prompt_len < total_len
    # 标志位，标志一条回答（一条轨迹）是否结束，初始化为0
    is_finished = torch.zeros(
        (bsz,), dtype=torch.bool, device=device)
    # 并行的预测下一个token
    for cur_pos in range(min_prompt_len, total_len):
        print(
            f"\r* 生成轨迹:{cur_pos-min_prompt_len:>4d}/{total_len-min_prompt_len:>4d}",
            flush=True,
            end="",
        )
        # 针对批次中的所有训练数据，并行采样下一个token
        # 根据文本的 prev_pos~cur_pos 部分生成下一个token
        with torch.autocast(device_type=device.type, dtype=dtype):
            logits = model.inference(
                tokens[:, prev_pos:cur_pos],
                prev_pos
            )
        # logits ---> probs
        probs = torch.softmax(logits[:, -1], dim=-1)
        # 采样下一个token，具体使用了多项分布来采样
        next_token = torch.multinomial(probs, num_samples=1)
        next_token = next_token.reshape(-1)
        # 如果cur_pos这个索引已经有token了，那么直接作为下一个token
        # cur_pos这个所有已经有token，说明是长问题，那么这个token不需要预测
        # 注意：这里cur_pos对应的token不能是pad
        next_token = torch.where(
            input_text_mask[:, cur_pos], # cur_pos是否已经存在token了
            tokens[:, cur_pos], # 对于长的问题，cur_pos对应的已经有token了
            next_token # 对于最小长度的问题，选择预测出来的next_token
        )
        # 如果生成回答已经结束，那么下一个token是pad，
        # 如果没有结束，那么是next_token
        next_token = torch.where(
            is_finished,
            pad_token_id, # 对于短回答，回答已经结束，需要继续填充pad
            next_token # 对于长回答，回答没有结束，需要使用预测出来的token
        )
        # 将cur_pos赋值为下一个token
        tokens[:, cur_pos] = next_token
        # 如果有结尾标记
        if end_token_id is not None:
            # 检查这个结尾标记是否为生成下一个token得到的
            is_end_token = next_token == end_token_id
            # 如果cur_pos对应的是False，说明cur_pos是填充符
            # 说明这个token是生成的next token
            is_generated_token = ~input_text_mask[:, cur_pos]
            # 如果eos token是生成的，那么结束。
            is_finished = is_finished \
                        | (is_end_token & is_generated_token)
        prev_pos = cur_pos
        # 如果全部结束，那么跳出循环
        if is_finished.all():
            break
    # 删除kv cache
    model.del_kv_cache()
    # 手动垃圾回收
    gc.collect()
    # 清空cuda显存
    torch.cuda.empty_cache()
    is_finished_list = is_finished.tolist()
    tokens_list = tokens.tolist()

    # 准备存放输出回合的数组
    episodes = []
    # 遍历批次中的问题数量
    for i in range(bsz // num_answer_per_question):
        # 遍历第i条问题的第j条回答
        for j in range(num_answer_per_question):
            idx = i * num_answer_per_question + j
            # 截取出回答部分
            generated_token_ids =                   \
                tokens_list                         \
                [idx]                               \
                [len(batch.prefix_token_ids[i]):]
            # 删除填充token
            if pad_token_id in generated_token_ids:
                generated_token_ids = generated_token_ids[
                    :generated_token_ids.index(pad_token_id)
                ]
            # 生成的文本
            generated_text = \
                tokenizer.detokenize(generated_token_ids)
            # 计算第i个问题的第j条回答的奖励
            rewards = reward_function(
                # 生成的文本
                response=generated_text,
                # 数字列表
                numbers=batch.numbers[i],
                # 正确答案数字
                target=batch.target[i],
                end_token=end_token,
            )
            episode = Episode(
                prefix=batch.prefix[i],
                text=batch.prefix[i] + generated_text,
                prefix_token_ids=batch.prefix_token_ids[i],
                prefix_tokens=batch.prefix_tokens[i],
                generated_token_ids=generated_token_ids,
                is_finished=is_finished_list[idx],
                reward=rewards["reward"],
                reward_info=rewards["reward_info"],
            )
            episodes.append(episode)
    # 清除输出内容
    print("\r", end=" " * 100, flush=True)
    return episodes


def normalize_rewards_per_group(
    episodes: List[Episode]
) -> List[Episode]:
    """归一化每个组的奖励. 使用 prefix（问题） 区分不同的组."""
    """每条轨迹的reward字段替换为轨迹的优势"""
    groups = defaultdict(list)
    for episode in episodes:
        groups[tuple(episode.prefix)].append(episode)
    output = []
    # 遍历每个组，一个问题对应一组回答
    for group in groups.values():
        # [r_{i,0}, r{i,1}, ...]
        group_rewards = [item.reward for item in group]
        # 每个组的回答的奖励的平均值
        mean_reward = np.mean(group_rewards)
        # 每个组的回答的奖励的标准差
        std_reward = np.std(group_rewards)
        # 遍历组中的每一条回答，然后计算这条回答的优势
        # (r_i - mean(r)) / (std(r)+特别小的数值)
        for episode in group:
            normalized_reward =                \
                (episode.reward - mean_reward) \
                /                              \
                (std_reward + 1e-4)
            # reward字段，使用回答的组内优势替换掉奖励
            episode = dataclasses.replace(
                episode,
                reward=normalized_reward
            )
            output.append(episode)
    return output


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """计算熵，熵越小不确定性越小，用来监控模型训练的稳定性，不参与反向传播"""
    probs = torch.nn.functional.softmax(logits, dim=-1)
    entropy =                           \
        torch.logsumexp(logits, dim=-1) \
        -                               \
        torch.sum(probs * logits, dim=-1)
    return entropy


def update_policy(
    model, # 微调的模型
    optimizer, # 优化器
    episodes: List[Episode], # 轨迹（问题+回答）的数组
    micro_batch_size: int,
    pad_token_id: int,
    max_grad_norm: float, # 梯度裁剪，1.0
    device: torch.device,
    dtype: torch.dtype,
):
    """使用GRPO算法更新策略."""
    # 计算出每一条回答的组内优势
    episodes = normalize_rewards_per_group(episodes)
    # 按照回合的token数量排序，更有效的微批次训练
    episodes.sort(
        key=lambda x:                   \
        len(x.prefix_token_ids)         \
        +                               \
        len(x.generated_token_ids))
    num_target_tokens = sum(
        len(episode.generated_token_ids)
        for episode in episodes
    )
    entropy = 0.0

    for i in range(0, len(episodes), micro_batch_size):
        print(
            f"\r* 计算策略梯度: {i:>2d}/{len(episodes):>2d}",
            flush=True,
            end="",
        )
        j = min(i + micro_batch_size, len(episodes))
        batch_episodes = episodes[i:j]
        batch_lengths = [
            len(episode.prefix_token_ids)     \
            +                                 \
            len(episode.generated_token_ids)
            for episode in batch_episodes
        ]
        # 微批次中最长的轨迹的长度
        batch_max_length = max(batch_lengths)
        batch_token_ids = [
            episode.prefix_token_ids      # 问题的input_ids
            + episode.generated_token_ids # 生成的回答的input_ids
            + [pad_token_id] * ( # 添加填充符pad
                  batch_max_length - batch_lengths[i]
              )
            for i, episode in enumerate(batch_episodes)
        ]
        batch_masks = [
            # 问题部分掩码是0
            [0] * len(episode.prefix_token_ids)
            # 回答部分掩码为1
            + [1] * len(episode.generated_token_ids)
            # 填充符掩码为0
            + [0] * (batch_max_length - batch_lengths[i])
            for i, episode in enumerate(batch_episodes)
        ]
        # 取出每个回合的优势(r_i-mean(r)) / std(r)
        batch_advantages = [
            episode.reward for episode in batch_episodes
        ]
        batch_token_ids = torch.tensor(
            batch_token_ids,
            device=device,
            dtype=torch.long
        )
        batch_masks = torch.tensor(
            batch_masks,
            device=device,
            dtype=torch.bool
        )
        batch_advantages = torch.tensor(
            batch_advantages, device=device, dtype=torch.float32
        )

        with torch.autocast(device_type=device.type, dtype=dtype):
            # 去掉最后一个token，输入
            input_token_ids = batch_token_ids[:, :-1]
            # 去掉第一个token，目标token
            # 真实的目标token是来自上一轮的模型输出的回答
            target_token_ids = batch_token_ids[:, 1:]
            target_masks = batch_masks[:, 1:]
            # logits是预测的下一个token
            logits = model.forward(input_token_ids).float()
        # 在 one-hot 分类里，
        # 交叉熵等于对正确类别概率取负对数，
        # 所以“负对数概率”与“交叉熵”指的是同一个目标函数。
        # log(π_θ(a|s)) = -cross_entropy
        # −∑ⱼaⱼ⋅logâⱼ = -logâₜ, aₜ是真实标签，âₜ是模型预测为aₜ的概率
        log_probs = -torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_token_ids.reshape(-1),
            ignore_index=pad_token_id,
            reduction="none",
        ).reshape(input_token_ids.shape[0], -1)

        with torch.no_grad():
            token_entropy = compute_entropy(logits)
            entropy = entropy                            \
                    +                                    \
                    (token_entropy * target_masks).sum() \
                    /                                    \
                    num_target_tokens
        # 对数概率乘以优势 log(π_θ(a|s)) * A
        obj = log_probs * batch_advantages[:, None]
        # 计算每个token的平均目标
        obj = (obj * target_masks).sum() / num_target_tokens
        loss = -obj
        # 每一轮都要进行反向传播，计算模型参数的导数，但不更新模型的参数
        loss.backward()

    # 梯度裁剪
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=max_grad_norm
    )
    # 梯度下降，更新策略的参数，θ = θ - α*grad
    optimizer.step()
    # 清空梯度
    optimizer.zero_grad(set_to_none=True)
    return {
        "loss": loss.item(),
        "grad_norm": grad_norm.item(),
        "entropy": entropy.item(),
    }