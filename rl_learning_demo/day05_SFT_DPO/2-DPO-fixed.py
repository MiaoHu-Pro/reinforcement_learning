from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import math
import os
import datasets
from dataclasses import dataclass
from pathlib import Path


"""
Updated only rl_learning_demo/day05_SFT_DPO/2-DPO-fixed.py. Other files were not changed.

Key fixes:

- Loads the SFT checkpoint from:

~/scratch/llms_model/post_trained_models/Qwen3-0.6B-SFT

- Uses all 61,135 local train_prefs records—no 30,000 limit.
- Loads local UltraFeedback Parquet files instead of ModelScope.
- Keeps each chosen/rejected pair aligned and validates their prompts.
- Fixes Transformers 5 Encoding compatibility.
- Freezes the reference model and disables its gradients.
- Scores only the final assistant response.
- Correctly aligns masks with shifted causal targets.
- Adds proper attention masks.
- Uses numerically stable log_softmax() and logsigmoid().
- Fixes learning-rate scheduling and final partial-batch accumulation.
- Enables gradient clipping and gradient checkpointing.
- Uses bfloat16 on the A100.
- Restores KV caching before saving for efficient inference.
- Expands ~ and creates the output directory automatically.

The result is saved to:

~/scratch/llms_model/post_trained_models/Qwen3-0.6B-SFT-DPO

After copying the updated file to the server, run:

conda activate rl_post_training_env

cd ~/scratch/dips_project/reinforcement_learning/rl_learning_demo/day05_SFT_DPO

python 2-DPO-fixed.py

"""

PROJECT_ROOT = Path(__file__).resolve().parents[2]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    raise RuntimeError("DPO training requires a CUDA GPU, but CUDA is unavailable")

# A100 supports bfloat16. It saves memory while retaining a much safer numeric
# range than float16. The fallback keeps this script correct on older GPUs.
model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
# model_path = "./Qwen3-0.6B-SFT"

model_path: Path = Path(
        "~/scratch/llms_model/post_trained_models/Qwen3-0.6B-SFT"
    ).expanduser()



# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    str(model_path),
    dtype=model_dtype,
    local_files_only=True
).to(device)
# 冻结的参考模型
ref_model = AutoModelForCausalLM.from_pretrained(
    str(model_path),
    dtype=model_dtype,
    local_files_only=True
).to(device)
# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

# DPO只更新策略模型。参考模型必须冻结并使用eval模式，否则dropout等训练层会让
# 同一个参考回答每次得到不同概率，也会浪费显存保存无用梯度。
ref_model.requires_grad_(False)
ref_model.eval()
model.gradient_checkpointing_enable()
model.config.use_cache = False
# 参考模型只需要当前序列的logits，DPO不会继续自回归生成；关闭KV cache可避免
# 四次forward期间保留完全不会使用的key/value张量。
ref_model.config.use_cache = False

print(model.generation_config)

model.generation_config.do_sample = True
model.generation_config.eos_token_id = [
    tokenizer.convert_tokens_to_ids("<|im_end|>"),
    tokenizer.eos_token_id,
]
model.generation_config.pad_token_id = tokenizer.pad_token_id
model.generation_config.temperature = 0.7
model.generation_config.top_p = 0.8
model.generation_config.top_k = 20
model.generation_config.repetition_penalty = 1.05

print(model.generation_config)

@dataclass
class DPOConfig:
    max_length:int = 1700 #根据自身具备的算力条件进行自适应更改
    # 为chosen/rejected共享的prompt设置独立上限，给回答保留token空间。
    max_prompt_length:int = 1024
    batch_size:int = 2
    gradient_accumulation_steps:int = 8
    beta:float = 0.5 # β是dpo公式中的超参数
    log_iter:int = 200
    max_lr:float = 1e-6
    min_lr:float = 1e-7
    warmup_steps:int = 300
    max_gradient_norm:float = 1.0

# 使用本地UltraFeedback偏好数据。DPO必须使用train_prefs，而不是train_sft：
# train_prefs中的每一行才包含同一prompt对应的chosen/rejected回答。
preference_data_dir = (
    PROJECT_ROOT / "data" / "ultrafeedback_binarized" / "data"
)
preference_shards = sorted(preference_data_dir.glob("train_prefs-*.parquet"))
if not preference_shards:
    raise FileNotFoundError(
        f"No train_prefs Parquet files found under {preference_data_dir}"
    )
binarized_data = datasets.load_dataset(
    "parquet",
    data_files=[str(path) for path in preference_shards],
    split="train",
)

# 使用全部记录之前先做固定seed的shuffle；这样训练顺序可复现，也减少文件顺序偏差。
binarized_data = binarized_data.shuffle(seed=42)

def tokenize_and_format(data, add_generation_prompt=False):
    encoded = tokenizer.apply_chat_template(
        data,
        tokenize = True,
        add_generation_prompt = add_generation_prompt,
        return_dict = True,
    )

    # Transformers 5可能返回tokenizers.Encoding；显式提取普通的token ID列表，
    # 否则后面的torch.tensor(original_sequence)无法推断dtype。
    return list(encoded["input_ids"])


def add_system_prompt(messages):
    """复制消息，并在需要时加入与SFT阶段相同的system prompt。"""
    copied_messages = [dict(message) for message in messages]
    if not copied_messages or copied_messages[0].get("role") != "system":
        copied_messages.insert(
            0,
            {"content": "You are a helpful assistant", "role": "system"}
        )
    return copied_messages


def tokenize_preference_pair(preference_row, pair_index):
    """对共同prompt只tokenize一次，并为两个回答保留相同上下文。

    原先直接右截断完整序列；长prompt会占满1700个token并删除回答。这里先
    分离共同prompt和两个completion，对prompt统一左截断，再拼接回答。
    """
    raw_chosen = preference_row['chosen']
    raw_rejected = preference_row['rejected']

    if raw_chosen[:-1] != raw_rejected[:-1]:
        raise ValueError(
            f"Preference pair {pair_index} does not share the same prompt"
        )
    if (
        not raw_chosen
        or not raw_rejected
        or raw_chosen[-1].get("role") != "assistant"
        or raw_rejected[-1].get("role") != "assistant"
    ):
        raise ValueError(
            f"Preference pair {pair_index} must end with assistant answers"
        )

    prompt_messages = add_system_prompt(raw_chosen[:-1])
    chosen_messages = add_system_prompt(raw_chosen)
    rejected_messages = add_system_prompt(raw_rejected)

    # generation prompt包含assistant header，但不包含回答内容。
    prompt_ids = tokenize_and_format(
        prompt_messages,
        add_generation_prompt=True
    )
    complete_chosen_ids = tokenize_and_format(chosen_messages)
    complete_rejected_ids = tokenize_and_format(rejected_messages)

    # 完整chosen/rejected必须以完全相同的prompt tokens开头。
    if (
        complete_chosen_ids[:len(prompt_ids)] != prompt_ids
        or complete_rejected_ids[:len(prompt_ids)] != prompt_ids
    ):
        raise RuntimeError(
            f"Chat-template prompt prefix mismatch in preference pair {pair_index}"
        )

    chosen_completion_ids = complete_chosen_ids[len(prompt_ids):]
    rejected_completion_ids = complete_rejected_ids[len(prompt_ids):]
    if not chosen_completion_ids or not rejected_completion_ids:
        raise ValueError(f"Preference pair {pair_index} has an empty completion")

    # 从左侧截断共同prompt，保留最接近回答的上下文和assistant header。
    prompt_ids = prompt_ids[-DPOConfig.max_prompt_length:]
    completion_budget = DPOConfig.max_length - len(prompt_ids)
    if completion_budget < 1:
        raise ValueError("max_prompt_length must be smaller than max_length")

    # completion从右侧截断以保留回答开头。两个序列仍共享完全相同的prompt。
    chosen_input_ids = prompt_ids + chosen_completion_ids[:completion_budget]
    rejected_input_ids = prompt_ids + rejected_completion_ids[:completion_budget]
    return chosen_input_ids, rejected_input_ids

## 同时生成chosen/rejected的input_ids，确保每个偏好对始终保持对齐
chosen_input_ids_list = []
rejected_input_ids_list = []

# 使用train_prefs中的全部偏好对，不再限制为前30000条。
number_of_preference_pairs = len(binarized_data)
for i in range(number_of_preference_pairs):
    chosen_input_ids, rejected_input_ids = tokenize_preference_pair(
        binarized_data[i], i
    )
    chosen_input_ids_list.append(chosen_input_ids)
    rejected_input_ids_list.append(rejected_input_ids)

    processed_count = i + 1
    if processed_count % 10000 == 0 or processed_count == number_of_preference_pairs:
        print(f"偏好对数据已处理{processed_count}条")

print('-' * 70)

## 确保数据条数一致
assert len(chosen_input_ids_list) == len(rejected_input_ids_list)

beta = DPOConfig.beta # β超参数
batch_size = DPOConfig.batch_size
gradient_accumulation_steps = DPOConfig.gradient_accumulation_steps
log_iter = DPOConfig.log_iter
max_lr = DPOConfig.max_lr
min_lr = DPOConfig.min_lr
warmup_steps = DPOConfig.warmup_steps
total_steps = math.ceil(len(chosen_input_ids_list) / batch_size)
total_optimizer_steps = math.ceil(total_steps / gradient_accumulation_steps)
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr)

##配置logging
import time

with open(f"log.txt", "a") as my_file:
    my_file.write(f' \
        time:{time.strftime("%Y-%m-%d, %H:%M:%S")}, \
        batch_size:{batch_size}, \
        warmup_steps:{warmup_steps}, \
        max_lr:{max_lr}, \
        min_lr:{min_lr}\n')

#定义一个日志记录函数
def log_call(iters, iters_average_loss):
    with open(f"log.txt", "a") as my_file:
        my_file.write(f' \
            time:{time.strftime("%Y-%m-%d, %H:%M:%S")}, \
            iters:{iters+1}, \
            iters_average_Loss:{iters_average_loss:.4f}\n')

def linear_warmup(current_step, warmup_steps, max_lr):
    if current_step < warmup_steps:
        # 第一次权重更新使用非零学习率；current_step从0开始。
        return max_lr * (current_step + 1) / warmup_steps
    else:
        return max_lr

def cosine_decay(current_step, warmup_steps, total_steps, max_lr, min_lr):
    if current_step < warmup_steps:
        return linear_warmup(current_step, warmup_steps, max_lr)
    else:
        decay_steps = total_steps - warmup_steps
        if decay_steps <= 1:
            return min_lr
        progress = (current_step - warmup_steps) / (decay_steps - 1)
        progress = min(max(progress, 0.0), 1.0)
        decay = 0.5 * (1 + np.cos(np.pi * progress))
        return (max_lr - min_lr) * decay + min_lr

def create_answer_mask(input_ids, tokenizer):
    """
    创建仅对助手回答部分计算损失的掩码
    
    Args:
        input_ids: 输入token序列 [batch_size, seq_len]
        tokenizer: 分词器
    
    Returns:
        answer_mask: 助手回答部分为1，其他部分为0的掩码
    """
    batch_size, _ = input_ids.shape
    answer_mask = torch.zeros_like(input_ids)

    # 不硬编码token ID和“+3”偏移；不同tokenizer可能把header编码成不同长度。
    assistant_header_ids = tokenizer.encode(
        '<|im_start|>assistant\n',
        add_special_tokens=False
    )
    message_end_ids = tokenizer.encode(
        '<|im_end|>',
        add_special_tokens=False
    )

    def find_subsequence(sequence, pattern, start=0):
        final_start = len(sequence) - len(pattern)
        for position in range(start, final_start + 1):
            if sequence[position:position + len(pattern)] == pattern:
                return position
        return None

    for batch_idx in range(batch_size):
        sequence = input_ids[batch_idx].tolist()

        # UltraFeedback train_prefs每条数据只有一个最终assistant回答。
        # 仍然查找最后一个header，避免未来数据出现历史assistant时错误地把历史
        # 回答也计入chosen/rejected概率。
        header_starts = []
        search_position = 0
        while True:
            header_start = find_subsequence(
                sequence, assistant_header_ids, search_position)
            if header_start is None:
                break
            header_starts.append(header_start)
            search_position = header_start + len(assistant_header_ids)

        if not header_starts:
            continue

        answer_start = header_starts[-1] + len(assistant_header_ids)
        message_end_start = find_subsequence(
            sequence, message_end_ids, answer_start)

        # 若回答右侧被max_length截断，仍训练现有回答token；若未截断，则包含
        # <|im_end|>，因为模型也需要学习何时停止回答。
        answer_end = (
            len(sequence)
            if message_end_start is None
            else message_end_start + len(message_end_ids)
        )
        answer_mask[batch_idx, answer_start:answer_end] = 1

    return answer_mask

def _compute_average_log_probability(logits, target_labels, mask):
    """
    计算带掩码的平均对数概率
    
    Args:
        logits: 模型输出 [batch_size, seq_len, vocab_size]
        target_labels: 目标标签 [batch_size, seq_len]
        mask: 计算掩码 [batch_size, seq_len]
    
    Returns:
        average_log_prob: 每个样本的平均对数概率 [batch_size]
    """
    # 直接使用log_softmax，避免低精度下softmax先下溢到0、随后log(0)=-inf。
    log_probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
    
    # 获取目标token的对数概率
    gathered_log_probs = torch.gather(
        log_probabilities, 
        dim=-1, 
        index=target_labels.unsqueeze(2)
    ).squeeze(2)
    
    # 应用掩码并计算平均值
    masked_log_probs = torch.mul(gathered_log_probs, mask)
    token_counts = mask.sum(dim=-1)
    if torch.any(token_counts == 0):
        raise RuntimeError("A DPO response has no valid assistant tokens")
    average_log_prob = masked_log_probs.sum(dim=-1) / token_counts
    
    return average_log_prob

model.train()

# ==================== 训练指标记录列表 ====================
training_losses = []
# 偏好的回答的概率
preferred_log_probabilities = []
# 讨厌的回答的概率
rejected_log_probabilities = []
# 偏好的回答的奖励
preferred_rewards = []
# 讨厌的回答的奖励
rejected_rewards = []
reward_margins = []

model.zero_grad()  # 训练开始时清空梯度
skipped_batches_count = 0
total_batches = math.ceil(len(chosen_input_ids_list) / batch_size)
optimizer_step = 0

for batch_idx in range(total_batches):
    ## ==================== 获取批次数据 ====================
    
    # 获取当前批次的偏好对数据
    preferred_batch_sequences = chosen_input_ids_list[
        batch_idx * batch_size:(batch_idx + 1) * batch_size
    ]
    rejected_batch_sequences = rejected_input_ids_list[
        batch_idx * batch_size:(batch_idx + 1) * batch_size
    ]

    ## ==================== 数据填充对齐 ====================
    
    # 计算各自批次的最大序列长度
    preferred_max_length = max([len(sequence) for sequence in preferred_batch_sequences])
    rejected_max_length = max([len(sequence) for sequence in rejected_batch_sequences])
    # 使用eos token作为pad token
    pad_token_id = model.generation_config.eos_token_id[-1]
    
    ### 偏好数据填充处理
    preferred_padded_sequences = []
    for seq_idx in range(len(preferred_batch_sequences)):
        original_sequence = preferred_batch_sequences[seq_idx]
        # 计算要填充多少个pad
        padding_length = preferred_max_length - len(original_sequence)
        # 在训练数据的末尾填充pad
        padded_sequence = torch.nn.functional.pad(
            torch.tensor(original_sequence), 
            (0, padding_length), 
            mode='constant', 
            value=pad_token_id
        ).tolist()
        # 将填充过的数据放入列表
        preferred_padded_sequences.append(padded_sequence)
    
    preferred_batch_tensor = torch.tensor(preferred_padded_sequences)

    # 根据原始长度构造attention mask，而不是用token值判断。Qwen的pad token
    # 可能同时也是合法的EOS token，按token值判断会误伤真实数据。
    preferred_attention_mask = torch.zeros_like(preferred_batch_tensor)
    for seq_idx, sequence in enumerate(preferred_batch_sequences):
        preferred_attention_mask[seq_idx, :len(sequence)] = 1
    
    ### 拒绝数据填充处理
    rejected_padded_sequences = []
    for seq_idx in range(len(rejected_batch_sequences)):
        original_sequence = rejected_batch_sequences[seq_idx]
        padding_length = rejected_max_length - len(original_sequence)
        
        padded_sequence = torch.nn.functional.pad(
            torch.tensor(original_sequence), 
            (0, padding_length), 
            mode='constant', 
            value=pad_token_id
        ).tolist()
        
        rejected_padded_sequences.append(padded_sequence)
    
    rejected_batch_tensor = torch.tensor(rejected_padded_sequences)

    rejected_attention_mask = torch.zeros_like(rejected_batch_tensor)
    for seq_idx, sequence in enumerate(rejected_batch_sequences):
        rejected_attention_mask[seq_idx, :len(sequence)] = 1

    ## ==================== 构建输入输出对 ====================
    
    # 构建因果语言模型的输入输出对：x->y（下一个词预测）
    # 模型的输入：偏好的回答
    preferred_model_inputs = preferred_batch_tensor[:, :-1].to(device)
    # 真实的标签
    preferred_target_labels = preferred_batch_tensor[:, 1:].to(device)
    preferred_model_attention_mask = preferred_attention_mask[:, :-1].to(device)
    
    rejected_model_inputs = rejected_batch_tensor[:, :-1].to(device)
    rejected_target_labels = rejected_batch_tensor[:, 1:].to(device)
    rejected_model_attention_mask = rejected_attention_mask[:, :-1].to(device)

    ## ==================== 构建训练掩码 ====================
    
    # 构建掩码矩阵：padding_mask（忽略填充token）+ answer_mask（只关注回答部分）
    
    # target是完整序列向左移动一位，因此padding mask也必须做同样的位移。
    preferred_padding_mask = preferred_attention_mask[:, 1:].to(device)
    rejected_padding_mask = rejected_attention_mask[:, 1:].to(device)
    
    # 助手回答的掩码：将助手回答的部分掩码为 1 。其它都是 0 。
    preferred_answer_mask = create_answer_mask(
        preferred_batch_tensor,
        tokenizer
    )[:, 1:].to(device)
    rejected_answer_mask = create_answer_mask(
        rejected_batch_tensor,
        tokenizer
    )[:, 1:].to(device)
    
    # 最终掩码：取交集
    preferred_final_mask = (preferred_answer_mask & preferred_padding_mask)
    rejected_final_mask = (rejected_answer_mask & rejected_padding_mask)

    ## ==================== 批次有效性检查 ====================
    
    # 检查偏好对数据是否都有有效的回答部分
    preferred_min_tokens = preferred_final_mask.sum(dim=-1).min().item()
    rejected_min_tokens = rejected_final_mask.sum(dim=-1).min().item()
    
    if preferred_min_tokens == 0 or rejected_min_tokens == 0:
        # 新的paired tokenization已为回答保留空间；若仍到达这里，说明chat
        # template或mask逻辑不兼容，不应静默跳过并破坏梯度累积边界。
        raise RuntimeError(
            f'第{batch_idx + 1}批次没有有效回答token，请检查chat template'
        )

    ## ==================== 模型前向传播 ====================
    
    # 训练模型对偏好数据的前向传播
    preferred_logits = model(
        input_ids=preferred_model_inputs,
        attention_mask=preferred_model_attention_mask
    ).logits

    # 训练模型对拒绝数据的前向传播
    rejected_logits = model(
        input_ids=rejected_model_inputs,
        attention_mask=rejected_model_attention_mask
    ).logits

    # 参考模型的前向传播（不计算梯度）
    with torch.no_grad():
        reference_preferred_logits = ref_model(                    \
            input_ids=preferred_model_inputs,                       \
            attention_mask=preferred_model_attention_mask           \
        )                                                           \
            .logits                                                    \
            .detach()
        reference_rejected_logits = ref_model(                       \
            input_ids=rejected_model_inputs,                         \
            attention_mask=rejected_model_attention_mask             \
        )                                                            \
            .logits                                                    \
            .detach()

    ## ==================== DPO损失计算 ====================
    """
    DPO (Direct Preference Optimization) 论文: https://arxiv.org/pdf/2305.18290.pdf
    核心思想：通过偏好对比学习，无需显式奖励模型
    """
    
    # 计算平均对数概率 (average_log_prob = True)
    # 参考: https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py#L924
    
    ### 训练模型的对数概率
    ### 正在微调的模型，接收到正例的logits，计算对数概率
    preferred_log_prob = _compute_average_log_probability(
        preferred_logits,
        preferred_target_labels,
        preferred_final_mask
    )
    rejected_log_prob = _compute_average_log_probability(
        rejected_logits,
        rejected_target_labels,
        rejected_final_mask
    )
    
    ### 参考模型的对数概率
    reference_preferred_log_prob = _compute_average_log_probability(
        reference_preferred_logits,
        preferred_target_labels,
        preferred_final_mask
    )
    reference_rejected_log_prob = _compute_average_log_probability(
        reference_rejected_logits,
        rejected_target_labels,
        rejected_final_mask
    )

    ## ==================== 奖励和边际计算 ====================
    
    # 计算隐式奖励 (基于KL散度)
    preferred_implicit_reward =                              \
        beta *                                               \
        (preferred_log_prob - reference_preferred_log_prob)
    rejected_implicit_reward =                               \
        beta *                                               \
        (rejected_log_prob - reference_rejected_log_prob)
    
    # 计算奖励边际 (偏好数据应该有更高的奖励)
    reward_margin = preferred_implicit_reward - rejected_implicit_reward
    
    # DPO损失：-log(sigmoid(margin))
    # logsigmoid在margin很小时仍保持数值稳定，避免log(sigmoid(...))产生-inf。
    sample_losses = -torch.nn.functional.logsigmoid(reward_margin)
    
    # 批次平均损失 + 梯度累积
    group_start = (batch_idx // gradient_accumulation_steps) \
                * gradient_accumulation_steps
    actual_accumulation_steps = min(
        gradient_accumulation_steps,
        total_batches - group_start
    )
    unscaled_batch_average_loss = torch.nanmean(sample_losses)

    # 全部61135条数据在batch_size=2时，最后一个microbatch只有1条偏好对。
    # 按本累积组的真实样本数缩放，保证这最后1条不会获得双倍权重。
    group_example_start = group_start * batch_size
    group_example_end = min(
        (group_start + actual_accumulation_steps) * batch_size,
        len(chosen_input_ids_list)
    )
    examples_in_group = group_example_end - group_example_start
    batch_average_loss = sample_losses.sum() / examples_in_group

    ## ==================== 反向传播和优化 ====================
    
    batch_average_loss.backward()

    # 学习率按真正的optimizer.step()次数变化，而不是按microbatch变化。
    if batch_idx % gradient_accumulation_steps == 0:
        current_learning_rate = cosine_decay(
            optimizer_step,
            warmup_steps,
            total_optimizer_steps,
            max_lr,
            min_lr
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_learning_rate

    # 梯度累积和权重更新
    is_accumulation_step = (batch_idx + 1) % gradient_accumulation_steps == 0
    is_final_batch = (batch_idx + 1) == total_batches
    
    if is_accumulation_step or is_final_batch:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), DPOConfig.max_gradient_norm)
        optimizer.step()        # 更新权重
        optimizer.zero_grad()   # 清空梯度
        optimizer_step += 1

    ## ==================== 训练指标记录 ====================
    
    # 记录各项训练指标（detach避免梯度追踪）
    training_losses.append(
        unscaled_batch_average_loss.detach().item())
    preferred_log_probabilities.append(
        torch.nanmean(preferred_log_prob.detach()).item())
    rejected_log_probabilities.append(
        torch.nanmean(rejected_log_prob.detach()).item())
    preferred_rewards.append(
        torch.nanmean(preferred_implicit_reward.detach()).item())
    rejected_rewards.append(torch.nanmean(
        rejected_implicit_reward.detach()).item())
    reward_margins.append(
        torch.nanmean(reward_margin.detach()).item())

    ## ==================== 训练日志输出 ====================
    
    should_log = (batch_idx + 1) % log_iter == 0 or is_final_batch
    
    if should_log:
        # 计算最近批次的平均指标
        recent_loss = np.nanmean(training_losses[-log_iter:])
        recent_preferred_logprob = np.nanmean(
            preferred_log_probabilities[-log_iter:])
        recent_rejected_logprob = np.nanmean(
            rejected_log_probabilities[-log_iter:])
        recent_preferred_reward = np.nanmean(preferred_rewards[-log_iter:])
        recent_rejected_reward = np.nanmean(rejected_rewards[-log_iter:])
        recent_margin = np.nanmean(reward_margins[-log_iter:])
        
        # 格式化输出训练状态
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f'⏰ 时间: {current_time}')
        print(f'📊 批次: {batch_idx + 1}/{total_batches}')
        print(f'📈 最近{log_iter}批次指标:')
        print(f'   - 平均损失: {recent_loss:.4f}')
        print(f'   - 偏好对数概率: {recent_preferred_logprob:.4f}')
        print(f'   - 拒绝对数概率: {recent_rejected_logprob:.4f}')
        print(f'   - 偏好奖励: {recent_preferred_reward:.4f}')
        print(f'   - 拒绝奖励: {recent_rejected_reward:.4f}')
        print(f'   - 奖励边际: {recent_margin:.4f}')
        print(f'🎯 学习率: {current_learning_rate:.2e}')
        print('-' * 80)
        
        # 调用外部日志记录
        log_call(batch_idx, recent_loss)

## ==================== 训练完成总结 ====================

print("🎉 DPO训练完成!")
print(f'📊 训练统计:')
print(f'   - 总批次数: {total_batches}')
print(f'   - 跳过批次数: {skipped_batches_count}')
print(f'   - 有效批次数: {total_batches - skipped_batches_count}')

# 输出最终训练指标
if training_losses:
    final_metrics = {
        'loss': np.nanmean(training_losses[-100:]),
        'preferred_logprob': np.nanmean(preferred_log_probabilities[-100:]),
        'rejected_logprob': np.nanmean(rejected_log_probabilities[-100:]),
        'preferred_reward': np.nanmean(preferred_rewards[-100:]),
        'rejected_reward': np.nanmean(rejected_rewards[-100:]),
        'margin': np.nanmean(reward_margins[-100:])
    }
    
    print(f'🎯 最终指标 (最近100批次平均):')
    for metric_name, metric_value in final_metrics.items():
        print(f'   - {metric_name}: {metric_value:.4f}')

if skipped_batches_count > 0:
    skip_ratio = skipped_batches_count / total_batches * 100
    print(f'⚠️ 跳过批次占比: {skip_ratio:.2f}%')
    if skip_ratio > 10:
        print('💡 建议: 跳过批次过多，考虑增加最大序列长度或优化数据预处理')

# Python库不会自动展开字符串中的~，因此先用Path.expanduser()得到真实目录。
output_model_path = Path(
    '~/scratch/llms_model/post_trained_models/Qwen3-0.6B-SFT-DPO'
).expanduser()
output_model_path.mkdir(parents=True, exist_ok=True)
# 训练时为gradient checkpointing关闭了KV cache；保存前恢复它，确保之后使用
# transformers serve/chat进行自回归生成时不会因缺少cache而明显变慢。
model.config.use_cache = True
model.save_pretrained(output_model_path)
tokenizer.save_pretrained(output_model_path)
print(f'DPO模型已保存到: {output_model_path}')
