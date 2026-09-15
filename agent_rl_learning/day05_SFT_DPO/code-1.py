from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import torch

data = [
    {
        "content": "你是谁？",
        "role": "user"
    },
    {
        "content": "我是左元。",
        "role": "assistant"
    },
    {
        "content": "你会强化学习吗？",
        "role": "user"
    },
    {
        "content": "略知一二。",
        "role": "assistant"
    }
]

device = "cuda"
# The shell expands ``~`` in commands, but Python libraries generally do not.
# Path.expanduser() converts it to /home/<username>/... before Transformers
# receives it. Converting to str also works across Transformers versions.
model_path = str(Path("~/scratch/llms_model/Qwen3-0.6B-Base").expanduser())
# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)


def tokenize_and_format(data):
    # Transformers 5 can return a ``tokenizers.Encoding`` when tokenization
    # options such as truncation are enabled. Request a mapping explicitly and
    # extract its token-ID list so the code below is version-stable.
    encoded = tokenizer.apply_chat_template(
        data,
        tokenize=True,
        add_generation_prompt=False,
        truncation=True,
        max_length=2500,
        return_dict=True,
    )

    return list(encoded["input_ids"])


data.insert(
    0,
    {"content": "You are a helpful assistant", "role": "system"}
)

input_ids = tokenize_and_format(data)

print(tokenizer.decode(input_ids))


def create_answer_mask(input_ids, tokenizer):
    """
    创建仅对助手回答部分计算损失的掩码

    Args:
        input_ids: 输入token序列 [batch_size, seq_len]
        tokenizer: 分词器

    Returns:
        answer_mask: 助手回答部分为1，其他部分为0的掩码
    """

    batch_size, seq_len = input_ids.shape
    answer_mask = torch.zeros_like(input_ids)  # 全零张量，后续 回答部分会赋值为1

    # 获取结束标记的token id
    eos_token_id = tokenizer.encode('<|im_end|>')[0]

    for batch_idx in range(batch_size):
        # 找到所有 <|im_end|> 的位置
        eos_positions = torch.where(
            input_ids[batch_idx] == eos_token_id
        )[0].tolist()

        if len(eos_positions) < 2:  # 至少需要user和assistant各一个结束标记
            continue

        # 解析对话轮次
        user_ends, assistant_ends = \
            _parse_conversation_turns(eos_positions)

        # 为每个助手回答设置掩码
        _set_answer_masks(
            answer_mask[batch_idx],
            user_ends,
            assistant_ends,
            seq_len
        )

    return answer_mask


def _parse_conversation_turns(eos_positions):
    """
    解析对话轮次，分离用户和助手的结束位置

    对话格式：
    <|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{assistant_msg}<|im_end|>\n

    eos_positions[0]: system结束 (如果有)
    eos_positions[1]: 第1轮user结束
    eos_positions[2]: 第1轮assistant结束
    eos_positions[3]: 第2轮user结束
    eos_positions[4]: 第2轮assistant结束
    ...
    """
    # 跳过system系统提示词部分，从第一个user开始
    conversation_eos = eos_positions[1:]  # 去掉system的<im_end>

    # 偶数索引：user结束位置，奇数索引：assistant结束位置
    user_ends = [pos + 1 for pos in conversation_eos[::2]]  # 每隔2个取一个，从0开始
    assistant_ends = [
        pos + 1 for pos in conversation_eos[1::2]]  # 每隔2个取一个，从1开始

    return user_ends, assistant_ends


def _set_answer_masks(mask, user_ends, assistant_ends, seq_len):
    """
    为助手回答部分设置掩码

    Args:
        mask: 当前样本的掩码 [seq_len]
        user_ends: 用户消息结束位置列表
        assistant_ends: 助手消息结束位置列表
        seq_len: 序列长度
    """

    num_user_turns = len(user_ends)
    num_assistant_turns = len(assistant_ends)

    if num_user_turns == num_assistant_turns:
        # 完整对话：每轮都有用户问题和助手回答
        for user_end, assistant_end in zip(user_ends, assistant_ends):
            answer_start = user_end + 3  # 跳过 <|im_start|>assistant\n
            answer_end = assistant_end - 1  # 不包含 <|im_end|>
            mask[answer_start:answer_end] = 1

    elif num_user_turns == num_assistant_turns + 1:
        # 未完成对话：最后一轮助手回答被截断

        # 处理完整的对话轮次
        for user_end, assistant_end in zip(user_ends[:-1], assistant_ends):
            answer_start = user_end + 3
            answer_end = assistant_end - 1
            mask[answer_start:answer_end] = 1

        # 处理最后一轮被截断的助手回答
        last_user_end = user_ends[-1]
        last_answer_start = last_user_end + 3
        mask[last_answer_start:] = 1  # 到序列结尾


input_ids = torch.tensor(input_ids).unsqueeze(0)
mask = create_answer_mask(input_ids, tokenizer)

# print(input_ids * mask)
# a1 = [198, 104198,  77559,  23305,   1773]
# a2 = [198, 151667,    271, 151668,    271,  99475, 52183, 112190,   1773]
#
# print("=================")
# print(tokenizer.decode(a1))
# print("=================")
# print(tokenizer.decode(a2))
