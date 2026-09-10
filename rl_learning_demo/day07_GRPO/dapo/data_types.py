from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Episode:
    """存储一个回合（Episode）或者说一条轨迹的所有相关信息"""
    """一个回合 = 问题 + 一条回答"""
    prefix: str # 问题
    text: str # “问题+回答”整个文本
    prefix_token_ids: List[int] # 问题的input_ids
    prefix_tokens: List[str] # 问题的token组成的列表
    generated_token_ids: List[int] # 生成的回答的token列表
    is_finished: bool # 回答是否结束标志位
    reward: float # 奖励
    reward_info: Dict[str, float] # 详细的奖励信息


@dataclass
class MiniBatch:
    """每个Step训练所需的微批次数据"""
    prefix: List[str] # 问题列表
    prefix_tokens: List[List[str]]
    prefix_token_ids: List[List[int]]
    numbers: List[List[int]] # 问题的数字列表
    target: List[int] # 问题对应的答案数字