import html
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from countdown_task import CountdownTasksDataset, reward_function
from grpo import rollout, update_policy
from qwen2_model import Transformer
from tokenizer import Tokenizer


def evaluate(model, tokenizer, device, dtype):
    """每隔10个step，使用测试数据集评估一下，看能做对多少题"""
    test_dataset = CountdownTasksDataset(
        data_path="Countdown-Tasks-3to4",
        tokenizer=tokenizer,
        split="test",
        test_size=128, # 128条测试数据
    )
    generator = torch.Generator(device=device)
    # 批次大小减半，我们就可以生成2倍长的轨迹了
    dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=CountdownTasksDataset.collate_fn,
        generator=generator,
        # 批次大小为256，减半
        batch_size=256 // 2,
        drop_last=False,
    )
    success = []
    for batch in dataloader:
        episodes = rollout(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            # 最大生成长度为1024，乘以2
            max_gen_len=1024 * 2,
            # 评估时，针对每个问题只生成1个回答
            num_answer_per_question=1,
            reward_function=reward_function,
            device=device,
            dtype=dtype,
        )
        success.extend([
            episode.reward_info["answer_reward"] \
            for episode in episodes
        ])
    return np.mean(success)


def main():
    # pretrained_model_path = Path(
    #     "./Qwen2.5-3B-Instruct/"
    # )

    pretrained_qwen253_b_instruct_model_path = Path(
        "~/scratch/llms_model/GRPO-Zero/Qwen2.5-3B-Instruct"
    ).expanduser()

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.set_default_device(device)
    torch.random.manual_seed(1337)
    # 批次大小
    BATCH_SIZE = 256
    # 每个批次32个问题
    NUM_QUESTIONS_PER_BATCH = 32
    # 每个问题产生8条回答
    NUM_ANSWERS_PER_QUESTION = \
        BATCH_SIZE // NUM_QUESTIONS_PER_BATCH

    current_time = datetime.now().strftime(r"%Y%m%d-%H%M%S")
    tb_writer = SummaryWriter(log_dir=f"./logs/{current_time}")
    # tokenizer = Tokenizer("~/scratch/llms_model/GRPO-Zero/Qwen2.5-3B-Instruct/tokenizer.json")
    tokenizer = Tokenizer("/iridisfs/scratch/mh1f25/llms_model/GRPO-Zero/Qwen2.5-3B-Instruct/tokenizer.json")
    train_dataset = CountdownTasksDataset(
        data_path="./Countdown-Tasks-3to4/",
        tokenizer=tokenizer,
        split="train",
        test_size=128,
    )
    generator = torch.Generator(device=device)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=CountdownTasksDataset.collate_fn,
        generator=generator,
        batch_size=NUM_QUESTIONS_PER_BATCH,
    )

    model = Transformer.from_pretrained(
        pretrained_qwen253_b_instruct_model_path,
        device=device
    ).train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1.0e-5,
        weight_decay=0.0,
        betas=[0.9, 0.999],
    )

    start_time = time.time()
    ckpt_dir = Path("~/scratch/llms_model/GRPO-Zero/Qwen2.5-3B-Instruct_ckpt").expanduser()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for step, batch in enumerate(train_dataloader, start=1):
        # 生成轨迹（问题+回答）数据
        episodes = rollout(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            max_gen_len=1024,
            num_answer_per_question=NUM_ANSWERS_PER_QUESTION,
            reward_function=reward_function,
            device=device,
            dtype=dtype,
        )
        # 更新策略
        results = update_policy(
            model=model,
            optimizer=optimizer,
            episodes=episodes,
            micro_batch_size=2, # 微批次大小为2
            pad_token_id=tokenizer.pad_token_id,
            max_grad_norm=1.0, # 梯度裁剪到1.0
            device=device,
            dtype=dtype,
        )
        torch.cuda.synchronize()
        end_time = time.time()
        duration = end_time - start_time
        start_time = end_time

        # 计算一些统计信息，然后保存到日志文件中
        reward = [episode.reward for episode in episodes]
        formatted_reward = [
            episode.reward_info["format_reward"] \
            for episode in episodes
        ]
        answer_reward = [
            episode.reward_info["answer_reward"] \
            for episode in episodes
        ]
        num_finished_episodes = sum(
            episode.is_finished for episode in episodes
        )
        mean_reward = np.mean(reward)
        std_reward = np.std(reward)
        success_rate = np.mean(answer_reward)
        format_reward = np.mean(formatted_reward)
        grad_norm = results["grad_norm"]
        entropy = results["entropy"]
        lr = optimizer.param_groups[0]["lr"]
        loss = results["loss"]
        mean_response_len = np.mean(
            [len(episode.generated_token_ids) \
             for episode in episodes]
        )
        print(
            f"\r步骤 {step}, 平均奖励: {mean_reward:.2f}, "
            f"计算正确率: {success_rate:.2f}, "
            f"梯度裁剪: {grad_norm:.2f}, 时长: {duration:.2f}, "
            f"结束的回合的数量: {num_finished_episodes}, "
            f"平均回答长度: {mean_response_len:.2f}, "
            f"熵: {entropy:.2f}"
        )
        # 每隔10步评估一次
        if step % 10 == 0:
            eval_success_rate = evaluate(
                model,
                tokenizer,
                device,
                dtype
            )
            print(f"\r评估数据集回答正确率: \
                   {eval_success_rate:.2f}" \
                   + " " * 100)
            tb_writer.add_scalar(
                "回答正确率/评估",
                eval_success_rate,
                step
            )

        tb_writer.add_scalar("损失", loss, step)
        tb_writer.add_scalar("平均奖励", mean_reward, step)
        tb_writer.add_scalar("奖励的标准差", std_reward, step)
        tb_writer.add_scalar(
            "回答正确率/训练", success_rate, step)
        tb_writer.add_scalar("格式奖励", format_reward, step)
        tb_writer.add_scalar("梯度裁剪", grad_norm, step)
        tb_writer.add_scalar("时长", duration, step)
        tb_writer.add_scalar(
            "结束的回合数量",
            num_finished_episodes,
            step
        )
        tb_writer.add_scalar("学习率", lr, step)
        tb_writer.add_scalar(
            "平均回答长度", mean_response_len, step)
        tb_writer.add_scalar("熵", entropy, step)
        for i, episode in enumerate(episodes):
            # TensorBoard 将文本处理为markdown格式
            text = html.escape(episode.text)
            tb_writer.add_text(
                f"text_{i}",
                f"<pre>{text}</pre>",
                step
            )

        # 每隔100步保存模型的检查点
        if step % 100 == 0:
            output_file = ckpt_dir / f"ckpt_{step:06d}.pt"
            torch.save(model.state_dict(), output_file)
            print(f"将检查点保存到 {output_file}")


if __name__ == "__main__":
    main()