import random
import numpy as np
import torch
import os
import csv
import pandas as pd
import math
import datetime
from cfg import get_cfg
from env import Locating
from data import Plate, generate_schedule
from network import SteelPlateConditionalMLPModel, MAX_SOURCE, MAX_DEST
import torch.nn.functional as F


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_eval_schedule(excel_file):
    """
    주어진 엑셀 파일에서 평가용 schedule을 구성합니다.
    엑셀 파일의 컬럼으로 pileno, inbound, outbound, unitw, topile 등이 있다고 가정합니다.
    각 Plate 객체에 대해 from_pile과 topile 속성을 설정하여 반환합니다.
    """
    df = pd.read_excel(excel_file, sheet_name="reshuffle")
    schedule = []
    for idx, row in df.iterrows():
        p = Plate(
            id=row["pileno"],
            inbound=row["inbound"] if "inbound" in row and not pd.isna(row["inbound"]) else random.randint(1, 10),
            outbound=row["outbound"] if "outbound" in row and not pd.isna(row["outbound"]) else random.randint(1, 30),
            unitw=row["unitw"] if "unitw" in row and not pd.isna(row["unitw"]) else random.uniform(0.5, 2.0)
        )
        p.from_pile = str(row["pileno"])
        if "topile" in row and not pd.isna(row["topile"]):
            p.topile = str(row["topile"]).strip()
        else:
            p.topile = str(random.randint(0, MAX_DEST - 1))
        schedule.append(p)
    return schedule


def reshape_state(state):
    """
    state가 numpy array인 경우,
    총 원소 수(state.size)가 (MAX_SOURCE+MAX_DEST)로 나누어떨어지면
    feature dimension을 자동으로 계산하여 [total_piles, feature_dim]으로 reshape합니다.
    그렇지 않으면 경고를 출력하고 원래 state를 그대로 반환합니다.
    """
    total_piles = MAX_SOURCE + MAX_DEST
    numel = state.size
    if numel % total_piles != 0:
        print(f"[WARNING] State size ({numel}) is not divisible by total_piles ({total_piles}). Using original state shape.")
        return state
    feature_dim = numel // total_piles
    return state.reshape(total_piles, feature_dim)


def evaluate_policy(model, envs, device, num_eval_episodes=5):
    """
    여러 평가 환경(envs)에 대해 학습된 모델을 실행합니다.
    각 환경마다 num_eval_episodes개의 에피소드를 실행하여, 에피소드 보상과 reversal 값을 누적한 후 평균을 계산합니다.
    각 step마다 디버그 메시지를 출력하여 상태, 선택된 행동, 보상 등을 추적합니다.
    """
    model.eval()
    all_rewards = []
    all_reversals = []

    for idx, env in enumerate(envs):
        print(f"[DEBUG] Eval Env {idx}: Checking outbound for each plate")
        for p_idx, plate in enumerate(env.inbound_plates):
            print(f"  Plate {p_idx} -> id={plate.id}, inbound={plate.inbound}, outbound={plate.outbound}")
        env_rewards = []
        env_reversals = []
        # 각 환경에서 num_eval_episodes번 반복
        for ep in range(num_eval_episodes):
            print(f"[DEBUG] Eval Env {idx} - Starting Episode {ep}")
            state = env.reset()
            # state가 1D이면 reshape 시도
            state_np = np.array(state)
            if state_np.ndim == 1:
                state = reshape_state(state_np)
            else:
                state = state_np
            done = False
            ep_reward = 0.0
            step_count = 0

            while not done:
                # 상태를 Tensor로 변환하고 배치 차원 추가: [1, total_piles, feature_dim]
                state_tensor = torch.tensor(state, dtype=torch.float).clone().detach().unsqueeze(0).to(device)
                print(f"[DEBUG] Episode {ep} Step {step_count}: state shape = {state_tensor.shape}")

                # 마스크 생성: env.get_masks()는 1D 텐서 또는 리스트라고 가정
                env_source_mask, env_dest_mask = env.get_masks()
                if not isinstance(env_source_mask, torch.Tensor):
                    env_source_mask = torch.tensor(env_source_mask, dtype=torch.bool, device=device)
                if not isinstance(env_dest_mask, torch.Tensor):
                    env_dest_mask = torch.tensor(env_dest_mask, dtype=torch.bool, device=device)

                source_mask = torch.zeros(1, MAX_SOURCE, dtype=torch.bool, device=device)
                dest_mask = torch.zeros(1, MAX_DEST, dtype=torch.bool, device=device)
                valid_source_length = min(int(env_source_mask.size(0)), MAX_SOURCE)
                valid_dest_length = min(int(env_dest_mask.size(0)), MAX_DEST)
                source_mask[0, :valid_source_length] = env_source_mask[:valid_source_length]
                dest_mask[0, :valid_dest_length] = env_dest_mask[:valid_dest_length]

                with torch.no_grad():
                    actions, _, _, _ = model.act_batch(state_tensor, source_mask, dest_mask, greedy=True)
                action = actions[0]
                source_idx = action[0].item()
                dest_idx = action[1].item()
                print(f"[DEBUG] Episode {ep} Step {step_count}: Selected action = (source: {source_idx}, dest: {dest_idx})")

                if valid_source_length == 0:
                    print(f"[DEBUG] Env {idx} 에피소드 {ep}: No valid source actions.")
                    break
                if source_idx >= valid_source_length or not bool(env_source_mask[source_idx]):
                    valid_indices = [i for i in range(valid_source_length) if bool(env_source_mask[i])]
                    if valid_indices:
                        print(f"[DEBUG] Episode {ep} Step {step_count}: Clipping source_idx {source_idx} to {valid_indices[0]}")
                        source_idx = valid_indices[0]
                    else:
                        break

                if valid_dest_length == 0:
                    print(f"[DEBUG] Env {idx} 에피소드 {ep}: No valid destination actions.")
                    break
                if dest_idx >= valid_dest_length or not bool(env_dest_mask[dest_idx]):
                    valid_indices = [i for i in range(valid_dest_length) if bool(env_dest_mask[i])]
                    if valid_indices:
                        print(f"[DEBUG] Episode {ep} Step {step_count}: Clipping dest_idx {dest_idx} to {valid_indices[0]}")
                        dest_idx = valid_indices[0]
                    else:
                        break

                next_output = env.step((source_idx, dest_idx))
                next_state = next_output[0]
                reward = next_output[1]
                done = next_output[2]
                print(f"[DEBUG] Episode {ep} Step {step_count}: Reward = {reward}, Done = {done}")

                next_state_np = np.array(next_state)
                if next_state_np.ndim == 1:
                    state = reshape_state(next_state_np)
                else:
                    state = next_state_np
                ep_reward += reward
                step_count += 1

            print(f"[DEBUG] Eval Env {idx} 에피소드 {ep} done => Reward={ep_reward}, Reversal={env.last_reversal}")
            env_rewards.append(ep_reward)
            env_reversals.append(env.last_reversal)

        avg_env_reward = sum(env_rewards) / len(env_rewards) if env_rewards else 0.0
        avg_env_reversal = sum(env_reversals) / len(env_reversals) if env_reversals else 0.0
        print(f"[DEBUG] Eval Env {idx}: Average Reward over {num_eval_episodes} episodes = {avg_env_reward:.2f}, Average Reversal = {avg_env_reversal:.2f}")
        all_rewards.append(avg_env_reward)
        all_reversals.append(avg_env_reversal)

    overall_avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
    overall_avg_reversal = sum(all_reversals) / len(all_reversals) if all_reversals else 0.0
    print(f"[DEBUG] Overall Eval: Avg Reward = {overall_avg_reward:.2f}, Avg Reversal = {overall_avg_reversal:.2f}")
    return overall_avg_reward, overall_avg_reversal


if __name__ == "__main__":
    set_seed(1)
    cfg = get_cfg()
    device = torch.device(cfg.device)

    # 모델 초기화 및 체크포인트에서 불러오기
    model = SteelPlateConditionalMLPModel(
        embed_dim=cfg.embed_dim,
        target_entropy=-math.log(1.0 / (MAX_SOURCE * MAX_DEST)),
        num_actor_layers=cfg.num_actor_layers,
        num_critic_layers=cfg.num_critic_layers,
        actor_init_std=cfg.actor_init_std,
        critic_init_std=cfg.critic_init_std,
        max_stack=cfg.max_stack,
        num_from_piles=MAX_SOURCE,
        num_to_piles=MAX_DEST
    ).to(device)
    checkpoint_path = "model_epoch1000.pth"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("체크포인트에서 모델을 불러왔습니다.")
    else:
        print("체크포인트 파일이 존재하지 않습니다.")
    model.eval()

    # 평가용 schedule을 "reshuffle_plan(for eval).xlsx"에서 불러오기
    eval_excel_file = "output/reshuffle_plan(for eval).xlsx"
    if os.path.exists(eval_excel_file):
        eval_schedule = load_eval_schedule(eval_excel_file)
        print("평가용 schedule을 엑셀 파일에서 불러왔습니다.")
    else:
        print("평가용 엑셀 파일이 존재하지 않습니다. 기본 generate_schedule() 사용.")
        eval_schedule = generate_schedule(num_plates=cfg.num_plates)
        for p in eval_schedule:
            p.from_pile = str(p.id)
            p.topile = str(random.randint(0, MAX_DEST - 1))

    # 평가 환경 구성 (Locating 환경)
    num_eval_envs = 1
    eval_envs = []
    for _ in range(num_eval_envs):
        env = Locating(
            num_pile=10,
            max_stack=cfg.max_stack,
            inbound_plates=eval_schedule,
            device=device,
            crane_penalty=cfg.crane_penalty,
        )
        eval_envs.append(env)

    avg_reward, avg_reversal = evaluate_policy(model, eval_envs, device, num_eval_episodes=5)
    print(f"평가 결과: 평균 보상 = {avg_reward:.2f}, 평균 reversal = {avg_reversal:.2f}")
