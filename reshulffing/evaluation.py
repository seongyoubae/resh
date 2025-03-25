import random
import numpy as np
import torch
import os
import math
import csv
import pandas as pd
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


def evaluate_policy(model, envs, device):
    model.eval()
    # 임시로 temperature_param = 0 세팅(결정론적)
    if hasattr(model, "temperature_param"):
        original_temp = model.temperature_param.data.clone()
        model.temperature_param.data.fill_(0.0)

    total_rewards = []
    total_reversals = []

    for idx, env in enumerate(envs):
        # 환경 내 plate 정보(특히 outbound) 디버깅
        # env.inbound_plates가 전체 plate 리스트라고 가정
        print(f"[DEBUG] Eval Env {idx}: Checking outbound of each plate")
        for p_idx, plate in enumerate(env.inbound_plates):
            print(f"  Plate {p_idx} -> id={plate.id}, inbound={plate.inbound}, outbound={plate.outbound}")

        state = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            state_tensor = state.unsqueeze(0).to(device)
            with torch.no_grad():
                # 평가용 마스크 (1,len(from_keys)) -> (1, MAX_SOURCE) 패딩
                s_mask_1d = torch.ones(len(env.from_keys), dtype=torch.bool, device=device)
                # 패딩 로직(예시)
                if s_mask_1d.size(0) < 30:
                    padded = torch.zeros(30, dtype=torch.bool, device=device)
                    padded[:s_mask_1d.size(0)] = s_mask_1d
                    s_mask_1d = padded
                s_mask_2d = s_mask_1d.unsqueeze(0)

                d_mask_1d = torch.ones(len(env.to_keys), dtype=torch.bool, device=device)
                # 패딩
                if d_mask_1d.size(0) < 30:
                    padded_d = torch.zeros(30, dtype=torch.bool, device=device)
                    padded_d[:d_mask_1d.size(0)] = d_mask_1d
                    d_mask_1d = padded_d
                d_mask_2d = d_mask_1d.unsqueeze(0)

                actions, _, _, _ = model.act_batch(
                    state_tensor,
                    s_mask_2d, d_mask_2d,
                    greedy=True
                )
            action = actions[0]
            source_idx = action[0].item()
            dest_idx = action[1].item()

            # 범위 체크
            if source_idx >= len(env.from_keys):
                source_idx = 0
            if dest_idx >= len(env.to_keys):
                dest_idx = 0

            next_state, reward, done, _ = env.step((source_idx, dest_idx))
            ep_reward += reward
            state = next_state

        print(f"[DEBUG] Eval Env {idx} done => Reward={ep_reward}, Reversal={env.last_reversal}")
        total_rewards.append(ep_reward)
        total_reversals.append(env.last_reversal)

    # 평가 결과 요약
    avg_reward = sum(total_rewards) / len(total_rewards) if total_rewards else 0
    avg_reversal = sum(total_reversals) / len(total_reversals) if total_reversals else 0
    print(f"[DEBUG] Eval total => Avg Reward={avg_reward:.2f}, Avg Reversal={avg_reversal:.2f}")

    # 온도 복원
    if hasattr(model, "temperature_param"):
        model.temperature_param.data.copy_(original_temp)

    return avg_reward, avg_reversal


def main():
    set_seed(970517)
    cfg = get_cfg()
    device = torch.device(cfg.device)

    # 모델 생성 (학습 때와 동일)
    model = SteelPlateConditionalMLPModel(
        embed_dim=cfg.embed_dim,
        num_actor_layers=cfg.num_actor_layers,
        num_critic_layers=cfg.num_critic_layers,
        max_source=MAX_SOURCE,
        max_dest=MAX_DEST
    ).to(device)

    # 체크포인트 로드
    if os.path.exists(cfg.save_model_dir):
        ckpts = [f for f in os.listdir(cfg.save_model_dir) if f.endswith('.pth')]
        if ckpts:
            latest = max(ckpts, key=lambda x: os.path.getmtime(os.path.join(cfg.save_model_dir, x)))
            cpath = os.path.join(cfg.save_model_dir, latest)
            model.load_state_dict(torch.load(cpath, map_location=device))
            print(f"Loaded checkpoint={cpath}")
        else:
            print("No .pth found, use untrained model.")
    else:
        print("No model dir found, use untrained model.")

    # 평가환경 생성
    schedule=[]
    try:
        df = pd.read_excel(cfg.plates_data_path, sheet_name="reshuffle")
        for idx, row in df.iterrows():
            plate_id = row["pileno"]
            inbound  = row["inbound"] if ("inbound" in df.columns and not pd.isna(row["inbound"])) else random.randint(cfg.inbound_min,cfg.inbound_max)
            outbound = row["outbound"] if("outbound" in df.columns and not pd.isna(row["outbound"])) else inbound+random.randint(cfg.outbound_extra_min,cfg.outbound_extra_max)
            unitw    = row["unitw"] if("unitw" in df.columns and not pd.isna(row["unitw"])) else random.uniform(cfg.unitw_min,cfg.unitw_max)
            to_pile  = str(row["topile"]).strip() if("topile" in df.columns and not pd.isna(row["topile"])) else "A01"
            p=Plate(plate_id,inbound,outbound,unitw)
            p.from_pile=plate_id
            p.topile   = to_pile
            schedule.append(p)
    except:
        schedule = generate_schedule(num_plates=cfg.num_plates)

    # 평가용 env 여러개
    eval_envs=[]
    for _ in range(5):
        env = Locating(
            num_pile=cfg.num_pile,
            max_stack=cfg.max_stack,
            inbound_plates=schedule,
            device=device,
            crane_penalty=cfg.crane_penalty,
            from_keys=None, # 자동
            to_keys=None    # 자동
        )
        eval_envs.append(env)

    avg_reward, avg_reversal = evaluate_policy(model, eval_envs, device)

    # 로그 기록
    eval_log = cfg.evaluation_log_file
    if os.path.dirname(eval_log):
        os.makedirs(os.path.dirname(eval_log), exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(eval_log, mode="a", newline="") as f:
        writer=csv.writer(f)
        writer.writerow([timestamp, avg_reward, avg_reversal])

if __name__=="__main__":
    main()
