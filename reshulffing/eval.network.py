import torch
import pandas as pd
import random
import numpy as np
import math
from cfg import get_cfg
from env import Locating
from data import Plate, generate_schedule
from network import SteelPlateConditionalMLPModel
import copy

##########################################
# 1) 마스크 패딩 함수 추가 (1D -> (1, max_size))
##########################################
MAX_SOURCE = 30
MAX_DEST   = 30
PAD_INPUT_DIM = (MAX_SOURCE + MAX_DEST) * 3  # 예: 180

def pad_boolean_mask_1d_to_2d(mask_1d: torch.Tensor, max_size: int) -> torch.Tensor:
    """
    1D Boolean mask를 (1, max_size)로 패딩
    - 부족분은 0(False)로 채우고, 초과분은 잘라냄
    """
    current_size = mask_1d.size(0)
    if current_size < max_size:
        out = torch.zeros(max_size, dtype=torch.bool, device=mask_1d.device)
        out[:current_size] = mask_1d
        mask_1d = out
    else:
        mask_1d = mask_1d[:max_size]
    return mask_1d.unsqueeze(0)

def load_model(model, model_path, device):
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    return model

def evaluate_model(model, eval_env, device, num_eval_episodes=20):
    model.eval()
    total_rewards = []
    total_reversals = []
    for epi in range(num_eval_episodes):

        env_schedule = copy.deepcopy(schedule)

        eval_env = Locating(
            num_pile=num_pile,
            max_stack=cfg.max_stack,
            inbound_plates=env_schedule,  # 고정된 schedule
            device=cfg.device,
            crane_penalty=cfg.crane_penalty,
            from_keys=from_piles,
            to_keys=allowed_piles
        )

        state = eval_env.reset(shuffle_schedule=False)
        done = False
        ep_reward = 0.0
        while not done:
            # state_tensor의 크기를 확인 (예: (240,))
            state_tensor = state.unsqueeze(0).to(device)
            # input_mask_eval: state의 원래 크기 (예: (1, state_dim))로 생성
            input_mask_eval = torch.ones((state_tensor.size(0), state_tensor.size(-1)), dtype=torch.bool, device=device)
            # 이렇게 생성하면, 예를 들어 state_tensor가 (1,240)이면 input_mask_eval는 (1,240),
            # pad_input_state_and_mask 함수 내부에서 둘 다 PAD_INPUT_DIM (예: 180)로 잘라내게 됩니다.

            # from_keys, to_keys용 마스크 생성 (기존대로)
            source_mask_1d = torch.ones(len(eval_env.from_keys), dtype=torch.bool, device=device)
            source_mask_2d = pad_boolean_mask_1d_to_2d(source_mask_1d, MAX_SOURCE)

            dest_mask_1d = torch.ones(len(eval_env.to_keys), dtype=torch.bool, device=device)
            dest_mask_2d = pad_boolean_mask_1d_to_2d(dest_mask_1d, MAX_DEST)

            with torch.no_grad():
                actions, _, _, _ = model.act_batch(
                    state_tensor,
                    input_mask_eval,  # (1, state_dim) → 예: (1,240), 이후 내부에서 (1,180)로 잘림
                    source_mask_2d,   # (1, MAX_SOURCE)
                    dest_mask_2d,     # (1, MAX_DEST)
                    greedy=True
                )

            action = actions[0]
            from_index = action[0].item()
            to_index = action[1].item()

            # 범위 초과 시 안전 처리
            if from_index >= len(eval_env.from_keys):
                from_index = 0
            if to_index >= len(eval_env.to_keys):
                to_index = 0

            state, reward, done, _ = eval_env.step((from_index, to_index))
            ep_reward += reward

        total_rewards.append(ep_reward)
        total_reversals.append(eval_env.last_reversal)
        print(f"Episode {epi}: Reward = {ep_reward}, Reversal = {eval_env.last_reversal}")

    avg_reward = sum(total_rewards) / num_eval_episodes
    avg_reversal = sum(total_reversals) / num_eval_episodes
    print(f"Evaluation Metrics: Avg Reward = {avg_reward:.2f}, Avg Reversal = {avg_reversal:.2f}")

if __name__ == "__main__":
    cfg = get_cfg()
    device = torch.device(cfg.device)

    random.seed(970517)
    np.random.seed(970517)
    torch.manual_seed(970517)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(970517)

    # 엑셀에서 schedule을 불러오거나, 실패하면 생성
    try:
        df = pd.read_excel(cfg.evaluation_plates_data_path, sheet_name="reshuffle")
        schedule = []
        for idx, row in df.iterrows():
            plate_id = row["pileno"]
            inbound = row["inbound"] if ("inbound" in df.columns and not pd.isna(row["inbound"])) else random.randint(cfg.inbound_min, cfg.inbound_max)
            outbound = row["outbound"] if ("outbound" in df.columns and not pd.isna(row["outbound"])) else inbound + random.randint(cfg.outbound_extra_min, cfg.outbound_extra_max)
            unitw = row["unitw"] if ("unitw" in df.columns and not pd.isna(row["unitw"])) else random.uniform(cfg.unitw_min, cfg.unitw_max)
            to_pile = str(row["topile"]).strip() if ("topile" in df.columns and not pd.isna(row["topile"])) else "A01"
            p = Plate(id=plate_id, inbound=inbound, outbound=outbound, unitw=unitw)
            p.from_pile = str(plate_id).strip()
            p.topile = to_pile
            schedule.append(p)
    except Exception as e:
        print("Error loading schedule from Excel:", e)
        schedule = generate_schedule(num_plates=cfg.num_plates)
        for p in schedule:
            p.from_pile = str(p.id)
            p.topile = str(random.randint(0, 10))

    # 환경에 사용할 pile 정보 설정
    from_piles = sorted(list(set([p.from_pile for p in schedule])))
    allowed_piles = sorted(list(set([p.topile for p in schedule])))
    num_source = len(from_piles)
    num_dest = len(allowed_piles)
    num_pile = len(set(from_piles).union(set(allowed_piles)))

    # 평가 환경 생성
    eval_env = Locating(
        num_pile=num_pile,
        max_stack=cfg.max_stack,
        inbound_plates=schedule,
        device=cfg.device,
        crane_penalty=cfg.crane_penalty,
        from_keys=from_piles,
        to_keys=allowed_piles
    )

    # 모델 생성 (학습 시 사용한 구성과 동일하게)
    model = SteelPlateConditionalMLPModel(
        embed_dim=cfg.embed_dim,
        target_entropy=-math.log(1.0 / (MAX_SOURCE * MAX_DEST)),
        use_temperature=True,
        num_actor_layers=cfg.num_actor_layers,
        num_critic_layers=cfg.num_critic_layers,
        actor_init_std=cfg.actor_init_std,  # 추가된 인자
        critic_init_std=cfg.critic_init_std  # 추가된 인자
    ).to(device)

    # 저장된 모델 파일 경로 (학습 때 생성했던 .pth 파일과 동일)
    model_path = "model_epoch1600.pth"
    model = load_model(model, model_path, device)

    evaluate_model(model, eval_env, device, num_eval_episodes=50)