import random
import numpy as np
import torch
import os
import csv
import pandas as pd
from cfg import get_cfg
from env import Locating
from data import generate_reshuffle_plan, save_reshuffle_plan_to_excel
from network_temp import SteelPlateAttentionModel  # 온도 자동 조정 및 double-critic 포함된 네트워크

import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import HeteroData, Data, Batch

# device로 state 옮기는 함수
def move_state_to_device(state, device):
    if isinstance(state, dict):
        return {k: move_state_to_device(v, device) for k, v in state.items()}
    elif hasattr(state, "to"):
        return state.to(device)
    else:
        return state

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# HeteroData 객체들을 배치로 합치는 함수 (네트워크와 호환)
def batch_heterodata(data_list, device):
    xs = []
    batch_vector = []
    for i, data in enumerate(data_list):
        x = data['plate'].x
        xs.append(x)
        batch_vector.append(torch.full((x.size(0),), i, dtype=torch.long, device=device))
    if xs:
        batched_x = torch.cat(xs, dim=0)
        batched_batch = torch.cat(batch_vector, dim=0)
    else:
        batched_x = torch.zeros((0, 1), dtype=torch.float, device=device)
        batched_batch = torch.tensor([], dtype=torch.long, device=device)
    batched_data = HeteroData()
    batched_data['plate'] = Data(x=batched_x, batch=batched_batch)
    batched_data[('plate', 'blocks', 'plate')] = Data(edge_index=torch.empty((2, 0), dtype=torch.long, device=device))
    return batched_data.to(device)

def export_final_state_to_excel(env, output_filepath):
    rows = []
    for key, pile in env.plates.items():
        for depth_idx, plate_obj in enumerate(pile):
            row = {
                "pileno": plate_obj.id,
                "inbound": plate_obj.inbound,
                "outbound": plate_obj.outbound,
                "unitw": plate_obj.unitw,
                "final_pile": key,
                "depth": depth_idx,
                "topile": getattr(plate_obj, "topile", None)
            }
            rows.append(row)
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    with pd.ExcelWriter(output_filepath, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="final_arrangement", index=False)
    print(f"Final state saved to {output_filepath}")

def test_evaluation(env, model, device, test_episodes, from_piles, allowed_piles):
    total_rewards = []
    total_reversal = []
    for _ in range(test_episodes):
        state = env.reset()
        state = move_state_to_device(state, device)
        done = False
        ep_reward = 0.0
        while not done:
            source_mask, dest_mask = env.get_masks()
            source_mask = source_mask.to(device)
            dest_mask = dest_mask.to(device)
            if not source_mask.any():
                source_mask = torch.ones_like(source_mask, dtype=torch.bool)
            if not dest_mask.any():
                dest_mask = torch.ones_like(dest_mask, dtype=torch.bool)
            action, _, _, _ = model.act(state, source_mask, dest_mask, greedy=True)
            source_key = from_piles[action[0]] if action[0] < len(from_piles) else env.from_keys[0]
            dest_key = allowed_piles[action[1]] if action[1] < len(allowed_piles) else env.to_keys[0]
            from_index = env.from_keys.index(source_key) if source_key in env.from_keys else 0
            to_index = env.to_keys.index(dest_key) if dest_key in env.to_keys else 0
            env_action = (from_index, to_index)
            next_state, reward, done, _ = env.step(env_action)
            next_state = move_state_to_device(next_state, device)
            ep_reward += reward
            state = next_state
        total_rewards.append(ep_reward)
        total_reversal.append(getattr(env, "last_reversal", 0))
    avg_reward = sum(total_rewards) / test_episodes
    avg_reversal = sum(total_reversal) / test_episodes
    print(f"[Test Evaluation] Average Reward: {avg_reward:.2f}, Average Reversal: {avg_reversal:.2f}")
    return avg_reward, avg_reversal

def make_batch(data_buffer, num_pile, device, model, from_piles, allowed_piles):
    s_lst, a_lst, r_lst, s_prime_lst = [], [], [], []
    a_logprob_lst, v_lst, mask_lst, done_lst = [], [], [], []
    for transition in data_buffer:
        s, a, r, s_prime, a_logprob, v, _, done = transition
        s_lst.append(s)
        a_lst.append(list(a))
        r_lst.append([r])
        s_prime_lst.append(s_prime)
        a_logprob_lst.append([a_logprob])
        mask = torch.ones(len(from_piles), dtype=torch.bool, device=device)
        mask_lst.append(mask.unsqueeze(0))
        done_mask = 0 if done else 1
        done_lst.append([done_mask])
    # 부트스트래핑: 마지막 상태에 대해 종료된 에피소드면 0, 아니면 모델로 추정한 value 사용
    if len(done_lst) == 0:
        v_lst.append([0.0])
    else:
        if done_lst[-1] == [0]:
            v_lst.append([0.0])
        else:
            source_mask = torch.ones(len(from_piles), dtype=torch.bool, device=device)
            dest_mask = torch.ones(len(allowed_piles), dtype=torch.bool, device=device)
            with torch.no_grad():
                _, _, v_val, _ = model.act(s_prime_lst[-1], source_mask, dest_mask)
            v_lst.append([v_val])
    s = batch_heterodata(s_lst, device)
    s_prime = batch_heterodata(s_prime_lst, device)
    a = torch.tensor(a_lst, device=device)
    r = torch.tensor(r_lst, dtype=torch.float, device=device)
    a_logprob = torch.tensor(a_logprob_lst, dtype=torch.float, device=device)
    v = torch.tensor(v_lst, dtype=torch.float, device=device)
    if len(mask_lst) == 0:
        mask = torch.empty((0, len(from_piles)), dtype=torch.bool, device=device)
    else:
        mask = torch.cat(mask_lst).to(device)
    if len(done_lst) == 0:
        done = torch.empty((0, 1), dtype=torch.float, device=device)
    else:
        done = torch.tensor(done_lst, dtype=torch.float, device=device)
    data_buffer.clear()
    return s, a, r, s_prime, a_logprob, v, mask, done

def main():
    set_seed(970517)
    cfg = get_cfg()
    device = torch.device(cfg.device)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f" -> CUDA Device Count: {torch.cuda.device_count()}")
        print(f" -> Current CUDA Device: {torch.cuda.current_device()}")
        print(f" -> Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    # reshuffle plan 생성 및 저장
    rows = ['A', 'B']
    df_plan, _, _, _, _ = generate_reshuffle_plan(
        rows, cfg.n_bays_in_area1, cfg.n_bays_in_area5,
        cfg.n_from_piles_reshuffle, cfg.n_to_piles_reshuffle,
        cfg.n_plates_reshuffle, cfg.safety_margin
    )
    excel_file_path = cfg.plates_data_path
    save_reshuffle_plan_to_excel(df_plan, excel_file_path)

    # pileno와 topile의 union으로 num_pile 결정
    all_piles = set(df_plan['pileno'].unique()).union(set(df_plan['topile'].unique()))
    num_pile = len(all_piles)
    print(f"Excel 파일 기반 num_pile: {num_pile}")

    # 액션 공간 결정: from_piles와 allowed_piles
    from_piles = list(df_plan['pileno'].unique())
    allowed_piles = list(df_plan['topile'].unique())
    num_source = len(from_piles)
    num_dest = len(allowed_piles)
    print(f"From piles: {from_piles}, 개수: {num_source}")
    print(f"Allowed piles: {allowed_piles}, 개수: {num_dest}")

    # Locating 환경 생성 (from_keys와 to_keys 전달)
    env = Locating(num_pile=num_pile, max_stack=cfg.max_stack, inbound_plates=None,
                   device=cfg.device, crane_penalty=cfg.crane_penalty,
                   from_keys=from_piles, to_keys=allowed_piles)
    state = env.reset()
    state = move_state_to_device(state, device)
    if state['plate'].x.shape[0] == 0:
        print("경고: 초기 상태에 강판 데이터가 없습니다.")
    else:
        print(f"초기 상태 강판 수: {state['plate'].x.shape[0]}")

    # 상태는 outbound 단일 값만 포함하도록 구성 (state_size 차원 1)
    state_size = {"plate": 1}
    meta_data = (["plate"], [("plate", "blocks", "plate")])
    model = SteelPlateAttentionModel(
        state_size=state_size,
        meta_data=meta_data,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        num_HGT_layers=cfg.num_HGT_layers,
        num_actor_layers=cfg.num_actor_layers,
        num_critic_layers=cfg.num_critic_layers,
        num_source=num_source,
        num_dest=num_dest,
        activation=cfg.activation,
        actor_init_std=cfg.actor_init_std,
        critic_init_std=cfg.critic_init_std
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = StepLR(optimizer, step_size=cfg.lr_step, gamma=cfg.lr_decay)

    # 온도 자동 조정 옵티마이저 및 목표 엔트로피
    temperature_optimizer = optim.Adam([model.log_temperature], lr=cfg.temp_lr)
    target_entropy = cfg.target_entropy

    data_buffer = []
    log_file = cfg.log_file
    with open(log_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Raw_Reward", "Avg_Loss",
                         "Actor_Loss", "Critic_Loss", "Entropy_Loss", "Temp_Loss",
                         "Reversal", "Crane_Move", "Steps"])

    num_episodes = cfg.n_episode
    T_horizon = cfg.T_horizon
    all_episode_raw_rewards = []

    for ep in range(num_episodes):
        state = env.reset()
        state = move_state_to_device(state, device)
        done = False
        ep_reward = 0.0
        env.crane_move = 0
        env.last_reversal = 0
        episode_rewards = []
        data_buffer.clear()

        # 초기 mask 업데이트
        source_mask, dest_mask = env.get_masks()
        source_mask = source_mask.to(device)
        dest_mask = dest_mask.to(device)
        if not source_mask.any():
            source_mask = torch.ones_like(source_mask, dtype=torch.bool)
        if not dest_mask.any():
            dest_mask = torch.ones_like(dest_mask, dtype=torch.bool)

        action, logprob, value, attn_weights = model.act(state, source_mask, dest_mask, greedy=False)
        source_key = from_piles[action[0]] if action[0] < len(from_piles) else env.from_keys[0]
        dest_key = allowed_piles[action[1]] if action[1] < len(allowed_piles) else env.to_keys[0]
        from_index = env.from_keys.index(source_key) if source_key in env.from_keys else 0
        to_index = env.to_keys.index(dest_key) if dest_key in env.to_keys else 0
        env_action = (from_index, to_index)

        next_state, reward, done, _ = env.step(env_action)
        next_state = move_state_to_device(next_state, device)
        episode_rewards.append(reward)
        data_buffer.append((state, action, reward, next_state, logprob, value, source_mask, done))
        state = next_state
        ep_reward += reward

        while not done and len(data_buffer) < T_horizon:
            source_mask, dest_mask = env.get_masks()
            source_mask = source_mask.to(device)
            dest_mask = dest_mask.to(device)
            if not source_mask.any():
                source_mask = torch.ones_like(source_mask, dtype=torch.bool)
            if not dest_mask.any():
                dest_mask = torch.ones_like(dest_mask, dtype=torch.bool)
            action, logprob, value, attn_weights = model.act(state, source_mask, dest_mask, greedy=False)
            source_key = from_piles[action[0]] if action[0] < len(from_piles) else env.from_keys[0]
            dest_key = allowed_piles[action[1]] if action[1] < len(allowed_piles) else env.to_keys[0]
            from_index = env.from_keys.index(source_key) if source_key in env.from_keys else 0
            to_index = env.to_keys.index(dest_key) if dest_key in env.to_keys else 0
            env_action = (from_index, to_index)
            next_state, reward, done, _ = env.step(env_action)
            next_state = move_state_to_device(next_state, device)
            episode_rewards.append(reward)
            data_buffer.append((state, action, reward, next_state, logprob, value, source_mask, done))
            state = next_state
            ep_reward += reward

        all_episode_raw_rewards.append(episode_rewards)
        episode_length = len(episode_rewards)
        raw_rewards = np.array(episode_rewards, dtype=np.float32)

        buffer_for_minibatch = data_buffer.copy()
        s_batch, a_batch, r_batch, s_prime_batch, a_logprob_batch, v_batch, mask_batch, done_batch = make_batch(
            data_buffer, env.num_pile, device, model, from_piles, allowed_piles)
        N = r_batch.shape[0]

        with torch.no_grad():
            td_target = r_batch + cfg.gamma * v_batch * done_batch
        delta = td_target - v_batch
        advantage_lst = []
        advantage = 0.0
        for delta_t in delta.flip(dims=(0,)):
            advantage = cfg.gamma * cfg.lmbda * advantage + delta_t
            advantage_lst.insert(0, advantage)
        advantage = torch.cat(advantage_lst).unsqueeze(-1).to(device)

        total_loss = 0.0
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy_loss = 0.0
        total_temp_loss = 0.0
        num_updates = 0

        if N < cfg.num_minibatches:
            num_minibatches = 1
            mini_batch_size = N
        else:
            num_minibatches = cfg.num_minibatches
            mini_batch_size = N // cfg.num_minibatches

        indices = torch.randperm(N)
        for _ in range(cfg.K_epoch):
            for i in range(num_minibatches):
                start = i * mini_batch_size
                end = N if i == num_minibatches - 1 else (i + 1) * mini_batch_size
                batch_idx = indices[start:end].to(device)
                mini_data = [buffer_for_minibatch[j] for j in batch_idx.cpu().tolist()]
                mini_s, mini_a, mini_r, mini_s_prime, mini_a_logprob, mini_v, mini_mask, mini_done = make_batch(
                    mini_data, env.num_pile, device, model, from_piles, allowed_piles)

                source_mask_mb = mini_mask
                dest_mask_mb = torch.ones(len(allowed_piles), dtype=torch.bool, device=device)

                with torch.no_grad():
                    mini_td_target = mini_r + cfg.gamma * mini_v * mini_done

                new_a_logprob, new_v, dist_entropy = model.evaluate(mini_s, mini_a, source_mask_mb, dest_mask_mb)
                ratio = torch.exp(new_a_logprob - mini_a_logprob)
                mini_advantage = advantage[batch_idx]
                surr1 = ratio * mini_advantage
                surr2 = torch.clamp(ratio, 1 - cfg.eps_clip, 1 + cfg.eps_clip) * mini_advantage
                actor_loss = - cfg.P_coeff * torch.min(surr1, surr2).mean()
                critic_loss = cfg.V_coeff * F.smooth_l1_loss(new_v, mini_td_target)
                entropy_loss = - cfg.E_coeff * dist_entropy.mean()

                temperature = model.log_temperature.exp()
                temperature_loss = - temperature * (dist_entropy.mean() - target_entropy)

                loss = actor_loss + critic_loss + entropy_loss + temperature_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                temperature_optimizer.step()
                optimizer.zero_grad()
                temperature_optimizer.zero_grad()
                total_loss += loss.mean().item()
                total_actor_loss += actor_loss.mean().item()
                total_critic_loss += critic_loss.mean().item()
                total_entropy_loss += entropy_loss.mean().item()
                total_temp_loss += temperature_loss.mean().item()
                num_updates += 1

        avg_loss = total_loss / num_updates if num_updates > 0 else 0
        avg_actor_loss = total_actor_loss / num_updates if num_updates > 0 else 0
        avg_critic_loss = total_critic_loss / num_updates if num_updates > 0 else 0
        avg_entropy_loss = total_entropy_loss / num_updates if num_updates > 0 else 0
        avg_temp_loss = total_temp_loss / num_updates if num_updates > 0 else 0

        reversal_value = getattr(env, "last_reversal", 0)
        crane_move_value = env.crane_move
        print(f"Episode {ep}: Raw Reward={ep_reward:.2f}, Loss={avg_loss:.4f} "
              f"(Actor={avg_actor_loss:.4f}, Critic={avg_critic_loss:.4f}, Entropy={avg_entropy_loss:.4f}, Temp={avg_temp_loss:.4f}), "
              f"Reversal={reversal_value}, Crane Move={crane_move_value:.2f}")

        with open(log_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ep, ep_reward, avg_loss, avg_actor_loss, avg_critic_loss, avg_entropy_loss, avg_temp_loss,
                             reversal_value, crane_move_value, episode_length])

        if ep % cfg.save_every == 0 and ep > 0:
            model_save_path = os.path.join(cfg.save_model_dir, f"model_ep{ep}.pth")
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print(f"모델 저장: {model_save_path}")

        if done and (ep % cfg.save_final_state_every == 0):
            output_filepath = os.path.join(cfg.output_dir, f"final_state_ep{ep}.xlsx")
            env.export_final_state_to_excel(output_filepath)

        scheduler.step()

if __name__ == "__main__":
    main()
