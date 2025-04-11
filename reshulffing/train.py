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
from data import Plate, generate_schedule, generate_reshuffle_plan, save_reshuffle_plan_to_excel
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import copy
import vessl
from torch.utils.tensorboard import SummaryWriter
from evaluation import evaluate_policy  # 평가 함수는 evaluation.py에 작성
from network import SteelPlateConditionalMLPModel, MAX_SOURCE, MAX_DEST

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_gae(episode_transitions, gamma, lam):
    advantages = []
    gae = 0
    for t in reversed(range(len(episode_transitions))):
        r = episode_transitions[t]['r']
        done = episode_transitions[t]['done']
        v = episode_transitions[t]['v']
        next_v = 0.0 if t == len(episode_transitions) - 1 else episode_transitions[t + 1]['v']
        delta = r + gamma * next_v * (1 - done) - v
        gae = delta + gamma * lam * (1 - done) * gae
        advantages.insert(0, gae)
    return advantages

def make_batch(data_buffer, device):
    s_lst, a_lst, r_lst, adv_lst, s_prime_lst = [], [], [], [], []
    a_logprob_lst, v_lst, source_mask_lst, done_lst = [], [], [], []
    for transition in data_buffer:
        s_lst.append(transition['state'])
        a_lst.append(list(transition['action']))
        r_lst.append([transition['r']])
        adv_lst.append([transition['advantage']])
        s_prime_lst.append(transition['next_state'])
        a_logprob_lst.append([transition['logprob']])
        v_val = transition['v'] if isinstance(transition['v'], float) else transition['v'].detach().cpu().item()
        v_lst.append([v_val])
        source_mask_lst.append(transition['source_mask'].unsqueeze(0))
        done_lst.append([1.0 if transition['done'] else 0.0])
    s = torch.stack(s_lst, dim=0).to(device)
    s_prime = torch.stack(s_prime_lst, dim=0).to(device)
    a = torch.tensor(a_lst, dtype=torch.long, device=device)
    r = torch.tensor(r_lst, dtype=torch.float, device=device)
    adv = torch.tensor(adv_lst, dtype=torch.float, device=device)
    a_logprob = torch.tensor(a_logprob_lst, dtype=torch.float, device=device)
    v = torch.tensor(v_lst, dtype=torch.float, device=device)
    source_mask_tensor = torch.cat(source_mask_lst, dim=0).to(device) if source_mask_lst else torch.empty((0,), dtype=torch.bool, device=device)
    done = torch.tensor(done_lst, dtype=torch.float, device=device)
    return s, a, r, adv, s_prime, a_logprob, v, source_mask_tensor, done

def main():
    set_seed(970517)
    cfg = get_cfg()
    device = torch.device(cfg.device)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f" -> CUDA Device Count: {torch.cuda.device_count()}")
        print(f" -> Current CUDA Device: {torch.cuda.current_device()}")
        print(f" -> Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    # TensorBoard 로그 디렉토리 설정
    base_tb_dir = getattr(cfg, "tensorboard_dir", "./runs")
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_log_dir = os.path.join(base_tb_dir, timestamp)
    tb_writer = SummaryWriter(log_dir=tb_log_dir)

    cfg_info = (
        f"Learning Rate: {cfg.lr}\nActor LR: {cfg.actor_lr}\nCritic LR: {cfg.critic_lr}\n"
        f"LR Step: {cfg.lr_step}, LR Decay: {cfg.lr_decay}\n"
        f"n_epoch: {cfg.n_epoch}, episodes_per_epoch: {cfg.episodes_per_epoch}\n"
        f"T_horizon: {cfg.T_horizon}\nGamma: {cfg.gamma}, Lambda: {cfg.lmbda}\n"
        f"New instance every: {cfg.new_instance_every}\n"
        f"P_coeff: {cfg.P_coeff}, V_coeff: {cfg.V_coeff}, E_coeff: {cfg.E_coeff}\n"
        f"Mini Batch Size: {cfg.mini_batch_size}\neps_clip: {cfg.eps_clip}, value_clip_range: {cfg.value_clip_range}\n"
        f"grad_clip_norm: {cfg.grad_clip_norm}"
    )
    tb_writer.add_text("Hyperparameters", cfg_info, global_step=0)
    print("=== Hyperparameters ===")
    print(cfg_info)
    print("=======================")

    # 최고 성능 모델 저장을 위한 변수 초기화 (최대 reward 기준)
    best_reward = float("-inf")

    # 1) Excel 기반 reshuffle plan 생성 및 저장
    rows = ['A', 'B']
    df_plan, _, _, _, _ = generate_reshuffle_plan(
        rows, cfg.n_from_piles_reshuffle, cfg.n_to_piles_reshuffle,
        cfg.n_plates_reshuffle, cfg.safety_margin
    )
    excel_path = cfg.plates_data_path
    save_reshuffle_plan_to_excel(df_plan, excel_path)
    all_piles = set(df_plan['pileno'].unique()).union(set(df_plan['topile'].unique()))
    num_pile = len(all_piles)
    print(f"Excel 기반 num_pile: {num_pile}")
    num_stack = df_plan.shape[0]
    print(f"사용 stack 개수(실제 강판 개수): {num_stack}")
    from_piles = list(df_plan['pileno'].unique())
    allowed_piles = list(df_plan['topile'].unique())
    num_source = len(from_piles)
    num_dest = len(allowed_piles)
    print(f"From piles: {from_piles}, 개수={num_source}")
    print(f"Allowed piles: {allowed_piles}, 개수={num_dest}")

    # 2) schedule 생성
    schedule = []
    try:
        df = pd.read_excel(cfg.plates_data_path, sheet_name="reshuffle")
        for idx, row in df.iterrows():
            plate_id = row["pileno"]
            inbound = row["inbound"] if ("inbound" in df.columns and not pd.isna(row["inbound"])) else random.randint(cfg.inbound_min, cfg.inbound_max)
            outbound = row["outbound"] if ("outbound" in df.columns and not pd.isna(row["outbound"])) else inbound + random.randint(cfg.outbound_extra_min, cfg.outbound_extra_max)
            unitw = row["unitw"] if ("unitw" in df.columns and not pd.isna(row["unitw"])) else random.uniform(cfg.unitw_min, cfg.unitw_max)
            to_pile = str(row["topile"]).strip() if ("topile" in df.columns and not pd.isna(row["topile"])) else "A01"
            p = Plate(id=plate_id, inbound=inbound, outbound=outbound, unitw=unitw)
            p.from_pile = plate_id
            p.topile = to_pile
            schedule.append(p)
    except Exception as e:
        print("Excel load 오류:", e)
        schedule = generate_schedule(num_plates=cfg.num_plates)
        for p in schedule:
            p.from_pile = str(p.id)
            p.topile = str(random.randint(0, num_pile - 1))

    # 3) 모델 생성 (새로운 Attention 기반 모델)
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

    # Optimizer & LR Scheduler
    actor_params = list(model.actor_head.parameters()) + \
                   list(model.dest_attention.parameters()) + \
                   list(model.dest_fc.parameters()) + [model.dest_embeddings]
    critic_params = list(model.critic_net.parameters())
    actor_optimizer = optim.Adam(actor_params, lr=cfg.actor_lr)
    critic_optimizer = optim.Adam(critic_params, lr=cfg.critic_lr)
    actor_lr_sched = StepLR(actor_optimizer, step_size=cfg.lr_step, gamma=cfg.lr_decay)
    critic_lr_sched = StepLR(critic_optimizer, step_size=cfg.lr_step, gamma=cfg.lr_decay)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.dirname(cfg.log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_filename = os.path.join(log_dir, f"train_log_{timestamp}.csv")
    else:
        log_filename = f"train_log_{timestamp}.csv"

    with open(log_filename, mode="w", newline="") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["Epoch", "Total_Episodes", "Avg_Episode_Reward", "Avg_Loss",
                             "Actor_Loss", "Critic_Loss", "Entropy_Loss",
                             "Avg_Reversal", "Crane_Move", "Total_Steps"])

    num_epochs = cfg.n_epoch
    episodes_per_epoch = cfg.episodes_per_epoch
    T_horizon = cfg.T_horizon
    data_buffer = []
    recent_rewards = []
    early_stop_patience = 10
    min_delta = 0.01
    global_step = 0

    for epoch in range(num_epochs):
        if epoch % cfg.new_instance_every == 0:
            print("새로운 시나리오 생성됨 (Epoch:", epoch, ")")
            df_plan2, _, _, _, _ = generate_reshuffle_plan(
                rows, cfg.n_from_piles_reshuffle, cfg.n_to_piles_reshuffle,
                cfg.n_plates_reshuffle, cfg.safety_margin
            )
            schedule = []
            for idx, row in df_plan2.iterrows():
                p = Plate(row["pileno"], row["inbound"], row["outbound"], row["unitw"])
                p.from_pile = row["pileno"]
                p.topile = row["topile"]
                schedule.append(p)
            save_reshuffle_plan_to_excel(df_plan2, excel_path)

        total_ep_reward = 0.0
        total_steps = 0
        ep_reversals = []
        ep_crane = []

        for ep in range(episodes_per_epoch):
            env = Locating(
                num_pile=num_pile,
                max_stack=cfg.max_stack,
                inbound_plates=schedule,
                device=device,
                crane_penalty=cfg.crane_penalty,
            )
            state = env.reset(shuffle_schedule=True)
            state = state.to(device)
            done = False
            ep_reward = 0.0
            ep_steps = 0
            episode_transitions = []
            t = 0

            while not done and t < T_horizon:
                s_mask, d_mask = env.get_masks()
                # 아래와 같이 as_tensor와 clone().detach()를 사용하여 경고 메시지를 해결합니다.
                source_mask = torch.zeros(1, MAX_SOURCE, dtype=torch.bool, device=device)
                dest_mask = torch.zeros(1, MAX_DEST, dtype=torch.bool, device=device)
                valid_source_length = len(s_mask)
                valid_dest_length = len(d_mask)
                source_mask[0, :valid_source_length] = torch.as_tensor(s_mask, dtype=torch.bool, device=device).clone().detach()
                dest_mask[0, :valid_dest_length] = torch.as_tensor(d_mask, dtype=torch.bool, device=device).clone().detach()

                state_tensor = state.unsqueeze(0)
                actions, logprobs, value, global_repr = model.act_batch(
                    state_tensor, source_mask, dest_mask, greedy=False, debug=False)
                action = actions[0]
                logprob = logprobs[0]
                value_val = value[0]
                src_idx = action[0].item()
                dst_idx = action[1].item()
                if src_idx >= valid_source_length or not s_mask[src_idx]:
                    src_idx = 0
                if dst_idx >= valid_dest_length or not d_mask[dst_idx]:
                    dst_idx = 0
                next_state, reward, done, _ = env.step((src_idx, dst_idx))
                next_state = next_state.to(device)
                transition = {
                    'state': state,
                    'action': action,
                    'r': reward,
                    'next_state': next_state,
                    'logprob': logprob.item() if hasattr(logprob, 'item') else logprob,
                    'v': value_val,
                    'source_mask': source_mask[0],
                    'done': done
                }
                episode_transitions.append(transition)
                ep_reward += reward
                ep_steps += 1
                state = next_state
                t += 1
                if done:
                    break

            total_ep_reward += ep_reward
            total_steps += ep_steps
            ep_reversal = env.last_reversal if hasattr(env, "last_reversal") else 0.0
            ep_crane.append(env.crane_move)
            ep_reversals.append(ep_reversal)
            if episode_transitions:
                advantages = compute_gae(episode_transitions, cfg.gamma, cfg.lmbda)
                for idx_trans, tr in enumerate(episode_transitions):
                    tr['advantage'] = advantages[idx_trans]
                    data_buffer.append(tr)
        avg_ep_reward = total_ep_reward / episodes_per_epoch
        avg_reversal = np.mean(ep_reversals) if ep_reversals else 0.0
        avg_crane_move = np.mean(ep_crane) if ep_crane else 0.0

        print(f"Epoch {epoch}: Avg Episode Reward = {avg_ep_reward:.2f}, Avg Reversal = {avg_reversal:.2f}, Total Steps = {total_steps}")
        tb_writer.add_scalar("Training/AverageReward", avg_ep_reward, epoch)
        tb_writer.add_scalar("Training/AverageReversal", avg_reversal, epoch)
        tb_writer.flush()
        vessl.log({
            "training_reward": avg_ep_reward,
            "training_reversal": avg_reversal,
        }, step=epoch)

        recent_rewards.append(avg_ep_reward)
        if len(recent_rewards) > early_stop_patience:
            recent_rewards.pop(0)
        if len(recent_rewards) == early_stop_patience:
            reward_diff = max(recent_rewards) - min(recent_rewards)
            if reward_diff < min_delta:
                print(f"조기 종료(Reward diff={reward_diff:.4f})")
                break

        if len(data_buffer) > 0:
            mini_data_buffer = data_buffer.copy()
            data_buffer.clear()
            N = len(mini_data_buffer)
            mini_batch_size = cfg.mini_batch_size
            num_minibatches = max(1, N // mini_batch_size)
            indices = torch.randperm(N, device=device)
            total_loss = 0.0
            total_actor_loss = 0.0
            total_critic_loss = 0.0
            total_entropy_loss = 0.0
            num_updates = 0
            for _ in range(cfg.K_epoch):
                for mb_i in range(num_minibatches):
                    start = mb_i * mini_batch_size
                    end = N if mb_i == (num_minibatches - 1) else (mb_i + 1) * mini_batch_size
                    batch_idx = indices[start:end]
                    mini_data = [mini_data_buffer[j] for j in batch_idx.cpu().tolist()]
                    mini_s, mini_a, mini_r, mini_adv, mini_s_prime, mini_a_logprob, mini_v, mini_source_mask, mini_done = make_batch(mini_data, device)
                    mini_adv = (mini_adv - mini_adv.mean()) / (mini_adv.std() + 1e-8)
                    with torch.no_grad():
                        _, _, v_next, _ = model.forward(mini_s_prime)
                    td_target = mini_r + cfg.gamma * v_next * (1.0 - mini_done)
                    B2 = mini_s.size(0)
                    dest_mask_full = torch.ones((B2, MAX_DEST), dtype=torch.bool, device=device)
                    new_a_logprob, new_v, dist_entropy = model.evaluate(mini_s, mini_source_mask, dest_mask_full, mini_a)
                    ratio = torch.exp(new_a_logprob - mini_a_logprob)
                    surr1 = ratio * mini_adv
                    surr2 = torch.clamp(ratio, 1.0 - cfg.eps_clip, 1.0 + cfg.eps_clip) * mini_adv
                    actor_loss = -cfg.P_coeff * torch.min(surr1, surr2).mean()
                    value_pred_clipped = mini_v + (new_v - mini_v).clamp(-cfg.value_clip_range, cfg.value_clip_range)
                    critic_loss_unclipped = F.smooth_l1_loss(new_v, td_target, reduction="none")
                    critic_loss_clipped = F.smooth_l1_loss(value_pred_clipped, td_target, reduction="none")
                    critic_loss = cfg.V_coeff * torch.max(critic_loss_unclipped, critic_loss_clipped).mean()
                    entropy_loss = -cfg.E_coeff * dist_entropy.mean()
                    ppo_loss = actor_loss + critic_loss + entropy_loss
                    actor_optimizer.zero_grad()
                    critic_optimizer.zero_grad()
                    ppo_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                    actor_optimizer.step()
                    critic_optimizer.step()
                    total_loss += ppo_loss.item()
                    total_actor_loss += actor_loss.item()
                    total_critic_loss += critic_loss.item()
                    total_entropy_loss += entropy_loss.item()
                    num_updates += 1
                    global_step += 1
            if num_updates > 0:
                avg_loss = total_loss / num_updates
                avg_actor_loss = total_actor_loss / num_updates
                avg_critic_loss = total_critic_loss / num_updates
                avg_entropy_loss = total_entropy_loss / num_updates
                print(f"Epoch {epoch}: Loss = {avg_loss:.4f} (Actor = {avg_actor_loss:.4f}, Critic = {avg_critic_loss:.4f}, Entropy = {avg_entropy_loss:.4f}), Crane = {avg_crane_move:.2f}")
            else:
                avg_loss = avg_actor_loss = avg_critic_loss = avg_entropy_loss = 0.0
                print(f"Epoch {epoch}: No update performed.")

            tb_writer.add_scalar("Training/TotalLoss", avg_loss, epoch)
            tb_writer.add_scalar("Training/ActorLoss", avg_actor_loss, epoch)
            tb_writer.add_scalar("Training/CriticLoss", avg_critic_loss, epoch)
            tb_writer.add_scalar("Training/EntropyLoss", avg_entropy_loss, epoch)
            tb_writer.add_scalar("Training/CraneMove", avg_crane_move, epoch)
            tb_writer.flush()
            vessl.log({
                "training_reward": avg_ep_reward,
                "training_reversal": avg_reversal,
                "total_loss": avg_loss,
                "actor_loss": avg_actor_loss,
                "critic_loss": avg_critic_loss,
                "entropy_loss": avg_entropy_loss
            }, step=epoch)
            with open(cfg.log_file, mode="a", newline="") as f:
                writer_csv = csv.writer(f)
                writer_csv.writerow([
                    epoch,
                    episodes_per_epoch,
                    avg_ep_reward,
                    avg_loss,
                    avg_actor_loss,
                    avg_critic_loss,
                    avg_entropy_loss,
                    avg_reversal,
                    avg_crane_move,
                    total_steps
                ])

        # 평가: 일정 에포크마다 (cfg.eval_every 마다 평가 진행)
        if epoch % cfg.eval_every == 0 and epoch > 0:
            eval_excel_file = cfg.evaluation_plates_data_path
            if os.path.exists(eval_excel_file):
                try:
                    eval_df = pd.read_excel(eval_excel_file, sheet_name="reshuffle")
                    eval_schedule = []
                    for idx, row in eval_df.iterrows():
                        p = Plate(
                            id=row["pileno"],
                            inbound=row["inbound"],
                            outbound=row["outbound"],
                            unitw=row["unitw"]
                        )
                        p.from_pile = str(row["pileno"])
                        p.topile = str(row["topile"]).strip() if not pd.isna(row["topile"]) else "A01"
                        eval_schedule.append(p)
                    print("평가용 schedule을 Excel 파일에서 불러왔습니다.")
                except Exception as e:
                    print("평가용 Excel 파일 로드 오류:", e)
                    eval_schedule = generate_schedule(num_plates=cfg.num_plates)
                    for p in eval_schedule:
                        p.from_pile = str(p.id)
                        p.topile = str(random.randint(0, MAX_DEST - 1))
            else:
                print("평가용 엑셀 파일이 존재하지 않습니다. 기본 generate_schedule() 사용.")
                eval_schedule = generate_schedule(num_plates=cfg.num_plates)
                for p in eval_schedule:
                    p.from_pile = str(p.id)
                    p.topile = str(random.randint(0, MAX_DEST - 1))
            # 단일 평가 환경 구성 (고정된 평가 문제 사용)
            eval_env = Locating(
                num_pile=num_pile,
                max_stack=cfg.max_stack,
                inbound_plates=eval_schedule,
                device=device,
                crane_penalty=cfg.crane_penalty,
            )
            eval_reward, eval_reversal = evaluate_policy(model, [eval_env], device)
            tb_writer.add_scalar("Evaluation/AverageReward", eval_reward, epoch)
            tb_writer.add_scalar("Evaluation/AverageReversal", eval_reversal, epoch)
            tb_writer.flush()
            vessl.log({
                "evaluation_reward": eval_reward,
                "evaluation_reversal": eval_reversal,
            }, step=epoch)

            # 최고 성능 모델 저장 (평가 reward 기준)
            if eval_reward > best_reward:
                best_reward = eval_reward
                best_model_path = os.path.join(cfg.save_model_dir, "best_policy.pth")
                os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                torch.save(model.state_dict(), best_model_path)
                print(f"[BEST] New best model saved to {best_model_path} (Reward: {eval_reward:.2f})")

        actor_lr_sched.step()
        critic_lr_sched.step()
        if epoch % cfg.save_every == 0 and epoch > 0:
            save_path = os.path.join(cfg.save_model_dir, f"model_epoch{epoch}.pth")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"모델 저장: {save_path}")

    tb_writer.close()

if __name__ == "__main__":
    main()
