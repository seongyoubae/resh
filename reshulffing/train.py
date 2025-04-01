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
# network.py에서 가져오는 모델 및 상수들
from network import SteelPlateConditionalMLPModel, pad_input_state_and_mask, MAX_SOURCE, MAX_DEST, PAD_INPUT_DIM
import vessl

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
        r    = episode_transitions[t]['r']
        done = episode_transitions[t]['done']
        v    = episode_transitions[t]['v']
        next_v = 0.0 if t == len(episode_transitions)-1 else episode_transitions[t+1]['v']
        delta  = r + gamma * next_v * (1-done) - v
        gae    = delta + gamma * lam * (1-done) * gae
        advantages.insert(0, gae)
    return advantages

def make_batch(data_buffer, device):
    s_lst, a_lst, r_lst, adv_lst, s_prime_lst = [], [], [], [], []
    a_logprob_lst, v_lst, source_mask_lst, done_lst  = [], [], [], []
    for transition in data_buffer:
        s_lst.append(transition['state'])
        a_lst.append(list(transition['action']))
        r_lst.append([transition['r']])
        adv_lst.append([transition['advantage']])
        s_prime_lst.append(transition['next_state'])
        a_logprob_lst.append([transition['logprob']])
        v_val = transition['v'] if isinstance(transition['v'], float) else transition['v'].detach().cpu().item()
        v_lst.append([v_val])
        # source_mask는 환경에서 얻은 actor용 mask (예: (MAX_SOURCE,))
        mask = transition['source_mask'].unsqueeze(0)
        source_mask_lst.append(mask)
        done = 1.0 if transition['done'] else 0.0
        done_lst.append([done])

    s       = torch.stack(s_lst, dim=0).to(device)
    s_prime = torch.stack(s_prime_lst, dim=0).to(device)
    a       = torch.tensor(a_lst,       dtype=torch.long,  device=device)
    r       = torch.tensor(r_lst,       dtype=torch.float, device=device)
    adv     = torch.tensor(adv_lst,     dtype=torch.float, device=device)
    a_logprob = torch.tensor(a_logprob_lst, dtype=torch.float, device=device)
    v       = torch.tensor(v_lst,       dtype=torch.float, device=device)
    source_mask_tensor = torch.cat(source_mask_lst, dim=0).to(device) if source_mask_lst else torch.empty((0,), dtype=torch.bool, device=device)
    done    = torch.tensor(done_lst,    dtype=torch.float, device=device)
    return s, a, r, adv, s_prime, a_logprob, v, source_mask_tensor, done

def main():
    cfg = get_cfg()
    device = torch.device(cfg.device)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f" -> CUDA Device Count: {torch.cuda.device_count()}")
        print(f" -> Current CUDA Device: {torch.cuda.current_device()}")
        print(f" -> Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    # 하이퍼파라미터 출력
    print("=== Hyperparameters ===")
    print(f"Learning Rate: {cfg.lr}")
    print(f"LR Step: {cfg.lr_step}, LR Decay: {cfg.lr_decay}")
    print(f"n_epoch: {cfg.n_epoch}, episodes_per_epoch: {cfg.episodes_per_epoch}")
    print(f"T_horizon: {cfg.T_horizon}")
    print(f"Gamma: {cfg.gamma}, Lambda: {cfg.lmbda}")
    print(f"New instance every: {cfg.new_instance_every}")
    print(f"P_coeff: {cfg.P_coeff}, V_coeff: {cfg.V_coeff}, E_coeff: {cfg.E_coeff}")
    print("=======================")

    # 1) 엑셀 기반 reshuffle plan 생성 + 저장
    rows = ['A','B']
    df_plan, _, _, _, _ = generate_reshuffle_plan(
        rows, cfg.n_from_piles_reshuffle, cfg.n_to_piles_reshuffle, cfg.n_plates_reshuffle, cfg.safety_margin
    )
    excel_path = cfg.plates_data_path
    save_reshuffle_plan_to_excel(df_plan, excel_path)

    all_piles = set(df_plan['pileno'].unique()).union(set(df_plan['topile'].unique()))
    num_pile  = len(all_piles)
    print(f"Excel 기반 num_pile: {num_pile}")
    num_stack = df_plan.shape[0]
    print(f"사용 stack 개수(실제 강판 개수): {num_stack}")

    from_piles    = list(df_plan['pileno'].unique())
    allowed_piles = list(df_plan['topile'].unique())
    num_source    = len(from_piles)
    num_dest      = len(allowed_piles)
    print(f"From piles: {from_piles}, 개수={num_source}")
    print(f"Allowed piles: {allowed_piles}, 개수={num_dest}")

    # 2) schedule 생성
    schedule = []
    try:
        df = pd.read_excel(cfg.plates_data_path, sheet_name="reshuffle")
        for idx, row in df.iterrows():
            plate_id = row["pileno"]
            inbound  = row["inbound"] if ("inbound" in df.columns and not pd.isna(row["inbound"])) else random.randint(cfg.inbound_min, cfg.inbound_max)
            outbound = row["outbound"] if ("outbound" in df.columns and not pd.isna(row["outbound"])) else inbound + random.randint(cfg.outbound_extra_min, cfg.outbound_extra_max)
            unitw    = row["unitw"]    if ("unitw" in df.columns and not pd.isna(row["unitw"])) else random.uniform(cfg.unitw_min, cfg.unitw_max)
            to_pile  = str(row["topile"]).strip() if ("topile" in df.columns and not pd.isna(row["topile"])) else "A01"
            p = Plate(id=plate_id, inbound=inbound, outbound=outbound, unitw=unitw)
            p.from_pile = plate_id
            p.topile    = to_pile
            schedule.append(p)
    except Exception as e:
        print("Excel load 오류:", e)
        schedule = generate_schedule(num_plates=cfg.num_plates)
        for p in schedule:
            p.from_pile = str(p.id)
            p.topile    = str(random.randint(0, num_pile-1))

    # 3) 모델 생성
    model = SteelPlateConditionalMLPModel(
        embed_dim=cfg.embed_dim,
        target_entropy=-math.log(1.0 / (MAX_SOURCE * MAX_DEST)),
        use_temperature=True,
        num_actor_layers=cfg.num_actor_layers,
        num_critic_layers=cfg.num_critic_layers,
        actor_init_std=cfg.actor_init_std,
        critic_init_std=cfg.critic_init_std
    ).to(device)

    # Optimizer & LR Scheduler
    actor_params = list(model.source_head.parameters()) + list(model.cond_dest_head.parameters())
    critic_params = list(model.critic_net.parameters())
    actor_optimizer = optim.Adam(actor_params, lr=cfg.actor_lr)
    critic_optimizer = optim.Adam(critic_params, lr=cfg.critic_lr)
    actor_lr_sched = StepLR(actor_optimizer, step_size=cfg.lr_step, gamma=cfg.lr_decay)
    critic_lr_sched = StepLR(critic_optimizer, step_size=cfg.lr_step, gamma=cfg.lr_decay)

    # 로그 파일 생성
    if os.path.dirname(cfg.log_file):
        os.makedirs(os.path.dirname(cfg.log_file), exist_ok=True)
    with open(cfg.log_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Episode", "Episode_Reward", "Avg_Loss",
                         "Actor_Loss", "Critic_Loss", "Entropy_Loss",
                         "Avg_Reversal", "Crane_Move", "Steps_in_Episode"])

    num_epochs         = cfg.n_epoch
    episodes_per_epoch = cfg.episodes_per_epoch
    T_horizon          = cfg.T_horizon
    data_buffer        = []
    recent_rewards     = []
    early_stop_patience= 10
    min_delta          = 0.01
    update_target_interval = cfg.update_target_interval
    tau = cfg.tau
    global_step = 0

    for epoch in range(num_epochs):
        if epoch % cfg.new_instance_every == 0:
            print("새로운 시나리오 생성됨 (Epoch:", epoch, ")")
            df_plan2, _, _, _, _ = generate_reshuffle_plan(
                rows, cfg.n_from_piles_reshuffle, cfg.n_to_piles_reshuffle, cfg.n_plates_reshuffle, cfg.safety_margin
            )
            schedule = []
            for idx, row in df_plan2.iterrows():
                p = Plate(row["pileno"], row["inbound"], row["outbound"], row["unitw"])
                p.from_pile = row["pileno"]
                p.topile    = row["topile"]
                schedule.append(p)
            save_reshuffle_plan_to_excel(df_plan2, excel_path)
            from_piles = list(df_plan2['pileno'].unique())
            allowed_piles = list(df_plan2['topile'].unique())
            num_pile = len(set(from_piles).union(set(allowed_piles)))
            total_stacks = df_plan2.shape[0]
            print("새로운 에피소드: from_piles =", from_piles, ", 개수 =", len(from_piles))
            print("새로운 에피소드: allowed_piles =", allowed_piles, ", 개수 =", len(allowed_piles))
            print("새로운 에피소드: num_pile =", num_pile)
            print("새로운 에피소드: 총 stack 갯수 =", total_stacks)

        envs = []
        for _ in range(episodes_per_epoch):
            env = Locating(
                num_pile=num_pile,
                max_stack=cfg.max_stack,
                inbound_plates=schedule,
                device=device,
                crane_penalty=cfg.crane_penalty,
                from_keys=from_piles,
                to_keys=allowed_piles
            )
            envs.append(env)

        states = [env.reset(shuffle_schedule=False) for env in envs]
        states = torch.stack(states, dim=0).to(device)
        input_mask = torch.ones(states.shape, dtype=torch.bool, device=states.device)

        dones      = [False]*episodes_per_epoch
        ep_rewards = [0.0]*episodes_per_epoch
        ep_data    = [[] for _ in range(episodes_per_epoch)]
        ep_steps   = [0]*episodes_per_epoch

        for t in range(T_horizon):
            batch_source_mask = []
            batch_dest_mask   = []

            # ---- 1) 각 env의 마스크 계산 & 패딩 ----
            for i, env in enumerate(envs):
                if dones[i]:
                    # 이미 done이면 스킵
                    continue

                s_mask, d_mask = env.get_masks()

                # 디버그 로그
                # print(f"[DEBUG][Env {i}] step={t}, from_keys={env.from_keys}")
                # print(f"[DEBUG][Env {i}] step={t}, s_mask(original): {s_mask.tolist()}")
                # print(f"[DEBUG][Env {i}] step={t}, to_keys={env.to_keys}")
                # print(f"[DEBUG][Env {i}] step={t}, d_mask(original): {d_mask.tolist()}")

                s_mask = s_mask.to(device)
                d_mask = d_mask.to(device)

                if s_mask.size(0) < MAX_SOURCE:
                    pad_s = torch.zeros(MAX_SOURCE, dtype=torch.bool, device=device)
                    pad_s[:s_mask.size(0)] = s_mask
                    s_mask = pad_s
                else:
                    s_mask = s_mask[:MAX_SOURCE]

                if d_mask.size(0) < MAX_DEST:
                    pad_d = torch.zeros(MAX_DEST, dtype=torch.bool, device=device)
                    pad_d[:d_mask.size(0)] = d_mask
                    d_mask = pad_d
                else:
                    d_mask = d_mask[:MAX_DEST]

                # print(f"[DEBUG][Env {i}] step={t}, s_mask(after pad): {s_mask.tolist()}")
                # print(f"[DEBUG][Env {i}] step={t}, d_mask(after pad): {d_mask.tolist()}")

                batch_source_mask.append(s_mask)
                batch_dest_mask.append(d_mask)
            if len(batch_source_mask) < episodes_per_epoch:
                # done되어 스킵된 env들은 전부 False mask로 채울 수도 있음 (옵션)
                for _ in range(episodes_per_epoch - len(batch_source_mask)):
                    dummy_s = torch.zeros(MAX_SOURCE, dtype=torch.bool, device=device)
                    dummy_d = torch.zeros(MAX_DEST,   dtype=torch.bool, device=device)
                    batch_source_mask.append(dummy_s)
                    batch_dest_mask.append(dummy_d)

            batch_source_mask = torch.stack(batch_source_mask, dim=0)
            batch_dest_mask   = torch.stack(batch_dest_mask,   dim=0)

            # print(f"[DEBUG] step={t}, batch_source_mask.shape={batch_source_mask.shape}")
            # print(f"[DEBUG] step={t}, batch_dest_mask.shape={batch_dest_mask.shape}")

            # ---- 2) 네트워크 액션 샘플링 ----
            actions_batch, logprobs_batch, values_batch, _ = model.act_batch(
                states,
                input_mask,
                batch_source_mask,
                batch_dest_mask,
                greedy=False,
                debug=False  # 네트워크 내부 디버깅
            )

            # ---- 3) 각 env step ----
            for i in range(episodes_per_epoch):
                if dones[i]:
                    continue

                action  = actions_batch[i]
                logprob = logprobs_batch[i]
                value   = values_batch[i]

                src_idx = action[0].item()
                dst_idx = action[1].item()

                if src_idx >= len(envs[i].from_keys):
                    src_idx = 0
                if dst_idx >= len(envs[i].to_keys):
                    dst_idx = 0

                source_key = envs[i].from_keys[src_idx]
                dest_key   = envs[i].to_keys[dst_idx]
                from_index = envs[i].from_keys.index(source_key)
                to_index   = envs[i].to_keys.index(dest_key)

                next_state, reward, done, _ = envs[i].step((from_index, to_index))
                next_state = next_state.to(device)

                transition = {
                    'state': states[i],
                    'action': action,
                    'r': reward,
                    'next_state': next_state,
                    'logprob': logprob.item() if hasattr(logprob, 'item') else logprob,
                    'v': value,
                    'source_mask': batch_source_mask[i],
                    'done': done
                }
                ep_data[i].append(transition)
                ep_rewards[i] += reward
                ep_steps[i] += 1

                dones[i] = done
                if done:
                    # 만약 한 에폭 내 여러 에피소드 돌리고 싶으면 아래 줄 활성화:
                    # states[i] = envs[i].reset().to(device)
                    # dones[i] = False
                    pass
                else:
                    states[i] = next_state
                    input_mask[i] = torch.ones_like(states[i], dtype=torch.bool)

            if all(dones):
                break  # 모든 env가 done이면 해당 T_horizon 도중이라도 종료

        # ---- GAE 계산 등 ----
        for i in range(episodes_per_epoch):
            if len(ep_data[i]) > 0:
                advantages = compute_gae(ep_data[i], cfg.gamma, cfg.lmbda)
                for idx, tr in enumerate(ep_data[i]):
                    tr['advantage'] = advantages[idx]
                    data_buffer.append(tr)

        avg_epoch_reward = np.mean(ep_rewards)
        avg_reversal     = np.mean([env.last_reversal for env in envs])
        print(f"Epoch {epoch}: AvgReward={avg_epoch_reward:.2f}, Reversal={avg_reversal:.2f}")

        recent_rewards.append(avg_epoch_reward)
        if len(recent_rewards) > early_stop_patience:
            recent_rewards.pop(0)
        if len(recent_rewards) == early_stop_patience:
            reward_diff = max(recent_rewards) - min(recent_rewards)
            if reward_diff < min_delta:
                print(f"조기 종료(Reward diff={reward_diff:.4f})")
                break

        # 평가 코드
        if epoch % cfg.eval_every == 0:
            print("----- Evaluation Start -----")
            backup_random = random.getstate()
            backup_numpy = np.random.get_state()
            backup_torch = torch.get_rng_state()
            set_seed(970517)

            num_eval_episodes = 20
            total_rewards = []
            total_reversals = []

            model.eval()
            if hasattr(model, "temperature_param"):
                original_temp = model.temperature_param.data.clone()
                model.temperature_param.data.fill_(0.0)

            for epi_idx in range(num_eval_episodes):
                env_eval = Locating(
                    num_pile=num_pile,
                    max_stack=cfg.max_stack,
                    inbound_plates=copy.deepcopy(schedule),
                    device=device,
                    crane_penalty=cfg.crane_penalty,
                    from_keys=from_piles,
                    to_keys=allowed_piles
                )
                s = env_eval.reset(shuffle_schedule=False).to(device)
                done = False
                ep_reward = 0.0
                while not done:
                    s_tensor = s.unsqueeze(0)
                    input_mask_eval = torch.ones(s_tensor.shape, dtype=torch.bool, device=device)

                    s_mask_1d = torch.ones(len(env_eval.from_keys), dtype=torch.bool, device=device)
                    if s_mask_1d.size(0) < MAX_SOURCE:
                        pad_s = torch.zeros(MAX_SOURCE, dtype=torch.bool, device=device)
                        pad_s[:s_mask_1d.size(0)] = s_mask_1d
                        s_mask_1d = pad_s
                    else:
                        s_mask_1d = s_mask_1d[:MAX_SOURCE]

                    d_mask_1d = torch.ones(len(env_eval.to_keys), dtype=torch.bool, device=device)
                    if d_mask_1d.size(0) < MAX_DEST:
                        pad_d = torch.zeros(MAX_DEST, dtype=torch.bool, device=device)
                        pad_d[:d_mask_1d.size(0)] = d_mask_1d
                        d_mask_1d = pad_d
                    else:
                        d_mask_1d = d_mask_1d[:MAX_DEST]

                    s_mask_2d = s_mask_1d.unsqueeze(0)
                    d_mask_2d = d_mask_1d.unsqueeze(0)

                    actions_eval, _, _, _ = model.act_batch(s_tensor, input_mask_eval, s_mask_2d, d_mask_2d,
                                                            greedy=True)
                    action_eval = actions_eval[0]
                    src_idx = action_eval[0].item()
                    dst_idx = action_eval[1].item()
                    if src_idx >= len(env_eval.from_keys):
                        src_idx = 0
                    if dst_idx >= len(env_eval.to_keys):
                        dst_idx = 0

                    s, rew, done, _ = env_eval.step((src_idx, dst_idx))
                    s = s.to(device)
                    ep_reward += rew

                print(f"Eval Ep{epi_idx}: reward={ep_reward}, reversal={env_eval.last_reversal}")
                total_rewards.append(ep_reward)
                total_reversals.append(env_eval.last_reversal)

            avg_eval = sum(total_rewards) / len(total_rewards)
            print("----- Evaluation End -----")
            print(f"[Eval] AvgReward={avg_eval:.2f}")
            print(f"[Eval] AvgReversal={np.mean(total_reversals):.2f}")

            vessl.log({
                "evaluation_reward": avg_eval,
                "evaluation_reversal": np.mean(total_reversals)
            }, step=epoch)

            if not hasattr(main, "best_eval_reward"):
                main.best_eval_reward = -float("inf")
            if avg_eval > main.best_eval_reward:
                main.best_eval_reward = avg_eval
                best_model_path = os.path.join(cfg.save_model_dir, "best_model.pth")
                torch.save(model.state_dict(), best_model_path)
                print(f"[Elitism] Best model updated: {best_model_path} with avg_eval: {avg_eval:.2f}")
                best_log_file = os.path.join(cfg.save_model_dir, "best_model_log.csv")
                header = ["Epoch", "AvgEvalReward"]
                if not os.path.exists(best_log_file):
                    with open(best_log_file, mode="w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(header)
                with open(best_log_file, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, avg_eval])

            random.setstate(backup_random)
            np.random.set_state(backup_numpy)
            torch.set_rng_state(backup_torch)
            model.train()
        # ---- PPO Update ----
        mini_data_buffer = data_buffer.copy()
        data_buffer.clear()
        N = len(mini_data_buffer)
        if N == 0:
            continue

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
                    input_mask_prime = torch.ones(mini_s_prime.shape, dtype=torch.bool, device=mini_s_prime.device)
                    selected_source = mini_a[:, 0]
                    v_next = model.forward(mini_s_prime, input_mask_prime, selected_source=selected_source)[2]

                td_target = mini_r + cfg.gamma * v_next * (1.0 - mini_done)

                B2 = mini_s.size(0)
                dest_mask = torch.ones((B2, MAX_DEST), dtype=torch.bool, device=device)
                input_mask_mini = torch.ones((B2, mini_s.shape[-1]), dtype=torch.bool, device=device)

                new_a_logprob, new_v, dist_entropy = model.evaluate(
                    mini_s, input_mask_mini, mini_a, mini_source_mask, dest_mask
                )

                ratio = torch.exp(new_a_logprob - mini_a_logprob)
                surr1 = ratio * mini_adv
                surr2 = torch.clamp(ratio, 1.0 - cfg.eps_clip, 1.0 + cfg.eps_clip) * mini_adv
                actor_loss = -cfg.P_coeff * torch.min(surr1, surr2).mean()

                value_pred_clipped = mini_v + (new_v - mini_v).clamp(-cfg.value_clip_range, cfg.value_clip_range)
                critic_loss_unclipped = F.smooth_l1_loss(new_v, td_target, reduction="none")
                critic_loss_clipped   = F.smooth_l1_loss(value_pred_clipped, td_target, reduction="none")
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

        avg_loss = total_loss / num_updates if num_updates > 0 else 0
        avg_actor_loss = total_actor_loss / num_updates if num_updates > 0 else 0
        avg_critic_loss = total_critic_loss / num_updates if num_updates > 0 else 0
        avg_entropy_loss = total_entropy_loss / num_updates if num_updates > 0 else 0

        crane_move_value = np.mean([env.crane_move for env in envs])
        print(f"Epoch {epoch}: Loss={avg_loss:.4f} (Actor={avg_actor_loss:.4f}, Critic={avg_critic_loss:.4f}, Entropy={avg_entropy_loss:.4f}), Crane={crane_move_value:.2f}")

        vessl.log({
            "training_reward": avg_epoch_reward,
            "training_reversal": avg_reversal,
            "total_loss": avg_loss,
            "actor_loss": avg_actor_loss,
            "critic_loss": avg_critic_loss,
            "entropy_loss": avg_entropy_loss
        }, step=epoch)

        with open(cfg.log_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                episodes_per_epoch,
                avg_epoch_reward,
                avg_loss,
                avg_actor_loss,
                avg_critic_loss,
                avg_entropy_loss,
                np.mean([env.last_reversal for env in envs]),
                crane_move_value,
                N
            ])

        actor_lr_sched.step()
        critic_lr_sched.step()

        if epoch % cfg.save_every == 0 and epoch > 0:
            save_path = os.path.join(cfg.save_model_dir, f"model_epoch{epoch}.pth")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"모델 저장: {save_path}")

if __name__ == "__main__":
    main()
