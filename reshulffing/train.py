import random
import numpy as np
import torch
import os
import csv
import pandas as pd
import math
import datetime
import copy
import warnings
from cfg import get_cfg
from env import Locating  # 수정된 Locating 클래스
from data import Plate, generate_schedule, generate_reshuffle_plan, save_reshuffle_plan_to_excel
from network import SteelPlateConditionalMLPModel, MAX_SOURCE, MAX_DEST
from evaluation import evaluate_policy  # 평가 함수

# PyTorch 관련 임포트
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

# Vessl 로깅 (사용하는 경우)
try:
    import vessl

    USE_VESSL = True
except ImportError:
    USE_VESSL = False
    print("Vessl not installed. Skipping Vessl logging.")

warnings.filterwarnings("ignore", category=FutureWarning)


def set_seed(seed):
    """재현성을 위한 시드 설정 함수"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_gae(episode_transitions, gamma, lam):
    """Generalized Advantage Estimation (GAE) 계산 함수"""
    advantages = []
    gae = 0
    # 마지막 전이부터 역순으로 계산
    for t in reversed(range(len(episode_transitions))):
        r = episode_transitions[t]['r']
        done = episode_transitions[t]['done']
        v = episode_transitions[t]['v']
        # 다음 상태의 가치 (마지막 스텝이면 0)
        next_v = 0.0 if (t == len(episode_transitions) - 1 or done) else episode_transitions[t + 1]['v']
        delta = r + gamma * next_v * (1 - done) - v
        gae = delta + gamma * lam * (1 - done) * gae
        advantages.insert(0, gae)  # 리스트 맨 앞에 추가
    return advantages


def make_batch(data_buffer, device):
    """데이터 버퍼에서 미니배치를 생성하는 함수 (dest_mask 포함)"""
    s_lst, a_lst, r_lst, adv_lst, s_prime_lst = [], [], [], [], []
    a_logprob_lst, v_lst, source_mask_lst, dest_mask_lst, done_lst = [], [], [], [], []

    for transition in data_buffer:
        s_lst.append(transition['state'])
        a_lst.append(list(transition['action']))
        r_lst.append([transition['r']])
        adv_lst.append([transition['advantage']])
        s_prime_lst.append(transition['next_state'])
        a_logprob_lst.append([transition['logprob']])
        # v가 이미 스칼라 값으로 저장되어 있음
        v_val = transition['v']
        v_lst.append([v_val])
        source_mask_lst.append(transition['source_mask'].unsqueeze(0))
        dest_mask_lst.append(transition['dest_mask'].unsqueeze(0))
        done_lst.append([1.0 if transition['done'] else 0.0])

    # 리스트들을 텐서로 변환
    s = torch.stack(s_lst, dim=0).to(device)
    s_prime = torch.stack(s_prime_lst, dim=0).to(device)
    a = torch.tensor(a_lst, dtype=torch.long, device=device)
    r = torch.tensor(r_lst, dtype=torch.float, device=device)
    adv = torch.tensor(adv_lst, dtype=torch.float, device=device)
    a_logprob = torch.tensor(a_logprob_lst, dtype=torch.float, device=device)
    v = torch.tensor(v_lst, dtype=torch.float, device=device)
    source_mask_tensor = torch.cat(source_mask_lst, dim=0).to(device)
    dest_mask_tensor = torch.cat(dest_mask_lst, dim=0).to(device)
    done = torch.tensor(done_lst, dtype=torch.float, device=device)

    return s, a, r, adv, s_prime, a_logprob, v, source_mask_tensor, dest_mask_tensor, done


def main():
    """메인 학습 함수"""
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
    os.makedirs(tb_log_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=tb_log_dir)
    print(f"TensorBoard log directory: {tb_log_dir}")

    # 하이퍼파라미터 로깅
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
    tb_writer.add_text("Hyperparameters", cfg_info.replace('\n', '  \n'), global_step=0)
    print("=== Hyperparameters (Effective Values) ===")
    print(cfg_info)
    print("==========================================")

    # --- 최고 성능 지표 추적 변수 초기화 (낮을수록 좋음) ---
    best_metric = float("inf")

    # --- 모델 초기화 ---
    _OBSERVED_TOP_N_PLATES = getattr(cfg, 'OBSERVED_TOP_N_PLATES', 10)
    _NUM_SUMMARY_STATS_DEEPER = getattr(cfg, 'NUM_SUMMARY_STATS_DEEPER', 4)
    _NUM_PILE_TYPE_FEATURES = 1
    _NUM_BLOCKING_FEATURES = 1
    actual_pile_feature_dim_for_model = (_OBSERVED_TOP_N_PLATES +
                                         _NUM_SUMMARY_STATS_DEEPER +
                                         _NUM_PILE_TYPE_FEATURES +
                                         _NUM_BLOCKING_FEATURES)

    model = SteelPlateConditionalMLPModel(
        embed_dim=cfg.embed_dim,
        num_actor_layers=cfg.num_actor_layers,
        num_critic_layers=cfg.num_critic_layers,
        actor_init_std=cfg.actor_init_std,
        critic_init_std=cfg.critic_init_std,
        pile_feature_dim=actual_pile_feature_dim_for_model,
        num_heads=cfg.num_heads,
        use_dropout_actor=getattr(cfg, 'use_dropout_actor', False),
        use_dropout_critic=getattr(cfg, 'use_dropout_critic', False),
        num_from_piles=MAX_SOURCE,
        num_to_piles=MAX_DEST
    ).to(device)

    # --- 옵티마이저 및 LR 스케줄러 설정 ---
    actor_params = list(model.pile_encoder.parameters()) + \
                   list(model.attn_norm.parameters()) + \
                   list(model.self_attention.parameters()) + \
                   list(model.source_logit_calculator.parameters()) + \
                   list(model.dest_kv_combiner.parameters()) + \
                   list(model.dest_attention.parameters()) + \
                   list(model.dest_fc.parameters()) + \
                   [model.dest_embeddings]

    critic_params = list(model.critic_attention.parameters()) + \
                    list(model.critic_net.parameters()) + \
                    [model.critic_query]

    actor_optimizer = optim.Adam(actor_params, lr=cfg.actor_lr, weight_decay=cfg.weight_decay)
    critic_optimizer = optim.Adam(critic_params, lr=cfg.critic_lr, weight_decay=cfg.weight_decay)

    num_epochs = float(cfg.n_epoch)
    lr_lambda = lambda epoch: max(0.0, 1.0 - (epoch / num_epochs))
    actor_lr_sched = torch.optim.lr_scheduler.LambdaLR(actor_optimizer, lr_lambda=lr_lambda)
    critic_lr_sched = torch.optim.lr_scheduler.LambdaLR(critic_optimizer, lr_lambda=lr_lambda)
    print(f"Using Linear LR Decay schedule over {int(num_epochs)} epochs.")

    # --- CSV 로그 파일 설정 ---
    log_dir = os.path.dirname(cfg.log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    log_filename = cfg.log_file if cfg.log_file else f"train_log_{timestamp}.csv"

    try:
        with open(log_filename, mode="w", newline="") as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow(["Epoch", "Total_Episodes", "Avg_Episode_Reward", "Avg_Loss",
                                 "Actor_Loss", "Critic_Loss", "Entropy_Loss",
                                 "Avg_Max_Move_Sum", "Crane_Move", "Total_Steps"])
    except IOError as e:
        print(f"Error opening log file {log_filename}: {e}")
        return

    # --- 학습 루프 변수 초기화 ---
    data_buffer = []
    global_step = 0
    start_epoch = 0  # 학습 재개를 위한 변수 (여기서는 0으로 시작)

    # --- 메인 학습 루프 (매 에피소드마다 새로운 시나리오 생성) ---
    for epoch in range(start_epoch, cfg.n_epoch):
        total_ep_reward = 0.0
        total_steps_epoch = 0
        ep_final_max_move_sums = []
        ep_crane_moves = []
        data_buffer.clear()  # 에포크 시작 시 데이터 버퍼를 비웁니다.

        print(f"\n--- Epoch {epoch} (Data Collection) ---")

        for ep_idx in range(cfg.episodes_per_epoch):
            # --- START: 매 에피소드마다 시나리오 동적 생성 ---
            # 50% 확률로 '저적재(low-pile)' 또는 '고적재(high-pile)' 문제 유형을 선택
            if random.random() < 0.5:
                # --- 시나리오 1: 저적재 (적은 파일, 적은 강판) ---
                current_n_from = random.randint(5, 15)
                current_n_to = random.randint(5, 15)
                current_n_plates_per_active_pile = random.randint(3, 10)
                scenario_type = "Low-pile"
            else:
                # --- 시나리오 2: 고적재 (많은 파일, 많은 강판) ---
                current_n_from = random.randint(16, MAX_SOURCE)
                current_n_to = random.randint(16, MAX_DEST)
                current_n_plates_per_active_pile = random.randint(10, 20)
                scenario_type = "High-pile"

            schedule_for_current_episode = []
            try:
                df_plan_new, _, _, _, _ = generate_reshuffle_plan(
                    rows=['A', 'B'],
                    n_from_piles_reshuffle=current_n_from,
                    n_to_piles_reshuffle=current_n_to,
                    n_plates_reshuffle=current_n_plates_per_active_pile,
                    safety_margin=cfg.safety_margin
                )
                for idx, row in df_plan_new.iterrows():
                    p = Plate(id=str(row["pileno"]),
                              inbound=int(row["inbound"]),
                              outbound=int(row["outbound"]),
                              unitw=float(row["unitw"]))
                    p.from_pile = str(row["pileno"])
                    p.topile = str(row["topile"]).strip()
                    schedule_for_current_episode.append(p)
            except Exception as e_gen:
                print(f"  [Error] Ep {ep_idx}: Failed to generate plan ({e_gen}). Skipping episode.")
                continue

            if not schedule_for_current_episode:
                print(f"  [Warning] Ep {ep_idx}: Generated schedule is empty. Skipping episode.")
                continue

            print(
                f"  [Ep {ep_idx + 1}/{cfg.episodes_per_epoch}] Scenario: {scenario_type}, From: {current_n_from}, To: {current_n_to}, Plates: {len(schedule_for_current_episode)}")

            # --- START: 환경 생성 및 에피소드 진행 ---
            env = Locating(
                max_stack=cfg.max_stack,
                inbound_plates=copy.deepcopy(schedule_for_current_episode),
                crane_penalty=cfg.crane_penalty,
                device=device
            )
            state = env.reset(shuffle_schedule=False).to(device)
            done = False
            ep_reward = 0.0
            ep_steps = 0
            episode_transitions = []
            last_info = {}

            t_step = 0
            while not done and t_step < cfg.T_horizon:
                s_mask_env, d_mask_env = env.get_masks()
                source_mask_padded = torch.zeros(1, MAX_SOURCE, dtype=torch.bool, device=device)
                dest_mask_padded = torch.zeros(1, MAX_DEST, dtype=torch.bool, device=device)

                valid_source_len = min(s_mask_env.size(0), MAX_SOURCE)
                valid_dest_len = min(d_mask_env.size(0), MAX_DEST)

                source_mask_padded[0, :valid_source_len] = s_mask_env[:valid_source_len].clone().detach()
                dest_mask_padded[0, :valid_dest_len] = d_mask_env[:valid_dest_len].clone().detach()

                if not source_mask_padded.any() or not dest_mask_padded.any():
                    # print(f"  [Debug] No valid source or destination. Ending episode.") # 디버깅용
                    done = True  # 에피소드를 정상적으로 종료 처리합니다.
                    break  # 현재 while 루프를 탈출합니다.

                state_tensor = state.unsqueeze(0)
                actions, logprobs, value, _ = model.act_batch(state_tensor, source_mask_padded, dest_mask_padded,
                                                              greedy=False)

                action = actions[0]
                logprob = logprobs[0]
                value_val = value[0].item()
                src_idx_model = action[0].item()
                dst_idx_model = action[1].item()

                next_state_cpu, reward, done, info = env.step((src_idx_model, dst_idx_model))
                next_state = next_state_cpu.to(device)
                last_info = info

                transition = {
                    'state': state, 'action': action, 'r': reward, 'next_state': next_state,
                    'logprob': logprob.item(), 'v': value_val,
                    'source_mask': source_mask_padded[0], 'dest_mask': dest_mask_padded[0], 'done': done
                }
                episode_transitions.append(transition)

                ep_reward += reward
                ep_steps += 1
                state = next_state
                t_step += 1

            if episode_transitions:
                advantages = compute_gae(episode_transitions, cfg.gamma, cfg.lmbda)
                for i, trans in enumerate(episode_transitions):
                    trans['advantage'] = advantages[i]
                    data_buffer.append(trans)

            total_ep_reward += ep_reward
            total_steps_epoch += ep_steps
            final_max_move = last_info.get('final_max_move_sum', float('inf'))
            ep_final_max_move_sums.append(final_max_move)
            ep_crane_moves.append(getattr(env, "crane_move", 0.0))
            # --- END: 에피소드 진행 ---

        # --- 에포크 결과 요약 및 로깅 ---
        avg_ep_reward = total_ep_reward / cfg.episodes_per_epoch if cfg.episodes_per_epoch > 0 else 0
        valid_max_moves = [m for m in ep_final_max_move_sums if m != float('inf')]
        avg_final_max_move_sum = np.mean(valid_max_moves) if valid_max_moves else float('inf')
        avg_crane_move = np.mean(ep_crane_moves) if ep_crane_moves else 0.0

        print(
            f"Epoch {epoch} Summary: Avg Reward={avg_ep_reward:.2f}, Avg Reversal={avg_final_max_move_sum:.2f}, Avg Crane={avg_crane_move:.2f}, Total Steps={total_steps_epoch}")

        tb_writer.add_scalar("Training/AverageReward", avg_ep_reward, epoch)
        tb_writer.add_scalar("Training/AvgFinalMaxMoveSum", avg_final_max_move_sum, epoch)
        tb_writer.add_scalar("Training/AverageCraneMove", avg_crane_move, epoch)
        tb_writer.add_scalar("Training/TotalSteps", total_steps_epoch, epoch)
        if USE_VESSL:
            vessl.log({
                "training_reward": avg_ep_reward,
                "training_reversal": avg_final_max_move_sum,
                "training_crane_move": avg_crane_move
            }, step=epoch)

        # --- PPO 업데이트 ---
        avg_loss_val, avg_actor_loss_val, avg_critic_loss_val, avg_entropy_loss_val = 0.0, 0.0, 0.0, 0.0

        if len(data_buffer) >= cfg.mini_batch_size:
            total_loss_accum, total_actor_loss_accum, total_critic_loss_accum, total_entropy_loss_accum = 0.0, 0.0, 0.0, 0.0
            num_updates = 0

            for _ in range(cfg.K_epoch):
                N = len(data_buffer)
                indices = torch.randperm(N)
                for mb_start_idx in range(0, N, cfg.mini_batch_size):
                    mb_end_idx = min(mb_start_idx + cfg.mini_batch_size, N)
                    if mb_end_idx <= mb_start_idx: continue

                    batch_indices = indices[mb_start_idx:mb_end_idx]
                    mini_batch_data = [data_buffer[j] for j in batch_indices.tolist()]

                    mini_s, mini_a, mini_r, mini_adv, mini_s_prime, \
                        mini_a_logprob, mini_v_old, mini_source_mask, mini_dest_mask, mini_done = make_batch(
                        mini_batch_data, device)

                    if mini_s.nelement() == 0: continue

                    # 미니배치에 요소가 2개 이상일 때만 정규화를 수행하여 안정성 확보
                    if mini_adv.numel() > 1:
                        mini_adv = (mini_adv - mini_adv.mean()) / (mini_adv.std() + 1e-8)

                    with torch.no_grad():
                        _, _, v_s_prime, _ = model.forward(mini_s_prime)
                    td_target = mini_r + cfg.gamma * v_s_prime * (1.0 - mini_done)

                    new_logprob, new_v, dist_entropy = model.evaluate(mini_s, mini_source_mask, mini_dest_mask, mini_a)

                    ratio = torch.exp(new_logprob - mini_a_logprob)
                    surr1 = ratio * mini_adv
                    surr2 = torch.clamp(ratio, 1.0 - cfg.eps_clip, 1.0 + cfg.eps_clip) * mini_adv
                    actor_loss = -torch.min(surr1, surr2).mean()

                    value_pred_clipped = mini_v_old + (new_v - mini_v_old).clamp(-cfg.value_clip_range,
                                                                                 cfg.value_clip_range)
                    beta = getattr(cfg, "smooth_l1_beta", 1.0)
                    critic_loss_unclipped = F.smooth_l1_loss(new_v, td_target.detach(), reduction="none", beta=beta)
                    critic_loss_clipped = F.smooth_l1_loss(value_pred_clipped, td_target.detach(), reduction="none",
                                                           beta=beta)
                    critic_loss = torch.max(critic_loss_unclipped, critic_loss_clipped).mean()

                    entropy_loss = -cfg.E_coeff * dist_entropy.mean()

                    total_actor_loss_accum += actor_loss.item()
                    total_critic_loss_accum += critic_loss.item()
                    total_entropy_loss_accum += entropy_loss.item()

                    ppo_loss = (cfg.P_coeff * actor_loss + cfg.V_coeff * critic_loss + entropy_loss)
                    total_loss_accum += ppo_loss.item()

                    actor_optimizer.zero_grad()
                    critic_optimizer.zero_grad()
                    ppo_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                    actor_optimizer.step()
                    critic_optimizer.step()

                    num_updates += 1
                    global_step += 1

            if num_updates > 0:
                avg_loss_val = total_loss_accum / num_updates
                avg_actor_loss_val = total_actor_loss_accum / num_updates
                avg_critic_loss_val = total_critic_loss_accum / num_updates
                avg_entropy_loss_val = total_entropy_loss_accum / num_updates

            print(
                f"  PPO Update: Avg Loss={avg_loss_val:.4f} (Actor={avg_actor_loss_val:.4f}, Critic={avg_critic_loss_val:.4f}, Entropy={avg_entropy_loss_val:.4f})")

            tb_writer.add_scalar("Loss/TotalLoss", avg_loss_val, epoch)
            tb_writer.add_scalar("Loss/ActorLoss", avg_actor_loss_val, epoch)
            tb_writer.add_scalar("Loss/CriticLoss", avg_critic_loss_val, epoch)
            tb_writer.add_scalar("Loss/EntropyLoss", avg_entropy_loss_val, epoch)
            if USE_VESSL:
                vessl.log({
                    "total_loss": avg_loss_val, "actor_loss": avg_actor_loss_val,
                    "critic_loss": avg_critic_loss_val, "entropy_loss": avg_entropy_loss_val
                }, step=epoch)

        else:
            print(
                f"  Skipping PPO update, data_buffer size ({len(data_buffer)}) < mini_batch_size ({cfg.mini_batch_size}).")

        # --- CSV 파일에 에포크 결과 기록 ---
        try:
            with open(log_filename, mode="a", newline="") as f:
                writer_csv = csv.writer(f)
                loss_str = f"{avg_loss_val:.4f}" if num_updates > 0 else "N/A"
                actor_loss_str = f"{avg_actor_loss_val:.4f}" if num_updates > 0 else "N/A"
                critic_loss_str = f"{avg_critic_loss_val:.4f}" if num_updates > 0 else "N/A"
                entropy_loss_str = f"{avg_entropy_loss_val:.4f}" if num_updates > 0 else "N/A"
                writer_csv.writerow([
                    epoch, cfg.episodes_per_epoch, f"{avg_ep_reward:.2f}",
                    loss_str, actor_loss_str, critic_loss_str, entropy_loss_str,
                    f"{avg_final_max_move_sum:.2f}", f"{avg_crane_move:.2f}", total_steps_epoch
                ])
        except IOError as e:
            print(f"Error writing to log file {log_filename}: {e}")

        # --- 평가 ---
        if epoch % cfg.eval_every == 0 and epoch >= 0:
            eval_excel_file = getattr(cfg, "evaluation_plates_data_path", None)
            eval_final_metric = float('inf')
            if eval_excel_file and os.path.exists(eval_excel_file):
                try:
                    eval_df = pd.read_excel(eval_excel_file, sheet_name="reshuffle")
                    if eval_df.empty or 'scenario_id' not in eval_df.columns:
                        raise ValueError("Evaluation Excel file is empty or missing 'scenario_id' column.")

                    eval_envs = []
                    for scenario_id, group_df in eval_df.groupby('scenario_id'):
                        eval_schedule_for_one_env = []
                        for _, row in group_df.iterrows():
                            p = Plate(id=row["pileno"], inbound=row.get("inbound", 0),
                                      outbound=row.get("outbound", 1), unitw=row.get("unitw", 1.0))
                            p.from_pile = str(row["pileno"]).strip()
                            p.topile = str(row.get("topile", "A01")).strip()
                            eval_schedule_for_one_env.append(p)

                        env_eval = Locating(max_stack=cfg.max_stack,
                                            inbound_plates=copy.deepcopy(eval_schedule_for_one_env),
                                            device=device, crane_penalty=cfg.crane_penalty)
                        eval_envs.append(env_eval)

                    print(f"--- Evaluation (Epoch {epoch}) on {len(eval_envs)} scenarios ---")
                    eval_reward, eval_final_metric = evaluate_policy(model, eval_envs, device)

                    print(f"Avg Reward: {eval_reward:.2f}")
                    print(f"Avg Final Metric (Reversals): {eval_final_metric:.2f}")
                    print("---------------------------------")

                    tb_writer.add_scalar("Evaluation/AverageReward", eval_reward, epoch)
                    tb_writer.add_scalar("Evaluation/AvgFinalMetric", eval_final_metric, epoch)
                    if USE_VESSL:
                        vessl.log({"evaluation_reward": eval_reward, "evaluation_reversal": eval_final_metric},
                                  step=epoch)

                    if eval_final_metric < best_metric:
                        best_metric = eval_final_metric
                        os.makedirs(cfg.save_model_dir, exist_ok=True)
                        best_model_path = os.path.join(cfg.save_model_dir, "best_policy_metric.pth")
                        torch.save(model.state_dict(), best_model_path)
                        print(f"[BEST] New best model saved to {best_model_path} (Metric: {eval_final_metric:.2f})")

                except Exception as e:
                    print(f"Evaluation error: {e}")
            else:
                print("Evaluation Excel file not found or path not set.")

            # 목표 기반 조기 종료
            target_metric_value = getattr(cfg, "target_metric_value", -float('inf'))
            if eval_final_metric <= target_metric_value:
                print(
                    f"\n[Target Reached!] Metric {eval_final_metric:.3f} reached target {target_metric_value} at epoch {epoch}. Stopping.")
                break

        # --- LR 스케줄러 스텝 및 주기적 모델 저장 ---
        actor_lr_sched.step()
        critic_lr_sched.step()

        if epoch > 0 and (epoch + 1) % cfg.save_every == 0:
            os.makedirs(cfg.save_model_dir, exist_ok=True)
            save_path = os.path.join(cfg.save_model_dir, f"model_epoch_{epoch}.pth")
            checkpoint_data = {
                'epoch': epoch, 'global_step': global_step, 'model_state_dict': model.state_dict(),
                'actor_optimizer_state_dict': actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': critic_optimizer.state_dict(),
                'loss_metric': float(avg_loss_val), 'best_metric_val': float(best_metric),
                'actor_scheduler_state_dict': actor_lr_sched.state_dict(),
                'critic_scheduler_state_dict': critic_lr_sched.state_dict()
            }
            torch.save(checkpoint_data, save_path)
            print(f"Checkpoint saved: {save_path}")

    # --- 학습 종료 후 최종 모델 저장 ---
    os.makedirs(cfg.save_model_dir, exist_ok=True)
    final_model_path = os.path.join(cfg.save_model_dir, f"final_policy_ep{cfg.n_epoch - 1}.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved: {final_model_path}")

    tb_writer.close()
    print("Training session complete.")


if __name__ == "__main__":
    main()