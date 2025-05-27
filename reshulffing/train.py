import random
import numpy as np
import torch
import os
import csv
import pandas as pd
import math
import datetime
import copy
from cfg import get_cfg
from env import Locating # 수정된 Locating 클래스
from data import Plate, generate_schedule, generate_reshuffle_plan, save_reshuffle_plan_to_excel
from network import SteelPlateConditionalMLPModel, MAX_SOURCE, MAX_DEST
from evaluation import evaluate_policy # 평가 함수

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
        advantages.insert(0, gae) # 리스트 맨 앞에 추가
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
        v_val = transition['v'].detach().cpu().item() if isinstance(transition['v'], torch.Tensor) else transition['v']
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
    source_mask_tensor = torch.cat(source_mask_lst, dim=0).to(device) if source_mask_lst else torch.empty((0, MAX_SOURCE), dtype=torch.bool, device=device)
    dest_mask_tensor = torch.cat(dest_mask_lst, dim=0).to(device) if dest_mask_lst else torch.empty((0, MAX_DEST), dtype=torch.bool, device=device)
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
    print("=== Hyperparameters (Effective Values) ===") # 실제 적용 값임을 명시
    print(cfg_info)
    print("==========================================")

    # --- 최고 성능 지표 추적 변수 초기화 (낮을수록 좋음) ---
    best_metric = float("inf")

    # --- 초기 스케줄 생성 또는 로드 ---
    try:
        df_plan = pd.read_excel(cfg.plates_data_path, sheet_name="reshuffle")
        schedule = []
        for idx, row in df_plan.iterrows():
            plate_id = row["pileno"]
            inbound = row["inbound"] if ("inbound" in df_plan.columns and not pd.isna(row["inbound"])) else random.randint(cfg.inbound_min, cfg.inbound_max)
            outbound = row["outbound"] if ("outbound" in df_plan.columns and not pd.isna(row["outbound"])) else inbound + random.randint(cfg.outbound_extra_min, cfg.outbound_extra_max)
            unitw = row["unitw"] if ("unitw" in df_plan.columns and not pd.isna(row["unitw"])) else random.uniform(cfg.unitw_min, cfg.unitw_max)
            to_pile = str(row["topile"]).strip() if ("topile" in df_plan.columns and not pd.isna(row["topile"])) else "A01"
            p = Plate(id=plate_id, inbound=inbound, outbound=outbound, unitw=unitw)
            p.from_pile = str(plate_id).strip()
            p.topile = to_pile
            schedule.append(p)
        if not schedule: raise ValueError("Excel에서 로드한 스케줄이 비어있습니다.")
        print(f"초기 스케줄을 {cfg.plates_data_path} 에서 로드했습니다.")
    except Exception as e:
        print(f"Excel 로드 오류 ({e}), 기본 스케줄 생성")
        schedule = generate_schedule(num_plates=cfg.num_plates)
        unique_from = set()
        unique_to = set()
        for i, p in enumerate(schedule):
            p.from_pile = f"Source_{i % MAX_SOURCE}"
            p.topile = f"Dest_{i % MAX_DEST}"
            unique_from.add(p.from_pile)
            unique_to.add(p.topile)
        print(f"기본 생성된 스케줄: {len(schedule)}개 강판, {len(unique_from)}개 출발 파일, {len(unique_to)}개 목적 파일")

    _OBSERVED_TOP_N_PLATES = getattr(cfg, 'OBSERVED_TOP_N_PLATES', 30)
    _NUM_SUMMARY_STATS_DEEPER = getattr(cfg, 'NUM_SUMMARY_STATS_DEEPER', 4)
    _NUM_PILE_TYPE_FEATURES = 1
    _NUM_BLOCKING_FEATURES = 1
    actual_pile_feature_dim_for_model = _OBSERVED_TOP_N_PLATES + \
                                        _NUM_SUMMARY_STATS_DEEPER + \
                                        _NUM_PILE_TYPE_FEATURES + \
                                        _NUM_BLOCKING_FEATURES

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
        num_from_piles=MAX_SOURCE,  # network.py의 MAX_SOURCE와 일치
        num_to_piles=MAX_DEST  # network.py의 MAX_DEST와 일치
    ).to(device)

    # train.py 에서 수정해야 할 부분 (기존 코드 예시)
    # actor_params = list(model.pile_encoder.parameters()) + \
    #                list(model.self_attention.parameters()) + \
    #                list(model.actor_head.parameters()) +    # list(model.dest_fc.parameters()) + [model.dest_embeddings]
    # Optimizer & LR Scheduler 설정

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

    actor_optimizer = optim.Adam(actor_params, lr=cfg.actor_lr)
    critic_optimizer = optim.Adam(critic_params, lr=cfg.critic_lr)

    num_epochs = float(cfg.n_epoch)
    lr_lambda = lambda epoch: max(0.0, 1.0 - (epoch / num_epochs))
    actor_lr_sched = torch.optim.lr_scheduler.LambdaLR(actor_optimizer, lr_lambda=lr_lambda)
    critic_lr_sched = torch.optim.lr_scheduler.LambdaLR(critic_optimizer, lr_lambda=lr_lambda)

    print(f"Using Linear LR Decay schedule over {int(num_epochs)} epochs.")

    # actor_lr_sched = StepLR(actor_optimizer, step_size=cfg.lr_step, gamma=cfg.lr_decay)
    # critic_lr_sched = StepLR(critic_optimizer, step_size=cfg.lr_step, gamma=cfg.lr_decay)

    # CSV 로그 파일 설정
    log_dir = os.path.dirname(cfg.log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    log_filename = cfg.log_file if cfg.log_file else f"train_log_{timestamp}.csv"

    # CSV 파일 헤더 작성
    try:
        with open(log_filename, mode="w", newline="") as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow(["Epoch", "Total_Episodes", "Avg_Episode_Reward", "Avg_Loss",
                                "Actor_Loss", "Critic_Loss", "Entropy_Loss",
                                "Avg_Max_Move_Sum", "Crane_Move", "Total_Steps"])
    except IOError as e:
        print(f"Error opening log file {log_filename}: {e}")
        return

    # 학습 루프 변수 초기화
    num_epochs = cfg.n_epoch
    episodes_per_epoch = cfg.episodes_per_epoch
    T_horizon = cfg.T_horizon
    data_buffer = []
    recent_rewards = []
    early_stop_patience = getattr(cfg, "early_stop_patience", 10)
    min_delta = getattr(cfg, "min_delta", 0.01)
    global_step = 0

    # --- 메인 학습 루프 ---
    for epoch in range(num_epochs):
        current_schedule = schedule
        if epoch > 0 and epoch % cfg.new_instance_every == 0:
            print(f"새로운 시나리오 생성됨 (Epoch: {epoch})")
            try:
                df_plan_new, _, _, _, _ = generate_reshuffle_plan(
                    rows=['A', 'B'],
                    n_from_piles_reshuffle=cfg.n_from_piles_reshuffle,
                    n_to_piles_reshuffle=cfg.n_to_piles_reshuffle,
                    n_plates_reshuffle=cfg.n_plates_reshuffle,
                    safety_margin=cfg.safety_margin
                )
                current_schedule = []
                for idx, row in df_plan_new.iterrows():
                    p = Plate(row["pileno"], row["inbound"], row["outbound"], row["unitw"])
                    p.from_pile = row["pileno"]
                    p.topile = row["topile"]
                    current_schedule.append(p)
            except Exception as e:
                print(f"새 시나리오 생성 오류 ({e}), 이전 스케줄 사용")

        total_ep_reward = 0.0
        total_steps = 0
        ep_final_max_move_sums = [] # 역적재
        ep_crane_moves = []
        last_infos = [] # 마지막 스텝의 info 저장용

        # --- 에피소드 반복 ---
        for ep in range(episodes_per_epoch):
            # 환경 생성 시 num_stack 인자 제거
            env = Locating(
                max_stack=cfg.max_stack,
                inbound_plates=copy.deepcopy(current_schedule),
                device=device,
                crane_penalty=cfg.crane_penalty,
            )
            state = env.reset(shuffle_schedule=False)
            state = state.to(device)
            done = False
            ep_reward = 0.0
            ep_steps = 0
            episode_transitions = []
            last_info = {} # 마지막 info 저장용
            t = 0

            # --- 한 에피소드 진행 ---
            while not done and t < T_horizon:
                s_mask, d_mask = env.get_masks()
                source_mask = torch.zeros(1, MAX_SOURCE, dtype=torch.bool, device=device)
                dest_mask = torch.zeros(1, MAX_DEST, dtype=torch.bool, device=device)
                valid_source_length = s_mask.size(0)
                valid_dest_length = d_mask.size(0)
                source_mask[0, :valid_source_length] = s_mask.clone().detach()
                dest_mask[0, :valid_dest_length] = d_mask.clone().detach()

                state_tensor = state.unsqueeze(0)
                actions, logprobs, value, _ = model.act_batch(
                    state_tensor, source_mask, dest_mask, greedy=False, debug=False)

                action = actions[0]
                logprob = logprobs[0]
                value_val = value[0]
                src_idx = action[0].item()
                dst_idx = action[1].item()

                next_state, reward, done, info  = env.step((src_idx, dst_idx))
                next_state = next_state.to(device)

                transition = {
                    'state': state, 'action': action, 'r': reward, 'next_state': next_state,
                    'logprob': logprob.item() if hasattr(logprob, 'item') else logprob,
                    'v': value_val, 'source_mask': source_mask[0], 'dest_mask': dest_mask[0], 'done': done
                }
                episode_transitions.append(transition)
                last_info = info # 마지막 info 덮어쓰기

                ep_reward += reward
                ep_steps += 1
                state = next_state
                t += 1
                if done: break

            total_ep_reward += ep_reward
            total_steps += ep_steps
            final_max_move = last_info.get('final_max_move_sum', None) # 마지막 info에서 값 추출
            if final_max_move is not None:
                ep_final_max_move_sums.append(final_max_move)
            else:
                 # done=True일 때 info에 값이 없는 예외적인 경우 처리
                 print(f"[Warning] Epoch {epoch}, Ep {ep}: final_max_move_sum not found in last info.")
                 # ep_final_max_move_sums.append(float('inf')) # 또는 다른 값
            ep_crane_move = env.crane_move if hasattr(env, "crane_move") else 0.0

            if episode_transitions:
                for tr in episode_transitions:
                     tr['v'] = tr['v'].detach().cpu().item() if isinstance(tr['v'], torch.Tensor) else tr['v']
                advantages = compute_gae(episode_transitions, cfg.gamma, cfg.lmbda)
                for idx_trans, tr in enumerate(episode_transitions):
                    tr['advantage'] = advantages[idx_trans]
                    data_buffer.append(tr)

        avg_ep_reward = total_ep_reward / episodes_per_epoch
        avg_final_max_move_sum = np.mean(ep_final_max_move_sums) if ep_final_max_move_sums else 0.0
        avg_crane_move = np.mean(ep_crane_moves) if ep_crane_moves else 0.0

        print(f"Epoch {epoch}: Avg Reward={avg_ep_reward:.2f}, Avg Final Max Move Sum={avg_final_max_move_sum:.2f}, Avg Crane={avg_crane_move:.2f}, Total Steps={total_steps}")

        # TensorBoard 및 Vessl 로깅 (에포크 단위)
        tb_writer.add_scalar("Training/AverageReward", avg_ep_reward, epoch)
        tb_writer.add_scalar("Training/AvgFinalMaxMoveSum", avg_final_max_move_sum, epoch)
        tb_writer.add_scalar("Training/AverageCraneMove", avg_crane_move, epoch)
        tb_writer.add_scalar("Training/TotalSteps", total_steps, epoch)
        if USE_VESSL:
            vessl.log({
                "training_reward": avg_ep_reward,
                "training_reversal": avg_final_max_move_sum,
                "training_crane_move": avg_crane_move,
            }, step=epoch)

        # 조기 종료 검사
        recent_rewards.append(avg_ep_reward)
        if len(recent_rewards) > early_stop_patience:
            recent_rewards.pop(0)
        if len(recent_rewards) == early_stop_patience:
            reward_diff = max(recent_rewards) - min(recent_rewards)
            if reward_diff < min_delta:
                print(f"Early stopping triggered at epoch {epoch} (Reward difference {reward_diff:.4f} < {min_delta})")
                break

        # --- PPO 업데이트 ---
        if len(data_buffer) >= cfg.mini_batch_size :
            mini_data_buffer = data_buffer.copy()
            data_buffer.clear()

            total_loss = 0.0        # 최종 가중합 손실 누적용
            total_actor_loss = 0.0  # 순수 액터 손실 누적용
            total_critic_loss = 0.0 # 순수 크리틱 손실 누적용
            total_entropy_loss = 0.0 # 계수 적용된 엔트로피 손실 누적용
            num_updates = 0

            for _ in range(cfg.K_epoch):
                N = len(mini_data_buffer)
                indices = torch.randperm(N)
                num_minibatches = N // cfg.mini_batch_size

                for mb_i in range(num_minibatches):
                    start = mb_i * cfg.mini_batch_size
                    end = (mb_i + 1) * cfg.mini_batch_size
                    if end > N: continue

                    batch_idx = indices[start:end]
                    mini_data = [mini_data_buffer[j] for j in batch_idx.tolist()]

                    mini_s, mini_a, mini_r, mini_adv, mini_s_prime, \
                    mini_a_logprob, mini_v, mini_source_mask, mini_dest_mask, mini_done = make_batch(mini_data, device)

                    mini_adv = (mini_adv - mini_adv.mean()) / (mini_adv.std() + 1e-8)

                    with torch.no_grad():
                        _, _, v_next, _ = model.forward(mini_s_prime)
                    td_target = mini_r + cfg.gamma * v_next * (1.0 - mini_done)

                    # --- PPO 손실 계산 ---
                    new_a_logprob, new_v, dist_entropy = model.evaluate(
                        mini_s, mini_source_mask, mini_dest_mask, mini_a
                    )

                    # Actor 손실 (텐서)
                    ratio = torch.exp(new_a_logprob - mini_a_logprob)
                    surr1 = ratio * mini_adv
                    surr2 = torch.clamp(ratio, 1.0 - cfg.eps_clip, 1.0 + cfg.eps_clip) * mini_adv
                    actor_loss = -torch.min(surr1, surr2).mean()

                    # Critic 손실 (텐서, Smooth L1)
                    value_pred_clipped = mini_v + (new_v - mini_v).clamp(-cfg.value_clip_range, cfg.value_clip_range)
                    beta = getattr(cfg, "smooth_l1_beta", 1.0)
                    critic_loss_unclipped = F.smooth_l1_loss(new_v, td_target, reduction="none", beta=beta)
                    critic_loss_clipped = F.smooth_l1_loss(value_pred_clipped, td_target, reduction="none", beta=beta)
                    critic_loss = torch.max(critic_loss_unclipped, critic_loss_clipped).mean()

                    # Entropy 손실
                    entropy_loss = -cfg.E_coeff * dist_entropy.mean()

                    # --- 누적 ---
                    total_actor_loss += actor_loss.item() # 순수 액터 손실 누적
                    total_critic_loss += critic_loss.item() # 순수 크리틱 손실 누적
                    total_entropy_loss += entropy_loss.item()

                    ppo_loss = (cfg.P_coeff * actor_loss +
                                cfg.V_coeff * critic_loss +
                                entropy_loss)

                    total_loss += ppo_loss.item() # 가중합된 최종 손실 누적

                    # 옵티마이저 스텝
                    actor_optimizer.zero_grad()
                    critic_optimizer.zero_grad()
                    ppo_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                    actor_optimizer.step()
                    critic_optimizer.step()

                    num_updates += 1
                    global_step += 1
                # --- 미니배치 루프 끝 ---

            # --- K_epoch 루프 끝 ---

            # 에포크 평균 손실 계산
            if num_updates > 0:
                avg_loss = total_loss / num_updates
                avg_actor_loss = total_actor_loss / num_updates
                avg_critic_loss = total_critic_loss / num_updates
                avg_entropy_loss = total_entropy_loss / num_updates

                # 콘솔 출력 (각 요소 평균 포함)
                print(f"Epoch {epoch}: Avg Loss={avg_loss:.4f} (Actor={avg_actor_loss:.4f}, Critic={avg_critic_loss:.4f}, Entropy={avg_entropy_loss:.4f})")

                # TensorBoard 및 Vessl 로깅
                tb_writer.add_scalar("Loss/TotalLoss", avg_loss, epoch)
                tb_writer.add_scalar("Loss/ActorLoss", avg_actor_loss, epoch)
                tb_writer.add_scalar("Loss/CriticLoss", avg_critic_loss, epoch)
                tb_writer.add_scalar("Loss/EntropyLoss", avg_entropy_loss, epoch)
                if USE_VESSL:
                    vessl.log({
                        "total_loss": avg_loss,
                        "actor_loss": avg_actor_loss,
                        "critic_loss": avg_critic_loss,
                        "entropy_loss": avg_entropy_loss
                    }, step=epoch)
            else:
                avg_loss = avg_actor_loss = avg_critic_loss = avg_entropy_loss = 0.0
                print(f"Epoch {epoch}: No update performed (data buffer size < mini_batch_size).")

            # CSV 파일에 에포크 결과 기록
            try:
                with open(log_filename, mode="a", newline="") as f:
                    writer_csv = csv.writer(f)
                    writer_csv.writerow([
                        epoch,
                        episodes_per_epoch,
                        avg_ep_reward,
                        avg_loss,
                        avg_actor_loss,
                        avg_critic_loss,
                        avg_entropy_loss,
                        avg_final_max_move_sum,
                        avg_crane_move,
                        total_steps
                    ])
            except IOError as e:
                print(f"Error writing to log file {log_filename}: {e}")

        # --- 평가 ---
        if epoch % cfg.eval_every == 0 and epoch >= 0:
            eval_excel_file = getattr(cfg, "evaluation_plates_data_path", None)
            if eval_excel_file and os.path.exists(eval_excel_file):
                try:
                    # 1. 평가용 Excel 파일 하나를 읽어옴
                    eval_df = pd.read_excel(eval_excel_file, sheet_name="reshuffle")
                    if eval_df.empty or 'scenario_id' not in eval_df.columns:
                        raise ValueError("평가용 Excel 파일이 비어 있거나 'scenario_id' 컬럼이 없습니다.")

                    # 2. 여러 환경 객체를 담을 빈 리스트를 생성
                    eval_envs = []

                    # 3. 'scenario_id'를 기준으로 데이터를 그룹화하여 각 시나리오를 개별적으로 처리
                    for scenario_id, group_df in eval_df.groupby('scenario_id'):

                        # 4. 각 그룹(하나의 시나리오)의 데이터로 스케줄 리스트를 만듬
                        eval_schedule_for_one_env = []
                        for idx, row in group_df.iterrows():
                            plate_id = row["pileno"]
                            inbound = row.get("inbound", 0)
                            outbound = row.get("outbound", 1)
                            unitw = row.get("unitw", 1.0)
                            to_pile = str(row.get("topile", "A01")).strip()
                            p = Plate(id=plate_id, inbound=inbound, outbound=outbound, unitw=unitw)
                            p.from_pile = str(plate_id).strip()
                            p.topile = to_pile
                            eval_schedule_for_one_env.append(p)

                        # 5. 한 개의 시나리오로 한 개의 환경(env) 객체를 생성
                        env = Locating(
                            max_stack=cfg.max_stack,
                            inbound_plates=copy.deepcopy(eval_schedule_for_one_env),
                            device=device,
                            crane_penalty=cfg.crane_penalty,
                        )
                        # 6. 생성된 환경을 큰 리스트에 추가
                        eval_envs.append(env)

                    print(f"총 {len(eval_envs)}개의 평가 환경을 생성하여 평가를 시작합니다.")

                    # 7. '평가 환경 리스트' 전체를 evaluate_policy 함수에 전달
                    eval_reward, eval_final_metric = evaluate_policy(model, eval_envs, device)

                    # (이하 평가 결과 로깅 코드는 원래 코드와 동일하게 유지)
                    print(f"--- Evaluation (Epoch {epoch}) ---")
                    print(f"Avg Reward: {eval_reward:.2f}")
                    print(f"Avg Final Metric (e.g., MaxMoveSum/Reversals): {eval_final_metric:.2f}")
                    print("-------------------------------")

                    tb_writer.add_scalar("Evaluation/AverageReward", eval_reward, epoch)
                    tb_writer.add_scalar("Evaluation/AvgFinalMetric", eval_final_metric, epoch)
                    if USE_VESSL:
                        vessl.log({
                            "evaluation_reward": eval_reward,
                            "evaluation_reversal": eval_final_metric,
                        }, step=epoch)

                    # 최고 모델 저장 로직 (원래 코드와 동일)
                    if eval_final_metric < best_metric:
                        best_metric = eval_final_metric
                        os.makedirs(cfg.save_model_dir, exist_ok=True)
                        best_model_path = os.path.join(cfg.save_model_dir, "best_policy_metric.pth")
                        torch.save(model.state_dict(), best_model_path)
                        print(f"[BEST] New best model saved to {best_model_path} (Metric: {eval_final_metric:.2f})")

                except Exception as e:
                    print(f"평가 중 오류 발생: {e}")

            else:
                print("평가용 Excel 파일이 없거나 경로가 설정되지 않았습니다.")
                # 2. 목표 기반 조기 종료 확인 (Metric 기준)
                target_metric_value = cfg.target_metric_value

                if eval_final_metric <= target_metric_value:
                    print(
                        f"\n[Target Reached!] Evaluation metric {eval_final_metric:.3f} reached target {target_metric_value} at epoch {epoch}.")
                    print("Stopping training.")
                    try:
                        target_reached_path = os.path.join(cfg.save_model_dir, f"target_reached_ep{epoch}.pth")
                        torch.save(
                            {'epoch': epoch, 'global_step': global_step, 'model_state_dict': model.state_dict(),
                             'actor_optimizer_state_dict': actor_optimizer.state_dict(),
                             'critic_optimizer_state_dict': critic_optimizer.state_dict(),
                             'best_metric': best_metric,
                             'final_eval_metric': eval_final_metric},
                            target_reached_path)
                        print(f"Model saved upon reaching target: '{target_reached_path}'")
                    except Exception as save_e:
                        print(f"[ERROR] Failed to save target reached model: {save_e}")
                    break
                    # --- 메인 학습 루프(for epoch)를 중단 ---

        # LR 스케줄러 스텝
        actor_lr_sched.step()
        critic_lr_sched.step()

        # 주기적 모델 저장
        if epoch > 0 and (epoch + 1) % cfg.save_every == 0:
            os.makedirs(cfg.save_model_dir, exist_ok=True)
            save_path = os.path.join(cfg.save_model_dir,
                                     f"model_epoch_{epoch}.pth")

            checkpoint = {
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'actor_optimizer_state_dict': actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': critic_optimizer.state_dict(),
                'loss_metric': avg_loss,
                'best_metric_val': best_metric,
            }
            if actor_lr_sched:
                checkpoint['actor_scheduler_state_dict'] = actor_lr_sched.state_dict()
            if critic_lr_sched:
                checkpoint['critic_scheduler_state_dict'] = critic_lr_sched.state_dict()

            torch.save(checkpoint, save_path)
            print(f"전체 체크포인트 저장됨: {save_path} (에폭 {epoch}, Global Step {global_step})")

    # 학습 종료 후 최종 모델 저장
    os.makedirs(cfg.save_model_dir, exist_ok=True)
    final_model_path = os.path.join(cfg.save_model_dir, "final_policy.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"최종 모델 저장됨: {final_model_path}")

    # TensorBoard writer 닫기
    tb_writer.close()
    print("학습 완료.")

if __name__ == "__main__":
    main()