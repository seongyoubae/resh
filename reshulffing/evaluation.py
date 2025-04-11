# evaluation.py
import torch
from network import MAX_SOURCE, MAX_DEST  # MAX_SOURCE, MAX_DEST는 평가 시 mask 생성 용도로 사용

def evaluate_policy(model, envs, device):
    """
    여러 평가 환경(envs)에 대해 학습된 모델을 실행하여 에피소드 보상과 reversal 값을 계산합니다.
    환경의 reset()에서 반환한 상태는 이미 [total_piles, pile_feature_dim] 형태라고 가정합니다.
    """
    model.eval()
    total_rewards = []
    total_reversals = []

    for idx, env in enumerate(envs):
        # 디버깅: 각 plate의 inbound/outbound 정보 출력
        print(f"[DEBUG] Eval Env {idx}: Checking outbound of each plate")
        for p_idx, plate in enumerate(env.inbound_plates):
            print(f"  Plate {p_idx} -> id={plate.id}, inbound={plate.inbound}, outbound={plate.outbound}")

        # env.reset()는 이미 [total_piles, pile_feature_dim] 형태의 상태를 반환합니다.
        state = env.reset()
        print(f"[DEBUG] Initial state shape: {state.shape}")

        done = False
        ep_reward = 0.0

        while not done:
            # 배치 차원 추가: [1, total_piles, pile_feature_dim]
            state_tensor = state.unsqueeze(0).to(device)

            # mask 생성:
            env_source_mask, env_dest_mask = env.get_masks()
            if not isinstance(env_source_mask, torch.Tensor):
                env_source_mask = torch.tensor(env_source_mask, dtype=torch.bool, device=device)
            if not isinstance(env_dest_mask, torch.Tensor):
                env_dest_mask = torch.tensor(env_dest_mask, dtype=torch.bool, device=device)
            source_mask = torch.zeros(1, MAX_SOURCE, dtype=torch.bool, device=device)
            dest_mask   = torch.zeros(1, MAX_DEST,   dtype=torch.bool, device=device)
            valid_source_length = len(env_source_mask)
            valid_dest_length   = len(env_dest_mask)
            source_mask[0, :valid_source_length] = env_source_mask.to(device)
            dest_mask[0, :valid_dest_length] = env_dest_mask.to(device)

            # 모델 추론 (greedy 선택)
            with torch.no_grad():
                actions, _, _, _ = model.act_batch(state_tensor, source_mask, dest_mask, greedy=True)
            action = actions[0]
            source_idx = action[0].item()
            dest_idx = action[1].item()

            # mask에서 유효하지 않은 인덱스 선택 시 첫 번째 유효 인덱스로 클리핑
            if source_idx >= valid_source_length or not env_source_mask[source_idx]:
                valid_indices = [i for i, flag in enumerate(env_source_mask.tolist()) if flag]
                if valid_indices:
                    print(f"[DEBUG] Clipping source_idx {source_idx} to {valid_indices[0]}")
                    source_idx = valid_indices[0]
                else:
                    break
            if dest_idx >= valid_dest_length or not env_dest_mask[dest_idx]:
                valid_indices = [i for i, flag in enumerate(env_dest_mask.tolist()) if flag]
                if valid_indices:
                    print(f"[DEBUG] Clipping dest_idx {dest_idx} to {valid_indices[0]}")
                    dest_idx = valid_indices[0]
                else:
                    break

            state, reward, done, _ = env.step((source_idx, dest_idx))
            # env.step()에서 반환한 상태는 역시 이미 [total_piles, pile_feature_dim]로 구성되어 있다고 가정
            ep_reward += reward

        print(f"[DEBUG] Eval Env {idx} done => Reward={ep_reward}, Reversal={env.last_reversal}")
        total_rewards.append(ep_reward)
        total_reversals.append(env.last_reversal)

    avg_reward = sum(total_rewards) / len(total_rewards) if total_rewards else 0.0
    avg_reversal = sum(total_reversals) / len(total_reversals) if total_reversals else 0.0
    print(f"[DEBUG] Eval total => Avg Reward={avg_reward:.2f}, Avg Reversal={avg_reversal:.2f}")
    return avg_reward, avg_reversal
