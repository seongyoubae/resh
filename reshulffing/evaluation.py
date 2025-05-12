import torch
import numpy as np # numpy 임포트 확인
from network import MAX_SOURCE, MAX_DEST

def evaluate_policy(model, envs, device):
    """
    여러 평가 환경(envs)에 대해 학습된 모델을 실행하여
    에피소드 보상과 '최종 상태의 최대 블로킹 합계(Final Max Move Sum)'를 계산합니다.
    """
    model.eval() # 평가 모드 설정
    total_rewards = []
    total_final_metrics = [] # 최종 상태 지표(max_move_sum) 저장용 리스트

    print(f"--- Starting Evaluation ({len(envs)} envs) ---")
    for idx, env in enumerate(envs):
        # # 디버깅: 초기 스케줄 플레이트 정보 (필요시 주석 해제)
        # print(f"[DEBUG] Eval Env {idx}: Initial Plates:")
        # if hasattr(env, 'inbound_plates'):
        #      for p_idx, plate in enumerate(env.inbound_plates):
        #           print(f"  Plate {p_idx} -> id={getattr(plate, 'id', 'N/A')}, outbound={getattr(plate, 'outbound', 'N/A')}")
        # else:
        #      print("  inbound_plates attribute not found.")

        try:
            # 평가는 보통 고정된 시작 상태 사용 (shuffle_schedule=False)
            state = env.reset(shuffle_schedule=False)
            # print(f"[DEBUG] Env {idx} Initial state shape: {state.shape}")
        except Exception as e:
            print(f"[Error] Env {idx} reset failed: {e}")
            total_final_metrics.append(float('inf')) # 실패 시 최악의 값 기록
            total_rewards.append(-float('inf'))      # 실패 시 최악의 값 기록
            continue # 다음 환경으로

        done = False
        ep_reward = 0.0
        final_metric_val = None # 에피소드 종료 시 값 저장용
        last_info = {} # 마지막 info 저장용

        # 에피소드 실행
        step_count = 0
        # total_plate_count가 0인 경우 등 대비하여 최소 스텝 보장 및 최대 스텝 설정
        max_steps = max(1, getattr(env, 'total_plate_count', 500)) * 2 # 무한 루프 방지 (넉넉하게 설정)

        while not done and step_count < max_steps:
            # 상태 텐서 준비
            state_tensor = state.unsqueeze(0).to(device)

            # 마스크 생성
            try:
                env_source_mask, env_dest_mask = env.get_masks()
            except Exception as e:
                print(f"[Error] Env {idx}, Step {step_count}: get_masks failed: {e}")
                final_metric_val = float('inf') # 에러 시 최악의 값
                done = True # 에피소드 중단 처리
                break

            # 마스크 텐서 변환 및 패딩
            source_mask = torch.zeros(1, MAX_SOURCE, dtype=torch.bool, device=device)
            dest_mask   = torch.zeros(1, MAX_DEST,   dtype=torch.bool, device=device)
            if isinstance(env_source_mask, torch.Tensor) and env_source_mask.ndim == 1:
                valid_source_length = min(len(env_source_mask), MAX_SOURCE)
                source_mask[0, :valid_source_length] = env_source_mask[:valid_source_length].to(device)
            if isinstance(env_dest_mask, torch.Tensor) and env_dest_mask.ndim == 1:
                valid_dest_length   = min(len(env_dest_mask), MAX_DEST)
                dest_mask[0, :valid_dest_length] = env_dest_mask[:valid_dest_length].to(device)

            # 모델 추론 (greedy)
            try:
                with torch.no_grad():
                    actions, _, _, _ = model.act_batch(state_tensor, source_mask, dest_mask, greedy=True)
                action = actions[0]
                source_idx = action[0].item()
                dest_idx = action[1].item()
            except Exception as e:
                print(f"[Error] Env {idx}, Step {step_count}: model inference failed: {e}")
                final_metric_val = float('inf') # 에러 시 최악의 값
                done = True # 에피소드 중단 처리
                break

            # 액션 유효성 검사/클리핑 (CPU 마스크 사용)
            source_mask_cpu = source_mask[0].cpu()
            dest_mask_cpu = dest_mask[0].cpu()
            # source 인덱스 검사
            if not (0 <= source_idx < MAX_SOURCE and source_mask_cpu[source_idx]):
                 valid_indices = torch.where(source_mask_cpu)[0].tolist()
                 if valid_indices: source_idx = valid_indices[0]
                 else: print(f"[Eval Warning] Env {idx}: No valid source actions left."); break
            # destination 인덱스 검사
            if not (0 <= dest_idx < MAX_DEST and dest_mask_cpu[dest_idx]):
                 valid_indices = torch.where(dest_mask_cpu)[0].tolist()
                 if valid_indices: dest_idx = valid_indices[0]
                 else: print(f"[Eval Warning] Env {idx}: No valid destination actions left."); break

            # 환경 스텝 실행
            try:
                # ### MODIFIED ### : info 캡처
                state_cpu, reward, done, info = env.step((source_idx, dest_idx))
                state = state_cpu.to(device) # 다음 상태 device로
                ep_reward += reward
                last_info = info # 마지막 info 업데이트
                step_count += 1
            except Exception as e:
                 print(f"[Error] Env {idx}, Step {step_count}: env.step failed: {e}")
                 final_metric_val = float('inf') # 에러 시 최악의 값
                 done = True # 에피소드 중단 처리
                 break

        # --- 에피소드 종료 후 처리 ---

        # ### MODIFIED ### : 최종 상태 지표 값 추출
        if final_metric_val is None: # 루프가 정상 종료되었거나 break로 빠져나왔으나 아직 값이 설정 안 된 경우
            if done: # 정상 종료 시 last_info에서 값 추출 시도
                 final_metric_val = last_info.get('final_max_move_sum', None)
                 if final_metric_val is None:
                      print(f"[Eval Warning] Env {idx}: 'final_max_move_sum' not in last info dict. Using inf.")
                      final_metric_val = float('inf') # 키가 없을 경우
            elif step_count >= max_steps: # 최대 스텝 도달 시
                 print(f"[Eval Warning] Env {idx}: Reached max steps ({max_steps}) without finishing.")
                 # 현재 상태 기준으로 계산 시도 (env에 해당 메서드 필요)
                 try:
                     final_metric_val = sum(env._get_total_blocking_pairs(env.plates.get(key, [])) for key in env.to_keys)
                     print(f"          Calculated max_move_sum at max steps: {final_metric_val}")
                 except Exception as e:
                     print(f"          Could not calculate max_move_sum at max steps: {e}")
                     final_metric_val = float('inf') # 계산 실패 시
            else: # 다른 이유로 루프 종료 (예: valid action 없음)
                 final_metric_val = float('inf') # 알 수 없는 종료는 최악으로 처리

        # 결과 기록
        # ### MODIFIED ### : 출력 메시지 변경
        print(f"  Env {idx} finished => Reward={ep_reward:.2f}, Final Max Move Sum={final_metric_val}")
        total_rewards.append(ep_reward)
        total_final_metrics.append(final_metric_val) # 이름 변경 및 올바른 값 누적

    # 최종 평균 계산
    avg_reward = np.mean(total_rewards) if total_rewards else 0.0
    # ### MODIFIED ### : 평균 지표 이름 변경
    avg_final_metric = np.mean(total_final_metrics) if total_final_metrics else float('inf')
    print(f"--- Evaluation Finished ---")
    print(f"Avg Reward: {avg_reward:.2f}")
    # ### MODIFIED ### : 최종 출력 레이블 변경
    print(f"Avg Final Metric (MaxMoveSum): {avg_final_metric:.2f}")

    # 반환 값 변경
    return avg_reward, avg_final_metric