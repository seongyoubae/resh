import random
import numpy as np
import torch
import os
import pandas as pd
# import math
from cfg import get_cfg

# === 사용자 정의 모듈 임포트 (network, env, data) ===
try:
    from network import SteelPlateConditionalMLPModel, MAX_SOURCE, MAX_DEST
    print("network.py 로드 성공.")
except ImportError as e:
    print(f"[오류] network.py 임포트 실패: {e}")
    exit()
except Exception as e:
    print(f"[오류] network.py 로드 중 예상치 못한 오류: {e}")
    exit()

# --- env.py 임포트 ---
try:
    # !!! 중요: env.py 의 Locating 클래스 내부에 export_final_state_to_excel 메서드가 정의되어 있어야 합니다 !!!
    from env import Locating
    print("env.py 로드 성공.")
except ImportError as e:
    print(f"[오류] env.py 임포트 실패: {e}")
    Locating = None
except Exception as e:
    print(f"[오류] env.py 로드 중 예상치 못한 오류: {e}")
    Locating = None
    exit()

# --- data.py 임포트 ---
try:
    from data import Plate, generate_schedule
    print("data.py 로드 성공.")
except ImportError as e:
    print(f"[오류] data.py 임포트 실패: {e}")
    Plate = None
    generate_schedule = None
except Exception as e:
    print(f"[오류] data.py 로드 중 예상치 못한 오류: {e}")
    Plate = None
    generate_schedule = None
    exit()

# === 헬퍼 함수 ===
def set_seed(seed):
    """지정된 시드로 random, numpy, torch의 난수 생성기를 초기화합니다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"시드 설정 완료: {seed}")

def load_eval_schedule(excel_file):
    """
    주어진 엑셀 파일 경로에서 평가용 schedule 리스트를 구성합니다.
    (data.py 에서 Plate 클래스가 성공적으로 임포트되어야 사용 가능)
    """
    if Plate is None:
        print("[오류] 'Plate' 클래스를 사용할 수 없어 스케줄을 로드할 수 없습니다.")
        return None

    if not os.path.exists(excel_file):
        print(f"[경고] 평가용 엑셀 파일을 찾을 수 없습니다: {excel_file}")
        return None

    try:
        df = pd.read_excel(excel_file, sheet_name="reshuffle")
        schedule = []
        print(f"평가용 스케줄 로딩 시작: {excel_file}")

        for idx, row in df.iterrows():
            plate_id = str(row["pileno"]) if "pileno" in df.columns and not pd.isna(row["pileno"]) else f"판_{idx}"
            inbound = int(row["inbound"]) if "inbound" in df.columns and not pd.isna(row["inbound"]) else random.randint(1, 5)
            outbound = int(row["outbound"]) if "outbound" in df.columns and not pd.isna(row["outbound"]) else random.randint(inbound + 1, inbound + 20)
            unitw = float(row["unitw"]) if "unitw" in df.columns and not pd.isna(row["unitw"]) else random.uniform(0.8, 1.5)
            default_topile = f"D_{random.randint(0, MAX_DEST - 1)}"
            topile = str(row["topile"]).strip() if "topile" in df.columns and not pd.isna(row["topile"]) else default_topile

            p = Plate(id=plate_id, inbound=inbound, outbound=outbound, unitw=unitw)
            p.from_pile = str(row["pileno"]) if "pileno" in df.columns and not pd.isna(row["pileno"]) else plate_id
            p.topile = topile
            schedule.append(p)

        print(f"엑셀 파일에서 {len(schedule)}개의 플레이트 정보를 성공적으로 로드했습니다.")
        return schedule
    except KeyError as e:
        print(f"[오류] 엑셀 파일 '{excel_file}'에 필요한 컬럼이 없습니다: {e}.")
        return None
    except Exception as e:
        print(f"[오류] 엑셀 파일 로드 중 오류 발생 ({excel_file}): {e}")
        return None

# === 평가 함수 ===
def evaluate_policy(model, envs, device):
    """
    여러 평가 환경(envs)에 대해 학습된 모델을 실행하여
    에피소드 보상과 '최종 상태의 최대 블로킹 합계(Final Max Move Sum)'를 계산합니다.
    """
    model.eval() # 평가 모드 설정
    total_rewards = []
    total_final_metrics = [] # 최종 상태 지표(max_move_sum) 저장용 리스트

    print(f"--- 평가 시작 ({len(envs)}개 환경) ---")
    for idx, env in enumerate(envs):
        print(f"[정보] 환경 {idx} 평가 시작...")

        try:
            state = env.reset(shuffle_schedule=False)
        except Exception as e:
            print(f"[오류] 환경 {idx} 리셋 실패: {e}")
            total_final_metrics.append(float('inf'))
            total_rewards.append(-float('inf'))
            continue

        done = False
        ep_reward = 0.0
        final_metric_val = None
        last_info = {}
        step_count = 0
        total_plates_in_env = getattr(env, 'total_plate_count', 500)
        max_steps = max(1, total_plates_in_env) * 3
        print(f"[정보] 환경 {idx}: 최대 스텝 수 설정 = {max_steps}")

        while not done and step_count < max_steps:
            state_tensor = state.unsqueeze(0).to(device)
            try:
                env_source_mask, env_dest_mask = env.get_masks()
            except Exception as e:
                print(f"[오류] 환경 {idx}, 스텝 {step_count}: get_masks 실패: {e}")
                final_metric_val = float('inf'); done = True; break

            source_mask = torch.zeros(1, MAX_SOURCE, dtype=torch.bool, device=device)
            dest_mask   = torch.zeros(1, MAX_DEST,   dtype=torch.bool, device=device)
            if isinstance(env_source_mask, torch.Tensor) and env_source_mask.ndim == 1:
                valid_source_length = min(len(env_source_mask), MAX_SOURCE)
                source_mask[0, :valid_source_length] = env_source_mask[:valid_source_length].clone().detach().to(device)
            if isinstance(env_dest_mask, torch.Tensor) and env_dest_mask.ndim == 1:
                valid_dest_length   = min(len(env_dest_mask), MAX_DEST)
                dest_mask[0, :valid_dest_length] = env_dest_mask[:valid_dest_length].clone().detach().to(device)

            try:
                with torch.no_grad():
                    actions, _, _, _ = model.act_batch(state_tensor, source_mask, dest_mask, greedy=True)
                action = actions[0]; source_idx = action[0].item(); dest_idx = action[1].item()
            except Exception as e:
                print(f"[오류] 환경 {idx}, 스텝 {step_count}: 모델 추론 실패: {e}")
                final_metric_val = float('inf'); done = True; break

            source_mask_cpu = source_mask[0].cpu(); dest_mask_cpu = dest_mask[0].cpu()
            if not (0 <= source_idx < MAX_SOURCE and source_mask_cpu[source_idx]):
                 valid_indices = torch.where(source_mask_cpu)[0].tolist()
                 if valid_indices:
                     original_idx = source_idx; source_idx = valid_indices[0]
                     # print(f"[경고][클리핑] 환경 {idx}, 스텝 {step_count}: 잘못된 Source({original_idx}) 선택! -> 유효한 {source_idx}로 변경.") # 필요시 주석 해제
                 else:
                     print(f"[오류] 환경 {idx}, 스텝 {step_count}: 유효한 Source 액션 없음! 중단."); final_metric_val = float('inf'); break
            if not (0 <= dest_idx < MAX_DEST and dest_mask_cpu[dest_idx]):
                 valid_indices = torch.where(dest_mask_cpu)[0].tolist()
                 if valid_indices:
                     original_idx = dest_idx; dest_idx = valid_indices[0]
                     # print(f"[경고][클리핑] 환경 {idx}, 스텝 {step_count}: 잘못된 Dest({original_idx}) 선택! -> 유효한 {dest_idx}로 변경.") # 필요시 주석 해제
                 else:
                     print(f"[오류] 환경 {idx}, 스텝 {step_count}: 유효한 Dest 액션 없음! 중단."); final_metric_val = float('inf'); break

            try:
                state_cpu, reward, done, info = env.step((source_idx, dest_idx))
                state = state_cpu.to(device); ep_reward += reward; last_info = info; step_count += 1
            except Exception as e:
                 print(f"[오류] 환경 {idx}, 스텝 {step_count}: env.step() 실행 실패: {e}")
                 final_metric_val = float('inf'); done = True; break

        print(f"[정보] 환경 {idx}: 에피소드 종료됨 (스텝 {step_count}, 종료 플래그: {done})")

        if final_metric_val is None:
            if done and step_count < max_steps:
                 final_metric_val = last_info.get('final_max_move_sum', None)
                 if final_metric_val is None:
                      print(f"[경고] 환경 {idx}: 정상 종료되었으나 'final_max_move_sum' 키 없음. 'inf'로 처리.")
                      final_metric_val = float('inf')
                 else:
                     print(f"[정보] 환경 {idx}: 최종 상태 지표 'final_max_move_sum' = {final_metric_val}")
            elif step_count >= max_steps:
                 print(f"[경고] 환경 {idx}: 최대 스텝({max_steps}) 도달 종료.")
                 try:
                     current_plates_state = getattr(env, 'plates', {}); current_to_keys = getattr(env, 'to_keys', [])
                     # env 에 _get_max_move_for_pile 메서드가 있어야 함
                     final_metric_val = sum(env._get_max_move_for_pile(current_plates_state.get(key, [])) for key in current_to_keys)
                     print(f"          -> 최대 스텝 시점 계산된 max_move_sum = {final_metric_val}")
                 except Exception as e:
                     print(f"          -> 현재 상태 지표 계산 실패: {e}. 'inf'로 처리.")
                     final_metric_val = float('inf')
            else:
                 print(f"[경고] 환경 {idx}: 예상치 못한 이유로 루프 종료. 'inf'로 처리.")
                 final_metric_val = float('inf')

        print(f"  환경 {idx} 평가 결과 => 보상={ep_reward:.2f}, 최종 지표(Final Max Move Sum)={final_metric_val}")
        total_rewards.append(ep_reward)
        total_final_metrics.append(float(final_metric_val))

    avg_reward = np.mean(total_rewards) if total_rewards else 0.0
    avg_final_metric = np.mean(total_final_metrics) if total_final_metrics else float('inf')

    print(f"\n--- 평가 완료 ---")
    print(f"평균 보상         : {avg_reward:.4f}")
    print(f"평균 최종 지표(MaxMoveSum): {avg_final_metric:.4f} (값이 낮을수록 좋음)")

    # 환경 객체 리스트 자체는 반환하지 않음 (결과만 반환)
    return avg_reward, avg_final_metric


# === 메인 실행 블록 ===
if __name__ == "__main__":
    print("="*50)
    print("모델 체크포인트 평가 및 결과 저장 스크립트 시작")
    print("="*50)

    # --- 1. 기본 설정 ---
    cfg = get_cfg()
    set_seed(970517)
    try:
        device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        print(f"사용 장치: {device}")
    except AttributeError:
        print("[오류] cfg 에 'device' 속성이 없습니다.")
        exit()

    # --- 2. 모델 초기화 ---
    model = None
    if 'SteelPlateConditionalMLPModel' in globals():
        try:
            model = SteelPlateConditionalMLPModel(
                embed_dim=cfg.embed_dim,
                num_actor_layers=cfg.num_actor_layers,
                num_critic_layers=cfg.num_critic_layers,
                actor_init_std=cfg.actor_init_std,
                critic_init_std=cfg.critic_init_std,
                max_stack=cfg.max_stack,
                num_from_piles=MAX_SOURCE,
                num_to_piles=MAX_DEST,
                num_heads=cfg.num_heads,
            ).to(device)
            print("모델 객체 초기화 성공.")
        except AttributeError as e:
             print(f"[오류] 모델 초기화 실패. cfg 속성 없음: {e}")
             exit()
        except Exception as e:
            print(f"[오류] 모델 초기화 중 오류: {e}")
            exit()
    else:
        print("[오류] 'SteelPlateConditionalMLPModel' 클래스 없음.")
        exit()

    # --- 3. 체크포인트 불러오기 ---
    checkpoint_path = "best_policy.pth" # <--- 평가할 체크포인트 파일
    print(f"체크포인트 로딩 시도 (직접 지정된 경로): {checkpoint_path}")

    if os.path.exists(checkpoint_path):
        try:
            if model is not None:
                state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
                model.load_state_dict(state_dict)
                print(f"체크포인트 로드 성공: {checkpoint_path}")
                model.eval()
            else:
                print("[오류] 모델 객체 초기화 안됨.")
                exit()
        except RuntimeError as e: # 로딩 실패 시
            print(f"[오류] 체크포인트 로드 실패 (구조/파라미터 불일치): {e}")
            print(">>> 체크포인트 저장 시점의 코드/cfg 와 현재 환경이 동일한지 확인! <<<")
            exit() # 로딩 실패 시 종료
        except Exception as e:
            print(f"[오류] 체크포인트 로드 중 오류 ({checkpoint_path}): {e}")
            exit() # 기타 로딩 오류 시 종료
    else:
        print(f"[오류] 체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}.")
        exit() # 파일 없으면 종료

    # --- 4. 평가용 데이터 준비 ---
    eval_schedule = None
    eval_excel_file = None # 입력 파일 경로 저장용 변수 초기화
    if Plate and generate_schedule:
        try:
            eval_excel_file = cfg.evaluation_plates_data_path # 경로 저장
            print(f"평가용 엑셀 파일 경로: {eval_excel_file}")
            eval_schedule = load_eval_schedule(eval_excel_file)
        except AttributeError:
             print("[오류] cfg 에 'evaluation_plates_data_path' 속성이 없습니다.")
             eval_excel_file = None

        if eval_schedule is None:
            print("평가용 엑셀 스케줄 로드 실패 또는 경로 미지정. 기본 스케줄 생성 시도...")
            try:
                num_default_plates = cfg.num_plates
                eval_schedule = generate_schedule(num_plates=num_default_plates)
                # 기본 스케줄 from/to 설정 (예시)
                unique_from_piles = [f"S_{i}" for i in range(MAX_SOURCE)]
                unique_to_piles = [f"D_{i}" for i in range(MAX_DEST)]
                for i, p in enumerate(eval_schedule):
                     p.from_pile = p.id if hasattr(p,'id') else f"plate_{i}"
                     p.topile = random.choice(unique_to_piles)
                print(f"기본 평가 스케줄 ({len(eval_schedule)}개 플레이트) 생성 완료.")
                eval_excel_file = "default_schedule" # 기본 스케줄 사용 시 이름 지정
            except AttributeError as e:
                print(f"[오류] 기본 스케줄 생성 실패. cfg 속성 없음: {e}")
                eval_schedule = None
            except Exception as e:
                print(f"[오류] 기본 스케줄 생성 중 오류: {e}")
                eval_schedule = None
    else:
        print("[경고] 'Plate'/'generate_schedule' 사용 불가.")

    if eval_schedule is None:
        print("[오류] 사용 가능한 평가 스케줄 없음. 종료.")
        exit()

    # --- 5. 평가 환경 구성 ---
    num_eval_envs = 1
    try:
        # cfg 에서 num_evaluation_environments 읽기 시도
        num_eval_envs = cfg.num_evaluation_environments
        print(f"{num_eval_envs}개의 평가 환경 생성을 시도합니다...")
    except AttributeError:
        # 없으면 기본값 1 사용 및 메시지 출력
        print("[정보] cfg 에 'num_evaluation_environments' 속성 없음. 기본값 1 사용.")
        print("       (필요시 cfg.py 에 '--num_evaluation_environments' 인자를 추가하세요.)")

    eval_envs = [] # 평가 환경 리스트
    if Locating:
        for i in range(num_eval_envs):
            print(f"  환경 {i} 생성 중...")
            try:
                # env.py Locating 클래스 생성자 확인 필수
                env = Locating(
                    max_stack=cfg.max_stack,
                    inbound_plates=eval_schedule, # 평가 데이터 전달
                    # device=device, # Locating 내부에서 device 사용 안하면 불필요
                    crane_penalty=cfg.crane_penalty,
                    # num_pile=cfg.num_pile, # Locating 이 인자를 받는 경우
                )
                eval_envs.append(env)
                print(f"  환경 {i} 생성 완료.")
            except AttributeError as e:
                 print(f"[오류] 환경 {i} 생성 실패. cfg 속성 없음: {e}")
            except TypeError as e:
                print(f"[오류] 환경 {i} 생성 실패 (Locating 생성자 인자 불일치): {e}")
            except Exception as e:
                print(f"[오류] 환경 {i} 생성 중 오류: {e}")

        if not eval_envs:
            print("[오류] 평가 환경 생성 실패. 종료.")
            exit()
        print(f"총 {len(eval_envs)}개의 평가 환경 생성 완료.")
    else:
        print("[오류] 'Locating' 클래스 사용 불가.")
        exit()

    # --- 6. 정책 평가 실행 ---
    output_filepath = None # 출력 파일 경로 변수 초기화
    if model is not None and eval_envs:
        print("\n정책 평가를 시작합니다...")
        avg_reward, avg_final_metric = evaluate_policy(model, eval_envs, device)

        # --- 6.5. 최종 상태 Excel 저장 ---
        print("\n--- 최종 상태 Excel 파일로 저장 시도 ---")

        # 출력 파일명 생성
        base_input_name = "default"
        if eval_excel_file and eval_excel_file != "default_schedule":
            base_input_name = os.path.splitext(os.path.basename(eval_excel_file))[0]
        base_checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        output_filename = f"result_{base_input_name}_{base_checkpoint_name}.xlsx"

        # 출력 디렉토리 설정
        output_dir = getattr(cfg, 'output_dir', 'output') # cfg.output_dir 사용, 없으면 'output'
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
                print(f"출력 디렉토리 생성: {output_dir}")
            except OSError as e:
                print(f"[오류] 출력 디렉토리 생성 실패: {e}. 현재 디렉토리에 저장 시도.")
                output_dir = "."
        output_filepath = os.path.join(output_dir, output_filename) # 최종 저장 경로

        # 평가 완료된 환경 객체 가져오기 (보통 첫번째 환경)
        final_env = eval_envs[0]

        # 환경 객체에 export_final_state_to_excel 메서드가 있는지 확인하고 호출
        if hasattr(final_env, 'export_final_state_to_excel') and callable(getattr(final_env, 'export_final_state_to_excel')):
            try:
                print(f"최종 상태 저장 중: {output_filepath}")
                final_env.export_final_state_to_excel(output_filepath)
                print(f"최종 상태 저장 완료: {output_filepath}")
            except Exception as e:
                print(f"[오류] 최종 상태 Excel 저장 실패: {e}")
                output_filepath = None
        else:
            print("[오류] 환경 객체에 'export_final_state_to_excel' 메서드가 없습니다.")
            print(">>> env.py 파일의 Locating 클래스 내부에 해당 메서드를 추가해주세요. <<<")
            output_filepath = None # 메서드 없으면 경로 None 처리

        # --- 7. 최종 결과 출력 ---
        print("\n" + "="*50)
        print("      최종 평가 결과 요약")
        print("="*50)
        print(f"평가에 사용된 모델: {checkpoint_path}")
        print(f"평가 데이터       : {eval_excel_file if eval_excel_file else '기본 생성 스케줄'}") # 평가 데이터 출처 명시
        print(f"평균 에피소드 보상  : {avg_reward:.4f}")
        print(f"평균 최종 상태 지표 : {avg_final_metric:.4f} (MaxMoveSum: 낮을수록 좋음)")
        # 저장 성공 시 파일 경로 출력, 실패 또는 해당 없으면 메시지 출력
        print(f"결과 저장 파일    : {output_filepath if output_filepath and os.path.exists(output_filepath) else '저장 실패 또는 해당 없음'}")
        print("="*50)
        print("평가 스크립트 종료.")
    else:
        print("[오류] 모델 또는 평가 환경 준비 안됨. 평가 불가.")
