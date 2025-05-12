import random
import numpy as np
import torch
import os
import pandas as pd
import copy
import math
from cfg import get_cfg
import data as plate
from data import Plate
from network import MAX_SOURCE, MAX_DEST

# === 키 정규화 함수 ===
def normalize_keys(schedule):
    """스케줄 내 파일(Pile) 키를 정규화하고 정규화된 키 목록 반환"""
    if not schedule: return [], [], []

    original_from_keys_set = set()
    original_to_keys_set = set()
    valid_schedule_input = []

    # 유효한 Plate 객체 필터링 및 원본 키 추출
    for p in schedule:
        if isinstance(p, Plate) and \
           hasattr(p, 'from_pile') and p.from_pile is not None and \
           hasattr(p, 'topile') and p.topile is not None and \
           hasattr(p, 'outbound') and isinstance(getattr(p, 'outbound'), (int, float)):
             valid_schedule_input.append(p)
             original_from_keys_set.add(str(p.from_pile).strip())
             original_to_keys_set.add(str(p.topile).strip())

    if not valid_schedule_input or not original_from_keys_set or not original_to_keys_set:
        print("[Warning] normalize_keys: 유효 스케줄 또는 파일 부족.")
        return [], [], []

    # 원본 키 목록 정렬 및 최대 개수 제한
    unique_original_from_list = sorted(list(original_from_keys_set))[:MAX_SOURCE]
    unique_original_to_list = sorted(list(original_to_keys_set))[:MAX_DEST]

    if not unique_original_from_list or not unique_original_to_list:
        print("[Warning] normalize_keys: MAX 제한 후 유효 파일 없음.")
        return [], [], []

    # 원본 키 -> 정규화 키 매핑 생성
    from_key_map = {key: f"from_{i:02d}" for i, key in enumerate(unique_original_from_list)}
    to_key_map = {key: f"to_{i:02d}" for i, key in enumerate(unique_original_to_list)}

    normalized_schedule_final = []
    default_norm_to_key = list(to_key_map.values())[0] # 기본 정규화된 도착 키

    # 스케줄 내 Plate 객체들의 키 속성 정규화 (덮어쓰기)
    for p in valid_schedule_input:
        original_from = str(p.from_pile).strip()
        original_to = str(p.topile).strip()
        norm_from = from_key_map.get(original_from)
        norm_to = to_key_map.get(original_to)

        # 유효한 출발 파일 키가 있는 경우만 처리
        if norm_from is not None:
            normalized_p = copy.copy(p) # 객체 복사
            setattr(normalized_p, 'from_pile', norm_from) # 정규화된 키로 덮어쓰기
            setattr(normalized_p, 'topile', norm_to if norm_to is not None else default_norm_to_key) # 정규화된 키로 덮어쓰기
            normalized_schedule_final.append(normalized_p)

    # 최종 사용될 정규화된 키 목록 반환
    final_from_keys = list(from_key_map.values())
    final_to_keys = list(to_key_map.values())

    return normalized_schedule_final, final_from_keys, final_to_keys

# --- Locating Environment Class ---
class Locating(object):
    """
    강판 재배치 RL 환경 클래스
    - 파일 키 정규화 사용
    - Potential-Based Reward Shaping 적용 (블로킹 최소화 목표)
    - 최종 상태 엑셀 출력 시 원본 파일 키 사용
    """
    def __init__(self, max_stack=30,
                 inbound_plates=None,
                 observe_inbounds=False, # 현재 로직에서 사용 안 됨
                 device="cuda", # 현재 로직에서 사용 안 됨
                 crane_penalty=0.0):

        cfg = get_cfg()
        self.max_stack = int(max_stack)
        self.OBSERVED_TOP_N_PLATES = getattr(cfg, 'OBSERVED_TOP_N_PLATES', 30)
        self.NUM_SUMMARY_STATS_DEEPER = getattr(cfg, 'NUM_SUMMARY_STATS_DEEPER', 4)
        self.NUM_PILE_TYPE_FEATURES = 1
        self.NUM_BLOCKING_FEATURES = 1

        self.actual_pile_feature_dim = self.OBSERVED_TOP_N_PLATES + \
                                       self.NUM_SUMMARY_STATS_DEEPER + \
                                       self.NUM_PILE_TYPE_FEATURES + \
                                       self.NUM_BLOCKING_FEATURES

        self.stage = 0 # 현재까지 이동한 횟수
        self.current_date = 0 # 사용되지 않음 (필요 시 로직 추가)
        self.crane_move = 0 # 크레인 이동 횟수 (단순 카운트)
        self.plates = {}    # 현재 파일 상태 (key: 정규화된 키, value: Plate 리스트)
        self.crane_penalty = float(crane_penalty)

        schedule_to_process = None # 원본 스케줄 리스트 (정규화 전)

        # === 1. 스케줄 준비 (외부 입력 > 파일 로드 > 기본 생성) ===
        if inbound_plates:
            schedule_to_process = copy.deepcopy(inbound_plates)
            # print("[정보] 외부 스케줄로 초기화.") # 정보 메시지 제거
        else:
            try:
                plates_data_path = getattr(cfg, 'plates_data_path', 'output/reshuffle_plan.xlsx')
                # print(f"[정보] 초기 스케줄 로드 시도: {plates_data_path}") # 정보 메시지 제거
                df = pd.read_excel(plates_data_path, sheet_name="reshuffle")
                loaded_schedule = []
                required_cols = ["pileno", "inbound", "outbound", "unitw", "topile"]
                if not all(col in df.columns for col in required_cols):
                    print(f"[경고] Excel 파일({plates_data_path}) 필수 컬럼 부족: {required_cols}") # 경고 유지

                # 기본값 설정을 위한 cfg 값 (없으면 기본값 사용)
                cfg_inbound_min = getattr(cfg, 'inbound_min', 1); cfg_inbound_max = getattr(cfg, 'inbound_max', 10)
                cfg_outbound_extra_min = getattr(cfg, 'outbound_extra_min', 1); cfg_outbound_extra_max = getattr(cfg, 'outbound_extra_max', 10)
                cfg_unitw_min = getattr(cfg, 'unitw_min', 0.1); cfg_unitw_max = getattr(cfg, 'unitw_max', 10.0)

                for idx, row in df.iterrows():
                    plate_id = row.get("pileno", f"Plate_{idx}")
                    inbound = int(row.get("inbound", random.randint(cfg_inbound_min, cfg_inbound_max)))
                    outbound = int(row.get("outbound", inbound + random.randint(cfg_outbound_extra_min, cfg_outbound_extra_max)))
                    unitw = float(row.get("unitw", random.uniform(cfg_unitw_min, cfg_unitw_max)))
                    to_pile = str(row.get("topile", f"D_{idx % MAX_DEST}")).strip()
                    from_pile_val = str(row.get("pileno", f"S_{idx % MAX_SOURCE}")).strip()

                    p = Plate(id=plate_id, inbound=inbound, outbound=outbound, unitw=unitw)
                    p.from_pile = from_pile_val
                    p.topile = to_pile
                    loaded_schedule.append(p)

                if not loaded_schedule: raise ValueError("Excel 스케줄 비어있음.")
                schedule_to_process = loaded_schedule
                # print(f"[정보] 초기 스케줄 로드 완료: {plates_data_path} ({len(schedule_to_process)}개)") # 정보 메시지 제거
            except FileNotFoundError:
                # print(f"[경고] Excel 파일({plates_data_path}) 없음. 기본 스케줄 생성.") # 정보 메시지 제거
                num_plates_default = getattr(cfg, 'num_plates', 50)
                schedule_to_process = plate.generate_schedule(num_plates=num_plates_default)
                for i, p in enumerate(schedule_to_process):
                    if not hasattr(p, 'from_pile'): p.from_pile = f"S_{i % MAX_SOURCE}"
                    if not hasattr(p, 'topile'): p.topile = f"D_{i % MAX_DEST}"
                # print(f"[정보] 기본 스케줄 생성 완료 ({len(schedule_to_process)}개).") # 정보 메시지 제거
            except Exception as e:
                print(f"[오류] 스케줄 로드/생성 중 오류: {e}") # 오류 메시지 유지
                raise ValueError("스케줄 초기화 실패")

        if not schedule_to_process:
            raise ValueError("처리할 스케줄 없음.") # 오류 메시지 유지

        # === 2. 원본 목표 파일 정보 저장 (`original_intended_topile` 속성 추가) ===
        for p in schedule_to_process:
            original_topile_val = None
            if isinstance(p, Plate) and hasattr(p, 'topile') and p.topile is not None:
                original_topile_val = str(p.topile).strip()
            setattr(p, 'original_intended_topile', original_topile_val)

        # === 3. 키 정규화 수행 ===
        normalized_schedule, self.from_keys, self.to_keys = normalize_keys(schedule_to_process)
        self.inbound_plates = normalized_schedule # 정규화된 스케줄 (Plate 객체 리스트)
        self.inbound_clone = copy.deepcopy(self.inbound_plates) # 리셋용 복사본

        if not self.from_keys or not self.to_keys:
             raise ValueError("키 정규화 후 유효 Source/Dest 파일 없음.") # 오류 메시지 유지

        # === 4. 역매핑 정보 생성 (`self.norm_to_orig_map`) ===
        self.norm_to_orig_map = {} # 초기화
        try:
            # 원본 스케줄에서 사용된 실제 키 목록 추출 (정렬 및 개수 제한)
            original_from_keys_used = sorted(list(set(str(p.from_pile).strip() for p in schedule_to_process if hasattr(p, 'from_pile') and p.from_pile is not None)))[:MAX_SOURCE]
            original_to_keys_used = sorted(list(set(p.original_intended_topile for p in schedule_to_process if hasattr(p, 'original_intended_topile') and p.original_intended_topile is not None)))[:MAX_DEST]

            # 정규화된 키와 원본 키 매칭 (개수 일치 확인)
            if len(self.from_keys) == len(original_from_keys_used):
                norm_to_orig_from_map = {norm_key: orig_key for norm_key, orig_key in zip(self.from_keys, original_from_keys_used)}
            else:
                print(f"[Warning] __init__: From 키 개수 불일치 ({len(self.from_keys)} vs {len(original_from_keys_used)}).") # 경고 유지
                norm_to_orig_from_map = {}
            if len(self.to_keys) == len(original_to_keys_used):
                norm_to_orig_to_map = {norm_key: orig_key for norm_key, orig_key in zip(self.to_keys, original_to_keys_used)}
            else:
                print(f"[Warning] __init__: To 키 개수 불일치 ({len(self.to_keys)} vs {len(original_to_keys_used)}).") # 경고 유지
                norm_to_orig_to_map = {}
            self.norm_to_orig_map = {**norm_to_orig_from_map, **norm_to_orig_to_map}

            if not self.norm_to_orig_map: print("[Warning] __init__: 역매핑 정보 비어있음.") # 경고 유지
            # else: print("[정보] 정규화 키 -> 원본 키 역매핑 생성 완료.") # --- 출력 제거 ---

        except Exception as e:
            print(f"[오류] __init__: 역매핑 생성 오류: {e}") # 오류 유지
            print("[경고] 역매핑 정보 생성 실패.") # 경고 유지
            self.norm_to_orig_map = {}
        # --- 역매핑 생성 끝 ---

        # === 5. 나머지 멤버 변수 설정 ===
        self.all_pile_keys = sorted(list(set(self.from_keys + self.to_keys)))
        self.source_index_to_key = {i: key for i, key in enumerate(self.from_keys)}
        self.source_key_to_index = {key: i for i, key in enumerate(self.from_keys)}
        self.dest_index_to_key = {i: key for i, key in enumerate(self.to_keys)}
        self.dest_key_to_index = {key: i for i, key in enumerate(self.to_keys)}
        self.plates = {} # 플레이트 상태 (reset에서 초기화)
        # print(f"[정보] 환경 초기화 완료: Source {len(self.from_keys)}({MAX_SOURCE}), Dest {len(self.to_keys)}({MAX_DEST}). Max Stack {self.max_stack}.") # --- 출력 제거 ---

    def reset(self, shuffle_schedule=False):
        """환경을 초기 상태로 리셋합니다."""
        schedule = copy.deepcopy(self.inbound_clone) # 정규화된 스케줄 복사본 사용
        if not schedule:
             print("[Warning] Reset: 스케줄 없음.") # 경고 유지
             self.plates = {key: [] for key in self.from_keys + self.to_keys}
             self.current_date = 0; self.crane_move = 0; self.stage = 0
             self.total_plate_count = 0; self.move_data = []
             return self._get_state()

        if shuffle_schedule: random.shuffle(schedule)

        # self.plates 딕셔너리 초기화 (모든 from/to 키 포함)
        self.plates = {key: [] for key in self.from_keys + self.to_keys}
        actual_plate_count = 0
        # 정규화된 스케줄의 플레이트를 정규화된 from_pile 키에 따라 분배
        for p in schedule:
            if hasattr(p, 'from_pile') and p.from_pile in self.from_keys:
                 self.plates[p.from_pile].append(p)
                 actual_plate_count += 1

        # 현재 날짜 설정 (사용되지 않음)
        valid_inbound_plates = [p for p in schedule if hasattr(p, 'inbound') and isinstance(p.inbound, (int, float))]
        self.current_date = min([p.inbound for p in valid_inbound_plates]) if valid_inbound_plates else 0

        # 상태 변수 초기화
        self.crane_move = 0
        self.stage = 0
        self.total_plate_count = actual_plate_count
        if self.total_plate_count == 0: print("[Warning] Reset 후 이동할 플레이트 없음.") # 경고 유지
        self.move_data = []
        return self._get_state() # 초기 상태 반환

    def _get_max_move_for_pile(self, pile):
        """주어진 파일 내 최대 블로킹 수 계산 (목표: 블로킹 최소화)"""
        n_pile = len(pile)
        if n_pile <= 1: return 0

        max_move = 0
        for i in range(n_pile - 1): # i: 아래 판 인덱스
            plate_i = pile[i]
            if not (hasattr(plate_i, 'outbound') and isinstance(plate_i.outbound, (int, float))): continue
            outbound_i = plate_i.outbound

            move = 0
            for j in range(i + 1, n_pile): # j: 위 판 인덱스
                plate_j = pile[j]
                if not (hasattr(plate_j, 'outbound') and isinstance(plate_j.outbound, (int, float))): continue
                outbound_j = plate_j.outbound

                # --- 핵심 로직: 블로킹 카운트 ---
                # 아래 판(i)이 위 판(j)보다 나중에 나가야 할 경우(outbound_i < outbound_j)
                if outbound_i < outbound_j:
                    move += 1

            max_move = max(max_move, move)

        return max_move

    def _get_total_blocking_pairs(self, pile):
        """주어진 파일 내 총 블로킹 쌍(pair) 개수 계산"""
        n_pile = len(pile)
        if n_pile <= 1: return 0
        total_blocking_pairs = 0
        for i in range(n_pile - 1): # 아래 판 인덱스 i
            plate_i = pile[i]
            if not (hasattr(plate_i, 'outbound') and isinstance(getattr(plate_i, 'outbound', None), (int, float))): continue
            outbound_i = plate_i.outbound

            for j in range(i + 1, n_pile): # 위 판 인덱스 j
                plate_j = pile[j]
                if not (hasattr(plate_j, 'outbound') and isinstance(getattr(plate_j, 'outbound', None), (int, float))): continue
                outbound_j = plate_j.outbound

                if outbound_i < outbound_j: # 블로킹 조건
                    total_blocking_pairs += 1

        return total_blocking_pairs

    def step(self, action):
        """환경 스텝 함수"""
        s_next, reward, done, info = self._composite_step(action)
        return s_next, reward, done, info

    def _composite_step(self, action):
        """행동 실행 및 결과 반환 (Potential-Based Reward Shaping)"""
        valid_source_mask, valid_dest_mask = self.get_masks()
        cfg = get_cfg()
        gamma = getattr(cfg, "gamma", 0.99)
        shaping_reward_scale = getattr(cfg, "shaping_reward_scale", 1.0)

        # 1. 행동 전 포텐셜 계산 (블로킹 지표 사용)
        potential_before = -sum(self._get_total_blocking_pairs(self.plates.get(key, [])) for key in self.to_keys)

        from_index, to_index = action

        # 2. 액션 유효성 검사 및 필요시 랜덤 유효 액션으로 대체
        is_action_invalid = False
        if not (0 <= from_index < len(self.from_keys)) or not valid_source_mask[from_index]: is_action_invalid = True
        if not (0 <= to_index < len(self.to_keys)) or not valid_dest_mask[to_index]: is_action_invalid = True

        if is_action_invalid:
            valid_source_indices = torch.where(valid_source_mask[:len(self.from_keys)])[0].tolist()
            valid_dest_indices = torch.where(valid_dest_mask[:len(self.to_keys)])[0].tolist()
            if not valid_source_indices or not valid_dest_indices:
                print(f"[오류] Step {self.stage}: 대체 유효 행동 없음. 종료.") # 오류 유지
                return self._get_state(), 0.0, True, {"error": "No valid moves available"}
            from_index = random.choice(valid_source_indices)
            to_index = random.choice(valid_dest_indices)

        source_key = self.from_keys[from_index] # 정규화된 키
        destination_key = self.to_keys[to_index] # 정규화된 키

        # 3. 이동 실행 전 상태 확인 (오류 방지)
        current_dest_pile = self.plates.get(destination_key, [])
        if len(current_dest_pile) >= self.max_stack:
             print(f"[오류] Step {self.stage}: 목적지 '{destination_key}' Full!") # 오류 유지
             return self._get_state(), 0.0, True, {"error": f"Destination '{destination_key}' full"}
        current_source_pile = self.plates.get(source_key)
        if not current_source_pile:
             print(f"[오류] Step {self.stage}: 출발지 '{source_key}' Empty!") # 오류 유지
             return self._get_state(), 0.0, True, {"error": f"Source '{source_key}' empty"}

        # 4. 플레이트 이동 실행
        try:
             moved_plate = current_source_pile.pop()
             self.plates[destination_key].append(moved_plate)
        except IndexError:
              print(f"[심각] Step {self.stage}: 비어있는 '{source_key}'에서 pop 시도!") # 오류 유지
              return self._get_state(), 0.0, True, {"error": "Pop from empty pile"}

        # 5. 행동 후 포텐셜 계산 (총 블로킹 쌍 기준)
        potential_after = -sum(self._get_total_blocking_pairs(self.plates.get(key, [])) for key in self.to_keys)

        # 6. 상태 업데이트
        self.move_data.append((source_key, destination_key))
        self.crane_move += 1
        self.stage += 1

        # 7. 종료 조건 확인
        done = self.stage >= self.total_plate_count
        info = {}
        if self.total_plate_count <= 0: done = True # 예외 처리

        # 8. 다음 상태 가져오기
        next_state = self._get_state() # CPU Tensor

        # --- 보상 계산 (PBRS + Terminal Reward) ---
        shaping_reward = (cfg.gamma * potential_after - potential_before) * cfg.shaping_reward_scale
        terminal_reward = 0.0

        if done:
            # 최종 블로킹 지표 계산 (총 블로킹 쌍 기준)
            final_blocking_metric = sum(self._get_total_blocking_pairs(self.plates.get(key, [])) for key in self.to_keys)
            info['final_max_move_sum'] = final_blocking_metric
            info['final_crane_move'] = self.crane_move

            # 터미널 보상 계산
            terminal_reward = 1.0 / (final_blocking_metric+1)

        # 최종 보상 = PBRS + 터미널 보상(done일 때만)
        total_reward = shaping_reward + terminal_reward

        return next_state, total_reward, done, info

    def _get_state(self):
        padding_feature = [0.0] * self.actual_pile_feature_dim
        state_features = []

        # Source Piles 특징 추출
        for i in range(MAX_SOURCE):  # network.py에서 import한 MAX_SOURCE 사용
            feature_vector_list = list(padding_feature)
            if i < len(self.from_keys):
                key = self.from_keys[i]
                pile = self.plates.get(key, [])
                if pile:
                    n_pile = len(pile)
                    for feature_idx in range(self.OBSERVED_TOP_N_PLATES):
                        plate_actual_idx_in_pile = n_pile - 1 - feature_idx
                        if plate_actual_idx_in_pile >= 0:
                            p_obj = pile[plate_actual_idx_in_pile]
                            ob_val = getattr(p_obj, 'outbound', 0.0)
                            feature_vector_list[feature_idx] = float(ob_val) if isinstance(ob_val,
                                                                                           (int, float)) else 0.0
                        else:
                            feature_vector_list[feature_idx] = 0.0

                    deeper_plates_end_idx_exclusive = max(0, n_pile - self.OBSERVED_TOP_N_PLATES)
                    deeper_plates = pile[0:deeper_plates_end_idx_exclusive]
                    summary_stats_start_idx = self.OBSERVED_TOP_N_PLATES
                    if deeper_plates:
                        deeper_outbounds = []
                        for p_deep in deeper_plates:
                            ob_deep = getattr(p_deep, 'outbound', 0.0)
                            if isinstance(ob_deep, (int, float)):
                                deeper_outbounds.append(float(ob_deep))

                        if deeper_outbounds:
                            feature_vector_list[summary_stats_start_idx + 0] = float(len(deeper_plates))
                            feature_vector_list[summary_stats_start_idx + 1] = min(deeper_outbounds)
                            feature_vector_list[summary_stats_start_idx + 2] = max(deeper_outbounds)
                            feature_vector_list[summary_stats_start_idx + 3] = sum(deeper_outbounds) / len(
                                deeper_outbounds)
                        elif self.NUM_SUMMARY_STATS_DEEPER > 0:  # count_deeper 만이라도 채움
                            feature_vector_list[summary_stats_start_idx + 0] = float(len(deeper_plates))

                    pile_type_idx = self.OBSERVED_TOP_N_PLATES + self.NUM_SUMMARY_STATS_DEEPER
                    feature_vector_list[pile_type_idx] = 1.0
                    blocking_count_idx = pile_type_idx + self.NUM_PILE_TYPE_FEATURES
                    feature_vector_list[blocking_count_idx] = float(self._get_total_blocking_pairs(pile))
            state_features.append(feature_vector_list)

        # Destination Piles 특징 추출 (Source와 동일 로직)
        for j in range(MAX_DEST):  # network.py에서 import한 MAX_DEST 사용
            feature_vector_list = list(padding_feature)
            if j < len(self.to_keys):
                key = self.to_keys[j]
                pile = self.plates.get(key, [])
                if pile:
                    n_pile = len(pile)
                    for feature_idx in range(self.OBSERVED_TOP_N_PLATES):
                        plate_actual_idx_in_pile = n_pile - 1 - feature_idx
                        if plate_actual_idx_in_pile >= 0:
                            p_obj = pile[plate_actual_idx_in_pile]
                            ob_val = getattr(p_obj, 'outbound', 0.0)
                            feature_vector_list[feature_idx] = float(ob_val) if isinstance(ob_val,
                                                                                           (int, float)) else 0.0
                        else:
                            feature_vector_list[feature_idx] = 0.0

                    deeper_plates_end_idx_exclusive = max(0, n_pile - self.OBSERVED_TOP_N_PLATES)
                    deeper_plates = pile[0:deeper_plates_end_idx_exclusive]
                    summary_stats_start_idx = self.OBSERVED_TOP_N_PLATES
                    if deeper_plates:
                        deeper_outbounds = []
                        for p_deep in deeper_plates:
                            ob_deep = getattr(p_deep, 'outbound', 0.0)
                            if isinstance(ob_deep, (int, float)):
                                deeper_outbounds.append(float(ob_deep))

                        if deeper_outbounds:
                            feature_vector_list[summary_stats_start_idx + 0] = float(len(deeper_plates))
                            feature_vector_list[summary_stats_start_idx + 1] = min(deeper_outbounds)
                            feature_vector_list[summary_stats_start_idx + 2] = max(deeper_outbounds)
                            feature_vector_list[summary_stats_start_idx + 3] = sum(deeper_outbounds) / len(
                                deeper_outbounds)
                        elif self.NUM_SUMMARY_STATS_DEEPER > 0:
                            feature_vector_list[summary_stats_start_idx + 0] = float(len(deeper_plates))

                    pile_type_idx = self.OBSERVED_TOP_N_PLATES + self.NUM_SUMMARY_STATS_DEEPER
                    feature_vector_list[pile_type_idx] = 2.0
                    blocking_count_idx = pile_type_idx + self.NUM_PILE_TYPE_FEATURES
                    feature_vector_list[blocking_count_idx] = float(self._get_total_blocking_pairs(pile))
            state_features.append(feature_vector_list)

        try:
            state_tensor = torch.tensor(state_features, dtype=torch.float)
            expected_shape = (MAX_SOURCE + MAX_DEST, self.actual_pile_feature_dim)
            if state_tensor.shape != expected_shape:
                if state_tensor.shape[1] != self.actual_pile_feature_dim:
                    print(
                        f"[ERROR] _get_state: Feature dimension mismatch! Got {state_tensor.shape[1]}, expected {self.actual_pile_feature_dim}.")
        except Exception as e:
            print(f"[ERROR] _get_state: Failed to create state tensor: {e}. Returning zeros.")
            expected_shape = (MAX_SOURCE + MAX_DEST, self.actual_pile_feature_dim)
            state_tensor = torch.zeros(expected_shape, dtype=torch.float)
        return state_tensor

    def export_final_state_to_excel(self, output_filepath):
        """최종 파일 배치 상태를 Excel 파일로 저장합니다. (원본 파일 이름 사용)"""
        rows = []
        all_normalized_keys = sorted(list(self.plates.keys())) # 정규화된 키

        # 역매핑 정보 가져오기 (없으면 빈 dict)
        norm_map = getattr(self, 'norm_to_orig_map', {})
        if not norm_map: print("[Warning] export: 역매핑 정보 없음.") # 경고 유지

        for normalized_key in all_normalized_keys:
            pile = self.plates.get(normalized_key, [])
            # 정규화된 키 -> 원본 파일 이름 변환
            original_pile_name = norm_map.get(normalized_key, normalized_key) # fallback 포함

            for depth_idx, plate_obj in enumerate(pile):
                if isinstance(plate_obj, Plate):
                    # 저장된 원본 목표 파일 이름 가져오기
                    original_intended_topile_name = getattr(plate_obj, 'original_intended_topile', None)
                    row = {
                        "pileno": getattr(plate_obj, 'id', f'Unknown_{original_pile_name}_{depth_idx}'),
                        "inbound": getattr(plate_obj, 'inbound', None),
                        "outbound": getattr(plate_obj, 'outbound', None),
                        "unitw": getattr(plate_obj, 'unitw', None),
                        "final_pile": original_pile_name, # 원본 이름 사용
                        "depth": depth_idx,
                        "original_topile": original_intended_topile_name # 원본 이름 사용
                    }
                    rows.append(row)
                # else: print(f"[Warning] export: Non-Plate object in pile '{original_pile_name}'") # 필요시 활성화

        if not rows:
            print("[Warning] 내보낼 플레이트 정보 없음.") # 경고 유지
            return

        df = pd.DataFrame(rows)
        try:
            output_dir = os.path.dirname(output_filepath)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                # print(f"[정보] 출력 디렉토리 생성: {output_dir}") # 정보 메시지 제거
            with pd.ExcelWriter(output_filepath, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="final_arrangement", index=False)
            # print(f"[정보] 최종 상태 저장 완료: {output_filepath}") # 정보 메시지 제거
        except Exception as e:
            print(f"[오류] 최종 상태 Excel 내보내기 오류: {e}") # 오류 메시지 유지


    def get_masks(self):
        """현재 상태에서 유효한 출발/도착 파일 마스크를 생성합니다."""
        source_flags = [False] * MAX_SOURCE
        for i in range(min(len(self.from_keys), MAX_SOURCE)):
            key = self.from_keys[i] # 정규화된 키
            # .get() 결과가 truthy 인지 확인 (빈 리스트는 False)
            if self.plates.get(key): source_flags[i] = True
        source_mask = torch.tensor(source_flags, dtype=torch.bool)

        dest_flags = [False] * MAX_DEST
        for j in range(min(len(self.to_keys), MAX_DEST)):
            key = self.to_keys[j] # 정규화된 키
            # 파일 길이가 max_stack 미만이면 True
            if len(self.plates.get(key, [])) < self.max_stack:
                dest_flags[j] = True
        dest_mask = torch.tensor(dest_flags, dtype=torch.bool)

        return source_mask, dest_mask