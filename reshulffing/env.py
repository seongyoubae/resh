import torch
import random
import copy
import math
import os
import pandas as pd

from cfg import get_cfg
from data import Plate, generate_schedule
from network import MAX_SOURCE, MAX_DEST # MAX_SOURCE, MAX_DEST는 network.py에 정의되어 있어야 합니다.

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
    # to_key_map이 비어있지 않은 경우에만 default_norm_to_key를 설정
    default_norm_to_key = list(to_key_map.values())[0] if to_key_map else "to_00"

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

## Locating Environment Class (with get_masks restored)

class Locating(object):
    """
    강판 재배치 RL 환경 클래스
    - [수정] 글로벌 특징(전체 문제 규모)을 포함한 상태 표현으로 일반화 성능 강화
    - get_masks 함수 복원
    """

    def __init__(self, max_stack=30, inbound_plates=None, crane_penalty=0.0, device="cpu"):
        cfg = get_cfg()
        self.max_stack = int(max_stack)

        # --- 특징 차원 정의 ---
        self.OBSERVED_TOP_N_PLATES = getattr(cfg, 'OBSERVED_TOP_N_PLATES', 10)
        self.NUM_SUMMARY_STATS_DEEPER = getattr(cfg, 'NUM_SUMMARY_STATS_DEEPER', 4)
        self.NUM_PILE_TYPE_FEATURES = 1
        self.NUM_BLOCKING_FEATURES = 1

        # [수정된 부분] 글로벌 특징의 개수를 2개로 정의합니다. (출발 파일 수, 도착 파일 수)
        self.NUM_GLOBAL_FEATURES = 2

        # [수정된 부분] 전체 특징 차원 계산식에 글로벌 특징 개수를 반영합니다.
        self.actual_pile_feature_dim = (self.OBSERVED_TOP_N_PLATES +
                                        self.NUM_SUMMARY_STATS_DEEPER +
                                        self.NUM_PILE_TYPE_FEATURES +
                                        self.NUM_BLOCKING_FEATURES +
                                        self.NUM_GLOBAL_FEATURES)

        # --- 스케줄 로드 및 기타 초기화 ---
        self.crane_penalty = float(crane_penalty)
        self.device = device
        self.plates = {} # 초기화 시 빈 딕셔너리로 설정
        self.stage = 0
        self.crane_move = 0
        self.total_plate_count = 0
        self.current_date = 0 # 현재 로직에서 사용되지 않음
        self.move_data = [] # move_data 초기화 (step 함수에서 사용되지 않더라도 초기화)


        schedule_to_process = self._load_or_generate_schedule(inbound_plates, cfg)

        # 원본 목표 파일 정보 저장 (`original_intended_topile` 속성 추가)
        for p in schedule_to_process:
            original_topile_val = None
            if isinstance(p, Plate) and hasattr(p, 'topile') and p.topile is not None:
                original_topile_val = str(p.topile).strip()
            setattr(p, 'original_intended_topile', original_topile_val)

        # 키 정규화 수행
        self.inbound_plates, self.from_keys, self.to_keys = normalize_keys(schedule_to_process)
        self.inbound_clone = copy.deepcopy(self.inbound_plates) # 리셋용 복사본

        if not self.from_keys or not self.to_keys:
            raise ValueError("키 정규화 후 유효 Source/Dest 파일 없음.")

        # 역매핑 정보 생성 (`self.norm_to_orig_map`)
        self.norm_to_orig_map = self._create_norm_to_orig_map(schedule_to_process)

        # 나머지 멤버 변수 설정
        self.all_pile_keys = sorted(list(set(self.from_keys + self.to_keys)))
        self.source_key_to_index = {key: i for i, key in enumerate(self.from_keys)}
        self.dest_key_to_index = {key: i for i, key in enumerate(self.to_keys)}
        # self.source_index_to_key = {i: key for i, key in enumerate(self.from_keys)} # 현재 사용되지 않아 주석처리
        # self.dest_index_to_key = {i: key for i, key in enumerate(self.to_keys)} # 현재 사용되지 않아 주석처리


        # 초기 상태로 리셋
        self.reset(shuffle_schedule=False)


    def _load_or_generate_schedule(self, inbound_plates, cfg):
        """스케줄을 로드하거나 생성합니다."""
        if inbound_plates:
            return copy.deepcopy(inbound_plates)
        else:
            try:
                plates_data_path = getattr(cfg, 'plates_data_path', 'output/reshuffle_plan.xlsx')
                df = pd.read_excel(plates_data_path, sheet_name="reshuffle")
                loaded_schedule = []
                required_cols = ["pileno", "inbound", "outbound", "unitw", "topile"]
                if not all(col in df.columns for col in required_cols):
                    print(f"[경고] Excel 파일({plates_data_path}) 필수 컬럼 부족: {required_cols}")

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
                return loaded_schedule
            except FileNotFoundError:
                num_plates_default = getattr(cfg, 'num_plates', 50)
                generated_schedule = generate_schedule(num_plates=num_plates_default)
                for i, p in enumerate(generated_schedule):
                    if not hasattr(p, 'from_pile'): p.from_pile = f"S_{i % MAX_SOURCE}"
                    if not hasattr(p, 'topile'): p.topile = f"D_{i % MAX_DEST}"
                return generated_schedule
            except Exception as e:
                print(f"[오류] 스케줄 로드/생성 중 오류: {e}")
                raise ValueError("스케줄 초기화 실패")

    def _create_norm_to_orig_map(self, original_schedule):
        """정규화된 키와 원본 키 간의 역매핑을 생성합니다."""
        norm_to_orig_map = {}
        try:
            original_from_keys_used = sorted(list(set(str(p.from_pile).strip() for p in original_schedule if hasattr(p, 'from_pile') and p.from_pile is not None)))[:MAX_SOURCE]
            original_to_keys_used = sorted(list(set(p.original_intended_topile for p in original_schedule if hasattr(p, 'original_intended_topile') and p.original_intended_topile is not None)))[:MAX_DEST]

            # from_keys와 to_keys가 정규화된 키 목록과 순서가 일치하는지 확인 후 매핑
            if len(self.from_keys) == len(original_from_keys_used):
                norm_from_map = {norm_key: orig_key for norm_key, orig_key in zip(self.from_keys, original_from_keys_used)}
            else:
                print(f"[Warning] _create_norm_to_orig_map: From 키 개수 불일치 ({len(self.from_keys)} vs {len(original_from_keys_used)}).")
                norm_from_map = {}

            if len(self.to_keys) == len(original_to_keys_used):
                norm_to_map = {norm_key: orig_key for norm_key, orig_key in zip(self.to_keys, original_to_keys_used)}
            else:
                print(f"[Warning] _create_norm_to_orig_map: To 키 개수 불일치 ({len(self.to_keys)} vs {len(original_to_keys_used)}).")
                norm_to_map = {}

            norm_to_orig_map = {**norm_from_map, **norm_to_map}
            if not norm_to_orig_map: print("[Warning] _create_norm_to_orig_map: 역매핑 정보 비어있음.")
        except Exception as e:
            print(f"[오류] _create_norm_to_orig_map: 역매핑 생성 오류: {e}")
            print("[경고] 역매핑 정보 생성 실패.")
            norm_to_orig_map = {}
        return norm_to_orig_map

    def reset(self, shuffle_schedule=False):
        """환경을 초기 상태로 리셋합니다."""
        schedule = copy.deepcopy(self.inbound_clone) # 정규화된 스케줄 복사본 사용
        if not schedule:
            print("[Warning] Reset: 스케줄 없음.")
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
            # from_pile 키가 self.from_keys에 포함되어 있는지 확인하여 유효한 출발지에만 배치
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
        if self.total_plate_count == 0: print("[Warning] Reset 후 이동할 플레이트 없음.")
        self.move_data = [] # reset 시 move_data 초기화
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
        cfg = get_cfg()
        gamma = getattr(cfg, "gamma", 0.99)
        shaping_reward_scale = getattr(cfg, "shaping_reward_scale", 1.0)

        # 1. 행동 전 포텐셜 계산 (블로킹 지표 사용)
        potential_before = -sum(self._get_total_blocking_pairs(self.plates.get(key, [])) for key in self.to_keys)

        from_index, to_index = action

        # 2. 액션 유효성 검사 및 필요시 랜덤 유효 액션으로 대체
        valid_source_mask, valid_dest_mask = self.get_masks()
        is_action_invalid = False
        # 인덱스 범위 확인 (from_keys, to_keys의 실제 길이와 비교)
        if not (0 <= from_index < len(self.from_keys)) or not valid_source_mask[from_index]: is_action_invalid = True
        if not (0 <= to_index < len(self.to_keys)) or not valid_dest_mask[to_index]: is_action_invalid = True


        if is_action_invalid:
            valid_source_indices = torch.where(valid_source_mask[:len(self.from_keys)])[0].tolist() # 실제 활성 소스 인덱스만 고려
            valid_dest_indices = torch.where(valid_dest_mask[:len(self.to_keys)])[0].tolist() # 실제 활성 데스트 인덱스만 고려

            if not valid_source_indices or not valid_dest_indices:
                print(f"[오류] Step {self.stage}: 대체 유효 행동 없음. 에피소드 종료.")
                return self._get_state(), 0.0, True, {"error": "No valid moves available, forced termination"}
            from_index = random.choice(valid_source_indices)
            to_index = random.choice(valid_dest_indices)
            # print(f"[경고] Step {self.stage}: 유효하지 않은 행동({action}) -> 유효 행동으로 대체 ({from_index}, {to_index}).") # 디버깅용

        source_key = self.from_keys[from_index] # 정규화된 키
        destination_key = self.to_keys[to_index] # 정규화된 키

        # 3. 이동 실행 전 상태 확인 (오류 방지)
        current_dest_pile = self.plates.get(destination_key, [])
        if len(current_dest_pile) >= self.max_stack:
            # 이 경우는 get_masks에서 이미 걸러져야 하지만, 만약을 위한 방어 코드
            print(f"[오류] Step {self.stage}: 목적지 '{destination_key}' Full! (get_masks 오류 가능성).")
            return self._get_state(), -1.0, True, {"error": f"Destination '{destination_key}' full unexpectedly"}
        current_source_pile = self.plates.get(source_key)
        if not current_source_pile:
            # 이 경우는 get_masks에서 이미 걸러져야 하지만, 만약을 위한 방어 코드
            print(f"[오류] Step {self.stage}: 출발지 '{source_key}' Empty! (get_masks 오류 가능성).")
            return self._get_state(), -1.0, True, {"error": f"Source '{source_key}' empty unexpectedly"}

        # 4. 플레이트 이동 실행
        try:
            moved_plate = current_source_pile.pop()
            self.plates[destination_key].append(moved_plate)
            # 이동 데이터를 기록 (필요시 활성화)
            # self.move_data.append((self.norm_to_orig_map.get(source_key, source_key),
            #                        self.norm_to_orig_map.get(destination_key, destination_key),
            #                        getattr(moved_plate, 'id', 'N/A')))
        except IndexError:
            print(f"[심각] Step {self.stage}: 비어있는 '{source_key}'에서 pop 시도! 논리 오류.")
            return self._get_state(), -10.0, True, {"error": "Pop from empty pile, serious logic error"}

        # 5. 행동 후 포텐셜 계산 (총 블로킹 쌍 기준)
        potential_after = -sum(self._get_total_blocking_pairs(self.plates.get(key, [])) for key in self.to_keys)

        # 6. 상태 업데이트
        self.crane_move += 1
        self.stage += 1

        # 7. 종료 조건 확인
        done = self.stage >= self.total_plate_count
        info = {}
        if self.total_plate_count <= 0: done = True # 예외 처리: 총 플레이트가 없으면 즉시 종료

        # 8. 다음 상태 가져오기
        next_state = self._get_state() # CPU Tensor

        # --- 보상 계산 (PBRS + Terminal Reward) ---
        shaping_reward = (gamma * potential_after - potential_before) * shaping_reward_scale
        terminal_reward = 0.0

        if done:
            # 최종 블로킹 지표 계산 (총 블로킹 쌍 기준)
            final_blocking_metric = sum(self._get_total_blocking_pairs(self.plates.get(key, [])) for key in self.to_keys)
            info['final_max_move_sum'] = final_blocking_metric
            info['final_crane_move'] = self.crane_move

            # 터미널 보상 계산
            # blocking_metric이 0일 때 가장 높은 보상 (10.0), blocking_metric이 커질수록 0에 수렴
            terminal_reward = 10.0 / (final_blocking_metric + 1)

        # 최종 보상 = PBRS + 터미널 보상(done일 때만) - 크레인 패널티
        total_reward = shaping_reward + terminal_reward - self.crane_penalty

        return next_state, total_reward, done, info

    def export_final_state_to_excel(self, output_filepath):
        """최종 파일 배치 상태를 Excel 파일로 저장합니다. (원본 파일 이름 사용)"""
        rows = []
        all_normalized_keys = sorted(list(self.plates.keys())) # 정규화된 키

        # 역매핑 정보 가져오기 (없으면 빈 dict)
        norm_map = getattr(self, 'norm_to_orig_map', {})
        if not norm_map: print("[Warning] export: 역매핑 정보 없음.")

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
            print("[Warning] 내보낼 플레이트 정보 없음.")
            return

        df = pd.DataFrame(rows)
        try:
            output_dir = os.path.dirname(output_filepath)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            with pd.ExcelWriter(output_filepath, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="final_arrangement", index=False)
            print(f"[정보] 최종 상태 저장 완료: {output_filepath}")
        except Exception as e:
            print(f"[오류] 최종 상태 Excel 내보내기 오류: {e}")

    def get_masks(self):
        """현재 상태에서 유효한 출발/도착 파일 마스크를 생성합니다."""
        source_flags = [False] * MAX_SOURCE
        for i in range(min(len(self.from_keys), MAX_SOURCE)):
            key = self.from_keys[i] # 정규화된 키
            # 해당 파일에 플레이트가 하나라도 있으면 True
            if self.plates.get(key):
                source_flags[i] = True
        source_mask = torch.tensor(source_flags, dtype=torch.bool, device=self.device)

        dest_flags = [False] * MAX_DEST
        for j in range(min(len(self.to_keys), MAX_DEST)):
            key = self.to_keys[j] # 정규화된 키
            # 파일 길이가 max_stack 미만이면 True (즉, 플레이트를 추가할 공간이 있으면 True)
            if len(self.plates.get(key, [])) < self.max_stack:
                dest_flags[j] = True
        dest_mask = torch.tensor(dest_flags, dtype=torch.bool, device=self.device)

        return source_mask, dest_mask

    def _get_state(self):
        """
        강판 환경의 현재 상태를 나타내는 특징 벡터를 생성합니다.
        각 파일의 로컬 특징과 정규화된 글로벌 특징을 포함합니다.
        """
        # 1. 글로벌 특징 계산 (실제 활성화된 출발/도착 파일의 개수만 포함, 정규화)
        # MAX_SOURCE와 MAX_DEST는 network.py에서 import 된 전역 상수입니다.
        num_active_sources = len(self.from_keys)
        num_active_dests = len(self.to_keys)

        # [핵심 수정] 글로벌 특징을 0-1 사이로 정규화합니다.
        # MAX_SOURCE와 MAX_DEST가 0일 경우를 대비하여 나눗셈 오류 방지
        normalized_num_sources = float(num_active_sources) / MAX_SOURCE if MAX_SOURCE > 0 else 0.0
        normalized_num_dests = float(num_active_dests) / MAX_DEST if MAX_DEST > 0 else 0.0

        global_features = [
            normalized_num_sources, # 정규화된 활성화 출발 파일 수
            normalized_num_dests    # 정규화된 활성화 도착 파일 수
        ]

        # 내부 헬퍼 함수 정의
        def create_feature_vector(pile, pile_type_id):
            feature_vector = [0.0] * self.actual_pile_feature_dim

            # 2. 개별 파일 특징 (Local Features)
            if pile:
                n_pile = len(pile)
                # 상위 OBSERVED_TOP_N_PLATES 개 강판의 outbound 값
                for i in range(self.OBSERVED_TOP_N_PLATES):
                    # 파일의 가장 위(top)부터 아래로 OBSERVED_TOP_N_PLATES 개를 관찰
                    plate_actual_idx_in_pile = n_pile - 1 - i
                    if plate_actual_idx_in_pile >= 0:
                        p_obj = pile[plate_actual_idx_in_pile]
                        ob_val = getattr(p_obj, 'outbound', 0.0)
                        feature_vector[i] = float(ob_val) if isinstance(ob_val, (int, float)) else 0.0
                    else:
                        feature_vector[i] = 0.0 # 스택에 플레이트가 충분치 않으면 0으로 패딩

                # 하위 강판 요약 통계 특징
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
                        feature_vector[summary_stats_start_idx + 0] = float(len(deeper_plates))
                        feature_vector[summary_stats_start_idx + 1] = min(deeper_outbounds)
                        feature_vector[summary_stats_start_idx + 2] = max(deeper_outbounds)
                        feature_vector[summary_stats_start_idx + 3] = sum(deeper_outbounds) / len(deeper_outbounds)
                    elif self.NUM_SUMMARY_STATS_DEEPER > 0: # 깊은 플레이트가 있지만 유효 outbound가 없을 경우
                        feature_vector[summary_stats_start_idx + 0] = float(len(deeper_plates))

                # 파일 타입 및 블로킹 특징
                pile_type_idx = self.OBSERVED_TOP_N_PLATES + self.NUM_SUMMARY_STATS_DEEPER
                feature_vector[pile_type_idx] = float(pile_type_id) # 1.0 for source, 2.0 for dest
                blocking_count_idx = pile_type_idx + self.NUM_PILE_TYPE_FEATURES
                feature_vector[blocking_count_idx] = float(self._get_total_blocking_pairs(pile))

            # 3. 계산된 글로벌 특징을 모든 파일 벡터의 끝에 동일하게 추가
            global_start_idx = self.OBSERVED_TOP_N_PLATES + self.NUM_SUMMARY_STATS_DEEPER + \
                               self.NUM_PILE_TYPE_FEATURES + self.NUM_BLOCKING_FEATURES
            for i, feat in enumerate(global_features):
                # 실제 feature_vector의 크기를 넘어가지 않도록 방어 코드 추가
                if global_start_idx + i < len(feature_vector):
                    feature_vector[global_start_idx + i] = feat

            return feature_vector

        # 최종 상태 텐서 생성
        state_features = []
        for i in range(MAX_SOURCE):
            # from_keys의 길이에 따라 실제 키를 사용하거나 None (패딩용)
            key = self.from_keys[i] if i < len(self.from_keys) else None
            state_features.append(create_feature_vector(self.plates.get(key, []), pile_type_id=1.0)) # 1.0 for Source

        for j in range(MAX_DEST):
            # to_keys의 길이에 따라 실제 키를 사용하거나 None (패딩용)
            key = self.to_keys[j] if j < len(self.to_keys) else None
            state_features.append(create_feature_vector(self.plates.get(key, []), pile_type_id=2.0)) # 2.0 for Destination

        try:
            state_tensor = torch.tensor(state_features, dtype=torch.float, device=self.device)
            expected_shape = (MAX_SOURCE + MAX_DEST, self.actual_pile_feature_dim)
            if state_tensor.shape != expected_shape:
                # 경고 메시지를 정확히 출력
                print(f"[ERROR] _get_state: 생성된 상태 텐서 크기 불일치! 예상: {expected_shape}, 실제: {state_tensor.shape}.")
                # 오류 발생 시 0으로 채워진 텐서 반환 (모델 에러 방지)
                return torch.zeros(expected_shape, dtype=torch.float, device=self.device)
            return state_tensor
        except Exception as e:
            print(f"[ERROR] _get_state: 상태 텐서 생성 실패: {e}. 0으로 채워진 텐서를 반환합니다.")
            expected_shape = (MAX_SOURCE + MAX_DEST, self.actual_pile_feature_dim)
            return torch.zeros(expected_shape, dtype=torch.float, device=self.device)