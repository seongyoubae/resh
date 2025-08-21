import torch
import random
import copy
import pandas as pd
from cfg import get_cfg
from data import Plate, generate_schedule
from network import MAX_SOURCE, MAX_DEST


def normalize_keys(schedule):
    """
    스케줄 내의 from_pile, topile 키를 정규화된 형식('from_XX', 'to_XX')으로 변환합니다.
    이는 다양한 파일 이름을 일관된 모델 입력으로 만들어줍니다.
    """
    if not schedule: return [], [], []

    # 원본 from/to 키를 중복 없이 정렬하여 추출 (최대 개수 제한)
    original_from_keys = sorted(list(set(str(p.from_pile).strip() for p in schedule if hasattr(p, 'from_pile'))))[
                         :MAX_SOURCE]
    original_to_keys = sorted(list(set(str(p.topile).strip() for p in schedule if hasattr(p, 'topile'))))[:MAX_DEST]

    # 원본 키 -> 정규화된 키 매핑 생성
    from_key_map = {key: f"from_{i:02d}" for i, key in enumerate(original_from_keys)}
    to_key_map = {key: f"to_{i:02d}" for i, key in enumerate(original_to_keys)}

    normalized_schedule = []
    default_to_key = list(to_key_map.values())[0] if to_key_map else "to_00"

    for p in schedule:
        norm_p = copy.copy(p)
        # Plate 객체의 from_pile, topile 속성을 정규화된 키로 덮어씁니다.
        norm_p.from_pile = from_key_map.get(str(p.from_pile).strip())
        norm_p.topile = to_key_map.get(str(p.topile).strip(), default_to_key)

        # 정규화된 키가 할당된 경우에만 최종 스케줄에 추가
        if norm_p.from_pile is not None:
            normalized_schedule.append(norm_p)

    return normalized_schedule, list(from_key_map.values()), list(to_key_map.values())


class Locating(object):
    """
    강판 재배치 RL 환경 클래스.
    상태 표현에 Min-Max 정규화를 적용하여 모델의 일반화 성능을 향상시킵니다.
    """

    def __init__(self, max_stack, inbound_plates, crane_penalty, device):
        cfg = get_cfg()
        self.device = device

        # ✨ [핵심] 정규화를 위한 파라미터 설정
        self.max_stack = float(max_stack)
        self.max_outbound_val = float(cfg.max_outbound)
        # 이론적으로 가능한 최대 블로킹 쌍의 수 (n C 2)
        self.max_possible_blocking_pairs = (self.max_stack * (self.max_stack - 1)) / 2.0
        self.max_source_piles = float(MAX_SOURCE)
        self.max_dest_piles = float(MAX_DEST)

        # --- 특징 차원 정의 ---
        self.OBSERVED_TOP_N_PLATES = cfg.OBSERVED_TOP_N_PLATES
        self.NUM_SUMMARY_STATS_DEEPER = cfg.NUM_SUMMARY_STATS_DEEPER
        self.NUM_PILE_TYPE_FEATURES = 1
        self.NUM_BLOCKING_FEATURES = 1
        self.actual_pile_feature_dim = (self.OBSERVED_TOP_N_PLATES + self.NUM_SUMMARY_STATS_DEEPER +
                                        self.NUM_PILE_TYPE_FEATURES + self.NUM_BLOCKING_FEATURES)

        # --- 스케줄 처리 및 초기화 ---
        self.crane_penalty = float(crane_penalty)
        schedule_to_process = copy.deepcopy(inbound_plates)

        # 키 정규화 수행
        self.inbound_plates, self.from_keys, self.to_keys = normalize_keys(schedule_to_process)
        self.inbound_clone = copy.deepcopy(self.inbound_plates)  # 리셋용 복사본

        if not self.from_keys or not self.to_keys:
            raise ValueError("키 정규화 후 유효한 Source 또는 Destination 파일이 없습니다.")

        self.all_pile_keys = sorted(list(set(self.from_keys + self.to_keys)))
        self.reset(shuffle_schedule=False)

    def reset(self, shuffle_schedule=False):
        """환경을 초기 상태로 리셋합니다."""
        schedule = copy.deepcopy(self.inbound_clone)
        if shuffle_schedule: random.shuffle(schedule)

        self.plates = {key: [] for key in self.all_pile_keys}

        actual_plate_count = 0
        for p in schedule:
            if p.from_pile in self.plates:
                self.plates[p.from_pile].append(p)
                actual_plate_count += 1

        self.crane_move = 0
        self.stage = 0
        # This ensures the count is always correct, regardless of input data
        self.total_plate_count = actual_plate_count

        return self._get_state()

    def _get_total_blocking_pairs(self, pile):
        """파일 내의 총 블로킹 쌍(inversion) 개수를 계산합니다."""
        n = len(pile)
        if n <= 1: return 0

        outbounds = [p.outbound for p in pile]
        blocking_pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                # 아래쪽 판(i)이 위쪽 판(j)보다 나중에 나가야 할 경우 (outbound_i < outbound_j)
                if outbounds[i] < outbounds[j]:
                    blocking_pairs += 1
        return float(blocking_pairs)

    def step(self, action):
        """환경 스텝 함수: 행동을 받아 다음 상태, 보상, 종료 여부를 반환합니다."""
        cfg = get_cfg()

        # 행동 전 포텐셜 계산 (블로킹 지표 기반)
        potential_before = -sum(self._get_total_blocking_pairs(self.plates.get(key, [])) for key in self.to_keys)

        from_index, to_index = action
        source_key = self.from_keys[from_index]
        destination_key = self.to_keys[to_index]

        # 플레이트 이동
        moved_plate = self.plates[source_key].pop()
        self.plates[destination_key].append(moved_plate)

        self.crane_move += 1
        self.stage += 1

        # 행동 후 포텐셜 계산
        potential_after = -sum(self._get_total_blocking_pairs(self.plates.get(key, [])) for key in self.to_keys)

        # 종료 조건 확인
        done = self.stage >= self.total_plate_count
        info = {}

        # 보상 계산 (PBRS + Terminal Reward)
        shaping_reward = (cfg.gamma * potential_after - potential_before) * cfg.shaping_reward_scale
        terminal_reward = 0.0

        if done:
            final_blocking_metric = sum(
                self._get_total_blocking_pairs(self.plates.get(key, [])) for key in self.to_keys)
            info['final_max_move_sum'] = final_blocking_metric
            info['final_crane_move'] = self.crane_move
            terminal_reward = 10.0 / (final_blocking_metric + 1.0)  # 블로킹이 적을수록 높은 보상

        total_reward = shaping_reward + terminal_reward - self.crane_penalty
        return self._get_state(), total_reward, done, info

    def get_masks(self):
        """현재 상태에서 유효한 출발/도착 파일 마스크를 생성합니다."""
        source_flags = [bool(self.plates.get(key)) for key in self.from_keys]
        dest_flags = [len(self.plates.get(key, [])) < self.max_stack for key in self.to_keys]

        # 패딩 추가
        source_flags.extend([False] * (MAX_SOURCE - len(source_flags)))
        dest_flags.extend([False] * (MAX_DEST - len(dest_flags)))

        return torch.tensor(source_flags, dtype=torch.bool, device=self.device), \
            torch.tensor(dest_flags, dtype=torch.bool, device=self.device)

    def _get_state(self):
        """
        환경의 현재 상태를 나타내는 **정규화된** 특징 벡터를 생성합니다.
        """
        # 1. 글로벌 특징 계산 (정규화)
        num_active_sources = len(self.from_keys)
        num_active_dests = len(self.to_keys)
        normalized_num_sources = num_active_sources / self.max_source_piles
        normalized_num_dests = num_active_dests / self.max_dest_piles
        global_features = [normalized_num_sources, normalized_num_dests]

        def create_feature_vector(pile, pile_type_id):
            feature_vector = [0.0] * self.actual_pile_feature_dim
            n_pile = len(pile)

            # 2. 로컬 특징 (개별 파일 특징)
            if n_pile > 0:
                # 상위 N개 강판의 정규화된 outbound 값
                for i in range(self.OBSERVED_TOP_N_PLATES):
                    if i < n_pile:
                        ob_val = pile[-(i + 1)].outbound
                        feature_vector[i] = float(ob_val) / self.max_outbound_val

                # 하위 강판 요약 통계 특징 (정규화)
                deeper_plates = pile[:-self.OBSERVED_TOP_N_PLATES]
                summary_start_idx = self.OBSERVED_TOP_N_PLATES
                if deeper_plates:
                    outbounds = [p.outbound for p in deeper_plates]
                    feature_vector[summary_start_idx + 0] = len(deeper_plates) / self.max_stack
                    feature_vector[summary_start_idx + 1] = min(outbounds) / self.max_outbound_val
                    feature_vector[summary_start_idx + 2] = max(outbounds) / self.max_outbound_val
                    feature_vector[summary_start_idx + 3] = (sum(outbounds) / len(outbounds)) / self.max_outbound_val

                # 파일 타입 및 블로킹 특징 (정규화)
                type_idx = self.OBSERVED_TOP_N_PLATES + self.NUM_SUMMARY_STATS_DEEPER
                feature_vector[type_idx] = float(pile_type_id)  # 0.0 for source, 1.0 for dest

                blocking_idx = type_idx + self.NUM_PILE_TYPE_FEATURES
                blocking_pairs = self._get_total_blocking_pairs(pile)
                feature_vector[
                    blocking_idx] = blocking_pairs / self.max_possible_blocking_pairs if self.max_possible_blocking_pairs > 0 else 0.0

            return feature_vector

        # 최종 상태 텐서 생성
        state_features = []
        # Source 파일 특징
        for i in range(MAX_SOURCE):
            key = self.from_keys[i] if i < len(self.from_keys) else None
            state_features.append(create_feature_vector(self.plates.get(key, []), pile_type_id=0.0))
        # Destination 파일 특징
        for j in range(MAX_DEST):
            key = self.to_keys[j] if j < len(self.to_keys) else None
            state_features.append(create_feature_vector(self.plates.get(key, []), pile_type_id=1.0))

        return torch.tensor(state_features, dtype=torch.float, device=self.device)