import random
import numpy as np
import torch
import os
import csv
import pandas as pd
import copy
import math
from cfg import get_cfg
import data as plate  # data.py에 정의된 Plate 클래스 및 함수들

def simulate_transfer(move_data, piles_from, piles_to):
    """
    여러 번의 이동(move_data)을 시뮬레이션하여 reversal(역순) 횟수를 계산합니다.
    """
    cfg = get_cfg()
    max_num = cfg.max_stack
    reversal = 0
    piles_from_copy = {k: [copy.copy(item) for item in v] for k, v in piles_from.items()}
    piles_to_copy = {k: [copy.copy(item) for item in v] for k, v in piles_to.items()}

    for target in move_data:
        src_key, dst_key = target
        if len(piles_to_copy[dst_key]) > max_num:
            reversal = 9999999999
            return reversal, piles_to_copy
        plate_to_move = piles_from_copy[src_key].pop()
        if piles_to_copy[dst_key] and (piles_to_copy[dst_key][-1].outbound < plate_to_move.outbound):
            reversal += 1
        piles_to_copy[dst_key].append(plate_to_move)

    return reversal, piles_to_copy

class Locating(object):
    def __init__(self, num_pile=6, max_stack=30,
                 inbound_plates=None,
                 observe_inbounds=False, display_env=False,
                 device="cuda",
                 crane_penalty=0.0,
                 from_keys=None, to_keys=None,
                 num_stack=None):  # num_stack: 각 pile에서 사용할 최대 강판 개수
        """
        강판 재배치 환경을 초기화합니다.
        """
        self.num_pile = num_pile
        self.max_stack = max_stack
        self.num_stack = num_stack  # 강판 개수(또는 stack 크기) 저장
        self.stage = 0
        self.current_date = 0
        self.crane_move = 0
        self.plates = {}
        self.observe_inbounds = observe_inbounds
        self.device = device
        self.crane_penalty = crane_penalty

        if inbound_plates:
            self.inbound_plates = inbound_plates
            self.inbound_clone = self.inbound_plates[:]
        else:
            cfg = get_cfg()
            try:
                df = pd.read_excel(cfg.plates_data_path, sheet_name="reshuffle")
                schedule = []
                for idx, row in df.iterrows():
                    plate_id = row["pileno"]
                    inbound = row["inbound"] if ("inbound" in df.columns and not pd.isna(row["inbound"])) else random.randint(cfg.inbound_min, cfg.inbound_max)
                    outbound = row["outbound"] if ("outbound" in df.columns and not pd.isna(row["outbound"])) else inbound + random.randint(cfg.outbound_extra_min, cfg.outbound_extra_max)
                    unitw = row["unitw"] if ("unitw" in df.columns and not pd.isna(row["unitw"])) else random.uniform(cfg.unitw_min, cfg.unitw_max)
                    if "topile" in df.columns and not pd.isna(row["topile"]):
                        to_pile = str(row["topile"]).strip()
                    else:
                        to_pile = "A01"
                    p = plate.Plate(id=plate_id, inbound=inbound, outbound=outbound, unitw=unitw)
                    p.from_pile = str(plate_id).strip()
                    p.topile = to_pile
                    schedule.append(p)
                if len(schedule) == 0:
                    raise ValueError("Excel 파일에서 불러온 schedule이 비어 있습니다")
                self.inbound_plates = schedule
                self.inbound_clone = self.inbound_plates[:]
            except Exception as e:
                print("Excel 파일 로드 오류:", e)
                self.inbound_plates = plate.generate_schedule(num_plates=cfg.num_plates)
                for p in self.inbound_plates:
                    p.from_pile = str(p.id)
                    p.topile = str(random.randint(0, self.num_pile - 1))
                self.inbound_clone = self.inbound_plates[:]

        schedule = self.inbound_clone[:]
        self.allowed_to = sorted(list({p.topile for p in schedule}))
        if from_keys is None:
            self.from_keys = sorted(list({p.from_pile for p in schedule}))
        else:
            self.from_keys = from_keys
        if to_keys is None:
            self.to_keys = self.allowed_to
        else:
            self.to_keys = to_keys

        self.index_to_key = {i: key for i, key in enumerate(self.from_keys)}
        self.key_to_index = {key: i for i, key in enumerate(self.from_keys)}
        self.plates = {key: [] for key in self.from_keys}
        self.pile_states = [False] * self.num_pile
        self.last_reversal = 0

    def reset(self, shuffle_schedule=False):
        cfg = get_cfg()
        schedule = copy.deepcopy(self.inbound_clone)  # shallow copy (필요 시 deep copy 고려)
        if shuffle_schedule:
            random.shuffle(schedule)
        self.plates = {key: [] for key in self.from_keys}
        for p in schedule:
            if p.from_pile not in self.plates:
                self.plates[p.from_pile] = []
            self.plates[p.from_pile].append(p)
        all_plates = []
        for key in self.plates:
            all_plates.extend(self.plates[key])
        self.current_date = min([p.inbound for p in all_plates]) if all_plates else 0
        self.crane_move = 0
        self.stage = 0
        self.total_plate_count = sum(len(lst) for lst in self.plates.values())
        self.move_data = []
        # deep copy로 초기 상태 저장
        self.initial_plates = copy.deepcopy(self.plates)
        self.initial_dest = {k: [] for k in self.to_keys}
        self.update_pile_states()
        return self._get_state()

    def step(self, action):
        s_next, reward, done, info = self._composite_step(action)
        self.update_pile_states()
        return s_next, reward, done, info

    def _calculate_reward(self, pile_key):
        """
        기존 보상 함수 (참고용)
        """
        pile = self.plates[pile_key]
        if len(pile) <= 1:
            return 0.0
        reversal = 0
        for i, plate_obj in enumerate(pile[:-1]):
            for upper in pile[i+1:]:
                if plate_obj.outbound < upper.outbound:
                    reversal += 1
        reward = 2.0 - math.log(1 + reversal)
        if reward < 0:
            reward = 0.0
        return reward

    def _composite_step(self, action):
        """
        주어진 action (from_index, to_index)을 환경에 적용하고,
        만약 유효하지 않다면 fallback 로직을 통해 softmax 확률 기반 재선택을 수행.
        이동된 강판(plate)에 대한 reversal 여부 등을 확인해 보상을 계산하고,
        다음 상태와 done, debug info를 반환한다.
        """
        # print("[DEBUG] _composite_step called with action:", action)

        from_index, to_index = action
        valid_source_mask, valid_dest_mask = self.get_masks()
        # print("[DEBUG] valid_source_mask:", valid_source_mask.tolist())
        # print("[DEBUG] valid_dest_mask:", valid_dest_mask.tolist())

        # (1) 선택된 액션이 유효하지 않다면 fallback
        if not valid_source_mask[from_index] or not valid_dest_mask[to_index]:
            # print("[DEBUG] Action indices invalid: from_index:", from_index, "to_index:", to_index)
            valid_source_indices = [i for i, flag in enumerate(valid_source_mask.tolist()) if flag]
            valid_dest_indices = [j for j, flag in enumerate(valid_dest_mask.tolist()) if flag]
            # print("[DEBUG] fallback: valid_source_indices:", valid_source_indices)
            # print("[DEBUG] fallback: valid_dest_indices:", valid_dest_indices)
            if not valid_source_indices or not valid_dest_indices:
                # 더 이상 이동할 수 있는 pile이 없다면 에피소드 종료
                done = True
                return self._get_state(), 0.0, done, {}

            # (2) softmax 기반 fallback (act_batch에서 계산된 확률 분포 사용)
            if hasattr(self, "last_source_probs") and hasattr(self, "last_dest_probs"):
                # valid_source_indices에 해당하는 확률만 추출
                src_probs = self.last_source_probs[valid_source_indices]
                dest_probs = self.last_dest_probs[valid_dest_indices]

                # 디버깅: 원래 전체 확률과 valid indices에 해당하는 확률 출력
                # print("[DEBUG] last_source_probs:", self.last_source_probs)
                # print("[DEBUG] valid_source_indices:", valid_source_indices)
                # print("[DEBUG] extracted src_probs:", src_probs)
                # print("[DEBUG] Sum of extracted src_probs:", src_probs.sum().item())
                # print("[DEBUG] last_dest_probs:", self.last_dest_probs)
                # print("[DEBUG] valid_dest_indices:", valid_dest_indices)
                # print("[DEBUG] extracted dest_probs:", dest_probs)
                # print("[DEBUG] Sum of extracted dest_probs:", dest_probs.sum().item())

                # 재정규화
                src_probs = src_probs / src_probs.sum()
                dest_probs = dest_probs / dest_probs.sum()

                # print("[DEBUG] normalized src_probs:", src_probs)
                # print("[DEBUG] normalized dest_probs:", dest_probs)

                from_index = np.random.choice(valid_source_indices, p=src_probs.cpu().numpy())
                to_index = np.random.choice(valid_dest_indices, p=dest_probs.cpu().numpy())
                print("[DEBUG] fallback (softmax-based): chosen from_index:", from_index, "chosen to_index:", to_index)

            else:
                # softmax 확률을 가지고 있지 않으면 uniform random choice로 fallback
                from_index = random.choice(valid_source_indices)
                to_index = random.choice(valid_dest_indices)
                print("[DEBUG] fallback (uniform random): chosen from_index:", from_index, "chosen to_index:", to_index)

        # (3) 최종 선택된 pile 인덱스로 source_key / destination_key를 확인
        source_key = self.from_keys[from_index]
        destination_key = self.to_keys[to_index] if to_index < len(self.to_keys) else self.to_keys[0]
        # print("[DEBUG] Selected source_key:", source_key, "destination_key:", destination_key)

        # 만약 목적지 pile이 이미 가득 찼다면 다시 fallback
        if destination_key not in self.plates:
            self.plates[destination_key] = []
        if len(self.plates[destination_key]) >= self.max_stack:
            # print("[DEBUG] Destination pile full for key:", destination_key)
            valid_dest_indices = [j for j, flag in enumerate(valid_dest_mask.tolist()) if flag]
            # print("[DEBUG] fallback for full destination: valid_dest_indices:", valid_dest_indices)
            if not valid_dest_indices:
                done = True
                # print("[DEBUG] No valid destination indices available. Terminating episode.")
                return self._get_state(), 0.0, done, {}
            if hasattr(self, "last_dest_probs"):
                # softmax 기반 fallback
                dest_probs = self.last_dest_probs[valid_dest_indices]
                dest_probs = dest_probs / dest_probs.sum()
                to_index = np.random.choice(valid_dest_indices, p=dest_probs.cpu().numpy())
            else:
                # uniform fallback
                to_index = random.choice(valid_dest_indices)
            destination_key = self.to_keys[to_index]
            # print("[DEBUG] fallback for full destination: chosen to_index:", to_index, "destination_key:", destination_key)

        # (4) 소스 pile이 비어 있으면 fallback
        if not self.plates.get(source_key, []):
            # print("[DEBUG] Source pile empty for key:", source_key)
            valid_source_indices = [i for i, flag in enumerate(valid_source_mask.tolist()) if flag]
            # print("[DEBUG] fallback for empty source: valid_source_indices:", valid_source_indices)
            if not valid_source_indices:
                done = True
                # print("[DEBUG] No valid source indices available. Terminating episode.")
                return self._get_state(), 0.0, done, {}
            if hasattr(self, "last_source_probs"):
                # softmax 기반 fallback
                src_probs = self.last_source_probs[valid_source_indices]
                src_probs = src_probs / src_probs.sum()
                from_index = np.random.choice(valid_source_indices, p=src_probs.cpu().numpy())
            else:
                # uniform fallback
                from_index = random.choice(valid_source_indices)
            source_key = self.from_keys[from_index]
            # print("[DEBUG] fallback for empty source: chosen from_index:", from_index, "source_key:", source_key)

        # (5) 이동할 강판 선택
        moved_plate = self.plates[source_key].pop()
        # print("[DEBUG] Moved plate id:", getattr(moved_plate, "id", moved_plate))

        # (6) 즉각 보상 계산 (reversal 여부 체크)
        if self.plates[destination_key]:
            last_plate = self.plates[destination_key][-1]
            if last_plate.outbound < moved_plate.outbound:
                immediate_reward = -1
                # print("[DEBUG] Reversal occurred: last_plate.outbound:", last_plate.outbound,
                #       "moved_plate.outbound:", moved_plate.outbound)
            else:
                immediate_reward = 2
                # print("[DEBUG] No reversal: last_plate.outbound:", last_plate.outbound,
                #       "moved_plate.outbound:", moved_plate.outbound)
        else:
            immediate_reward = 0
            # print("[DEBUG] Destination pile empty, immediate reward:", immediate_reward)

        # (7) 실제 이동 수행
        moved_plate.topile = destination_key
        self.plates[destination_key].append(moved_plate)
        self.move_data.append((source_key, destination_key))
        self.crane_move += 1
        self.stage += 1
        # print(f"[DEBUG] stage: {self.stage}, total_plate_count: {self.total_plate_count}")
        # (8) 에피소드 종료 여부 및 최종 reversal 계산
        done = self.stage >= self.total_plate_count
        # 여기서 상태가 전부 0이면 강제 종료
        next_state = self._get_state()
        if next_state.abs().sum().item() < 1e-6:
            # print("[WARNING] next_state is all zeros — forcing done=True")
            done = True

        if done:
            final_reversal, _ = simulate_transfer(self.move_data, self.initial_plates, self.initial_dest)
            self.last_reversal = final_reversal
            # print("[DEBUG] Episode done. Final reversal:", final_reversal)

        return next_state, immediate_reward, done, {}

    # def _composite_step(self, action):
    #     from_index, to_index = action
    #     valid_source_mask, valid_dest_mask = self.get_masks()
    #     if not valid_source_mask[from_index] or not valid_dest_mask[to_index]:
    #         valid_source_indices = [i for i, flag in enumerate(valid_source_mask.tolist()) if flag]
    #         valid_dest_indices = [j for j, flag in enumerate(valid_dest_mask.tolist()) if flag]
    #         if not valid_source_indices or not valid_dest_indices:
    #             done = True
    #             return self._get_state(), 0.0, done, {}
    #         from_index = random.choice(valid_source_indices)
    #         to_index = random.choice(valid_dest_indices)
    #     source_key = self.from_keys[from_index]
    #     destination_key = self.to_keys[to_index] if to_index < len(self.to_keys) else self.to_keys[0]
    #     if destination_key not in self.plates:
    #         self.plates[destination_key] = []
    #     if len(self.plates[destination_key]) >= self.max_stack:
    #         valid_dest_indices = [j for j, flag in enumerate(valid_dest_mask.tolist()) if flag]
    #         if not valid_dest_indices:
    #             done = True
    #             return self._get_state(), 0.0, done, {}
    #         to_index = random.choice(valid_dest_indices)
    #         destination_key = self.to_keys[to_index]
    #     if not self.plates.get(source_key, []):
    #         valid_source_indices = [i for i, flag in enumerate(valid_source_mask.tolist()) if flag]
    #         if not valid_source_indices:
    #             done = True
    #             return self._get_state(), 0.0, done, {}
    #         from_index = random.choice(valid_source_indices)
    #         source_key = self.from_keys[from_index]
    #
    #     # 이동할 강판 선택
    #     moved_plate = self.plates[source_key].pop()
    #
    #     # 즉각 보상: 도착 pile에 이미 있는 강판과 비교하여 reversal 여부 체크
    #     if self.plates[destination_key]:
    #         last_plate = self.plates[destination_key][-1]
    #         if last_plate.outbound < moved_plate.outbound:
    #             immediate_reward = -1
    #         else:
    #             immediate_reward = 2
    #     else:
    #         immediate_reward = 0
    #
    #     moved_plate.topile = destination_key
    #     self.plates[destination_key].append(moved_plate)
    #     self.move_data.append((source_key, destination_key))
    #     self.crane_move += 1
    #     self.stage += 1
    #
    #     # 에피소드 종료 여부 확인 및 최종 상태의 reversal 패널티 적용
    #     done = self.stage >= self.total_plate_count
    #     if done:
    #         final_reversal, _ = simulate_transfer(self.move_data, self.initial_plates, self.initial_dest)
    #         self.last_reversal = final_reversal
    #         # immediate_reward += -final_reversal
    #
    #     return self._get_state(), immediate_reward, done, {}

    def update_pile_states(self):
        self.pile_states = [False] * self.num_pile
        for i, key in enumerate(self.from_keys):
            if i < self.num_pile:
                self.pile_states[i] = len(self.plates.get(key, [])) > 0

    def _get_state(self):
        """
        각 from pile과 to pile의 상위 3개 강판의 outbound 값을 flat 1D 벡터로 반환합니다.
        강판이 3개 미만이면 부족한 부분은 0으로 채웁니다.
        """
        state_values = []

        # from pile 상태
        for key in self.from_keys:
            pile = self.plates.get(key, [])
            # 상위 3개 강판의 outbound 값 (강판이 없다면 0으로 채움)
            if len(pile) >= 3:
                top3 = [float(plate.outbound) for plate in pile[-3:]]
            else:
                top3 = [float(plate.outbound) for plate in pile] + [0.0] * (3 - len(pile))
            state_values.extend(top3)

        # to pile 상태
        for key in self.to_keys:
            pile = self.plates.get(key, [])
            if len(pile) >= 3:
                top3 = [float(plate.outbound) for plate in pile[-3:]]
            else:
                top3 = [float(plate.outbound) for plate in pile] + [0.0] * (3 - len(pile))
            state_values.extend(top3)

        # flat 1D 벡터 반환 (배치 차원은 이후 처리에서 추가 가능)
        return torch.tensor(state_values, dtype=torch.float)

    def export_final_state_to_excel(self, output_filepath):
        rows = []
        for key, pile in self.plates.items():
            for depth_idx, plate_obj in enumerate(pile):
                row = {
                    "pileno": plate_obj.id if hasattr(plate_obj, "id") else str(plate_obj),
                    "inbound": plate_obj.inbound if hasattr(plate_obj, "inbound") else None,
                    "outbound": plate_obj.outbound if hasattr(plate_obj, "outbound") else None,
                    "unitw": plate_obj.unitw if hasattr(plate_obj, "unitw") else None,
                    "final_pile": key,
                    "depth": depth_idx,
                    "topile": getattr(plate_obj, "topile", None)
                }
                rows.append(row)
        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        with pd.ExcelWriter(output_filepath, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="final_arrangement", index=False)

    def get_masks(self):
        source_mask = torch.tensor(
            [len(self.plates.get(key, [])) > 0 for key in self.from_keys],
            dtype=torch.bool
        )
        dest_mask = torch.tensor(
            [len(self.plates.get(key, [])) < self.max_stack for key in self.to_keys],
            dtype=torch.bool
        )

        # # 디버깅 로그 추가
        # print("[DEBUG][Env.get_masks()] from_keys:", self.from_keys)
        # print("[DEBUG][Env.get_masks()] to_keys:", self.to_keys)
        # for i, k in enumerate(self.from_keys):
        #     print(
        #         f"   from_keys[{i}]={k}, pile_size={len(self.plates.get(k, []))}, source_mask={source_mask[i].item()}")
        # for j, k in enumerate(self.to_keys):
        #     print(f"   to_keys[{j}]={k}, pile_size={len(self.plates.get(k, []))}, dest_mask={dest_mask[j].item()}")

        return source_mask, dest_mask

