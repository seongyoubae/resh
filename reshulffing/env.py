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


def normalize_keys(schedule):
    """
    schedule 내 각 Plate 객체의 from_pile, topile 값을 정규화합니다.
    원래 값(예: "A01", "A15" 등)을 고유하게 모은 후,
    "from_00", "from_01", ... 및 "to_00", "to_01", ... 형태로 매핑합니다.
    """
    # 원래 키들을 문자열로 정리한 후 정렬
    unique_from = sorted(list({str(p.from_pile).strip() for p in schedule}))
    unique_to = sorted(list({str(p.topile).strip() for p in schedule}))

    # 정규화된 키 생성 (예: "from_00", "from_01", ...)
    from_key_map = {key: f"from_{i:02d}" for i, key in enumerate(unique_from)}
    to_key_map = {key: f"to_{i:02d}" for i, key in enumerate(unique_to)}

    # 모든 Plate 객체에 대해 키를 정규화
    for p in schedule:
        p.from_pile = from_key_map[str(p.from_pile).strip()]
        p.topile = to_key_map[str(p.topile).strip()]
    return schedule, list(from_key_map.values()), list(to_key_map.values())


class Locating(object):
    def __init__(self, num_pile=10, max_stack=30,
                 inbound_plates=None,
                 observe_inbounds=False, display_env=False,
                 device="cuda",
                 crane_penalty=0.0,
                 # from_keys, to_keys 인자는 제거하거나 사용하지 않음
                 num_stack=None):
        """
        강판 재배치 환경을 초기화합니다.
        """
        self.num_pile = num_pile
        self.max_stack = max_stack
        self.num_stack = num_stack
        self.stage = 0
        self.current_date = 0
        self.crane_move = 0
        self.plates = {}
        self.observe_inbounds = observe_inbounds
        self.device = device
        self.crane_penalty = crane_penalty

        cfg = get_cfg()
        if inbound_plates:
            # 전달된 schedule에 대해 normalize_keys를 호출하여 키를 정규화
            self.inbound_plates, normalized_from, normalized_to = normalize_keys(inbound_plates)
            # print("DEBUG: 정규화된 from_keys:", normalized_from)
            # print("DEBUG: 정규화된 to_keys:", normalized_to)
            self.inbound_clone = self.inbound_plates[:]
            self.from_keys = normalized_from
            self.to_keys = normalized_to
        else:
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
                # normalize_keys 호출
                schedule, normalized_from, normalized_to = normalize_keys(schedule)
                # print("DEBUG: 정규화된 from_keys:", normalized_from)
                # print("DEBUG: 정규화된 to_keys:", normalized_to)
                self.inbound_plates = schedule
                self.inbound_clone = self.inbound_plates[:]
                self.from_keys = normalized_from
                self.to_keys = normalized_to
            except Exception as e:
                print("Excel 파일 로드 오류:", e)
                self.inbound_plates = plate.generate_schedule(num_plates=cfg.num_plates)
                for p in self.inbound_plates:
                    p.from_pile = str(p.id)
                    p.topile = str(random.randint(0, self.num_pile - 1))
                self.inbound_clone = self.inbound_plates[:]
                self.from_keys = sorted(list({p.from_pile for p in self.inbound_clone}))
                self.to_keys = sorted(list({p.topile for p in self.inbound_clone}))

        # 외부 인자는 사용하지 않고, 내부에서 정규화된 키를 사용합니다.
        schedule = self.inbound_clone[:]
        self.allowed_to = sorted(list({p.topile for p in schedule}))
        if self.from_keys is None:
            self.from_keys = sorted(list({p.from_pile for p in schedule}))
        if self.to_keys is None:
            self.to_keys = self.allowed_to

        self.index_to_key = {i: key for i, key in enumerate(self.from_keys)}
        self.key_to_index = {key: i for i, key in enumerate(self.from_keys)}
        self.plates = {key: [] for key in self.from_keys}
        self.pile_states = [False] * self.num_pile
        self.last_reversal = 0


    def reset(self, shuffle_schedule=False):
        cfg = get_cfg()
        schedule = copy.deepcopy(self.inbound_clone)
        if shuffle_schedule:
            random.shuffle(schedule)
        self.plates = {key: [] for key in self.from_keys}
        for p in schedule:
            if p.from_pile not in self.plates:
                self.plates[p.from_pile] = []
            self.plates[p.from_pile].append(p)
        # 디버깅: 각 정규화된 from_key에 할당된 plate 개수 출력
        # for key in self.from_keys:
            # print(f"DEBUG: {key} has {len(self.plates.get(key, []))} plates")
        all_plates = []
        for key in self.plates:
            all_plates.extend(self.plates[key])
        self.current_date = min([p.inbound for p in all_plates]) if all_plates else 0
        self.crane_move = 0
        self.stage = 0
        self.total_plate_count = sum(len(lst) for lst in self.plates.values())
        self.move_data = []
        self.initial_plates = copy.deepcopy(self.plates)
        self.initial_dest = {k: [] for k in self.to_keys}
        self.update_pile_states()
        return self._get_state()

    def step(self, action):
        s_next, reward, done, info = self._composite_step(action)
        self.update_pile_states()
        return s_next, reward, done, info

    def _calculate_reward(self, pile_key):
        pile = self.plates[pile_key]
        if len(pile) <= 1:
            return 0.0
        reversal = 0
        for i, plate_obj in enumerate(pile[:-1]):
            for upper in pile[i + 1:]:
                if plate_obj.outbound < upper.outbound:
                    reversal += 1
        reward = 2.0 - math.log(1 + reversal)
        if reward < 0:
            reward = 0.0
        return reward

    def _composite_step(self, action):
        from_index, to_index = action
        valid_source_mask, valid_dest_mask = self.get_masks()
        # print(f"[DEBUG] _composite_step: valid_source_mask = {valid_source_mask}")
        # print(f"[DEBUG] _composite_step: valid_dest_mask = {valid_dest_mask}")

        if not valid_source_mask[from_index] or not valid_dest_mask[to_index]:
            valid_source_indices = [i for i, flag in enumerate(valid_source_mask.tolist()) if flag]
            valid_dest_indices = [j for j, flag in enumerate(valid_dest_mask.tolist()) if flag]
            if not valid_source_indices or not valid_dest_indices:
                done = True
                return self._get_state(), 0.0, done, {}
            if hasattr(self, "last_source_probs") and hasattr(self, "last_dest_probs"):
                src_probs = self.last_source_probs[valid_source_indices]
                dest_probs = self.last_dest_probs[valid_dest_indices]
                src_probs = src_probs / src_probs.sum()
                dest_probs = dest_probs / dest_probs.sum()
                from_index = np.random.choice(valid_source_indices, p=src_probs.cpu().numpy())
                to_index = np.random.choice(valid_dest_indices, p=dest_probs.cpu().numpy())
                print("[DEBUG] fallback (softmax-based): chosen from_index:", from_index, "chosen to_index:", to_index)
            else:
                from_index = random.choice(valid_source_indices)
                to_index = random.choice(valid_dest_indices)
                print("[DEBUG] fallback (uniform random): chosen from_index:", from_index, "chosen to_index:", to_index)
        source_key = self.from_keys[from_index]
        destination_key = self.to_keys[to_index] if to_index < len(self.to_keys) else self.to_keys[0]
        if destination_key not in self.plates:
            self.plates[destination_key] = []
        if len(self.plates[destination_key]) >= self.max_stack:
            valid_dest_indices = [j for j, flag in enumerate(valid_dest_mask.tolist()) if flag]
            if not valid_dest_indices:
                done = True
                return self._get_state(), 0.0, done, {}
            if hasattr(self, "last_dest_probs"):
                dest_probs = self.last_dest_probs[valid_dest_indices]
                dest_probs = dest_probs / dest_probs.sum()
                to_index = np.random.choice(valid_dest_indices, p=dest_probs.cpu().numpy())
            else:
                to_index = random.choice(valid_dest_indices)
            destination_key = self.to_keys[to_index]
        if not self.plates.get(source_key, []):
            valid_source_indices = [i for i, flag in enumerate(valid_source_mask.tolist()) if flag]
            if not valid_source_indices:
                done = True
                return self._get_state(), 0.0, done, {}
            if hasattr(self, "last_source_probs"):
                src_probs = self.last_source_probs[valid_source_indices]
                src_probs = src_probs / src_probs.sum()
                from_index = np.random.choice(valid_source_indices, p=src_probs.cpu().numpy())
            else:
                from_index = random.choice(valid_source_indices)
            source_key = self.from_keys[from_index]
        moved_plate = self.plates[source_key].pop()
        if self.plates[destination_key]:
            last_plate = self.plates[destination_key][-1]
            if last_plate.outbound < moved_plate.outbound:
                immediate_reward = -1
            else:
                immediate_reward = 1
        else:
            immediate_reward = 0
        moved_plate.topile = destination_key
        self.plates[destination_key].append(moved_plate)
        self.move_data.append((source_key, destination_key))
        self.crane_move += 1
        self.stage += 1
        done = self.stage >= self.total_plate_count
        next_state = self._get_state()
        if next_state.abs().sum().item() < 1e-6:
            done = True
        if done:
            final_reversal, _ = simulate_transfer(self.move_data, self.initial_plates, self.initial_dest)
            self.last_reversal = final_reversal

        # print(f"[DEBUG] _composite_step: from_key = {source_key}, to_key = {destination_key}, reward = {immediate_reward}")
        return next_state, immediate_reward, done, {}

    def update_pile_states(self):
        self.pile_states = [False] * self.num_pile
        for i, key in enumerate(self.from_keys):
            if i < self.num_pile:
                self.pile_states[i] = len(self.plates.get(key, [])) > 0
        # print(f"[DEBUG] update_pile_states: {self.pile_states}")

    def _get_state(self):
        state_values = []
        # from_keys에 해당하는 각 pile 처리 (예: 5개)
        for key in self.from_keys:
            pile = self.plates.get(key, [])
            top_outbounds = [float(p.outbound) for p in pile[:self.max_stack]]
            if len(top_outbounds) < self.max_stack:
                top_outbounds += [0.0] * (self.max_stack - len(top_outbounds))
            avg_outbound = np.mean([float(p.outbound) for p in pile]) if pile else 0.0
            state_values.extend(top_outbounds + [avg_outbound])
        # to_keys에 해당하는 각 pile 처리 (예: 5개)
        for key in self.to_keys:
            pile = self.plates.get(key, [])
            top_outbounds = [float(p.outbound) for p in pile[:self.max_stack]]
            if len(top_outbounds) < self.max_stack:
                top_outbounds += [0.0] * (self.max_stack - len(top_outbounds))
            avg_outbound = np.mean([float(p.outbound) for p in pile]) if pile else 0.0
            state_values.extend(top_outbounds + [avg_outbound])

        total_piles = len(self.from_keys) + len(self.to_keys)
        pile_feature_dim = self.max_stack + 1
        state_tensor = torch.tensor(state_values, dtype=torch.float).view(total_piles, pile_feature_dim)
        return state_tensor

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
        # print(f"[DEBUG] get_masks: source_mask = {source_mask}, dest_mask = {dest_mask}")

        return source_mask, dest_mask
