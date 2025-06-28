import random
import pandas as pd
import numpy as np
import os
from cfg import get_cfg

class Plate:
    def __init__(self, id, inbound, outbound, unitw=0.0):
        self.id = id
        self.inbound = inbound
        self.outbound = outbound
        self.unitw = unitw

    def __repr__(self):
        return f"Plate({self.id}, inbound={self.inbound}, outbound={self.outbound}, unitw={self.unitw:.2f})"

    def __copy__(self):
        # 새로운 Plate 객체를 생성하여 반환
        return Plate(self.id, self.inbound, self.outbound, self.unitw)

def generate_schedule(num_plates=None):
    cfg = get_cfg()
    if num_plates is None:
        num_plates = cfg.num_plates
    schedule = []
    for i in range(num_plates):
        inbound = random.randint(cfg.inbound_min, cfg.inbound_max)
        outbound = inbound + random.randint(cfg.outbound_extra_min, cfg.outbound_extra_max)
        unitw = random.uniform(cfg.unitw_min, cfg.unitw_max)
        schedule.append(Plate(f"P{i:03d}", inbound, outbound, unitw))
    return schedule

def import_plates_schedule(filepath, sheet_name="reshuffle"):
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    if 'outbound' not in df.columns:
        df['outbound'] = df['inbound'] + df['inbound'].apply(lambda _: random.randint(1, 20))
    else:
        df['outbound'] = df['outbound'].fillna(df['inbound'] + df['inbound'].apply(lambda _: random.randint(1, 20)))
    df['inbound'] = df['inbound'].astype(int)
    df['outbound'] = df['outbound'].astype(int)
    plates = []
    for i, row in df.iterrows():
        plate_obj = Plate(row['pileno'], row['inbound'], row['outbound'], row['unitw'])
        plates.append(plate_obj)
    return plates

def generate_reshuffle_plan(rows, n_from_piles_reshuffle, n_to_piles_reshuffle,
                            n_plates_reshuffle, safety_margin):
    mapping_from_pile_to_x = {}

    # 초기 pile 생성
    piles_all = []
    for row_id in rows:
        for col_id in range(1, 30):  # 열 개수는 고정 가능
            pile = row_id + str(col_id).rjust(2, '0')
            piles_all.append(pile)
            mapping_from_pile_to_x[pile] = col_id + 1

    x_max = max(mapping_from_pile_to_x.values()) + 1

    # reshuffle을 위한 DataFrame 초기화
    df_reshuffle = pd.DataFrame(columns=["pileno", "pileseq", "markno", "unitw", "topile", "inbound", "outbound"])

    # From piles와 To piles 샘플링
    from_piles_reshuffle = random.sample(piles_all, n_from_piles_reshuffle)
    candidates = [i for i in piles_all if i not in from_piles_reshuffle]
    to_piles_reshuffle = random.sample(candidates, n_to_piles_reshuffle)

    for pile in from_piles_reshuffle:
        x = mapping_from_pile_to_x[pile]
        if x < 1 + safety_margin:
            to_piles_reshuffle_rev = [i for i in to_piles_reshuffle if mapping_from_pile_to_x[i] <= x_max - safety_margin]
        elif x > x_max - safety_margin:
            to_piles_reshuffle_rev = [i for i in to_piles_reshuffle if mapping_from_pile_to_x[i] >= 1 + safety_margin]
        else:
            to_piles_reshuffle_rev = to_piles_reshuffle

        # plate 수 설정 (고정)
        num_of_plates = n_plates_reshuffle

        pileno = [pile] * num_of_plates
        pileseq = [str(i).rjust(3, '0') for i in range(1, num_of_plates + 1)]
        markno = [f"SP-RS-{pile}-{seq}" for seq in pileseq]
        unitw = np.random.uniform(0.141, 19.294, num_of_plates)
        topile = random.choices(to_piles_reshuffle_rev, k=num_of_plates)
        inbound_vals = [random.randint(1, 10) for _ in range(num_of_plates)]
        outbound_vals = [inb + random.randint(1, 20) for inb in inbound_vals]

        df_temp = pd.DataFrame({
            "pileno": pileno,
            "pileseq": pileseq,
            "markno": markno,
            "unitw": unitw,
            "topile": topile,
            "inbound": inbound_vals,
            "outbound": outbound_vals
        })

        df_reshuffle = pd.concat([df_reshuffle, df_temp], ignore_index=True)

    return df_reshuffle, from_piles_reshuffle, mapping_from_pile_to_x, piles_all, x_max

def save_reshuffle_plan_to_excel(df_reshuffle, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        df_reshuffle.to_excel(writer, sheet_name="reshuffle", index=False)
    abs_path = os.path.abspath(file_path)
    print(f"Reshuffle plan saved to {abs_path}")

# -----------------------------
# Main 함수: 파일 경로 지정
# -----------------------------
def main():
    cfg = get_cfg()
    # Excel 파일 저장 경로 지정 (원하는 폴더와 파일명)
    output_file = "output/reshuffle_plan.xlsx"

    # 설정값을 이용해 reshuffle plan 생성
    rows = ['A', 'B']
    df_plan, from_piles_reshuffle, mapping_from_pile_to_x, piles_all, x_max = generate_reshuffle_plan(
        rows,
        n_from_piles_reshuffle=10,
        n_to_piles_reshuffle=10,
        n_plates_reshuffle=22,
        safety_margin=0
    )
    print(f"Generated reshuffle plan with {df_plan.shape[0]} rows.")

    # 엑셀 파일로 저장
    save_reshuffle_plan_to_excel(df_plan, output_file)

if __name__ == "__main__":
    main()
