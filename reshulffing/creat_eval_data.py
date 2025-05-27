import pandas as pd
from data import generate_reshuffle_plan # 사용자님의 기존 데이터 생성 함수를 가져옵니다.
from cfg import get_cfg # 설정값을 가져오기 위해 필요할 수 있습니다.

def create_evaluation_dataset(num_scenarios=30, output_filename="reshuffle_plan(for eval).xlsx"):
    """
    지정된 개수(num_scenarios)만큼 새로운 시나리오를 생성하여,
    하나의 Excel 파일로 저장하는 함수.
    """
    cfg = get_cfg()
    all_scenarios_list = [] # 각 시나리오의 DataFrame을 저장할 리스트

    print(f"총 {num_scenarios}개의 평가 시나리오 생성을 시작합니다...")

    # 지정된 횟수만큼 반복
    for i in range(1, num_scenarios + 1):
        print(f"  - 시나리오 {i} 생성 중...")
        try:
            # 사용자님의 기존 데이터 생성 함수를 그대로 호출합니다.
            df_plan, _, _, _, _ = generate_reshuffle_plan(
                rows=['A', 'B'],
                n_from_piles_reshuffle=cfg.n_from_piles_reshuffle,
                n_to_piles_reshuffle=cfg.n_to_piles_reshuffle,
                n_plates_reshuffle=cfg.n_plates_reshuffle,
                safety_margin=cfg.safety_margin
            )

            # 가장 중요한 부분: 'scenario_id' 컬럼을 추가하고 현재 번호를 할당합니다.
            df_plan['scenario_id'] = i

            all_scenarios_list.append(df_plan)

        except Exception as e:
            print(f"[Error] 시나리오 {i} 생성 중 오류 발생: {e}")
            continue

    # 모든 시나리오 DataFrame들을 하나로 합칩니다.
    if not all_scenarios_list:
        print("생성된 시나리오가 없습니다. Excel 파일을 만들 수 없습니다.")
        return

    final_df = pd.concat(all_scenarios_list, ignore_index=True)

    # Excel 파일로 저장합니다.
    final_df.to_excel(output_filename, index=False)
    print(f"\n완료! {num_scenarios}개의 시나리오가 '{output_filename}' 파일에 저장되었습니다.")
    print("파일 내용 확인:")
    print(final_df.head()) # 상위 5줄 출력
    print("...")
    print(final_df.tail()) # 하위 5줄 출력


if __name__ == '__main__':
    # 이 스크립트를 직접 실행하면 평가 데이터셋을 생성합니다.
    create_evaluation_dataset(num_scenarios=30)