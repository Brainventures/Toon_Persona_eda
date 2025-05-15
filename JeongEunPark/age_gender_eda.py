import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from itertools import chain
from utils.json_loader import load_json_to_dataframe
from utils.font_setting import set_korean_font

set_korean_font()

# === 공통 함수 정의 ===
def get_top_items(series, common_items=None, top_n=5):
    raw = Counter(series)
    filtered = Counter([x for x in series if x not in (common_items or set())])
    return raw.most_common(top_n), filtered.most_common(top_n)

def draw_heatmap(data_dict, title, filename, cmap="YlGnBu"):
    df = pd.DataFrame(data_dict).fillna(0).T
    plt.figure(figsize=(14, 6))
    sns.heatmap(df, annot=True, fmt=".0f", cmap=cmap)
    plt.title(title)
    plt.xlabel("요소")
    plt.ylabel("연령대_성별 그룹")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# === 경로 설정 ===
data_dir = "/HDD/toon_persona/Training/label"
save_dir = "/home/jepark/dev/Toon_Persona_eda/JeongEunPark/results"
os.makedirs(save_dir, exist_ok=True)

# === 폴더 순회 ===
for subfolder in os.listdir(data_dir):
    subfolder_path = os.path.join(data_dir, subfolder)
    print(f"\n====={subfolder} EDA 진행 중=====")

    df = load_json_to_dataframe(subfolder_path)
    if df.empty:
        print(f"[WARN] {subfolder}에 데이터가 없습니다.")
        continue

    # 연령-성별 조합 필드 생성
    df["age_gender"] = df["age"].fillna("미상") + "_" + df["gender"].fillna("미상")

    # 공통 요소 정의
    all_shapes = list(chain.from_iterable(df["shape_list"].dropna()))
    all_clothes = list(chain.from_iterable(df["clothing_list"].dropna()))
    common_shapes = set([s for s, _ in Counter(all_shapes).most_common(10)])
    common_clothing = set([c for c, _ in Counter(all_clothes).most_common(10)])
    print(f"📌 공통 Shape 요소 Top 10:", common_shapes)
    print(f"📌 공통 Clothing 요소 Top 10:", common_clothing)

    # 그룹별 분석 결과 저장용
    shape_filtered_dist = defaultdict(dict)
    clothing_filtered_dist = defaultdict(dict)

    for group, group_df in df.groupby("age_gender"):
        shapes = list(chain.from_iterable(group_df["shape_list"].dropna()))
        clothes = list(chain.from_iterable(group_df["clothing_list"].dropna()))

        top_shapes, top_shapes_f = get_top_items(shapes, common_shapes)
        top_clothes, top_clothes_f = get_top_items(clothes, common_clothing)

        for k, v in top_shapes_f: shape_filtered_dist[group][k] = v
        for k, v in top_clothes_f: clothing_filtered_dist[group][k] = v

        print(f"\n📌 {group}")
        print("Top Shape (기존):", top_shapes)
        print("Top Shape (filtered):", top_shapes_f)
        print("Top Clothing (기존):", top_clothes)
        print("Top Clothing (filtered):", top_clothes_f)

    # 히트맵 시각화
    draw_heatmap(
        shape_filtered_dist,
        title=f"{subfolder} - 그룹별 비공통 Shape 빈도 (Top5 기준)",
        filename=os.path.join(save_dir, f"{subfolder}_heatmap_shape_filtered.png"),
        cmap="YlGnBu"
    )
    draw_heatmap(
        clothing_filtered_dist,
        title=f"{subfolder} - 그룹별 비공통 Clothing 빈도 (Top5 기준)",
        filename=os.path.join(save_dir, f"{subfolder}_heatmap_clothing_filtered.png"),
        cmap="PuRd"
    )

    print(f"[DONE] {subfolder} EDA 결과 저장 완료: {save_dir}")