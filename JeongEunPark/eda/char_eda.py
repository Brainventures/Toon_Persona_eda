import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from collections import Counter
from itertools import chain

from utils.json_loader import load_json_to_dataframe
from utils.font_setting import set_korean_font
set_korean_font()

data_dir = "/HDD/toon_persona/Training/label"
save_dir = "/home/jepark/dev/Toon_Persona_eda/JeongEunPark/results"
os.makedirs(save_dir, exist_ok=True)

for subfolder in os.listdir(data_dir):
    subfolder_path = os.path.join(data_dir, subfolder)
    
    print(f"====={subfolder} eda 진행중=====")

    df = load_json_to_dataframe(subfolder_path)
    if df.empty:
        print(f"{subfolder}안에 데이터가 없음")
        continue
    
    # === 성별 분포 ===
    df["gender"].value_counts()

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="gender", order=df["gender"].value_counts().index)
    plt.title(f"{subfolder} 캐릭터 성별 분포")
    plt.xlabel("성별")
    plt.ylabel("수")
    plt.xticks()
    plt.yticks()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{subfolder}_gender_dist_all.png"))
    plt.close()

    # === 성별이 기타인 character ===
    df_gender_etc = df[df["gender"] == "기타"]
    print(f"'기타' 성별 샘플 수: {len(df_gender_etc)}")

    # 전체 shape top 10
    all_shapes = sum(df["shape_list"], [])
    all_shapes_counter = Counter(all_shapes)
    print("전체 데이터 shape 분포 top 10")
    print(all_shapes_counter.most_common(10))

    # 연령대
    print(df_gender_etc["age"].value_counts())

    # kind (인간 / 비인간)
    print(df_gender_etc["kind"].value_counts())

    # 대사 유무
    has_dialogue = df_gender_etc["dialogue_count"] > 0
    print("대사 있음:", has_dialogue.sum())
    print("대사 없음:", (~has_dialogue).sum())

    # === 성별 기타 & 종류 인간 ===
    df_target = df[(df["gender"] == "기타") & (df["kind"] == "인간")]

    # shape 분포
    target_shapes = list(chain.from_iterable(df_target["shape_list"]))
    target_shape_counter = Counter(target_shapes)
    print("공통된 shape 분포 Top 10")
    print(target_shape_counter.most_common(10))

    # 전체 데이터에서 파마/무표정을 한 사람의 성별 비율
    df["has_parma"] = df["shape_list"].apply(lambda lst: "파마" in lst)
    df["has_neutral_face"] = df["shape_list"].apply(lambda lst: "무표정" in lst)

    print("전체 데이터에서 파마를 한 사람의 성별 비율")
    print(df.groupby("gender")["has_parma"].mean())
    print("전체 데이터에서 무표정을 한 사람의 성별 비율")
    print(df.groupby("gender")["has_neutral_face"].mean())

    # 성별 + 파마 여부 조합 비율 시각화
    parma_ct = pd.crosstab(df["gender"], df["has_parma"], normalize='index')

    parma_ct.plot(kind='bar', stacked=True, figsize=(6,4), colormap='Set2')
    plt.title("성별별 파마 여부 구성비율")
    plt.ylabel("비율")
    plt.xlabel("성별")
    plt.legend(["파마 아님", "파마 있음"])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{subfolder}_gender_parma_stacked.png"))
    plt.close()

    # 성별 + 무표정 조합 비율 시각화
    neutral_face_ct = pd.crosstab(df["gender"], df["has_neutral_face"], normalize='index')

    neutral_face_ct.plot(kind='bar', stacked=True, figsize=(6,4), colormap='Set2')
    plt.title("성별별 무표정 여부 구성비율")
    plt.ylabel("비율")
    plt.xlabel("성별")
    plt.legend(["무표정 아님", "무표정"])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{subfolder}_gender_nuetral_face_stacked.png"))
    plt.close()

    print(f"{subfolder}EDA 결과 저장 완료:", save_dir)