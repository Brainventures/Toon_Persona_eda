import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.json_loader import load_json_to_dataframe
import os

# df_era1 = load_json_to_dataframe("/HDD/toon_persona/Training/label/TL_01. 생성기/")
# df_era2 = load_json_to_dataframe("/HDD/toon_persona/Training/label/TL_02. 중폭기/")
# df_era3 = load_json_to_dataframe("/HDD/toon_persona/Training/label/TL_03. 전환기/")

# df_all = pd.concat([df_era1, df_era2, df_era3], ignore_index=True)

# ✅ 한글 폰트 설정 함수 불러오기 & 적용
from utils.font_setting import get_korean_font
font_prop = get_korean_font()

df = load_json_to_dataframe("/HDD/toon_persona/Training/label/TL_03. 전환기/")

save_dir = "/home/jepark/dev/Toon_Persona_eda/JeongEunPark/results"
os.makedirs(save_dir, exist_ok=True)

# === 성별 분포 시각화 ===
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="char_gender", order=df["char_gender"].value_counts().index)
plt.title("캐릭터 성별 분포_03", fontproperties=font_prop)
plt.xlabel("성별", fontproperties=font_prop)
plt.ylabel("수", fontproperties=font_prop)
plt.xticks(fontproperties=font_prop)
plt.yticks(fontproperties=font_prop)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "char_gender_dist03.png"))
plt.close()

# === 연령대 분포 시각화 ===
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="char_age", order=df["char_age"].value_counts().index)
plt.title("캐릭터 나이 분포_03", fontproperties=font_prop)
plt.xlabel("나이", fontproperties=font_prop)
plt.ylabel("수", fontproperties=font_prop)
plt.xticks(fontproperties=font_prop)
plt.yticks(fontproperties=font_prop)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "char_age_dist03.png"))
plt.close()

print("EDA 결과 저장 완료:", save_dir)
