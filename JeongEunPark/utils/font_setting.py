import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

def set_korean_font():
    font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf" 
    if os.path.exists(font_path):
        fontprop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = fontprop.get_name()
        plt.rcParams['axes.unicode_minus'] = False
        print(f"[INFO] 한글 폰트 설정 완료: {fontprop.get_name()}")
    else:
        print(f"[WARN] 지정한 폰트 경로가 존재하지 않습니다: {font_path}")

# import matplotlib.font_manager as fm

# def get_korean_font():
#     font_path = "/home/jepark/dev/Toon_Persona_eda/JeongEunPark/.fonts/malgun.ttf"
#     return fm.FontProperties(fname=font_path)