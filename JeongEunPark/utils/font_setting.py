# import matplotlib.pyplot as plt
# import matplotlib.font_manager as fm

# def set_korean_font():
#     font_path = "/home/jepark/dev/Toon_Persona_eda/JeongEunPark/.fonts/malgun.ttf"
#     fontprop = fm.FontProperties(fname=font_path)
#     plt.rcParams['font.family'] = fontprop.get_name()
#     plt.rcParams['axes.unicode_minus'] = False

import matplotlib.font_manager as fm

def get_korean_font():
    font_path = "/home/jepark/dev/Toon_Persona_eda/JeongEunPark/.fonts/malgun.ttf"
    return fm.FontProperties(fname=font_path)
