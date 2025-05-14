import os
import json
from dotenv import load_dotenv
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib as mpl
from collections import Counter

plt.rc('font', family='NanumGothicCoding')

load_dotenv()

def char_info_list_num(path, era, img_save_dir):
    results = []
    json_files = glob.glob(os.path.join(path, "*.json"))    

    for json_file in tqdm(json_files, desc=f"Processing {os.path.basename(path)}"):
        char_num = 0
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                char_info = data.get('label',{}).get('character',{}).get('char_info',[])
                character = data.get('label',{}).get('character',{})
                char_num = data['label']['character']['char_num']
                results.append(character)
                if len(char_info) != char_num:
                    print('등장인물 수가 일치하지 않습니다.')
                # 구체적인 파일 로그찍기
                if char_num == 4:
                    print(json_file)
        except Exception as e:
            print(f'num of char: {char_num}')
            print(f"Error processing {json_file}: {e}")
    df = pd.DataFrame(results)

    # 등장인물 수 분포 출력
    print(df['char_num'].value_counts())

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='char_num', palette='viridis')
    plt.title(f'{era} 등장인물 수별 데이터 개수')
    plt.xlabel('등장인물 수 (char_num)')
    plt.ylabel('데이터 개수')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # 그래프 저장
    save_path = os.path.join(img_save_dir, f"{era}_character_num.png")
    plt.savefig(save_path)
    plt.show()

    print(f"그래프 저장됨: {save_path}")


generating_path = os.getenv('label_01')
rising_path = os.getenv('label_02')
changing_path = os.getenv('label_03')
img_save_dir = os.getenv('img_save_dir')

char_info_list_num(generating_path, 'generating', img_save_dir)
char_info_list_num(rising_path, 'rising', img_save_dir)
char_info_list_num(changing_path, 'changing', img_save_dir)