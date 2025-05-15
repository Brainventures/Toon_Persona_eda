import os
import json
from dotenv import load_dotenv
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import Counter
from pprint import pprint as pp
import csv

plt.rc('font', family='NanumGothicCoding')

load_dotenv()

def char_trait_distribution(path, era, img_save_dir):
    results = []
    key_counts = Counter()
    key_values = {
        "gender": Counter(),
        "age": Counter(),
        "kind": Counter(),
        "shape": Counter(),
        "movement": Counter(),
        "clothing": Counter(),
        "props": Counter()
    }
    json_files = glob.glob(os.path.join(path, "*.json"))

    for json_file in tqdm(json_files, desc=f'Processing {os.path.basename(path)}'):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                char_info = data.get('label', {}).get('character', {}).get('char_info', [])
                for char in char_info:
                    for key in char.keys():
                        key_counts[key] += 1
                        if key in key_values.keys():
                            s = char[key].split(',')
                            for item in s:
                                # 특정 파일 로깅
                                if item == '햄버거':
                                    print(f'특정한 파일: {json_file}')
                                key_values[key][item] += 1
                    char_data = {
                        "file": os.path.basename(json_file),
                        "era": era
                    }
                    results.append(char_data)
        except Exception as e:
            print(f'Error processing {json_file}: {e}')
    
    print(f"\n== {era} 키 분포 분석 ==")
    print(key_counts)
    print(f"발견된 키 종류: {len(key_counts)}")
    print(f"키별 출현 횟수: {dict(key_counts)}")
    dic = dict(key_counts)
    del dic['id']
    del dic['bbox']

    return key_values

def merge_key_values(*key_values_list):
    merged = {}
    keys = key_values_list[0].keys()
    for key in keys:
        merged[key] = sum((kv[key] for kv in key_values_list), Counter())
    return merged


generating_path = os.getenv('label_01')
rising_path = os.getenv('label_02')
changing_path = os.getenv('label_03')
img_save_dir = os.getenv('img_save_dir')
csv_save_dir = os.getenv('csv_save_dir')

generating_key_values = char_trait_distribution(generating_path, 'generating', img_save_dir)
rising_key_values = char_trait_distribution(rising_path, 'rising', img_save_dir)
changing_key_values =  char_trait_distribution(changing_path, 'changing', img_save_dir)


merged_key_values = merge_key_values(generating_key_values, rising_key_values, changing_key_values)

# 전체 데이터 로그 찍는 코드
# pp(merged_key_values)

writer = pd.ExcelWriter(os.path.join(csv_save_dir, 'char_trait.xlsx'), engine='xlsxwriter')
for key in merged_key_values:
    dic = dict(merged_key_values[key])
    df = pd.DataFrame(dic, index=[0])
    df.to_excel(writer, sheet_name = key, index=False)

writer.close()
