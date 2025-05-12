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


load_dotenv()

def image_size_distribution_eda(path, era, img_save_dir):
    results = []
    over_four = 0
    over_five = 0

    json_files = glob.glob(os.path.join(path, "*.json"))
    print(f"{path}에서 {len(json_files)}개의 JSON 파일을 찾았습니다.")

    for json_file in tqdm(json_files, desc=f"Processing {os.path.basename(path)}"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                width = data['meta']['images']['width']
                height = data['meta']['images']['height']
                if height >= 5000:
                    over_five += 1
                elif height >= 4000 and height < 5000:
                    over_four += 1
                results.append({
                    "width": width,
                    "height": height,
                    "resolution":width * height
                })
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    print(f'{era}에서 height가 5000 이상인 img 개수: {over_five}, 4000 이상 5000 이상인 개수: {over_four}')
    df = pd.DataFrame(results)

    plt.figure(figsize=(5, 4))
    sns.scatterplot(data=df, x='width', y='height', alpha=0.6)
    plt.title(f'{era} era Image width vs height')
    plt.xlabel('width')
    plt.ylabel('height')

    plt.savefig(f'{img_save_dir}/{era}_era_image_size_distribution.png', dpi=300)

generating_path = os.getenv('label_01')
rising_path = os.getenv('label_02')
changing_path = os.getenv('label_03')
img_save_dir = os.getenv('img_save_dir')


image_size_distribution_eda(generating_path, 'generating', img_save_dir)
image_size_distribution_eda(rising_path, 'rising', img_save_dir)
image_size_distribution_eda(changing_path, 'changing', img_save_dir)