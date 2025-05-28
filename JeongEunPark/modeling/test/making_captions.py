import json
import os
import csv
from pathlib import Path
import glob

label_path_list = ["/HDD/toon_persona/Training/label/TL_01. 생성기",
                    "/HDD/toon_persona/Training/label/TL_02. 증폭기",
                    "/HDD/toon_persona/Training/label/TL_03. 전환기"]

output_csv_path = "/home/jepark/dev/Toon_Persona_eda/JeongEunPark/modeling/toon_caption.csv"
base_image_root = "/HDD/toon_persona/Training/origin"
data_rows = []

# label path 순회하며 data 가져오는 함수
for dir_name in label_path_list:
    label_path = Path(dir_name)

    print(f"{label_path}안의 파일을 처리 시작합니다.")

    json_files = list(label_path.glob("*.json"))

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                label = json.load(f)

            source_path = label['meta']['dataset']['source_path']
            # 상대 경로에서 "01. 생성기/SC102629.JPEG" 추출
            relative_path = Path(source_path).relative_to("../원천")
            # "TS_" 접두사 붙이고 완전 경로 구성
            ts_dir = "TS_" + relative_path.parts[0]
            filename = relative_path.parts[1]
            image_file = str(Path(base_image_root) / ts_dir / filename)
            
            caption = label['caption'].replace('\n', '').replace('\r', '').replace('\t', '').strip()

            data_rows.append([image_file, caption])

        except Exception as e:
            print(f"오류 발생 : {json_file} -> {e}")
            continue

with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['image', 'caption'])  # 헤더
    writer.writerows(data_rows)

print(f"\n CSV 저장 완료: {output_csv_path}")
print(f" 총 {len(data_rows)}개 이미지 처리됨")