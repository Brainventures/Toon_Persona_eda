import json
import os
import csv
from pathlib import Path

def extract_image_filename_from_path(source_path):
    """
    source_path에서 이미지 파일명을 추출합니다.
    예: "../원천/03. 전환기/ST091254.JPEG" -> "ST091254.JPEG"
    """
    return os.path.basename(source_path)

def process_label_files(label_directories, output_csv_path, base_path):
    """
    라벨 디렉토리들을 순회하며 JSON 파일들을 읽어 CSV 파일을 생성합니다.
    
    Args:
        label_directories (list): 라벨 디렉토리 경로들의 리스트
        output_csv_path (str): 출력할 CSV 파일 경로
    """
    data_rows = []
    
    for label_dir in label_directories:
        label_path = Path(label_dir)
        
        if not label_path.exists():
            print(f"경고: 디렉토리가 존재하지 않습니다: {label_dir}")
            continue
            
        print(f"처리 중: {label_dir}")
        
        # JSON 파일들을 찾아서 처리
        json_files = list(label_path.glob("*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # source_path에서 이미지 파일명 추출
                source_path = data['meta']['dataset']['source_path']
                image_filename = extract_image_filename_from_path(source_path)
                relative_path = os.path.relpath(source_path, start='../원천')
                # 'TS_' 접두사 추가
                new_relative_path = os.path.join('TS_' + relative_path)
                # base_path와 결합
                final_path = os.path.join(base_path, new_relative_path)
                # caption 추출 및 정리
                caption = data['caption']
                # 개행 문자, 탭, 불필요한 공백 제거
                caption = caption.replace('\n', '').replace('\r', '').replace('\t', '').strip()
                
                # 데이터 행 추가
                data_rows.append([final_path, caption])
                
            except Exception as e:
                print(f"오류 발생 - 파일: {json_file}, 오류: {str(e)}")
                continue
    
    # CSV 파일로 저장
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        
        # 헤더 작성
        writer.writerow(['origin', 'caption'])
        
        # 데이터 행들 작성
        writer.writerows(data_rows)
    
    print(f"CSV 파일이 생성되었습니다: {output_csv_path}")
    print(f"총 {len(data_rows)}개의 데이터가 처리되었습니다.")

def main():
    # 라벨 디렉토리 경로들
    label_directories = [
        "/HDD/toon_persona/Training/label/TL_01. 생성기",
        "/HDD/toon_persona/Training/label/TL_02. 증폭기", 
        "/HDD/toon_persona/Training/label/TL_03. 전환기"
    ]
    base_path = '/HDD/toon_persona/Training/origin'
    # 출력 CSV 파일 경로
    output_csv_path = "/home/thkim/dev/eda/Toon_Persona_eda/toon_caption_dataset.csv"
    
    # CSV 파일 생성
    process_label_files(label_directories, output_csv_path, base_path)

if __name__ == "__main__":
    main()