import json
import glob
import os
import pandas as pd
import numpy as np

def load_json_to_dataframe(path):
    file_paths = glob.glob(os.path.join(path, "*.json"))
    data_records = []

    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # === meta 영역 ===
        meta = data.get("meta", {})
        # dataset = meta.get("dataset", {})
        product = meta.get("product", {})
        images = meta.get("images", {})

        # === label 영역 ===
        label = data.get("label", {})
        char_info = label.get("character", {}).get("char_info", [])
        obj_info = label.get("object", {}).get("obj_info", [])
        char_num = int(label.get("character", {}).get("char_num", 0))
        obj_num = int(label.get("object", {}).get("obj_num", 0))
        background = label.get("background", {})
        directing = label.get("directing", {})
        composition = directing.get("composition", {})
        context_list = directing.get("context", [])

        # === caption & prompt ===
        caption = data.get("caption", "")
        prompt = label.get("prompt", "")

        # 장면 공통 정보
        base_info = {
            "file_path": file_path,
            # "dataset_id": dataset.get("id"),
            # "dataset_type": dataset.get("type"),
            # "source_path": dataset.get("source_path"),
            # "label_path": dataset.get("label_path"),
            # "caption_path": dataset.get("caption_path"),

            "title": str(product.get("title", "")),
            # "writer": product.get("writer"),
            # "illustrator": product.get("illustrator"),
            # "platform": product.get("platform"),
            # "company": product.get("company"),
            # "post_date": product.get("post"),
            "genre": str(product.get("genre", "")),
            "era": str(product.get("era", "")),
            # "category": product.get("category"),

            # "image_type": images.get("type"),
            "width": int(images.get("width", 0) or 0),
            "height": int(images.get("height", 0) or 0),
            # "is_sketch": images.get("sketch"),

            "background_exist": bool(background.get("exist", False)),
            "background_info_list": background.get("background_info", "").split(","),

            # "angle": composition.get("angle"),
            # "lighting": composition.get("lighting"),
            # "shot": composition.get("shot"),

            "dialogue_count": len(context_list),
            "dialogue_texts": [c.get("dialogue") for c in context_list],
            "bubble_types": [c.get("bubble") for c in context_list],

            "prompt": prompt,
            "caption": caption,
        }

        # 캐릭터가 있는 경우
        if char_num > 0:
            for char in char_info:
                record = base_info.copy()
                record.update({
                    "entity_type": "character",
                    "gender": str(char.get("gender", "")),
                    "age": str(char.get("age", "")),
                    "kind": str(char.get("kind", "")),
                    "shape_list": str(char.get("shape", "") or "").split(","),
                    "movement_list": str(char.get("movement", "") or "").split(","),
                    "clothing_list": str(char.get("clothing", "") or "").split(","),
                    "obj_name": "", 
                    "bbox_x": np.float32(char.get("bbox", {}).get("x", 0.0) or 0.0),
                    "bbox_y": np.float32(char.get("bbox", {}).get("y", 0.0) or 0.0),
                    "bbox_w": np.float32(char.get("bbox", {}).get("w", 0.0) or 0.0),
                    "bbox_h": np.float32(char.get("bbox", {}).get("h", 0.0) or 0.0),
                        })
                data_records.append(record)

        # 오브젝트가 있는 경우
        elif obj_num > 0:
            for obj in obj_info:
                record = base_info.copy()
                record.update({
                    "entity_type": "object",
                    "gender": "",         
                    "age": "",
                    "kind": "",
                    "shape_list": [],      
                    "movement_list": [],
                    "clothing_list": [],
                    "obj_name": str(obj.get("obj_name", "")),
                    "bbox_x": np.float32(obj.get("bbox", {}).get("obj_x", 0.0)),
                    "bbox_y": np.float32(obj.get("bbox", {}).get("obj_y", 0.0)),
                    "bbox_w": np.float32(obj.get("bbox", {}).get("obj_w", 0.0)),
                    "bbox_h": np.float32(obj.get("bbox", {}).get("obj_h", 0.0)),
                })
                data_records.append(record)

    df = pd.DataFrame(data_records)
    if not df.empty:
        df["bbox_area"] = df["bbox_w"] * df["bbox_h"]
        df["area"] = df["width"] * df["height"]
    return df
