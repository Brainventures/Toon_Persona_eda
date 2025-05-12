import json
import glob
import os
import pandas as pd

def load_json_to_dataframe(path):
    file_paths = glob.glob(os.path.join(path, "*.json"))
    data_records = []

    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # === meta 영역 ===
        meta = data.get("meta", {})
        dataset = meta.get("dataset", {})
        product = meta.get("product", {})
        images = meta.get("images", {})

        # === label 영역 ===
        label = data.get("label", {})
        character_info = label.get("character", {}).get("char_info", [])
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
            "dataset_id": dataset.get("id"),
            "dataset_type": dataset.get("type"),
            "source_path": dataset.get("source_path"),
            "label_path": dataset.get("label_path"),
            "caption_path": dataset.get("caption_path"),

            "title": product.get("title"),
            "writer": product.get("writer"),
            "illustrator": product.get("illustrator"),
            "platform": product.get("platform"),
            "company": product.get("company"),
            "post_date": product.get("post"),
            "genre": product.get("genre"),
            "era": product.get("era"),
            "category": product.get("category"),

            "image_type": images.get("type"),
            "image_width": images.get("width"),
            "image_height": images.get("height"),
            "is_sketch": images.get("sketch"),

            "background_exist": background.get("exist"),
            "background_info": background.get("background_info"),

            "angle": composition.get("angle"),
            "lighting": composition.get("lighting"),
            "shot": composition.get("shot"),

            "dialogue_count": len(context_list),
            "dialogue_texts": [c.get("dialogue") for c in context_list],
            "bubble_types": [c.get("bubble") for c in context_list],

            "prompt": prompt,
            "caption": caption,
        }

        # 캐릭터 수만큼 반복
        for char in character_info:
            record = base_info.copy()
            record.update({
                "char_gender": char.get("gender"),
                "char_age": char.get("age"),
                "char_kind": char.get("kind"),
                "char_shape_list": char.get("shape", "").split(","),
                "char_movement_list": char.get("movement", "").split(","),
                "char_clothing_list": char.get("clothing", "").split(","),
                "bbox_x": char.get("bbox", {}).get("x"),
                "bbox_y": char.get("bbox", {}).get("y"),
                "bbox_w": char.get("bbox", {}).get("w"),
                "bbox_h": char.get("bbox", {}).get("h"),
            })
            data_records.append(record)

    df = pd.DataFrame(data_records)
    if not df.empty:
        df["bbox_area"] = df["bbox_w"] * df["bbox_h"]
    return df
