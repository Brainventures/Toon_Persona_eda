import os
import random
from PIL import Image
import torch
from tqdm import tqdm
from transformers import BlipProcessor, Blip2ForConditionalGeneration

# processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")
processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl").to("cuda")

image_folder = "/HDD/toon_persona/Training/origin/TS_01. 생성기"

image_paths = [
    os.path.join(image_folder, fname)
    for fname in os.listdir(image_folder)
    if fname.lower().endswith(".jpeg")
]

random.seed(42)  # 재현 가능성 유지
image_paths = random.sample(image_paths, k=min(100, len(image_paths)))

print(f"<<<<< 선택된 이미지 수: {len(image_paths)}>>>>>")

for path in tqdm(image_paths, desc="이미지 캡션 생성 중"):
    try:
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to("cuda")
        output = model.generate(**inputs, max_new_tokens=50)
        caption = processor.decode(output[0], skip_special_tokens=True)

        print(f" {os.path.basename(path)} → {caption}")

    except Exception as e:
        print(f"❌ 오류 ({path}): {e}")
