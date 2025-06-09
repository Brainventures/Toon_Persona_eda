import csv
import torch
import pandas as pd
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from preprocess import normalize_caption, Vocab
from dataset import CustomDataset
from model import EncoderCNN, DecoderRNN
from train import train_model 
from inference import generate_caption
import json, os
from torch import optim
from transformers import PreTrainedTokenizerFast

CSV_PATH = "/home/jepark/dev/Toon_Persona_eda/JeongEunPark/modeling/toon_caption.csv"
BATCH_SIZE = 32
MAX_SAMPLES = 40000 # 불러올 데이터 개수
NUM_EPOCHS = 70
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TOKENIZER_NAME = "skt/kogpt2-base-v2"
SEED = 42

# Load data
img_filenames_list = []
captions_list = []

with open(CSV_PATH, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)
    for i, row in enumerate(reader):
        if i >= MAX_SAMPLES:
            break
        img_filenames_list.append(row[0])
        captions_list.append(normalize_caption(row[1]))

print(f"총 {len(captions_list)}개 캡션 로드 완료")

max_cap_length = max(len(c.split(" ")) for c in captions_list)

print(f"최대 캡션 길이: {max_cap_length}")

# tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_NAME)
tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_NAME,
  bos_token='</s>', eos_token='</s>', unk_token='<unk>',
  pad_token='<pad>', mask_token='<mask>')


# DataLoader
full_dataset = CustomDataset(
    img_filenames_list=img_filenames_list,
    captions_list=captions_list,
    tokenizer=tokenizer,
    max_cap_length=max_cap_length
)

train_dataset, val_dataset = random_split(full_dataset, [0.9, 0.1], generator=torch.Generator().manual_seed(SEED))
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) # 여러 개의 샘플을 모아서 미니 배치 생성
val_dataloader = DataLoader(val_dataset, batch_size=1)

print("전체 데이터 수:", len(full_dataset))
print("학습 데이터 수:", len(train_dataset))
print("테스트 데이터 수:", len(val_dataset))

EMBED_SIZE = 256
# HIDDEN_SIZE = 512
# VOCAB_SIZE = vocab.nwords

encoder = EncoderCNN(EMBED_SIZE).to(DEVICE)
decoder = DecoderRNN(EMBED_SIZE, model_name=TOKENIZER_NAME).to(DEVICE)
decoder.kogpt2.resize_token_embeddings(len(tokenizer))

print("모델 초기화 완료")
print(f"Encoder: {encoder.__class__.__name__}")
print(f"Decoder: {decoder.__class__.__name__}")

optimizer = optim.AdamW(
    list(encoder.parameters()) + list(decoder.parameters()),
    lr=5e-5,
    weight_decay=0.01 # 모델이 학습 데이터에만 너무 잘 맞고, 테스트 성능이 떨어질 때 넣어주는게 안전
) # 스케줄링 기법 사용

train_model(
    encoder=encoder,
    decoder=decoder,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    optimizer = optimizer,
    device=DEVICE,
    num_epochs=NUM_EPOCHS,
    # learning_rate=1e-3 
)

# print("\n 추론 결과 (샘플 5개):")
# for i in range(5):  # 예시로 3개 추론
#     img_path = test_dataset[i][0]
#     caption = generate_caption(
#         image_path=img_path,
#         encoder=encoder,
#         decoder=decoder,
#         tokenizer=tokenizer,
#         device=DEVICE
#     )
#     print(f"\n이미지: {img_path}")
#     print(f"생성된 캡션: {caption}")