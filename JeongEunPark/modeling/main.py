import csv
import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from preprocess import normalize_caption, Vocab
from dataset import CustomDataset
from modeling import EncoderCNN, DecoderRNN
from train import train_model 
from inference import generate_caption
import json, os

CSV_PATH = "/home/jepark/dev/Toon_Persona_eda/JeongEunPark/modeling/toon_caption.csv"
BATCH_SIZE = 16
MAX_SAMPLES = 100 # 불러올 데이터 개수
NUM_EPOCHS = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# vocab
vocab = Vocab()
for caption in captions_list:
    vocab.build_vocab(caption)

print(f"\nVocab 크기: {vocab.nwords}")

# max caption length
max_cap_length = max(len(c.split(" ")) for c in captions_list)

print(f"최대 캡션 길이: {max_cap_length}")

# DataLoader
full_dataset = CustomDataset(
    img_filenames_list=img_filenames_list,
    captions_list=captions_list,
    vocab=vocab,
    max_cap_length=max_cap_length
)

train_dataset, test_dataset = random_split(full_dataset, [0.9, 0.1])
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1)

print("전체 데이터 수:", len(full_dataset))
print("학습 데이터 수:", len(train_dataset))
print("테스트 데이터 수:", len(test_dataset))

EMBED_SIZE = 256
HIDDEN_SIZE = 512
VOCAB_SIZE = vocab.nwords

encoder = EncoderCNN(EMBED_SIZE).to(DEVICE)
decoder = DecoderRNN(EMBED_SIZE, HIDDEN_SIZE, VOCAB_SIZE).to(DEVICE)

print("모델 초기화 완료")
print(f"Encoder: {encoder.__class__.__name__}")
print(f"Decoder: {decoder.__class__.__name__}")

train_model(
    encoder=encoder,
    decoder=decoder,
    train_loader=train_dataloader,
    vocab_size=VOCAB_SIZE,
    device=DEVICE,
    num_epochs=NUM_EPOCHS,
    learning_rate=1e-3
)

# 모델 저장
if not os.path.exists("state_dict"):
    os.makedirs("state_dict")
torch.save(encoder.state_dict(), "state_dict/encoder_epoch_0.pt")
torch.save(decoder.state_dict(), "state_dict/decoder_epoch_0.pt")

print("\n 추론 결과 (샘플 3개):")
for i in range(3):  # 예시로 3개 추론
    img_path = img_filenames_list[i]
    caption = generate_caption(
        image_path=img_path,
        encoder_path="state_dict/encoder_epoch_0.pt",
        decoder_path="state_dict/decoder_epoch_0.pt",
        vocab=vocab,
        device=DEVICE
    )
    print(f"\n이미지: {img_path}")
    print(f"생성된 캡션: {caption}")