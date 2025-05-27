import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import os

def train_one_epoch(encoder, decoder, dataloader, criterion, encoder_optimizer, decoder_optimizer, device):
    encoder.train()
    decoder.train()
    total_loss = 0

    for images, captions in tqdm(dataloader, desc=">>> Training", leave=False):
        images, captions = images.to(device), captions.to(device)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        # 이미지 → 특징 벡터
        features = encoder(images)  # [B, embed_size]

        # 특징 + 캡션 → 단어 예측
        outputs = decoder(features, captions[:, :-1])  # [B, T, vocab_size]
        targets = captions[:, 1:]

        # Loss 계산 (정답: captions[:, 1:])
        loss = criterion(
            outputs.view(-1, outputs.size(2)), 
            targets.reshape(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def train_model(encoder, decoder, train_loader, vocab_size, device,
                num_epochs=10, learning_rate=1e-3):

    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0은 padding
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        avg_loss = train_one_epoch(
            encoder, decoder, train_loader, criterion,
            encoder_optimizer, decoder_optimizer, device
        )
        print(f"[Epoch {epoch+1}/{num_epochs}] 평균 Loss: {avg_loss:.4f}")

        # 저장 경로가 없다면 생성
        if not os.path.exists("state_dict"):
            os.makedirs("state_dict")

        # 모델 저장
        torch.save(encoder.state_dict(), f"state_dict/encoder_epoch_{epoch}.pt")
        torch.save(decoder.state_dict(), f"state_dict/decoder_epoch_{epoch}.pt")