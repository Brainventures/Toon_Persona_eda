import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import os

def train_model(encoder, decoder, dataloader, optimizer, device, num_epochs=1):
    encoder.train()
    decoder.train()

    for epoch in range(num_epochs):
        total_loss = 0
        loop = tqdm(dataloader, desc=f"[Epoch {epoch+1}]")
        
        for batch in loop:
            images, input_ids, attention_mask = batch

            images = images.to(device)                 # [B, 3, 224, 224]
            input_ids = input_ids.to(device)           # [B, T]
            attention_mask = attention_mask.to(device) # [B, T]

            optimizer.zero_grad()

            features = encoder(images)

            outputs = decoder(
                features=features,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids  # Hugging Face가 자동으로 CrossEntropyLoss 계산
            )
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch+1}] 평균 Loss: {avg_loss:.4f}")

    # for images, captions in tqdm(dataloader, desc=">>> Training", leave=False):
    #     images, captions = images.to(device), captions.to(device)

    #     encoder_optimizer.zero_grad()
    #     decoder_optimizer.zero_grad()
    #     # 이미지 → 특징 벡터
    #     features = encoder(images)  # [B, embed_size]

    #     # 특징 + 캡션 → 단어 예측
    #     outputs = decoder(
    #         features=features,
    #         input_id=input_ids,
    #         attention_mask=attention_mask,
    #         labels=input_ids 
    #     )  # [B, T, vocab_size]

    #     # Loss 계산 
    #     loss = outputs.loss
    #     loss.backward()

    #     # 추후 optimizer encoder, decoder 나눌 예정
    #     optimizer.step()
    #     decoder_optimizer.step()

    #     total_loss += loss.item()

    # avg_loss = total_loss / len(dataloader)
    # return avg_loss


# def train_model(encoder, decoder, train_loader, vocab_size, device,
#                 num_epochs=10, learning_rate=1e-3):

#     criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0은 padding
#     encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
#     decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

#     for epoch in range(num_epochs):
#         avg_loss = train_one_epoch(
#             encoder, decoder, train_loader, criterion,
#             encoder_optimizer, decoder_optimizer, device
#         )
#         print(f"[Epoch {epoch+1}/{num_epochs}] 평균 Loss: {avg_loss:.4f}")

#         # 저장 경로가 없다면 생성
#         if not os.path.exists("state_dict"):
#             os.makedirs("state_dict")

#         # 모델 저장
#         torch.save(encoder.state_dict(), f"state_dict/encoder_epoch_{epoch}.pt")
#         torch.save(decoder.state_dict(), f"state_dict/decoder_epoch_{epoch}.pt")