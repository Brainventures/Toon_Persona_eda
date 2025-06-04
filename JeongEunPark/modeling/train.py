import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import os

def train_model(encoder, decoder, dataloader, optimizer, device, num_epochs=1, patience=3, save_dir="state_dict"):
    encoder.train()
    decoder.train()

    # Early stopping 관련 변수
    best_loss = float('inf')  # 최적의 손실값을 추적
    epochs_without_improvement = 0  # 성능 개선이 없었던 에폭 수

    for epoch in range(num_epochs):
        total_loss = 0
        loop = tqdm(dataloader, desc=f"[Epoch {epoch+1}]")
        
        for step,batch in enumerate(loop):
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
            if step % 10 == 0:  # 매 10 step마다만 출력
                loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch+1}] 평균 Loss: {avg_loss:.4f}")

        # Early stopping 조건 체크
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_without_improvement = 0
            # 성능이 개선되었으므로 모델 저장
            torch.save(encoder.state_dict(), f"{save_dir}/encoder_best_2.pt")
            torch.save(decoder.state_dict(), f"{save_dir}/decoder_best_2.pt")
            print(f"[Epoch {epoch+1}] 성능 개선, 모델 저장 완료 (Loss: {avg_loss:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"[Epoch {epoch+1}] 성능 향상 없음 (연속 개선 없는 에폭: {epochs_without_improvement}/{patience})")

        # Early stopping 조건
        if epochs_without_improvement >= patience:
            print(f"[Epoch {epoch+1}] Early stopping triggered! 성능 향상이 없으므로 학습을 종료합니다.")
            break