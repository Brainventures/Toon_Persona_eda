import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from early_stopping import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train_model(encoder, decoder, train_dataloader, val_dataloader, optimizer, device, num_epochs=1, patience=10, save_dir="state_dict"):
    os.makedirs(save_dir, exist_ok=True)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    early_stopper = EarlyStopping(patience=5, path=save_dir)

    # 학습 및 검증 손실을 저장할 리스트
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        total_loss = 0

        loop = tqdm(train_dataloader, desc=f"[Epoch {epoch+1}]", mininterval=1.0)
        
        for images, input_ids, attention_mask, _  in loop:
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

        avg_loss = total_loss / len(train_dataloader)
        val_loss = evaluate_model(encoder, decoder, val_dataloader, device)

        train_losses.append(avg_loss)
        val_losses.append(val_loss)

        print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f}")
        
        scheduler.step(val_loss)

        if avg_loss <= 0.1:
            early_stopper(val_loss, encoder, decoder)
            if early_stopper.early_stop:
                print(f"[Epoch {epoch+1}] Early stopping triggered")
                break
        else:
            print(f"[Epoch {epoch+1}] 아직 train loss > 0.1 → early stopping 미적용")
        
    # plot 저장
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_plot_ep70.png') 

def evaluate_model(encoder, decoder, val_dataloader, device):
    encoder.eval()
    decoder.eval()
    total_loss = 0

    with torch.no_grad():
        for images, input_ids, attention_mask, _ in val_dataloader:
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            features = encoder(images)
            outputs = decoder(
                features=features,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            loss = outputs.loss
            total_loss += loss.item()

    avg_loss = total_loss / len(val_dataloader)
    return avg_loss