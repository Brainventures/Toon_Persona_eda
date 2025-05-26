import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, PreTrainedTokenizerFast
import torchvision.models as models
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import json

# 한국어 GPT-2 모델 사용을 위한 설정
MODEL_NAME = "skt/kogpt2-base-v2"

class ImageCaptionDataset(Dataset):
    def __init__(self, csv_file, image_dir, tokenizer, transform=None, max_length=128):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length
        
        # 특수 토큰 설정
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.image_dir, row['origin'])
        caption = row['caption']
        
        # 이미지 로드
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # 빈 이미지로 대체
            image = Image.new('RGB', (224, 224), color='white')
            if self.transform:
                image = self.transform(image)
        
        # 캡션 토큰화
        caption_encoded = self.tokenizer(
            caption,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'image': image,
            'caption_ids': caption_encoded['input_ids'].squeeze(),
            'caption_mask': caption_encoded['attention_mask'].squeeze(),
            'caption_text': caption
        }

class ImageCaptionModel(nn.Module):
    def __init__(self, vocab_size, d_model=768, max_seq_len=128):
        super().__init__()
        
        # Vision Encoder (EfficientNet-B0) - 메모리 효율적
        self.vision_encoder = models.efficientnet_b0(pretrained=True)
        # EfficientNet의 마지막 분류 레이어 제거
        self.vision_encoder.classifier = nn.Identity()
        
        # EfficientNet-B0의 출력 차원은 1280
        efficientnet_dim = 1280
        
        # Vision feature를 GPT-2 차원으로 변환
        self.vision_projection = nn.Sequential(
            nn.Linear(efficientnet_dim, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model)
        )
        
        # Adaptive pooling으로 시퀀스 길이 조정 (197 -> 더 작은 수로)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(49)  # 7x7 = 49 토큰으로 줄임
        
        # Text Decoder (GPT-2 based)
        gpt_config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=d_model,
            n_layer=6,  # 레이어 수를 12에서 6으로 줄임 (메모리 절약)
            n_head=12,
            n_positions=max_seq_len,
            add_cross_attention=True,
            use_cache=False
        )
        self.text_decoder = GPT2LMHeadModel(gpt_config)
        
        # pad_token_id 저장
        self.pad_token_id = None
        
    def forward(self, images, caption_ids=None, caption_mask=None):
        # Vision encoding with EfficientNet
        vision_features = self.vision_encoder(images)  # [batch, 1280]
        
        # 차원 확장 및 projection
        vision_features = vision_features.unsqueeze(1)  # [batch, 1, 1280]
        vision_features = self.vision_projection(vision_features)  # [batch, 1, d_model]
        
        # 시퀀스 길이를 늘리기 위해 반복 (cross-attention을 위해)
        vision_features = vision_features.repeat(1, 49, 1)  # [batch, 49, d_model]
        
        if caption_ids is not None:
            # Training mode
            # Text decoding with cross-attention
            text_outputs = self.text_decoder(
                input_ids=caption_ids,
                attention_mask=caption_mask,
                encoder_hidden_states=vision_features,
                encoder_attention_mask=torch.ones(vision_features.shape[:2], device=vision_features.device),
                return_dict=True
            )
            
            # Loss 계산
            if text_outputs.loss is not None:
                return text_outputs.loss, text_outputs.logits
            else:
                # 수동으로 loss 계산
                logits = text_outputs.logits
                labels = caption_ids.clone()
                labels[labels == self.pad_token_id] = -100
                
                # Shift labels for next token prediction
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                # Cross entropy loss 계산
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                return loss, logits
        else:
            # Inference mode
            return vision_features

class ImageCaptionTrainer:
    def __init__(self, model, tokenizer, train_loader, val_loader, device, lr=5e-5):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # pad_token_id를 모델에 전달
        self.model.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(self.train_loader, desc="Training"):
            images = batch['image'].to(self.device)
            caption_ids = batch['caption_ids'].to(self.device)
            caption_mask = batch['caption_mask'].to(self.device)
            
            self.optimizer.zero_grad()
            
            loss, logits = self.model(images, caption_ids, caption_mask)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                images = batch['image'].to(self.device)
                caption_ids = batch['caption_ids'].to(self.device)
                caption_mask = batch['caption_mask'].to(self.device)
                
                loss, logits = self.model(images, caption_ids, caption_mask)
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train(self, epochs=50, save_path="image_caption_model.pth"):
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'tokenizer': self.tokenizer,
                    'epoch': epoch,
                    'val_loss': val_loss
                }, save_path)
                print(f"Best model saved with val_loss: {val_loss:.4f}")
            
            self.scheduler.step()
    
    def plot_losses(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

class ImageCaptionInference:
    def __init__(self, model_path, device):
        self.device = device
        
        # 모델 로드
        # checkpoint = torch.load(model_path, map_location=device)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        self.tokenizer = checkpoint['tokenizer']
        
        # 모델 초기화
        self.model = ImageCaptionModel(
            vocab_size=self.tokenizer.vocab_size,
            d_model=768,
            max_seq_len=128
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # 이미지 전처리
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def generate_caption(self, image_path, max_length=50, temperature=0.7):
        # 이미지 로드 및 전처리
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Vision features 추출
            vision_features = self.model(image)
            
            # 캡션 생성 - 더 간단한 방식으로 수정
            if self.tokenizer.bos_token_id:
                generated_ids = [self.tokenizer.bos_token_id]
            else:
                generated_ids = [self.tokenizer.eos_token_id]
            
            for _ in range(max_length):
                input_ids = torch.tensor([generated_ids]).to(self.device)
                
                # 다음 토큰 예측
                outputs = self.model.text_decoder(
                    input_ids=input_ids,
                    encoder_hidden_states=vision_features,
                    encoder_attention_mask=torch.ones(vision_features.shape[:2], device=self.device),
                    use_cache=False  # 추론 시에도 캐시 사용 안 함
                )
                
                logits = outputs.logits[0, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                
                if next_token == self.tokenizer.eos_token_id:
                    break
                
                generated_ids.append(next_token)
            
            # 토큰을 텍스트로 변환
            caption = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            return caption

def main():
    # 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 데이터 경로 설정
    csv_file = "/home/thkim/dev/eda/Toon_Persona_eda/toon_caption_dataset.csv"
    image_dirs = [
        "/HDD/toon_persona/Training/origin/TL_01. 생성기",
        "/HDD/toon_persona/Training/origin/TL_02. 증폭기", 
        "/HDD/toon_persona/Training/origin/TL_03. 전환기"
    ]
    
    # 모든 이미지 경로를 하나로 통합
    all_image_dir = "/HDD/toon_persona/Training/origin"
    
    # 토크나이저 로드
    # try:
    # tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                bos_token='</s>', eos_token='</s>', unk_token='<unk>',
                pad_token='<pad>', mask_token='<mask>')
    # except:
        # 한국어 모델이 없을 경우 영어 모델 사용
        # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        # print("한국어 GPT-2 모델을 찾을 수 없어 영어 GPT-2 모델을 사용합니다.")
    
    # 데이터 전처리
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 데이터셋 로드 (100개로 제한)
    df = pd.read_csv(csv_file)
    
    # 전체 데이터에서 100개만 샘플링
    if len(df) > 100:
        df = df.sample(n=5000, random_state=42).reset_index(drop=True)
        print(f"데이터를 5000개로 제한했습니다. (원본: {len(pd.read_csv(csv_file))}개)")
    else:
        print(f"전체 데이터가 {len(df)}개입니다.")
    
    # 8:2로 분할 (80개 훈련, 20개 검증)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"훈련 데이터: {len(train_df)}개, 검증 데이터: {len(val_df)}개")
    
    # 임시 CSV 파일 생성
    train_df.to_csv("/home/thkim/dev/eda/Toon_Persona_eda/train_temp.csv", index=False)
    val_df.to_csv("/home/thkim/dev/eda/Toon_Persona_eda/val_temp.csv", index=False)
    
    # 데이터로더 생성 (작은 배치 크기로 설정)
    train_dataset = ImageCaptionDataset("/home/thkim/dev/eda/Toon_Persona_eda/train_temp.csv", all_image_dir, tokenizer, transform)
    val_dataset = ImageCaptionDataset("/home/thkim/dev/eda/Toon_Persona_eda/val_temp.csv", all_image_dir, tokenizer, transform)
    
    # 작은 데이터셋이므로 배치 크기를 더 줄임
    batch_size = 4  # 4에서 2로 줄임 (메모리 절약)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # 모델 초기화
    model = ImageCaptionModel(vocab_size=tokenizer.vocab_size)
    
    # 트레이너 초기화 및 학습 (빠른 테스트를 위해 에포크 수 줄임)
    trainer = ImageCaptionTrainer(model, tokenizer, train_loader, val_loader, device)
    trainer.train(epochs=10, save_path="/home/thkim/dev/eda/Toon_Persona_eda/best_image_caption_model.pth")  # 30에서 10으로 줄임
    
    # 손실 그래프 출력
    trainer.plot_losses()
    
    # 추론 예제
    print("\n=== 추론 예제 ===")
    inference = ImageCaptionInference("/home/thkim/dev/eda/Toon_Persona_eda/best_image_caption_model.pth", device)
    
    # 테스트 이미지로 캡션 생성
    test_images = val_df.head(5)
    for idx, row in test_images.iterrows():
        image_path = os.path.join(all_image_dir, row['origin'])
        if os.path.exists(image_path):
            predicted_caption = inference.generate_caption(image_path)
            print(f"이미지: {row['origin']}")
            print(f"실제 캡션: {row['caption']}")
            print(f"예측 캡션: {predicted_caption}")
            print("-" * 50)
    
    # 임시 파일 정리
    os.remove("/home/thkim/dev/eda/Toon_Persona_eda/train_temp.csv")
    os.remove("/home/thkim/dev/eda/Toon_Persona_eda/val_temp.csv")

if __name__ == "__main__":
    main()