import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from transformers import GPT2LMHeadModel, GPT2Config, PreTrainedTokenizerFast
from transformers import ViTModel, ViTConfig  # Vision Transformer
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split    

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
        
        # 캡션을 BOS + caption + EOS 형태로 구성
        caption_text = f"</s>{caption}</s>"  # BOS + caption + EOS
        
        # 캡션 토큰화
        caption_encoded = self.tokenizer(
            caption_text,
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

class VisionTransformerCaptionModel(nn.Module):
    def __init__(self, vocab_size, d_model=768, max_seq_len=128, vit_model_name='google/vit-base-patch16-224', freeze_vit=True):
        super().__init__()
        
        # Vision Transformer Encoder
        print(f"Loading Vision Transformer: {vit_model_name}")
        self.vision_encoder = ViTModel.from_pretrained(vit_model_name)
        
        # ViT freeze 설정
        if freeze_vit:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            print("ViT parameters frozen for training")
        
        # ViT의 hidden size 가져오기
        vit_hidden_size = self.vision_encoder.config.hidden_size
        
        # Vision feature를 GPT-2 차원으로 변환
        self.vision_projection = nn.Sequential(
            nn.Linear(vit_hidden_size, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model)
        )
        
        # Text Decoder (GPT-2 based)
        gpt_config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=d_model,
            n_layer=12,  # 레이어 수 조정 가능
            n_head=12,
            n_positions=max_seq_len,
            add_cross_attention=True,
            use_cache=False
        )
        self.text_decoder = GPT2LMHeadModel(gpt_config)
        
        # pad_token_id 저장
        self.pad_token_id = None
        
    def forward(self, images, caption_ids=None, caption_mask=None):
        # Vision encoding with ViT
        vision_outputs = self.vision_encoder(pixel_values=images)
        
        # ViT는 [CLS] 토큰과 패치 토큰들을 모두 반환
        # last_hidden_state: [batch_size, seq_len, hidden_size]
        # seq_len = 1 (CLS) + num_patches (196 for 224x224 with 16x16 patches)
        vision_features = vision_outputs.last_hidden_state  # [batch, 197, vit_hidden_size]
        
        # Vision features를 GPT-2 차원으로 projection
        vision_features = self.vision_projection(vision_features)  # [batch, 197, d_model]
        
        if caption_ids is not None:
            # Training mode - Teacher Forcing
            # 입력: BOS + caption (EOS 제외)
            # 타겟: caption + EOS (BOS 제외)
            input_ids = caption_ids[:, :-1]  # 마지막 토큰(EOS) 제외
            target_ids = caption_ids[:, 1:]   # 첫 번째 토큰(BOS) 제외
            input_mask = caption_mask[:, :-1]
            
            # Text decoding with cross-attention
            text_outputs = self.text_decoder(
                input_ids=input_ids,
                attention_mask=input_mask,
                encoder_hidden_states=vision_features,
                encoder_attention_mask=torch.ones(vision_features.shape[:2], device=vision_features.device),
                return_dict=True
            )
            
            # Loss 계산 - 다음 토큰 예측
            logits = text_outputs.logits
            
            # 타겟에서 패딩 토큰은 -100으로 설정 (loss 계산에서 제외)
            target_ids = target_ids.clone()
            target_ids[target_ids == self.pad_token_id] = -100
            
            # Cross entropy loss 계산
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            
            return loss, logits
        else:
            # Inference mode - 이미지 특성만 반환
            return vision_features

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, train_loss, model):
        if self.best_loss is None:
            self.best_loss = train_loss
            self.save_checkpoint(model)
        elif train_loss < self.best_loss - self.min_delta:
            self.best_loss = train_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

class ImageCaptionTrainer:
    def __init__(self, model, tokenizer, train_loader, val_loader, device, lr=1e-4, patience=7):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # pad_token_id를 모델에 전달
        self.model.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        
        # ViT를 사용할 때는 더 작은 learning rate 사용
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        
        # Early stopping 초기화
        self.early_stopping = EarlyStopping(patience=patience, min_delta=1e-3)
        
        self.train_losses = []
        self.val_losses = []
        
        # Resume 관련 변수들
        self.start_epoch = 0
        self.best_train_loss = float('inf')
    
    def load_checkpoint(self, checkpoint_path):
        """체크포인트에서 모델과 옵티마이저 상태를 로드합니다."""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # 모델 상태 로드
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 옵티마이저 상태 로드
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer state loaded")
        
        # 스케줄러 상태 로드 (있다면)
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Scheduler state loaded")
        
        # 에포크와 손실 정보 로드
        self.start_epoch = checkpoint.get('epoch', 0) + 1  # 다음 에포크부터 시작
        self.best_train_loss = checkpoint.get('train_loss', float('inf'))
        
        # 손실 기록 로드 (있다면)
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
        if 'val_losses' in checkpoint:
            self.val_losses = checkpoint['val_losses']
        
        # Early stopping 상태 로드 (있다면)
        if 'early_stopping_best_loss' in checkpoint:
            self.early_stopping.best_loss = checkpoint['early_stopping_best_loss']
            self.early_stopping.counter = checkpoint.get('early_stopping_counter', 0)
        
        print(f"Resuming from epoch {self.start_epoch}, best train loss: {self.best_train_loss:.4f}")
        
        return checkpoint
    
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
    
    def train(self, epochs=50, save_path="vit_caption_model.pth", resume_from=None):
        """
        모델을 학습합니다.
        
        Args:
            epochs: 총 학습할 에포크 수
            save_path: 모델을 저장할 경로
            resume_from: 이어서 학습할 체크포인트 경로 (None이면 처음부터 학습)
        """
        
        # 체크포인트에서 이어서 학습하는 경우
        if resume_from and os.path.exists(resume_from):
            self.load_checkpoint(resume_from)
            print(f"Resuming training from epoch {self.start_epoch}")
        else:
            print("Starting training from scratch")
        
        for epoch in range(self.start_epoch, epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # 최고 성능 모델 저장
            if train_loss < self.best_train_loss:
                self.best_train_loss = train_loss
                checkpoint = {
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'tokenizer': self.tokenizer,
                    'vit_model_name': getattr(self.model, 'vit_model_name', 'google/vit-base-patch16-224'),
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'early_stopping_best_loss': self.early_stopping.best_loss,
                    'early_stopping_counter': self.early_stopping.counter
                }
                torch.save(checkpoint, save_path)
                print(f"Best model saved with train_loss: {train_loss:.4f}")
            
            # Early stopping 체크
            if self.early_stopping(train_loss, self.model):
                print(f"Early stopping triggered at epoch {epoch+1}")
                print(f"Best train loss: {self.early_stopping.best_loss:.4f}")
                break
            
            self.scheduler.step()
        
        print("Training completed!")
    
    def plot_losses(self, save_path=None):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss (Vision Transformer)')
        plt.legend()
        plt.grid(True)

         # 파일 저장
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"Loss plot saved to: {save_path}")

        plt.show()

class ImageCaptionInference:
    def __init__(self, model_path, device, vit_model_name=None):
        self.device = device
        
        # 모델 로드
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        self.tokenizer = checkpoint['tokenizer']
        
        # ViT 모델명 확인
        if vit_model_name is None:
            vit_model_name = checkpoint.get('vit_model_name', 'google/vit-base-patch16-224')
        
        print(f"Loading model with ViT: {vit_model_name}")
        
        # 모델 초기화 (추론 시에는 freeze 여부는 중요하지 않음)
        self.model = VisionTransformerCaptionModel(
            vocab_size=self.tokenizer.vocab_size,
            d_model=768,
            max_seq_len=128,
            vit_model_name=vit_model_name,
            freeze_vit=False  # 추론 시에는 상관없음
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # 이미지 전처리 (ViT용)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                               std=[0.5, 0.5, 0.5])  # ViT는 [-1, 1] 범위 사용
        ])
    
    def generate_caption(self, image_path, max_length=50, temperature=0.8):
        # 이미지 로드 및 전처리
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Vision features 추출
            vision_features = self.model(image)
            
            # BOS 토큰으로 시작
            generated_ids = [self.tokenizer.bos_token_id if self.tokenizer.bos_token_id else self.tokenizer.eos_token_id]
            
            for _ in range(max_length):
                input_ids = torch.tensor([generated_ids]).to(self.device)
                
                # 다음 토큰 예측
                outputs = self.model.text_decoder(
                    input_ids=input_ids,
                    encoder_hidden_states=vision_features,
                    encoder_attention_mask=torch.ones(vision_features.shape[:2], device=self.device),
                    use_cache=False
                )
                
                logits = outputs.logits[0, -1, :]
                
                # Temperature scaling
                logits = logits / temperature
                
                # 샘플링
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                
                # EOS 토큰이면 생성 종료
                if next_token == self.tokenizer.eos_token_id:
                    break
                
                generated_ids.append(next_token)
            
            # 토큰을 텍스트로 변환
            caption = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            return caption

def main():
    # 설정
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Vision Transformer 모델 선택
    vit_models = {
        'base': 'google/vit-base-patch16-224',
        'large': 'google/vit-large-patch16-224',
        'deit': 'facebook/deit-base-distilled-patch16-224',
        'swin': 'microsoft/swin-base-patch4-window7-224',
        'beit': 'microsoft/beit-base-patch16-224'
    }
    
    selected_vit = 'large'  # 원하는 모델로 변경 가능
    vit_model_name = vit_models[selected_vit]
    print(f"Using Vision Transformer: {vit_model_name}")
    
    # 데이터 경로 설정
    csv_file = "/home/thkim/dev/eda/Toon_Persona_eda/toon_caption_dataset.csv"
    all_image_dir = "/HDD/toon_persona/Training/origin"
    
    # 토크나이저 로드
    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                bos_token='</s>', eos_token='</s>', unk_token='<unk>',
                pad_token='<pad>', mask_token='<mask>')
    
    # 데이터 전처리 (ViT용)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                           std=[0.5, 0.5, 0.5])  # ViT는 [-1, 1] 범위 사용
    ])
    
    # 데이터셋 로드
    df = pd.read_csv(csv_file)
    print(f"전체 데이터: {len(df)}개")
    
    # 데이터 샘플링 (테스트용)
    if len(df) > 100:
        df = df.sample(n=42897, random_state=42).reset_index(drop=True)  # ViT 테스트용으로 더 적게
        print(f"데이터를 42897개로 제한했습니다.")
    
    # 8:2로 분할
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"훈련 데이터: {len(train_df)}개, 검증 데이터: {len(val_df)}개")
    
    # 임시 CSV 파일 생성
    train_df.to_csv("/home/thkim/dev/eda/Toon_Persona_eda/train_temp_vit.csv", index=False)
    val_df.to_csv("/home/thkim/dev/eda/Toon_Persona_eda/val_temp_vit.csv", index=False)
    
    # 데이터로더 생성
    train_dataset = ImageCaptionDataset("/home/thkim/dev/eda/Toon_Persona_eda/train_temp_vit.csv", all_image_dir, tokenizer, transform)
    val_dataset = ImageCaptionDataset("/home/thkim/dev/eda/Toon_Persona_eda/val_temp_vit.csv", all_image_dir, tokenizer, transform)
    
    batch_size = 32  # ViT는 메모리를 더 많이 사용하므로 배치 크기 줄임
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # 모델 초기화 (ViT freeze 활성화)
    model = VisionTransformerCaptionModel(
        vocab_size=tokenizer.vocab_size, 
        vit_model_name=vit_model_name, 
        freeze_vit=True
    )
    
    # 트레이너 초기화 (ViT는 더 작은 learning rate 사용)
    trainer = ImageCaptionTrainer(model, tokenizer, train_loader, val_loader, device, lr=1e-4, patience=10)
    
    # 체크포인트 경로
    checkpoint_path = f"/home/thkim/dev/eda/Toon_Persona_eda/TaehongKim/model/best_vit_{selected_vit}_caption_model.pth"
    
    # 이어서 학습할지 선택
    resume_training = False  # False로 설정하면 처음부터 학습
    
    if resume_training and os.path.exists(checkpoint_path):
        print("=== 이어서 학습 모드 ===")
        trainer.train(epochs=100, save_path=checkpoint_path, resume_from=checkpoint_path)
    else:
        print("=== 새로운 학습 모드 ===")
        trainer.train(epochs=50, save_path=checkpoint_path)  # ViT는 더 빨리 수렴할 수 있음
    
    # 손실 그래프 출력
    loss_plot_path = f"/home/thkim/dev/eda/Toon_Persona_eda/TaehongKim/vit_{selected_vit}_training_losses.png"
    trainer.plot_losses(loss_plot_path)
    
    # 추론 예제
    print("\n=== 추론 예제 ===")
    inference = ImageCaptionInference(checkpoint_path, device, vit_model_name)
    
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
    os.remove("/home/thkim/dev/eda/Toon_Persona_eda/train_temp_vit.csv")
    os.remove("/home/thkim/dev/eda/Toon_Persona_eda/val_temp_vit.csv")

if __name__ == "__main__":
    main()