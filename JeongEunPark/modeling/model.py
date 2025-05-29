import torch
import torch.nn as nn
import torchvision.models as models
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

# 이미지 -> 임베딩
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__() # 부모 클래스를 초기화한 후, 모델의 레이어와 파라미터를 초기화
        # 사전학습된 ResNet50 불러오기
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        modules = list(resnet.children())[:-1]  # 마지막 FC 레이어 제거(분류 레이어)
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size) # [B, 2048]벡터를 embed_size로 줄이는 선형 변환 레이어
        self.bn = nn.BatchNorm1d(embed_size) # 배치 정규화로 학습 안정화(gradient 흐름 개선)

    def forward(self, images): # 입력 데이터에 대한 연산을 정의 < 여기서 images는 PIL 이미지가 아니라 이미 전처리된 pytorch Tensor
        with torch.no_grad():  # ResNet 가중치를 학습하지 않기 위해 파라미터 고정
            features = self.resnet(images)  # [B, 2048, 1, 1] 이미지수, 채널 수, 이미지 크기
        features = features.view(features.size(0), -1)  # [B, 2048] 이미지수, 채널수*이미지 넓이
        features = self.bn(self.linear(features))       # [B, embed_size]
        return features

# KoGP2로 변경 
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, model_name='skt/kogpt2-base-v2'): # config 설정
        super().__init__()
        self.kogpt2 = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
        self.image_proj = nn.Linear(embed_size, self.kogpt2.config.n_embd) # 이미지 임베딩을 KoGPT의 hidden size로 변환

        # self.linear = nn.Linear(2048, self.gpt2.config.n_embd)
        # self.embed = nn.Embedding(vocab_size, embed_size)
        # self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        # self.linear = nn.Linear(hidden_size, vocab_size)
        # self.dropout = nn.Dropout(0.3)
        # self.relu = nn.ReLU()

    """
    KoGPT2는 Transformer 기반 GPT 모델 
    - features : [B, embed_size] encoder의 output
    - input_ids : 토크나이저로 토큰화된 인덱스 시퀀스 [B, T]
    - attention_mask : 어떤 토큰이 유효한지 마스킹하는 벡터
    - labels : 정답 시퀀스 보통 input_ids와 동일
    """

    def forward(self, features, input_ids, attention_mask=None, labels=None):
        B = input_ids.size(0)
        # 이미지 특징을 KoGPT2의 임베딩 차원으로 투영
        image_embeds = self.image_proj(features) # [B, hidden_size]
        # 입력 토큰을 koGPT2의 임베딩으로 변환
        token_embeds = self.kogpt2.transformer.wte(input_ids) # [B, T, hidden_size]
        # 이미지 벡터를 첫 위치에 삽입
        image_embeds = image_embeds.unsqueeze(1) # [B, T, hidden_size]
        inputs_embeds = torch.cat([image_embeds, token_embeds[:, :-1]], dim=1) # [B, T, hidden_size]

        # attention mask 앞에 1추가
        if attention_mask is not None:
            attention_mask = torch.cat([
                torch.ones((B, 1), dtype=attention_mask.dtype, device=attention_mask.device),
                attention_mask[:, :-1]
            ], dim=1)

        # KoGPT2 forward
        outputs = self.kogpt2(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )

        return outputs
        # embeddings = self.embed(captions[:, :-1])  # [B, T-1, embed_size]
        # embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        # lstm_out, _ = self.lstm(embeddings)
        # outputs = self.linear(self.dropout(lstm_out))
        # return outputs

    # def decode_step(self, input_token, hidden):
    #     embedded = self.embed(input_token)              # [1, 1, embed_size]
    #     embedded = self.relu(embedded)
    #     output, hidden = self.lstm(embedded, hidden)    
    #     output = self.linear(output.squeeze(1))         # [1, vocab_size]
    #     return output, hidden