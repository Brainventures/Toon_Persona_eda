import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        # 사전학습된 ResNet50 불러오기
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        modules = list(resnet.children())[:-1]  # 마지막 FC 레이어 제거
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)

    def forward(self, images):
        with torch.no_grad():  # ResNet 파라미터 고정
            features = self.resnet(images)  # [B, 2048, 1, 1]
        features = features.view(features.size(0), -1)  # [B, 2048]
        features = self.bn(self.linear(features))       # [B, embed_size]
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, features, captions):
        embeddings = self.embed(captions[:, :-1])  # [B, T-1, embed_size]
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        lstm_out, _ = self.lstm(embeddings)
        outputs = self.linear(self.dropout(lstm_out))
        return outputs

    def decode_step(self, input_token, hidden):
        embedded = self.embed(input_token)              # [1, 1, embed_size]
        embedded = self.relu(embedded)
        output, hidden = self.lstm(embedded, hidden)    
        output = self.linear(output.squeeze(1))         # [1, vocab_size]
        return output, hidden
