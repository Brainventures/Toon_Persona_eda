import torch
import os

class EarlyStopping:
    """
        patience: 성능이 개선되지 않아도 몇 에폭 기다릴지 설정
        delta: 개선으로 간주할 최소한의 loss 감소 수치
        path: 모델을 저장할 디렉토리 경로
        verbose: 로그를 출력할지 여부
    """
    def __init__(self, patience=8, delta=0.0, path="state_dict", verbose=True):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.delta = delta
        self.early_stop = False
        self.verbose = verbose
        self.path = path
    # 매 에폭마다 호출되어 성능 비교
    def __call__(self, avg_loss, encoder, decoder):
        if self.best_loss is None or avg_loss < self.best_loss - self.delta:
            self.best_loss = avg_loss
            self.counter = 0
            self.save_checkpoint(encoder, decoder)
        else:
            self.counter += 1
            if self.verbose:
                print(f">>>>> EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, encoder, decoder):
        torch.save(encoder.state_dict(), os.path.join(self.path, "encoder_best_ep70.pt"))
        torch.save(decoder.state_dict(), os.path.join(self.path, "decoder_best_ep70.pt"))
        if self.verbose:
            print(f"✅ Validation loss improved. Saved models to '{self.path}'")