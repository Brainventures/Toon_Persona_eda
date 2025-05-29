import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import Resize, ToTensor, Normalize, Compose
import os
from transformers import PreTrainedTokenizerFast

class CustomDataset(Dataset):
    def __init__(self, img_filenames_list, captions_list, tokenizer, max_cap_length=82):
        super().__init__()
        self.img_filenames_list = img_filenames_list
        self.captions_list = captions_list
        self.tokenizer = tokenizer
        self.max_cap_length = max_cap_length
        

        # 이미지 전처리 정의
        self.transform = Compose([
            Resize((224, 224), antialias=True),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225])
        ])
        

    def __len__(self):
        return len(self.captions_list)

    # def get_input_ids(self, sentence):
    #     input_ids = [0] * (self.max_cap_length + 2)
    #     i = 1
    #     for word in sentence.split(" "):
    #         input_ids[i] = self.vocab.word2index.get(word, self.vocab.word2index['EOS'])
    #         i += 1
    #         if i >= self.max_cap_length + 1:
    #             break
    #     input_ids[0] = self.vocab.word2index['SOS']
    #     input_ids[i] = self.vocab.word2index['EOS']
    #     return torch.tensor(input_ids)

    def __getitem__(self, idx):
        img_path = self.img_filenames_list[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new("RGB", (224, 224))
        image = self.transform(image)

        caption = self.captions_list[idx]
        caption_with_token = f"</s>{caption}</s>"
        tokenized = self.tokenizer(
            caption_with_token,
            max_length = self.max_cap_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)

        return image, input_ids, attention_mask