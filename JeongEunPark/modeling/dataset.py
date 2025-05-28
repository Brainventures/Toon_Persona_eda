import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import Resize, ToTensor, Normalize, Compose
import os

class CustomDataset(Dataset):
    def __init__(self, img_filenames_list, captions_list, vocab, max_cap_length):
        super().__init__()
        self.img_filenames_list = img_filenames_list
        self.captions_list = captions_list
        self.length = len(self.captions_list)
        self.transform = Compose([
            Resize((224, 224), antialias=True),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.vocab = vocab
        self.max_cap_length = max_cap_length

    def __len__(self):
        return self.length

    def get_input_ids(self, sentence):
        input_ids = [0] * (self.max_cap_length + 2)
        i = 1
        for word in sentence.split(" "):
            input_ids[i] = self.vocab.word2index.get(word, self.vocab.word2index['EOS'])
            i += 1
            if i >= self.max_cap_length + 1:
                break
        input_ids[0] = self.vocab.word2index['SOS']
        input_ids[i] = self.vocab.word2index['EOS']
        return torch.tensor(input_ids)

    def __getitem__(self, idx):
        img_path = self.img_filenames_list[idx]
        caption = self.captions_list[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            img = Image.new("RGB", (224, 224))
        img = self.transform(img)
        caption_tensor = self.get_input_ids(caption)