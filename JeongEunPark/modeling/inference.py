import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from modeling import EncoderCNN, DecoderRNN
from preprocess import Vocab, normalize_caption
import csv
import os


def build_vocab_from_csv(csv_path, max_samples=100):
    captions = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            captions.append(normalize_caption(row[1]))
            if len(captions) >= max_samples:
                break
    vocab = Vocab()
    for cap in captions:
        vocab.build_vocab(cap)
    return vocab


def generate_caption(image_path, encoder_path, decoder_path, vocab, device):
    transform = Compose([
        Resize((224, 224), antialias=True),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    embed_size = 256
    hidden_size = 512
    vocab_size = vocab.nwords

    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size).to(device)

    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        features = encoder(image_tensor)
        hidden = features.unsqueeze(0)
        input_token = torch.tensor([[vocab.word2index['SOS']]], device=device, dtype=torch.long)
        generated = []

        for _ in range(30):
            output, hidden = decoder.decode_step(input_token, hidden)
            predicted_id = output.argmax(-1)
            word = vocab.index2word[predicted_id.item()]
            if word == 'EOS':
                break
            generated.append(word)
            input_token = predicted_id

    return " ".join(generated)