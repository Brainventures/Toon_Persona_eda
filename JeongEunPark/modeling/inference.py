import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from transformers import PreTrainedTokenizerFast
from model import EncoderCNN, DecoderRNN

def generate_caption(image_path, encoder, decoder, tokenizer, device, max_length=82):
    transform = Compose([
        Resize((224, 224), antialias=True),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], 
                  std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    encoder.eval()
    decoder.eval()
    # 이미지 임베딩
    with torch.no_grad():
        features = encoder(image)  # [1, embed_size]
        generated = [tokenizer.bos_token_id]  # 시작 토큰

        for _ in range(max_length):
            input_ids = torch.tensor([generated]).to(device)  # [1, T]
            attention_mask = torch.ones_like(input_ids).to(device)

            outputs = decoder(
                features, input_ids,
                attention_mask=attention_mask
            )
            next_token_logits = outputs.logits[:, -1, :]  # [1, vocab_size]
            next_token = torch.argmax(next_token_logits, dim=-1).item()

            if next_token == tokenizer.eos_token_id:
                break
            generated.append(next_token)

    # 토큰 시퀀스를 텍스트로 변환
    caption = tokenizer.decode(generated, skip_special_tokens=True)
    return caption

    # def build_vocab_from_csv(csv_path, max_samples=100):
#     captions = []
#     with open(csv_path, 'r', encoding='utf-8') as f:
#         reader = csv.reader(f)
#         next(reader)
#         for row in reader:
#             captions.append(normalize_caption(row[1]))
#             if len(captions) >= max_samples:
#                 break
#     vocab = Vocab()
#     for cap in captions:
#         vocab.build_vocab(cap)
#     return vocab