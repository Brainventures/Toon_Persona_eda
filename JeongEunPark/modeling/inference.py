import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast
from model import EncoderCNN, DecoderRNN
from preprocess import normalize_caption
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import csv
from tqdm import tqdm 
import pandas as pd 

def generate_caption(image_path, encoder, decoder, tokenizer, device, max_length=82, beam_width=3, use_beam_search=False):
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
        if not use_beam_search:
            # 기존 Greedy decoding
            generated = [tokenizer.bos_token_id]
            for _ in range(max_length):
                input_ids = torch.tensor([generated]).to(device)
                attention_mask = torch.ones_like(input_ids).to(device)

                outputs = decoder(features, input_ids, attention_mask=attention_mask)
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).item()

                generated.append(next_token)
                if next_token == tokenizer.eos_token_id:
                    break

            caption = tokenizer.decode(generated, skip_special_tokens=True)
            return caption

        else:
            # Beam Search
            beams = [( [tokenizer.bos_token_id], 0 )]  # (토큰 시퀀스, 누적 로그확률)

            for _ in range(max_length):
                new_beams = []
                for seq, score in beams:
                    input_ids = torch.tensor([seq]).to(device)
                    attention_mask = torch.ones_like(input_ids).to(device)

                    outputs = decoder(features, input_ids, attention_mask=attention_mask)
                    logits = outputs.logits[:, -1, :]  # [1, vocab_size]
                    log_probs = F.log_softmax(logits, dim=-1).squeeze(0)  # [vocab_size]

                    topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)

                    for log_prob, idx in zip(topk_log_probs, topk_indices):
                        next_seq = seq + [idx.item()]
                        next_score = score + log_prob.item()
                        new_beams.append((next_seq, next_score))

                # top-k 빔 선택
                beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

                # EOS로 끝나는 시퀀스가 있으면 종료
                if any(seq[-1] == tokenizer.eos_token_id for seq, _ in beams):
                    break

            # 가장 확률 높은 시퀀스 선택
            best_seq = beams[0][0]
            caption = tokenizer.decode(best_seq, skip_special_tokens=True)
            return caption


def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EMBED_SIZE = 256
    TOKENIZER_NAME = "skt/kogpt2-base-v2"

    ENCODER_PATH = "state_dict/encoder_best.pt"
    DECODER_PATH = "state_dict/decoder_best.pt"
    CSV_PATH = "test_data.csv"
    DEBUG_SAMPLES = 5

    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        TOKENIZER_NAME,
        bos_token='</s>',
        eos_token='</s>',
        unk_token='<unk>',
        pad_token='<pad>',
        mask_token='<mask>'
    )
    encoder = EncoderCNN(EMBED_SIZE).to(DEVICE)
    decoder = DecoderRNN(EMBED_SIZE, model_name=TOKENIZER_NAME).to(DEVICE)
    decoder.kogpt2.resize_token_embeddings(len(tokenizer))

    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
    decoder.load_state_dict(torch.load(DECODER_PATH, map_location=DEVICE))

    encoder.eval()
    decoder.eval()

    # test_data.csv 불러오기
    df = pd.read_csv(CSV_PATH)

    # 추론 결과 저장용
    results = []
    MAX_SAMPLES = 50  # 추론할 샘플 수
    print(f"\n[✓] 총 {len(df)}개 테스트 샘플 추론 시작...\n")

    for i, row in tqdm(df.iterrows(), total=MAX_SAMPLES):
        img_path = row["img_path"]
        ref_caption = row["caption"]

        pred_caption = generate_caption(
            image_path=img_path,
            encoder=encoder,
            decoder=decoder,
            tokenizer=tokenizer,
            device=DEVICE,
            max_length=82,
            beam_width=5,
            use_beam_search=True
        )

        results.append({
            "img_path": img_path,
            "reference": ref_caption,
            "generated": pred_caption
        })

    # 결과 저장
    output_df = pd.DataFrame(results)
    output_df.to_csv("inference_results_beamsearch.csv", index=False, encoding="utf-8-sig")
    print("\n[✓] inference_results_beamsearch.csv 저장 완료!")

if __name__ == "__main__":
    main()