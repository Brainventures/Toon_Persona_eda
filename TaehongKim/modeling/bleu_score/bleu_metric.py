import pandas as pd
import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.tokenize import word_tokenize
import numpy as np

# NLTK 데이터 다운로드 (처음 실행 시 필요)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def calculate_bleu_scores(csv_file):
    """
    CSV 파일에서 caption(정답)과 predict(예측) 컬럼을 읽어 BLEU score를 계산합니다.
    
    Args:
        csv_file (str): CSV 파일 경로
    
    Returns:
        dict: 다양한 BLEU score 결과
    """
    
    # CSV 파일 읽기
    df = pd.read_csv(csv_file)
    
    # 빈 값이 있는 행 제거
    df = df.dropna(subset=['caption', 'predict'])
    
    print(f"총 {len(df)}개의 데이터를 처리합니다.")
    
    # 토큰화된 문장들을 저장할 리스트
    references = []  # 정답 문장들
    candidates = []  # 예측 문장들
    individual_scores = []
    
    # 각 행에 대해 처리
    for idx, row in df.iterrows():
        # 한국어 텍스트를 공백 기준으로 토큰화 (또는 형태소 분석기 사용 가능)
        reference = row['caption'].strip().split()
        candidate = row['predict'].strip().split()
        
        # BLEU score는 reference를 리스트의 리스트로 받음 (여러 정답 가능)
        references.append([reference])
        candidates.append(candidate)
        
        # 개별 문장 BLEU score 계산
        try:
            individual_bleu = sentence_bleu([reference], candidate)
            individual_scores.append(individual_bleu)
        except:
            individual_scores.append(0.0)
    
    # 전체 코퍼스에 대한 BLEU score 계산
    corpus_bleu_score = corpus_bleu(references, candidates)
    
    # 개별 문장 BLEU score들의 평균
    avg_sentence_bleu = np.mean(individual_scores)
    
    # N-gram별 BLEU score 계산
    bleu_1 = corpus_bleu(references, candidates, weights=(1, 0, 0, 0))
    bleu_2 = corpus_bleu(references, candidates, weights=(0.5, 0.5, 0, 0))
    bleu_3 = corpus_bleu(references, candidates, weights=(0.33, 0.33, 0.33, 0))
    bleu_4 = corpus_bleu(references, candidates, weights=(0.25, 0.25, 0.25, 0.25))
    
    results = {
        'corpus_bleu': corpus_bleu_score,
        'avg_sentence_bleu': avg_sentence_bleu,
        'bleu_1': bleu_1,
        'bleu_2': bleu_2,
        'total_samples': len(df)
    }
    
    return results, individual_scores

def print_results(results):
    """결과를 보기 좋게 출력합니다."""
    print("\n" + "="*50)
    print("BLEU Score 계산 결과")
    print("="*50)
    print(f"총 샘플 수: {results['total_samples']}")
    print("\nN-gram별 BLEU Scores:")
    print(f"BLEU-1 (unigram): {results['bleu_1']:.4f}")
    print(f"BLEU-2 (bigram): {results['bleu_2']:.4f}")

def analyze_individual_scores(individual_scores, top_n=10):
    """개별 점수들을 분석합니다."""
    scores_array = np.array(individual_scores)
    
    print(f"\n개별 문장 BLEU Score 분석:")
    print(f"최고 점수: {scores_array.max():.4f}")
    print(f"최저 점수: {scores_array.min():.4f}")
    print(f"중간값: {np.median(scores_array):.4f}")
    print(f"표준편차: {np.std(scores_array):.4f}")
    
    # 점수 분포
    print(f"\n점수 분포:")
    print(f"0.8 이상: {np.sum(scores_array >= 0.8)} 개 ({np.sum(scores_array >= 0.8)/len(scores_array)*100:.1f}%)")
    print(f"0.6 이상: {np.sum(scores_array >= 0.6)} 개 ({np.sum(scores_array >= 0.6)/len(scores_array)*100:.1f}%)")
    print(f"0.4 이상: {np.sum(scores_array >= 0.4)} 개 ({np.sum(scores_array >= 0.4)/len(scores_array)*100:.1f}%)")
    print(f"0.2 이상: {np.sum(scores_array >= 0.2)} 개 ({np.sum(scores_array >= 0.2)/len(scores_array)*100:.1f}%)")

# 사용 예시
if __name__ == "__main__":
    # CSV 파일 경로를 여기에 입력하세요
    csv_file_path = "/home/thkim/dev/eda/Toon_Persona_eda/output_v5.csv"  # 실제 파일 경로로 변경
    
    try:
        results, individual_scores = calculate_bleu_scores(csv_file_path)
        print_results(results)
        analyze_individual_scores(individual_scores)
        
        # 결과를 CSV로 저장하고 싶다면
        # df_results = pd.DataFrame({
        #     'individual_bleu_scores': individual_scores
        # })
        # df_results.to_csv('bleu_scores_results_v4.csv', index=False)
        
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {csv_file_path}")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")