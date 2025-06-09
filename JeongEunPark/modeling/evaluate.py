from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

def compute_bleu(references, predictions):
    smoothie = SmoothingFunction().method4
    scores = [
        sentence_bleu([ref.split()], pred.split(), smoothing_function=smoothie)
        for ref, pred in zip(references, predictions)
    ]
    return sum(scores) / len(scores)

def compute_rouge(references, predictions):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge1_list, rougel_list = [], []

    for ref, pred in zip(references, predictions):
        scores = scorer.score(ref, pred)
        rouge1_list.append(scores['rouge1'].fmeasure)
        rougel_list.append(scores['rougeL'].fmeasure)

    return sum(rouge1_list)/len(rouge1_list), sum(rougel_list)/len(rougel_list)
