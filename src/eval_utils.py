from collections import Counter
import math

def calculate_bleu_score(reference, hypothesis, max_n=4, smooth_method='exp'):
    """
    Calculate BLEU score for a single reference-hypothesis pair.
    
    Args:
        reference: reference text (string)
        hypothesis: generated text (string)
        max_n: maximum n-gram size (default: 4 for BLEU-4)
        smooth_method: smoothing method ('exp' for exponential, 'add-k' for add-k, 'none' for no smoothing)
    
    Returns:
        BLEU score (float between 0 and 1)
    """
    # Tokenize by splitting on whitespace
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    
    # Handle empty cases
    if not hyp_tokens or not ref_tokens:
        return 0.0
    
    # Calculate brevity penalty
    ref_len = len(ref_tokens)
    hyp_len = len(hyp_tokens)
    
    if hyp_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / hyp_len) if hyp_len > 0 else 0.0
    
    # Calculate n-gram precisions
    precisions = []
    for n in range(1, max_n + 1):
        # Count n-grams in reference
        ref_ngrams = Counter()
        for i in range(len(ref_tokens) - n + 1):
            ngram = tuple(ref_tokens[i:i+n])
            ref_ngrams[ngram] += 1
        
        # Count n-grams in hypothesis
        hyp_ngrams = Counter()
        for i in range(len(hyp_tokens) - n + 1):
            ngram = tuple(hyp_tokens[i:i+n])
            hyp_ngrams[ngram] += 1
        
        # Calculate clipped counts
        clipped_count = 0
        total_count = sum(hyp_ngrams.values())
        
        for ngram, count in hyp_ngrams.items():
            clipped_count += min(count, ref_ngrams.get(ngram, 0))
        
        # Calculate precision for this n
        if total_count > 0:
            precision = clipped_count / total_count
        else:
            precision = 0.0
            
        # Apply smoothing if precision is 0
        if precision == 0.0 and smooth_method != 'none':
            if smooth_method == 'exp':
                # Exponential smoothing (used by SacreBLEU by default)
                # When n-gram precision is 0, use smoothing
                precision = 1.0 / (2.0 * total_count) if total_count > 0 else 0.0
            elif smooth_method == 'add-k':
                # Add-k smoothing
                k = 1
                precision = (clipped_count + k) / (total_count + k) if total_count > 0 else 0.0
        
        precisions.append(precision)
    
    # Calculate geometric mean of precisions
    if all(p > 0 for p in precisions):
        log_sum = sum(math.log(p) for p in precisions)
        geo_mean = math.exp(log_sum / len(precisions))
    else:
        geo_mean = 0.0
    
    # Final BLEU score
    bleu = bp * geo_mean
    return bleu


if __name__ == "__main__":
    # import sacrebleu
    # References: list of reference corpora. 
    # Each reference corpus is a list of strings (one per sentence).
    refs = [
        [
            "there is a cat on the mat look at the beautiful garden"
        ]
    ]

    # Hypotheses: system outputs
    hyps = [
        "the cat is on the mat see the nice garden"
    ]

    # sacrebleu wants references as list-of-lists: [refs1, refs2, ...]
    # bleu = sacrebleu.corpus_bleu(hyps, refs)

    # print("BLEU score:", bleu.score)          # e.g. 46.12
    # print("Precisions:", bleu.precisions)     # n-gram precisions
    # print("BP:", bleu.bp)                     # Brevity penalty
    # print("sys_len:", bleu.sys_len)           # total hypothesis length
    # print("ref_len:", bleu.ref_len)           # total reference length

    # Test custom BLEU implementation
    # Pass the original strings (not tokenized)
    print("\nDebug info:")
    print("Reference string:", repr(refs[0][0]))
    print("Hypothesis string:", repr(hyps[0]))
    print("Reference tokens:", refs[0][0].split())
    print("Hypothesis tokens:", hyps[0].split())
    
    score = calculate_bleu_score(refs[0][0], hyps[0], max_n=4, smooth_method='exp')
    print("\nCustom BLEU (with exp smoothing):", score)
    print("Custom BLEU (%):", score * 100)