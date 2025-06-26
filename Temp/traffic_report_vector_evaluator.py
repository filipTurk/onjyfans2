"""
traffic_report_vector_evaluator.py

A script to compare two traffic report texts using sentence embeddings and cosine similarity.
Uses the sentence-transformers library and a multilingual model suitable for Slovenian.

Usage (from command line):
    python traffic_report_vector_evaluator.py "text1" "text2"

Or import and use the compare_reports function in your own code.
"""

import sys
from sentence_transformers import SentenceTransformer, util

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# Load model
model = SentenceTransformer(MODEL_NAME)

def compare_reports(text1: str, text2: str, verbose: bool = True) -> float:
    """
    Compute cosine similarity between two texts using sentence embeddings.
    Returns a float between 0 and 1 (1 = identical, ~0.8-0.95 = semantically similar, <0.7 = diverging).
    """
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    similarity = util.cos_sim(emb1, emb2).item()
    if verbose:
        
        print(f"Cosine similarity: {similarity:.4f}")
        
    return similarity

if __name__ == "__main__":
    if len(sys.argv) == 3:
        text1 = sys.argv[1]
        text2 = sys.argv[2]
        compare_reports(text1, text2)
    else:

        h = ""  # human report
        ai = ""  # AI report
        compare_reports(h, ai)
