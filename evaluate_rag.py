import fitz
import re
import hashlib
import time
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi


# ==========================================
# 1. Data Processing and Chunking
# ==========================================
def clean_text(t):
    t = t.replace("\u00ad", "")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


print(" [Phase 1] Reading PDF and performing chunking...")
pdf = fitz.open("test_manual.pdf")  # Ensure filename is correct
pages = [
    {"page": i + 1, "text": clean_text(pdf[i].get_text("text"))}
    for i in range(len(pdf))
]

chunks = []
for p in pages:
    text = p["text"]
    if not text:
        continue
    start, max_chars, overlap = 0, 1200, 150
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append({"page": p["page"], "text": text[start:end].strip()})
        if end == len(text):
            break
        start = end - overlap
print(f" Successfully generated {len(chunks)} chunks.")

# ==========================================
# 2. Dual Engine Initialization (BM25 & FAISS)
# ==========================================
print("\n [Phase 2] Initializing retrieval engines...")


# --- Sparse (BM25) Initialization ---
def tokenize(text):
    return re.findall(r"[A-Za-z0-9_./-]+", text.lower())


bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
print(" BM25 (Sparse) engine ready!")

# --- Dense (FAISS) Initialization ---
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
corpus_embeddings = embedder.encode([c["text"] for c in chunks], convert_to_numpy=True)
index = faiss.IndexFlatL2(corpus_embeddings.shape[1])
index.add(corpus_embeddings)
print(" FAISS (Dense) engine ready!")

# ==========================================
# 3. 23-Question Golden Dataset
# ==========================================
eval_set_pages = [
    {"question": "What is a tree object?", "gold_pages": {54}},
    {"question": "What does git diff compare?", "gold_pages": {14}},
    {"question": "What is a commit object?", "gold_pages": {33}},
    {"question": "What does git fsck do?", "gold_pages": {20}},
    {"question": "What is a blob object?", "gold_pages": {48}},
    {"question": "What does git checkout-index do?", "gold_pages": {43}},
    {"question": "What does git update-index do?", "gold_pages": {42}},
    {"question": "What does git write-tree do?", "gold_pages": {42}},
    {"question": "How do you show an object's type?", "gold_pages": {44}},
    {"question": "What does git cat-file do?", "gold_pages": {44}},
    {"question": "What does git merge-base do?", "gold_pages": {44}},
    {"question": "What does git submodule add <repo> <path> do?", "gold_pages": {40}},
    {"question": "What is a commit?", "gold_pages": {49}},
    {"question": "What does git log master.test show?", "gold_pages": {10}},
    {"question": "What does git diff master.test produce?", "gold_pages": {10}},
    {"question": "What does git diff master.test do?", "gold_pages": {10}},
    {"question": "What does git format-patch master.test do?", "gold_pages": {10}},
    {"question": "How do you create a lightweight tag?", "gold_pages": {9}},
    {"question": "What is a blob object?", "gold_pages": {48}},
    {"question": "What is a branch in Git?", "gold_pages": {48}},
    {"question": "What is a ref?", "gold_pages": {53}},
    {"question": "What is HEAD?", "gold_pages": {49}},
    {"question": "What is a tag object?", "gold_pages": {54}},
]

# ==========================================
# 4. Automated Evaluation (A/B Testing)
# ==========================================
print("\n [Phase 3] Starting 23-question A/B testing evaluation (K=5)...")
k_val = 5
bm25_correct = 0
faiss_correct = 0
bm25_total_time = 0
faiss_total_time = 0

for item in eval_set_pages:
    q = item["question"]
    gold = set(item["gold_pages"])

    # Test BM25 (Sparse)
    t0 = time.perf_counter()
    bm25_scores = bm25.get_scores(tokenize(q))
    bm25_top_idx = sorted(
        range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
    )[:k_val]
    bm25_got = {chunks[i]["page"] for i in bm25_top_idx}
    bm25_total_time += time.perf_counter() - t0
    if len(gold & bm25_got) > 0:
        bm25_correct += 1

    # Test FAISS (Dense)
    t0 = time.perf_counter()
    q_vec = embedder.encode([q], convert_to_numpy=True)
    distances, indices = index.search(q_vec, k_val)
    faiss_got = {chunks[i]["page"] for i in indices[0]}
    faiss_total_time += time.perf_counter() - t0
    if len(gold & faiss_got) > 0:
        faiss_correct += 1

# Calculate final metrics
num_q = len(eval_set_pages)
bm25_recall = (bm25_correct / num_q) * 100
faiss_recall = (faiss_correct / num_q) * 100
bm25_avg_time = bm25_total_time / num_q
faiss_avg_time = faiss_total_time / num_q

# ==========================================
# 5. Output Comparison Table (Ready for Poster)
# ==========================================
print("\n" + "=" * 65)
print(f" [Final Experimental Report] Sparse vs. Dense Retrieval (K={k_val})")
print("=" * 65)
print(f"{'Metrics':<25} | {'BM25 (Sparse)':<15} | {'FAISS (Dense)':<15}")
print("-" * 65)
print(f"{'Correct Hits':<25} | {bm25_correct:<15} | {faiss_correct:<15}")
print(
    f"{'Retrieval Accuracy (Recall@5)':<25} | {bm25_recall:>5.1f}%          | {faiss_recall:>5.1f}%"
)
print(
    f"{'Average Retrieval Latency':<25} | {bm25_avg_time:>6.4f} sec    | {faiss_avg_time:>6.4f} sec"
)
print("=" * 65)
