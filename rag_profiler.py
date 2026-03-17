import fitz
import re
import json
import hashlib
import time
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss  # Meta's open-source high-performance vector search library


# ==========================================
# 1. Data Processing and Chunking
# ==========================================
def clean_text(t):
    t = t.replace("\u00ad", "")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def make_chunk_id(doc_id, page, idx, text):
    h = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    return f"{doc_id}_p{page}_c{idx}_{h}"


print("Reading PDF...")
pdf = fitz.open("test_manual.pdf")  # Ensure this PDF exists in your directory
pages = []
for i in range(len(pdf)):
    text = pdf[i].get_text("text")
    pages.append({"page": i + 1, "text": clean_text(text)})

chunks = []
idx = 0
for p in pages:
    text = p["text"]
    if not text:
        continue
    start, max_chars, overlap = 0, 1200, 150
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk_text = text[start:end].strip()
        chunks.append(
            {
                "chunk_id": make_chunk_id("git_manual", p["page"], idx, chunk_text),
                "page": p["page"],
                "text": chunk_text,
            }
        )
        idx += 1
        if end == len(text):
            break
        start = end - overlap

print(f"Successfully generated {len(chunks)} chunks.")

# ==========================================
# 2. Dense Retrieval (Embedding + FAISS) Initialization
# ==========================================
print("Loading Embedding model and vectorizing (takes a few dozen seconds)...")
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

corpus_texts = [c["text"] for c in chunks]
corpus_embeddings = embedder.encode(corpus_texts, convert_to_numpy=True)

dimension = corpus_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(corpus_embeddings)
print("FAISS vector database built successfully!")


def dense_retrieve(query: str, k: int = 5):
    query_vector = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vector, k)
    return [chunks[i] for i in indices[0]]


# ==========================================
# 3. Core Function: Standard Ollama RAG Generation
# ==========================================
def rag_answer_ollama_dense(question: str, k_inject: int = 3):
    t0_search = time.perf_counter()
    retrieved_chunks = dense_retrieve(question, k=k_inject)
    t_search = time.perf_counter() - t0_search

    t0_prompt = time.perf_counter()
    src_blocks = [f"[Page {c['page']}]\n{c['text'][:500]}" for c in retrieved_chunks]
    context_text = "\n\n".join(src_blocks)

    prompt = f"Answer ONLY using the SOURCES. Answer concisely.\n\nSOURCES:\n{context_text}\n\nQUESTION: {question}\nANSWER:"
    t_prompt = time.perf_counter() - t0_prompt

    payload = {
        "model": "phi3",
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 100},
    }

    response = requests.post("http://localhost:11434/api/generate", json=payload).json()

    t_prefill = response.get("prompt_eval_duration", 0) / 1e9
    t_decode = response.get("eval_duration", 0) / 1e9

    return {
        "answer": response.get("response", "").strip(),
        "t_search": t_search,
        "t_prompt": t_prompt,
        "t_prefill": t_prefill,
        "t_decode": t_decode,
        "retrieved_pages": [c["page"] for c in retrieved_chunks],
    }


# ==========================================
# 3.5. NEW: Ollama API Tool Use (Function Calling)
# ==========================================
def rag_answer_ollama_with_tool_use(question: str, k_inject: int = 3):
    # Define tool schema provided to the model
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_git_manual",
                "description": "Search the Git User Manual for technical documentation, commands, or concepts.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_query": {
                            "type": "string",
                            "description": "The specific keyword or question to search for in the manual.",
                        }
                    },
                    "required": ["search_query"],
                },
            },
        }
    ]

    messages = [{"role": "user", "content": question}]

    # Phase 1: Let the model decide whether to call a tool
    t0_think = time.perf_counter()
    payload = {
        "model": "phi3",
        "messages": messages,
        "tools": tools,
        "stream": False,
        "options": {"temperature": 0.0},
    }

    # Tool use must use the /api/chat endpoint
    response1 = requests.post("http://localhost:11434/api/chat", json=payload).json()
    t_think = time.perf_counter() - t0_think

    message1 = response1.get("message", {})

    # Phase 2: Intercept tool call request and execute FAISS search
    if "tool_calls" in message1 and message1["tool_calls"]:
        tool_call = message1["tool_calls"][0]
        function_name = tool_call["function"]["name"]
        arguments = tool_call["function"]["arguments"]
        search_query = arguments.get("search_query", question)

        print(
            f"\n   [System] Model autonomously invoked tool: {function_name}(search_query='{search_query}')"
        )

        t0_search = time.perf_counter()
        retrieved_chunks = dense_retrieve(search_query, k=k_inject)
        t_search = time.perf_counter() - t0_search

        src_blocks = [
            f"[Page {c['page']}]\n{c['text'][:500]}" for c in retrieved_chunks
        ]
        context_text = "\n\n".join(src_blocks)

        # Add model request and tool execution results back to the conversation history
        messages.append(message1)
        messages.append({"role": "tool", "content": context_text})

        # Phase 3: Generate the final answer using the retrieved data
        t0_final = time.perf_counter()
        payload2 = {
            "model": "phi3",
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.2},
        }
        response2 = requests.post(
            "http://localhost:11434/api/chat", json=payload2
        ).json()
        t_final = time.perf_counter() - t0_final

        return {
            "answer": response2.get("message", {}).get("content", "").strip(),
            "t_think": t_think,
            "t_search": t_search,
            "t_final_gen": t_final,
            "used_tool": True,
        }
    else:
        print("\n   [System] Model decided NOT to use the tool and answered directly.")
        return {
            "answer": message1.get("content", "").strip(),
            "t_think": t_think,
            "t_search": 0.0,
            "t_final_gen": 0.0,
            "used_tool": False,
        }


# ==========================================
# 4. Recall Evaluation Framework
# ==========================================
eval_set_pages = [
    {"question": "What is a tree object?", "gold_pages": {54}},
    {"question": "What does git diff compare?", "gold_pages": {14}},
    {"question": "What is a commit object?", "gold_pages": {33}},
    {"question": "What does git fsck do?", "gold_pages": {20}},
    {"question": "What is a blob object?", "gold_pages": {48}},
    {"question": "What does git checkout-index do?", "gold_pages": {43}},
    {"question": "What does git update-index do?", "gold_pages": {43}},
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


def evaluate_dense_pages(eval_set, k=5):
    print("\n" + "=" * 50)
    print(f" Starting Dense Retrieval Recall@{k} Evaluation")
    print("=" * 50)
    correct = 0

    for item in eval_set:
        q = item["question"]
        gold_pages = set(item["gold_pages"])
        ctx = dense_retrieve(q, k=k)
        got_pages = {c["page"] for c in ctx}

        hit = len(gold_pages & got_pages) > 0
        correct += int(hit)

        print(f"\nQ: {q}")
        print(
            f"Hit: {hit} | gold: {sorted(list(gold_pages))} | got: {sorted(list(got_pages))}"
        )

    recall = correct / len(eval_set)
    print(f"\n Final Dense Recall@{k}: {recall:.2f}")
    return recall


evaluate_dense_pages(eval_set_pages, k=5)

# ---------------------------------------------------------
# Test 1: Standard Pipeline (Direct Context Injection)
# ---------------------------------------------------------
print("\n" + "=" * 50)
print("Test 1: Standard Pipeline (Direct Context Injection)")
print("=" * 50)
result1 = rag_answer_ollama_dense("What is a commit object?", k_inject=3)

print(f"[Model Answer] {result1['answer']}")
print("\n[Latency Breakdown]")
print(f"1. Search (T_search): {result1['t_search']:.4f} seconds")
print(f"2. Prefill (T_prefill): {result1['t_prefill']:.4f} seconds")
print(f"3. Decode (T_decode): {result1['t_decode']:.4f} seconds")

# ---------------------------------------------------------
# Test 2: Tool Use Pipeline (Autonomous API Calling)
# ---------------------------------------------------------
print("\n" + "=" * 50)
print("Test 2: Agentic Tool Use (Function Calling) Pipeline")
print("=" * 50)
# Updated prompt to explicitly instruct the small model to utilize the tool
result2 = rag_answer_ollama_with_tool_use(
    "You must use the search_git_manual tool to find the definition of 'commit object'.",
    k_inject=3,
)

print(f"\n[Model Answer] {result2['answer']}")
print("\n[Latency Breakdown (Double Inference)]")
print(
    f"1. Model Thinking Time (Deciding to use tool): {result2['t_think']:.4f} seconds"
)
print(f"2. Tool Execution Time (FAISS Search): {result2['t_search']:.4f} seconds")
print(
    f"3. Final Generation Time (Reading Context & Decode): {result2['t_final_gen']:.4f} seconds"
)
print(
    f"-> Total Agentic Latency: {(result2['t_think'] + result2['t_search'] + result2['t_final_gen']):.4f} seconds"
)
print("=" * 50)
