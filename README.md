# ECE_226_group17
RAG &amp; Tool Use for Small Models

**Edge-Native RAG: Evaluating Context Injection Latency and Retrieval Efficacy for Small LLMs**

This repository contains the implementation and evaluation framework for an edge-optimized Retrieval-Augmented Generation (RAG) system. The project investigates the trade-offs between Sparse and Dense retrieval and profiles the system bottlenecks on resource-constrained hardware.

##  Group Information
- **Group Number:** 17
- **Target Model:** Phi-3-mini-4k-instruct (via Ollama)
- **Primary Corpus:** Official Git User Manual

---

##  Key Findings
* **Lexical vs. Semantic:** In highly technical domains, **BM25 (Sparse)** outperformed **FAISS (Dense)** with a **47.8% vs. 34.8% Recall@5**, proving that exact command-line syntax matching is superior to semantic embedding for technical documentation.
* **Latency Bottleneck:** Profiling reveals that **Context Injection overhead is negligible (~1.5%)**, with the majority of latency residing in the LLM decoding phase.
* **Agentic Constraints:** While deterministic RAG is highly efficient, autonomous **API Tool Use** causes severe latency spikes (>2.0s) and reasoning failures on edge-native SLMs.

---

##  Hardware Environment
- **GPU:** NVIDIA RTX 5060 (8GB VRAM) - Dedicated to LLM Inference.
- **CPU:** AMD Ryzen AI 7 350 (32GB RAM) - Offloaded Retrieval Tasks (BM25/FAISS).
- **Inference Engine:** Ollama (4-bit quantization).

---

##  File Descriptions
- **`evaluate_rag.py`**: The core pipeline implementing the heterogeneous CPU/GPU RAG workflow and tool-calling logic.
- **`rag_profiler.py`**: The benchmarking suite used to generate the Recall@5 and latency metrics presented in the paper.
- **`test_manual.pdf`**: The technical source document (Git User Manual).
- **`requirements.txt`**: Python dependencies.

---

##  Quick Start

### 1. Prerequisites
Install [Ollama](https://ollama.ai/) and pull the model:
```bash
ollama pull phi3
```

### 2. Setup Environment
```bash
pip install -r requirements.txt
```


### 3. Run Evaluation
Execute the profiling script to replicate our Sparse vs. Dense retrieval comparison:
```bash
python rag_profiler.py
```
To test the RAG pipeline with a custom query:
```bash
python evaluate_rag.py
```

## Methodology and Evaluation
The system evaluates retrieval performance based on the Recall@5 metric. The experimental results, including retrieval latency and accuracy comparisons, are detailed in the final paper and can be replicated by running the profiling script. 

