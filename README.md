# ECE_226_group17
RAG &amp; Tool Use for Small Models

# RAG System Performance Evaluation and Retrieval Profiling

This repository contains the implementation and evaluation framework for a Retrieval-Augmented Generation (RAG) system, developed as part of the final project. The system utilizes the Git User Manual as a primary knowledge base to investigate different retrieval strategies and their impact on generative performance.

## Project Overview

The project focuses on two main objectives:
1. Comparing the effectiveness of Sparse Retrieval (BM25) and Dense Retrieval (FAISS).
2. Implementing and profiling an Agentic RAG workflow with autonomous tool-calling capabilities.

## File Descriptions

- **evaluate_rag.py**: Implements the core RAG pipeline using FAISS vector search and the Phi-3 model via Ollama. It includes a framework for evaluating retrieval recall and testing autonomous tool-use logic.
- **rag_profiler.py**: A benchmarking script designed to perform A/B testing between BM25 (Sparse) and FAISS (Dense) retrieval engines across 23 standardized technical questions.
- **test_manual.pdf**: The source documentation used for data chunking and knowledge retrieval.
- **requirements.txt**: A list of Python dependencies required to execute the scripts.

## Installation and Setup

### Prerequisites
- Ollama: Ensure Ollama is installed and the Phi-3 model is available locally.
  ```bash
  ollama pull phi3

### Dependency Installation
- Install the required Python packages using the following command:
  ```bash
  pip install -r requirements.txt

## Methodology and Evaluation
The system evaluates retrieval performance based on the Recall@5 metric. The experimental results, including retrieval latency and accuracy comparisons, are detailed in the final paper and can be replicated by running the profiling script.
