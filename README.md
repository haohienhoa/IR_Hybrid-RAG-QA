# Hybrid RAG System - Cranfield Dataset

## Overview
This project implements a highly optimized Retrieval-Augmented Generation (RAG) pipeline designed to query and answer technical questions based on the Cranfield dataset (a collection of aerodynamics and aerospace engineering abstracts). 

## Architecture & Key Features

* **Hybrid Search Retrieval:** Combines lexical search (BM25) to capture exact academic keywords and semantic search (ChromaDB) to understand context. The results are merged using Reciprocal Rank Fusion (RRF).
* **Local Embedding for Privacy:** Uses `all-MiniLM-L6-v2` running entirely locally (CPU-friendly) to ensure sensitive documents are never exposed to external APIs.
* **LangChain Expression Language (LCEL):** Replaces heavy graph-based workflows with a streamlined, linear LCEL pipeline for maximum memory efficiency and speed.
* **Zero-Hallucination Mechanism:** Implements Few-Shot Chain-of-Thought (CoT) prompting. The LLM is forced to analyze the context in a `<thought>` block before outputting the `<answer>`, ensuring strict adherence to the provided documents and explicit refusal when information is absent.

## Technology Stack
* **Framework:** LangChain 
* **Vector Database:** ChromaDB
* **Lexical Search:** Rank-BM25
* **LLM Inference:** Hugging Face Serverless API (`Qwen/Qwen2.5-7B-Instruct`)
* **Embedding Model:** Hugging Face Local (`sentence-transformers/all-MiniLM-L6-v2`)

## Project Structure
```text
.
├── data/                  
├── src/
│   ├── config.py           
│   ├── data_loader.py     
│   ├── retriever.py        
│   └── rag_chain.py       
├── .env                   
├── .gitignore
├── requirements.txt       
└── main.py                 
