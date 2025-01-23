# Multi-Task Sentence Transformer Project

This repository contains a minimal demonstration of how to:
1. Implement a **Sentence Transformer** model to produce fixed-length sentence embeddings.
2. Expand that model into a **Multi-Task** architecture capable of producing outputs for multiple NLP tasks simultaneously.
3. Provide a brief **discussion** on training strategies and considerations for multi-task learning, including **when to freeze** parts of the network, **when to use** multi-task models vs. separate models, and **how to handle** data imbalance.

---

## Overview

### Step 1: Sentence Transformer

- **Goal**: Convert input sentences into fixed-length embeddings.
- **Approach**: 
  1. Leverage a pre-trained model (BERT).
  2. Optionally apply different pooling strategies (CLS pooling, mean pooling) for the final sentence representation.
  3. Add a projection layer to modify the embedding dimension.

### Step 2: Multi-Task Expansion

- **Goal**: Share the same backbone but produce outputs for multiple tasks.
- **Tasks**:
  1. **Sentence Classification (Task A)** — e.g., classify a sentence into different categories (positive/negative/neutral).
  2. **Another NLP task (Task B)** — e.g., Named Entity Recognition (NER)
- **Approach**:
  1. Use the same backbone (transformer) for the shared feature representation.
  2. Add separate heads for each task.

### Step 3: Discussion

- **Freezing vs. Training**, which parts of the network are used for multi-task learning?
- **Choosing** between multi-task vs. separate models.
- **Managing data imbalance** between tasks.

---

## Project Structure
```bash
my_sentence_transformer_project/├─ README.md
                                ├─ requirements.txt
                                ├─ src
                                │   ├─ test_sentence_transformers.py
                                │   ├─ test_multi_task_model.py 
                                │   ├─ models.py
                                └─ discussion.md 
```
                                





