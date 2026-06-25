# Understanding the Transformer: Foundations of Large Language Models

> A paper review tracing the architectural evolution from **RNN → LSTM → Seq2Seq → Attention → Transformer → GPT**, synthesized from key literature to understand *why* each model emerged and *what problem* it solved.

Prepared as part of undergraduate research activities at the **Applied Modeling and Simulation Lab (AMAS), Hankyong National University**, under the guidance of Prof. Un Chang Jeoung.

---

## 📄 Papers Reviewed

| # | Paper | Venue |
|---|-------|-------|
| 1 | Rumelhart et al., "Learning representations by back-propagating errors" | *Nature*, 1986 |
| 2 | Hochreiter & Schmidhuber, "Long Short-Term Memory" | *Neural Computation*, 1997 |
| 3 | Sutskever et al., "Sequence to Sequence Learning with Neural Networks" | *NeurIPS*, 2014 |
| 4 | Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align and Translate" | *ICLR*, 2015 |
| 5 | Vaswani et al., "Attention Is All You Need" | *NeurIPS*, 2017 |
| 6 | He et al., "Deep Residual Learning for Image Recognition" | *CVPR*, 2016 |
| 7 | Radford et al., "Improving Language Understanding by Generative Pre-Training" | *OpenAI*, 2018 |

**Supplementary reading:** 세바스찬 라시카, 『밑바닥부터 만들면서 배우는 LLM』, 박해선 옮김, 길벗, 2025

---

## 📌 Overview

This repository summarizes a paper review series on the foundations of the Transformer architecture. Rather than reviewing each paper in isolation, the goal was to understand the **chain of problems and solutions** that led from early RNNs to modern LLMs — tracing how each paper motivated the next.

---

## 📚 Contents

1. [Background](#1-background)
   - [Recurrent Neural Network (RNN)](#11-recurrent-neural-network-rnn)
   - [Long Short-Term Memory (LSTM)](#12-long-short-term-memory-lstm)
   - [Sequence to Sequence (Seq2Seq)](#13-sequence-to-sequence-seq2seq)
   - [Attention Mechanism](#14-attention-mechanism)
2. [Transformer — "Attention is All You Need"](#2-transformer--attention-is-all-you-need)
3. [GPT-1: Transformer in Practice](#3-gpt-1-transformer-in-practice)
4. [Conclusion](#4-conclusion)

---

## 1. Background

### 1.1 Recurrent Neural Network (RNN)

RNN is a neural network designed to handle **sequential data** (e.g., time series, text) by feeding the output of each hidden state back as input to the next step.

```
h_t = f(W · x_t + U · h_{t-1} + b)
```

**RNN variants by input/output structure:**

| Type | Structure | Example Use Case |
|------|-----------|-----------------|
| One-to-One | Single input → Single output | Standard classification |
| One-to-Many | Single input → Sequence output | Image captioning |
| Many-to-One | Sequence input → Single output | Sentiment analysis |
| Many-to-Many | Sequence input → Sequence output | Machine translation |

**Core problem:** RNNs suffer from the **vanishing gradient problem** — gradients diminish exponentially during backpropagation through long sequences, making it difficult to learn long-range dependencies.

→ Solution: **Gated RNNs (LSTM, GRU)**

---

### 1.2 Long Short-Term Memory (LSTM)

LSTM addresses the vanishing gradient problem by introducing a **cell state** and **gate mechanisms** that control information flow.

**Key components:**

| Gate | Role |
|------|------|
| **Forget Gate** | Decides what information to discard from the previous cell state |
| **Input Gate** | Determines what new information to store |
| **Cell State** | Combines old memory with new information |
| **Output Gate** | Produces the final hidden state (output) |

**RNN vs. LSTM internal structure:**
- RNN repeating module: **1 layer** (simple tanh)
- LSTM repeating module: **4 interacting layers** (gates + cell state)

The gate structure acts as a "valve" that controls gradient flow, enabling stable learning over long sequences.

---

### 1.3 Sequence to Sequence (Seq2Seq)

Seq2Seq models handle **variable-length input → variable-length output** mapping, primarily designed for machine translation.

**Architecture:**
```
"I am a student"
        ↓ [Encoder: LSTM]
    Context Vector
        ↓ [Decoder: LSTM]
"je suis étudiant"
```

**How it works:**
1. The **Encoder** processes the input sequence and compresses it into a fixed-size **Context Vector**
2. The **Decoder** uses the Context Vector to generate the output sequence token by token
3. Special tokens `<sos>` (start of sequence) and `<eos>` (end of sequence) mark boundaries

**Limitation — Bottleneck Problem:**
The entire input sequence is compressed into a single fixed-size Context Vector. For long sentences, this causes **information loss**, degrading translation quality.

→ Solution: **Attention Mechanism**

---

### 1.4 Attention Mechanism

Attention allows the Decoder to **directly reference all Encoder hidden states** at each decoding step, rather than relying solely on a single Context Vector.

**Core idea:**
> Instead of compressing all input information into one vector, let the Decoder *attend* to the most relevant parts of the input at each generation step.

**How it works:**
1. At each decoding step, compute a relevance score between the current Decoder state and every Encoder hidden state
2. Apply softmax to get attention weights (sum to 1)
3. Compute a weighted sum of Encoder hidden states → dynamic Context Vector
4. Use this context + previous output to predict the next token

**Key advantage:** The model learns *which input words to focus on* for each output word, enabling more accurate and interpretable translations.

---

## 2. Transformer — "Attention is All You Need"

> Vaswani et al., NeurIPS 2017

**Motivation:**
- RNN/LSTM-based Seq2Seq requires **sequential computation** — each step depends on the previous
- This prevents parallelization and limits training speed on long sequences
- Attention partially solved the dependency problem, but was still attached to RNN

**The Transformer eliminates RNN entirely — using only Attention.**

---

### Encoder

| Component | Role |
|-----------|------|
| **Input Embedding** | Converts tokens into dense vector representations |
| **Positional Encoding** | Injects position information (since Attention has no inherent order) |
| **Multi-Head Self-Attention** | Each token attends to all other tokens in the input |
| **Residual Connection** | Preserves original information; stabilizes gradient flow |
| **Layer Normalization** | Normalizes activations for stable training |
| **Feed-Forward Network** | Projects attended representations into richer feature vectors |

### Decoder

| Component | Role |
|-----------|------|
| **Masked Multi-Head Self-Attention** | Attends to previous output tokens only (future tokens are masked) |
| **Encoder-Decoder Attention** | Decoder Query attends to Encoder Key/Value — learns input-output alignment |
| **Residual + LayerNorm** | Same as Encoder |
| **Feed-Forward Network** | Same as Encoder |
| **Linear + Softmax** | Maps output vector to vocabulary probability distribution |

---

### Scaled Dot-Product Attention

The fundamental attention operation:

```
Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V
```

| Symbol | Meaning |
|--------|---------|
| Q (Query) | "What am I looking for?" |
| K (Key) | "What information do I have?" |
| V (Value) | "What is the actual content?" |
| √d_k | Scaling factor to prevent softmax saturation in high dimensions |

---

### Multi-Head Attention

Instead of computing one attention function, Multi-Head Attention runs **h attention operations in parallel**, each learning different aspects of the relationships:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_O
where head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)
```

**Benefit:** Different heads can capture different types of relationships (syntactic, semantic, positional) simultaneously.

---

### Three Types of Attention in the Transformer

| Type | Query Source | Key/Value Source | Purpose |
|------|-------------|-----------------|---------|
| **Encoder Self-Attention** | Encoder | Encoder | Learn relationships between all input tokens |
| **Decoder Self-Attention** (Masked) | Decoder | Decoder (past only) | Generate output autoregressively |
| **Encoder-Decoder Attention** | Decoder | Encoder | Align input and output sequences |

---

## 3. GPT-1: Transformer in Practice

> Radford et al., OpenAI 2018 — *"Improving Language Understanding by Generative Pre-Training"*

GPT-1 demonstrated that a **single Transformer-based language model** can be adapted to a wide range of NLP tasks through a two-stage training approach.

### Stage 1 — Pre-training (Unsupervised)
- Trained on large-scale unlabeled text data
- Objective: **Next-token prediction** (language modeling)
- Architecture: **Transformer Decoder only** (causal/masked self-attention)
- Learns general language representations and long-range context

### Stage 2 — Fine-tuning (Supervised)
- Pre-trained model is adapted to specific downstream tasks
- Only a **task-specific output layer** is added — no architectural changes
- Tasks: Question Answering, Text Classification, Natural Language Inference (NLI), etc.

### Key Results
- Achieved **state-of-the-art on 9 out of 12 NLP benchmarks**
- Story Cloze: **+8.9%** improvement
- RACE: **+5.7%** improvement
- Demonstrated clear advantages over LSTM-based models on long-range dependency tasks

---

## 4. Conclusion

| Model | Key Innovation | Limitation Addressed |
|-------|---------------|---------------------|
| RNN | Sequential hidden state | — |
| LSTM | Gate mechanism + cell state | Vanishing gradient |
| Seq2Seq | Encoder-Decoder structure | Variable-length sequences |
| Attention | Dynamic context from all encoder states | Bottleneck problem |
| Transformer | Self-attention only, no RNN | Sequential computation, parallelization |
| GPT-1 | Pre-training + Fine-tuning | Task-specific model requirements |

**Key takeaways:**
- The Transformer solves the parallelization and long-range dependency problems of RNN-based models through pure attention
- Multi-Head Attention enables simultaneous learning of multiple types of token relationships
- The Pre-training + Fine-tuning paradigm established by GPT-1 became the foundation for all subsequent LLMs (GPT-2/3/4, BERT, etc.)

### Future Directions
- Study of GPT-2/3 and more recent LLM architectures (e.g., instruction tuning, RLHF)
- Exploration of Transformer applications in engineering domains — particularly for time-series fault diagnosis and structural health monitoring (PHM)

---

## 📁 Repository Structure

```
llm-transformer-paper-review/
│
├── README.md
└── slides/
    └── LLM_Transformer_Foundations.pptx
```

---

*Studied and documented as part of undergraduate research at AMAS Lab, Hankyong National University.*
