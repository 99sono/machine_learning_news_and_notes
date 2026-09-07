# AI Class: Attention Mechanisms & Autoregressive Models

## 1. The Autoregressive Setup

Neural language models function as **autoregressive systems**: they generate discrete token sequences sequentially, one position at a time from left to right.

Given an input prompt of length $N$, processing occurs across two distinct computational phases:

### Phase A: Prefill (The Parallel Computation Phase)

* Processes the **entire prompt sequence simultaneously** in parallel across all layers.
* Computes vector representations, query-key-value projections, and intermediate activations for every position $t \in \{1, \dots, N\}$.
* **Computationally heavy**: Dominated by dense matrix-matrix multiplications ($GEMM$) scaling quadratically with sequence length $N$.
* **Functional definition**: Corresponds to reading and encoding the global context of the user's message.

### Phase B: Decode (The Autoregressive Generation Phase)

* Generates **a single new token** per forward pass.
* Each newly predicted token depends autoregressively on all historical tokens generated up to that step.
* Iterates sequentially until a termination token or maximum length constraint is reached.
* **Memory-bandwidth bound**: Limited by memory throughput (loading weights and the Key-Value cache from High-Bandwidth Memory) rather than raw floating-point compute capacity.

---

## 2. The Architecture Chain (Hybrid Design)

Modern efficient architectures interleave linear recurrence layers with periodic global attention layers:

$$\underbrace{[\text{Linear Attention Layer}] \times M}_{\text{Cheap propagation, bounded state}} \longrightarrow \underbrace{[\text{Global/Full Attention or MLA Layer}]}_{\text{Precise lossless global retrieval}} \longrightarrow \text{repeat}$$

* **Linear recurrent layers** (e.g., KDA, GatedDeltaNet, Mamba): Maintain a fixed-size internal state, yield $O(1)$ per-token generation complexity, and efficiently propagate local context.
* **Global/MLA layers**: Act as periodic structural anchors that provide lossless key-value retrieval and correct long-range dependencies.
* **Stacking Ratio**: An engineered ratio (e.g., a 5:1 KDA-to-MLA ratio in Ling-3.0) optimizes the trade-off between computational efficiency and retrieval accuracy.

---

## 3. The Algorithms (Comparative Evaluation)

| Mechanism | Mathematical Type | Recurrent State Dimension | Per-Token Gen Cost | Total Seq Complexity | Inductive Bias | Hardware Acceleration Fit |
| --- | --- | --- | --- | --- | --- | --- |
| **Standard Attention** | Softmax $QK^T$ | $O(N)$ KV cache | $O(N)$ | $O(N^2)$ | Strong, dense global pairing | Highly optimized for Tensor Cores (FlashAttention) |
| **GatedDeltaNet** (Qwen) | Linear recurrent | Matrix $S_t \in \mathbb{R}^{d_k \times d_v}$ per head | $O(1)$ | $O(N)$ | Associative recall via scalar gating | Efficient; simpler scalar state updates |
| **KDA** (Kimi) | Linear + diagonal gating | Matrix $S_t \in \mathbb{R}^{d_k \times d_v}$ per head | $O(1)$ | $O(N)$ | Stronger retrieval via channel-wise decay | Efficient, though higher register pressure |
| **Mamba / SSM** | Selective State-Space | Vector $\mathbf{h}_t \in \mathbb{R}^{N_{\text{state}}}$ | $O(1)$ | $O(N)$ | Strong sequence compression, weaker exact retrieval | **Optimal** — compact state fits entirely in SRAM |
| **MLA** (DeepSeek) | Low-rank KV compression | Latent vector $\mathbf{c}_t^{\text{KV}} \in \mathbb{R}^{d_c}$ | $O(1)$* | $O(N)$ | Retrieval via low-rank projections | Eliminates HBM bandwidth bottlenecks |

**Note: MLA does not alter theoretical asymptotic FLOP counts; it reduces memory bandwidth overhead during decoding.*

### Why NVIDIA Hardware Optimizes for Mamba-First
* Mamba's hidden state is represented as a **compact vector** ($\mathbf{h}_t \in \mathbb{R}^{N}$, where $N \in [64, 128]$), allowing the entire recurrence state to reside directly within fast on-chip SRAM tiles.
* Scalar state-space transitions map cleanly onto hardware scalar execution units.
* Conversely, KDA and GatedDeltaNet maintain a **full matrix** $S_t \in \mathbb{R}^{d_k \times d_v}$ per attention head, resulting in higher register pressure, larger memory footprints, and more complex tiling requirements.
* Consequently, low-level compilation toolchains (e.g., cuDNN, TensorRT-LLM) natively prioritize Mamba-style state-space operator kernels.

### Why Qwen Iterates on GatedDeltaNet over KDA

* GatedDeltaNet utilizes a scalar decay gate ($\alpha_t \in \mathbb{R}$), representing a simpler architectural baseline.
* KDA extends this formulation by introducing channel-wise diagonal gating ($\text{Diag}(\boldsymbol{\alpha}_t) \in \mathbb{R}^{d_k \times d_k}$), which increases parameter expressivity at the cost of implementation complexity.
* Qwen prioritizes engineering stability and hardware kernel simplicity ("sufficient retrieval with streamlined kernels") over maximal theoretical state capacity.

---

## 4. Mathematical Notation (Rigorous Formulation)

Consider an input prompt string mapped to discrete token indices, corresponding to the sequence:


$$\text{Prompt: } \text{"I love"} \longrightarrow \text{Target Prediction: } \text{"You"}$$

Let $\mathbf{x}_t \in \mathbb{R}^{d_{\text{model}}}$ denote the continuous embedding vector representation of the token at sequence index $t$. The sequence of input tokens is indexed explicitly as:

* $\mathbf{x}_1$: Embedding vector for `"I"`
* $\mathbf{x}_2$: Embedding vector for `"love"`
* $\mathbf{x}_3$: Embedding vector for `"You"` (the target token to be predicted)

### Standard Attention (Single-Head Formulation)

Let the input matrix representation of the sequence be $X \in \mathbb{R}^{N \times d_{\text{model}}}$. The linear projection matrices are defined as $W_Q, W_K \in \mathbb{R}^{d_{\text{model}} \times d_k}$ and $W_V \in \mathbb{R}^{d_{\text{model}} \times d_v}$.

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

* **Complexity Bottleneck:** The matrix product $QK^T$ yields an $N \times N$ matrix, driving computational complexity to $O(N^2)$.

### Linear Recurrent Mechanisms (KDA / GatedDeltaNet)

Instead of retaining historical keys and values, these models maintain a fixed-size internal recurrent state matrix $S_t \in \mathbb{R}^{d_k \times d_v}$ per head.

**GatedDeltaNet Update Rule (Scalar Gating):**


$$S_t = (I - \beta_t \mathbf{k}_t \mathbf{k}_t^T)\,\alpha_t\, S_{t-1} + \beta_t \mathbf{k}_t \mathbf{v}_t^T \quad \text{where } \alpha_t \in \mathbb{R}$$

**KDA Update Rule (Diagonal Gating):**


$$S_t = (I - \beta_t \mathbf{k}_t \mathbf{k}_t^T)\,\text{Diag}(\boldsymbol{\alpha}_t)\, S_{t-1} + \beta_t \mathbf{k}_t \mathbf{v}_t^T \quad \text{where } \boldsymbol{\alpha}_t \in \mathbb{R}^{d_k}$$

**Recurrent Output Projection at Step $t$:**


$$\mathbf{o}_t = S_t\, \mathbf{q}_t$$

**Instantiation at $t = 3$ (Predicting $\mathbf{x}_3$ = `"You"`):**

* The historical context vectors $\mathbf{x}_1$ and $\mathbf{x}_2$ have already been compressed into the recurrent state matrix $S_2$.
* When processing the incoming query vector $\mathbf{q}_3$, the model updates the state to $S_3$ and computes output logits over the vocabulary without re-scanning past tokens.

### Multi-head Latent Attention (MLA) Compression

MLA compresses explicit Key-Value caches via low-rank projections. Rather than storing full matrices $K_t, V_t \in \mathbb{R}^{N \times d}$, it projects them into a shared latent vector space:


$$\mathbf{c}_t^{\text{KV}} = W_c [K_t; V_t] \in \mathbb{R}^{d_c} \quad \text{where } d_c \ll d$$


During autoregressive decoding, memory traffic is minimized by fetching the compressed latent vector $\mathbf{c}_t^{\text{KV}}$ from High-Bandwidth Memory instead of uncompressed multi-head tensors.

---

## 5. Summary Cheat Sheet

* **Prefill Phase:** Parallel processing of prompt tokens; dominated by heavy matrix multiplications ($O(N^2)$).
* **Decode Phase:** Sequential single-token generation; bounded by memory bandwidth.
* **Linear Attention Layers:** Achieve $O(1)$ per-token generation complexity via fixed-size recurrent states ($S_t$).
* **MLA:** Low-rank KV cache compression designed to mitigate memory-bandwidth constraints.
* **Hybrid Architecture:** Couples linear recurrent layers for efficient token propagation with periodic global/MLA layers for exact long-range retrieval.

---

## Appendix: Mathematical Notation Reference

| Symbol / Notation | Mathematical Definition / Meaning |
| --- | --- |
| $N$ | Total sequence length (number of tokens in the input context). |
| $d_{\text{model}}$ | Hidden dimensionality of the model's transformer representation space. |
| $d_k, d_v$ | Dimensionality of the Query/Key subspaces and Value subspaces, respectively. |
| $\mathbf{x}_t \in \mathbb{R}^{d_{\text{model}}}$ | Column vector representing the embedding or hidden state at sequence position $t$. |
| $X \in \mathbb{R}^{N \times d_{\text{model}}}$ | Matrix formed by stacking sequence token vectors row-wise. |
| $Q, K, V$ | Query, Key, and Value matrices derived via linear projections of $X$. |
| $\mathbf{q}_t, \mathbf{k}_t, \mathbf{v}_t$ | Vector representations of Query, Key, and Value at a specific time step $t$. |
| $S_t \in \mathbb{R}^{d_k \times d_v}$ | Recurrent state matrix maintained across time steps in linear attention models. |
| $\alpha_t \in \mathbb{R}$ / $\boldsymbol{\alpha}_t \in \mathbb{R}^{d_k}$ | Scalar decay factor or channel-wise diagonal decay vector governing state persistence. |
| $\beta_t \in \mathbb{R}$ | Step-size or update gating coefficient in delta-rule recurrent formulations. |
| $\text{Diag}(\cdot)$ | Operator constructing a diagonal matrix from a vector input. |
| $O(1), O(N), O(N^2)$ | Asymptotic computational complexity bounds using Big-O notation. |
