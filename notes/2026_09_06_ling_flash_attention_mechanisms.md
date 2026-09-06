# AI Class: Attention Mechanisms & Autoregressive Models

## 1. The Autoregressive Setup

Neural language models are **autoregressive systems**: they generate text one token at a time, left to right.

Given a prompt, the model does two phases:

### Phase A: Prefill (the "mega computation")
- Processes the **entire prompt at once** — all tokens in parallel.
- Computes every layer's output for every position.
- **Computationally heavy**: matrix multiplications over the full sequence length $N$.
- Equivalent to: "reading and understanding the user's message."

### Phase B: Decode (generation)
- Generates **one new token** at a time.
- Each new token depends on all previous ones (autoregressive).
- Repeated until stop token or max length.
- **Memory-bandwidth bound**, not compute-bound — the KV cache dominates.

---

## 2. The Architecture Chain (Hybrid Design)

Modern efficient models often stack layers like:


[Linear Attention Layer] × N  →  [Global/Full Attention or MLA Layer]  →  repeat

- **Linear layers** (KDA, GatedDeltaNet, Mamba): cheap per-token, bounded state, good for propagation.
- **Global/MLA layer**: periodic "reset" or anchor for precise retrieval (lossless key-value access).
- The ratio (e.g., 5:1 KDA:MLA in Ling-3.0) balances cost vs. recall.

---

## 3. The Algorithms (Subjective Evaluation)

| Mechanism | Type | State Size | Per-Token Gen | Total Seq | Inductive Bias | Hardware Fit |
|---|---|---|---|---|---|---|
| **Standard Attention** | Softmax $QK^T$ | $O(N)$ KV cache | $O(N)$ | $O(N^2)$ | Strong, explicit | Tensor cores, FlashAttention |
| **GatedDeltaNet** (Qwen) | Linear recurrent | $d_k \times d_v$ per head | $O(1)$ | $O(N)$ | Associative, scalar gate | Good, simpler state |
| **KDA** (Kimi) | Linear + diagonal gating | $d_k \times d_v$ per head | $O(1)$ | $O(N)$ | Stronger recall, per-channel decay | Good but heavier state |
| **Mamba / SSM** | Selective state-space | Vector $\mathbb{R}^{N_{\text{state}}}$ | $O(1)$ | $O(N)$ | Weak retrieval, strong compression | **Excellent** — small state fits SRAM |
| **MLA** (DeepSeek) | KV-cache compression | Latent $d_c \ll d$ | $O(1)$* | $O(N)$ | Retrieval via projection | HBM bandwidth saver |

\*MLA doesn't change FLOPs, reduces memory reads.

### Why NVIDIA "likes" Mamba more
- Mamba's state is a **small vector** ($\sim$64–128 dims) — fits neatly into GPU SRAM tiles.
- Scalar recurrence maps to GPU scalar units cleanly.
- KDA / DeltaNet maintain a **matrix** $S_t \in \mathbb{R}^{d_k \times d_v}$ per head — larger, harder to tile, more register pressure.
- So NVIDIA tooling (cuDNN, TensorRT) optimizes for Mamba-style kernels first.

### Why Qwen iterates on GatedDeltaNet, not KDA
- GatedDeltaNet is the **ancestor**: scalar gate $\alpha_t \in \mathbb{R}$, simpler.
- KDA adds **diagonal gating** $\text{Diag}(\alpha_t) \in \mathbb{R}^{d_k}$ — more parameters, more expressivity, more complexity.
- Qwen likely prefers: "good enough recall + simpler kernel" over "better recall + heavier state."
- They've released multiple DeltaNet variants (DeltaNet → GatedDeltaNet → ...), suggesting incremental refinement, not a leap to KDA.

### Which linear mechanism is more costly?
- **KDA** is the most expensive per layer (largest recurrent state).
- **GatedDeltaNet** sits in the middle.
- **Mamba** is cheapest in state memory but weaker at retrieval.
- Trade-off: **recall quality vs. hardware efficiency**.

---

## 4. Mathematical Notation (Mock Example)

Prompt: `"I love"` → predict `"You"`

Tokens: $x_1 = \text{"I"},\; x_2 = \text{"love"},\; x_3 = \text{"You"}$ (to be predicted)

### Standard Attention (one head简化)

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

$$
\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

- $QK^T$ is $N \times N \rightarrow O(N^2)$ — the bottleneck.

### Linear Recurrent (KDA / GatedDeltaNet)

Maintain state $S_t \in \mathbb{R}^{d_k \times d_v}$:

**GatedDeltaNet:**

$$
S_t = (I - \beta_t k_t k_t^T)\,\alpha_t\, S_{t-1} + \beta_t k_t v_t^T \quad (\alpha_t \in \mathbb{R})
$$

**KDA:**

$$
S_t = (I - \beta_t k_t k_t^T)\,\text{Diag}(\alpha_t)\, S_{t-1} + \beta_t k_t v_t^T \quad (\alpha_t \in \mathbb{R}^{d_k})
$$

Output at step $t$:

$$
o_t = S_t\, q_t
$$

For $t=3$ (predicting "You"):
- Input: $x_1, x_2$ already processed $\rightarrow$ state $S_2$ exists.
- New token $x_3$ uses $q_3$, updates to $S_3$, outputs logits over vocabulary.
- The "I love" context is compressed into $S_2$; no need to re-read all previous tokens.

### MLA (compressed cache)

Instead of storing full $K, V$ matrices, store latent:

$$
c_t^{KV} = W_{c}[K_t; V_t] \in \mathbb{R}^{d_c}
$$

During decode, load $c_t^{KV}$ (small) instead of full $K, V$ (large) $\rightarrow$ less HBM traffic.

---

## 5. Summary Cheat Sheet

- **Prefill**: process all prompt tokens in parallel, heavy compute.
- **Decode**: one token at a time, memory-bandwidth bound.
- **Linear attention** (KDA, GatedDeltaNet, Mamba): $O(1)$ per token, fixed state, replaces $O(N^2)$ softmax.
- **MLA**: compresses KV cache, saves bandwidth, not FLOPs.
- **Hybrid**: linear layers for cheap propagation + occasional global/MLA for precise recall.
- **NVIDIA** prefers Mamba (small state, SRAM-friendly).
- **Qwen** prefers GatedDeltaNet (simplicity, sufficient).
- **KDA** offers better retrieval at higher state cost.

The field is converging on: *keep attention's retrieval strength, but compress its memory cost via linear recurrence + KV compression.*

