# Inside the Global Attention Machine: A Step-by-Step Walkthrough

Imagine we are building a miniature language model to see how it "reads" a sentence during its warmup phase. We are going to play with a really simple scenario using the phrase **"I love you"**.

In our prompt context right now, we only have so far **two tokens**: the token **"I"** and the token **"love"**. The token **"you"** is the target word that will need to be predicted further down the road during the decode phase.

Before we can ever get to predicting the token "you", we first need to go to the gym and calculate the so-called attention for `"I love"` across all the attention layers of our network. The underlying math is identical for every single layer, so we can just pretend that our input `"I love"` is hitting the **very first global attention layer** of our model.

To understand how this works, let's first introduce our notation glossary. Don't worry if these terms look abstract right now; they will become completely intuitive as we walk through the math:

* **$N$ (Sequence Length):** In our example here, this refers to the number of tokens currently in our prompt context: just `"I"` and `"love"`. Therefore, our sequence length is $N = 2$. *(Note: In real tokenizers, tokens aren't always whole words—they can be sub-words or syllables—but for learning purposes, pretending tokens are words makes things much easier to digest).*
* **$d_{\text{model}}$ (Hidden Dimension):** To put this in even simpler terms: if `"I"` is our very first token and it must be represented as a vector, our $d_{\text{model}}$ being $4$ implies that our `"I"` will be a little vector with just 4 dimensions. For example, `"I"` could be the vector $(1, 1, 1, 1)$. And `"love"` would be another vector of dimension $4$ with different values, such as $(2, 2, 2, 2)$. *Note on numbers:* These dimensions aren't just whole integer numbers; each dimension can be any real number floating in $\mathbb{R}^{d_{\text{model}}}$.
* **$h$ (Number of Attention Heads):** When we think about 1 global attention layer, we can think of it as a factory of correlations between tokens. Each head is a different floor of that factory that allows the math to run in parallel. A global attention mechanism with a single head looks at everything through one lens, but splitting it into multiple heads lets the model analyze different relationships simultaneously across parallel lanes.
* **The Important Catch ($h$ must divide cleanly):** $h$ cannot be just any arbitrary number. It must be a number that can divide cleanly into the total number of dimensions of our token vector ($d_{\text{model}}$).
* If our vector has $4$ dimensions per token, the allowed number of attention heads are:
* **$h = 1$:** The $4$ dimensions are not divided at all; all computations happen in a monolithic block, so our vector $(1, 1, 1, 1)$ stays exactly $(1, 1, 1, 1)$.
* **$h = 2$:** Our input tokens are sliced up into two dimensions each, splitting $(1, 1, 1, 1)$ into chunks of $(1, 1)$ and $(1, 1)$.
* **$h = 4$:** Each input vector is sliced down to a single dimension of size $1$, splitting $(1, 1, 1, 1)$ into four separate $1$-dimensional values.
* For our example, we will choose **$h = 2$** attention heads.




* **$X$ (Input Matrix):** A matrix where each row (or column, depending on convention) represents a prompt input token vector. Here, $X$ stacks our tokens $\mathbf{x}_1$ and $\mathbf{x}_2$ together into one neat table.
* **$W_Q, W_K, W_V$ (Weight Matrices):** The magical, heavy-duty weight matrices that start out as random garbage noise and are trained via gradient descent to become elite routing tools.

Now that our glossary is set, let's trace exactly how the input vector flows through this first global attention layer, ignoring the decode phase entirely and focusing strictly on this prefill warmup.

---

## 1. Setting Up the Playground (The Input Matrix $X$)

In our simplified universe, every token is represented by a vector with **4 dimensions** ($d_{\text{model}} = 4$).

Let's assign simple numerical values to our two words:

* Token 1 (`"I"`): represented by the vector $\mathbf{x}_1 = (1, 1, 1, 1)$
* Token 2 (`"love"`): represented by the vector $\mathbf{x}_2 = (2, 2, 2, 2)$

We stack them together into our input matrix $X$ of size $2 \times 4$:

$$
X = \begin{pmatrix} 1 & 1 & 1 & 1 \\\\ 2 & 2 & 2 & 2 \end{pmatrix}
$$


**Note on Conventions:** Unadorned vectors like ($\mathbf{x}$) default to column vectors in pure linear algebra. However, transformer literature frequently treats them as row vectors when stacked into $X$, or uses explicit transposes ($\mathbf{x}^T$) for horizontal rows.
test02

In this layout, each row represents a separate token ($\mathbf{x}_1, \mathbf{x}_2$), which is the standard convention in deep learning frameworks where $X$ has shape $\text{Sequence Length} \times d_3{\text{model}}$. 


If vectors are written without a transpose superscript ($\mathbf{x}$), they are conventionally treated as column vectors by default in pure linear algebra, but transformer literature frequently treats unadorned vectors as row vectors when stacked into $X$, or uses explicit transposes ($\mathbf{x}^T$) when writing them out as horizontal rows.



---

## 2. The Architectural Design: Splitting Into "Baby Matrices"

Since our model has **2 attention heads** ($h = 2$), and our total dimension is $4$, each head gets allocated a slice of **2 dimensions** ($4 / 2 = 2$).

We call these sliced sub-components **"baby matrices."**

* **Head 1** handles dimensions 1 and 2.
* **Head 2** handles dimensions 3 and 4.

Instead of processing all 4 dimensions in one giant, messy calculation, the model runs Head 1 and Head 2 simultaneously in parallel factory floors. Let's trace what happens inside **Head 1** (dimensions 1 and 2).

First, Head 1 extracts its slice of the input tokens, creating our "baby input" matrix for Head 1:


$$X_{\text{head1}} = \begin{pmatrix} 1 & 1 \\\\ 2 & 2 \end{pmatrix}$$

---

## 3. The Trained Weight Matrices ($W_Q, W_K, W_V$)

During training, gradient descent tunes our weight matrices so they know how to route information. For Head 1, we have three weight matrices, each sized $2 \times 2$ (matching our 2-dimensional head subspace):

* **$W_Q$ (Query Weights):** What this head is *looking for*.
* **$W_K$ (Key Weights):** What each token *advertises* about itself.
* **$W_V$ (Value Weights):** The actual *content payload* each token carries.

Let's assume gradient descent successfully tuned Head 1's weight matrices to look like this:


$$W_Q = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}, \quad W_K = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}, \quad W_V = \begin{pmatrix} 2 & 0 \\ 0 & 2 \end{pmatrix}$$

---

## 4. Computing Queries ($Q$), Keys ($K$), and Values ($V$)

Now, Head 1 multiplies its baby input matrix $X_{\text{head1}}$ by its weight matrices. This transforms our raw tokens into Queries, Keys, and Values:

### Step A: The Value Vectors ($V$)

$$V = X_{\text{head1}} W_V = \begin{pmatrix} 1 & 1 \\ 2 & 2 \end{pmatrix} \begin{pmatrix} 2 & 0 \\ 0 & 2 \end{pmatrix} = \begin{pmatrix} 2 & 2 \\ 4 & 4 \end{pmatrix}$$


*Meaning:* These are the refined semantic content payloads for `"I"` and `"love"` ready to be shared.

### Step B: The Query ($Q$) and Key ($K$) Vectors

$$Q = X_{\text{head1}} W_Q = \begin{pmatrix} 1 & 1 \\ 2 & 2 \end{pmatrix} \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 1 & 1 \\ 2 & 2 \end{pmatrix}$$

$$K = X_{\text{head1}} W_K = \begin{pmatrix} 1 & 1 \\ 2 & 2 \end{pmatrix} \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}$$

---

## 5. The Query-Key Voodoo ($QK^T$)

To find out how much attention token 1 should pay to token 2 (and itself), we take the dot product of Queries and transposed Keys ($QK^T$). This creates an $N \times N$ correlation grid ($2 \times 2$ in our case):

$$QK^T = \begin{pmatrix} 1 & 1 \\ 2 & 2 \end{pmatrix} \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}^T = \begin{pmatrix} 1 & 1 \\ 2 & 2 \end{pmatrix} \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix} = \begin{pmatrix} 3 & 6 \\ 6 & 12 \end{pmatrix}$$

Next, the model scales this grid down by dividing by $\sqrt{d_k}$ (here $\sqrt{2} \approx 1.41$) to keep numbers stable, and runs it through a **softmax** function. Softmax turns those raw scores into clean percentages (probabilities that sum to 1).

Let's assume our softmax matrix turns into clean attention weights:


$$\text{Attention Scores} = \begin{pmatrix} 0.2 & 0.8 \\ 0.1 & 0.9 \end{pmatrix}$$


*Read this grid like this:*

* Token 1 (`"I"`) spends 20% of its attention on itself and 80% on `"love"`.
* Token 2 (`"love"`) spends 10% of its attention on `"I"` and 90% on itself.

---

## 6. The Grand Finale: Multiplying by Values ($V$)

The final step of Head 1 takes our attention percentage grid and multiplies it by our Value matrix ($V$):

$$\text{Output} = (\text{Attention Scores}) \times V = \begin{pmatrix} 0.2 & 0.8 \\ 0.1 & 0.9 \end{pmatrix} \begin{pmatrix} 2 & 2 \\ 4 & 4 \end{pmatrix} = \begin{pmatrix} 3.6 & 3.6 \\ 3.8 & 3.8 \end{pmatrix}$$

Head 1 has successfully taken raw input tokens, routed them through trained weight matrices, cross-compared them via Queries and Keys, and outputted brand-new, context-infused representations. Head 2 is doing the exact same thing in parallel on dimensions 3 and 4.

During this prefill warmup, these computed Key and Value vectors are saved into the **KV cache**, completing our gym session and leaving the model fully prepared to predict `"you"` next!
