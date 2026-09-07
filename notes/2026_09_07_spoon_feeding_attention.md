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

**3. The Trained Weight Matrices ($W_Q, W_K, W_V$)**

During training, gradient descent tunes our weight matrices so they know how to route information. But before looking at their values, let's examine their **size and shape geometry**, which reveals a fascinating twist about how AI uses linear algebra compared to traditional math textbooks.

* **The Native Size Rule:** The native dimensions of these weight matrices are determined by the head subspace size, written as $d_k \times d_k$ (where $d_k = d_{\text{model}} / h$).
* If our model had only a single attention head ($h = 1$), our total dimension would stay at $4$, meaning our weight matrices would have to be a monolithic **$4 \times 4$** block.
* Because we chose **$h = 2$ attention heads**, our total dimension is sliced in half ($4 / 2 = 2$), shrinking each head's weight matrices down to a tidy **$2 \times 2$** grid.


* **The Traditional Math Paradox (Left vs. Right):** In standard linear algebra classes, if you want to transform a vector using a matrix, you typically write the matrix on the left and the vector on the right ($W \cdot \mathbf{x}$). But if you tried that with a single row token vector (shape $1 \times 2$) and a $2 \times 2$ weight matrix, standard math rules would break because the inner dimensions wouldn't match!
* **The Deep Learning Solution (The Left-Side Rule):** Deep learning frameworks flip this convention entirely to make processing massive batches of tokens blazing fast on GPUs. Instead of putting the input on the right, **the input vector or matrix comes from the LEFT, and the weight matrix sits on the RIGHT.**
* **Why This Works for Any Sequence Length:** Because of this left-side rule, our input matrix $X_{\text{head1}}$ acts as the driver. It has shape $N \times d_k$ (in our example, $2 \text{ tokens} \times 2 \text{ dimensions}$). When we post-multiply it by our $2 \times 2$ weight matrix, the dimensions match up perfectly:

$$\underbrace{X_{\text{head1}}}_{(2 \times 2)} \times \underbrace{W_Q}_{(2 \times 2)} = \underbrace{Q}_{(2 \times 2)}$$



If we had $100$ tokens instead of just $2$, our input matrix would be $100 \times 2$, multiplying smoothly against the exact same $2 \times 2$ weight matrix to spit out a brand new $100 \times 2$ Query matrix. The weight matrix size never changes—it acts like a universal translator tailored strictly to the head's dimension width ($d_k$), no matter how long your prompt sequence grows!

For Head 1, let's assume gradient descent successfully tuned these $2 \times 2$ matrices to look like this:

$$W_Q = \begin{pmatrix} 1 & 0 \\\\ 0 & 1 \end{pmatrix}, \quad W_K = \begin{pmatrix} 1 & 1 \\\\ 0 & 1 \end{pmatrix}, \quad W_V = \begin{pmatrix} 2 & 0 \\\\ 0 & 2 \end{pmatrix}$$

---

## 4. Computing Queries ($Q$), Keys ($K$), and Values ($V$)

Now, Head 1 multiplies its baby input matrix $X_{\text{head1}}$ by its weight matrices. This transforms our raw tokens into Queries, Keys, and Values:

### Step A: The Value Vectors ($V$) — *The Content Payload*

$$
V = X_{\text{head1}} W_V = \begin{pmatrix} 1 & 1 \\\\ 2 & 2 \end{pmatrix} \begin{pmatrix} 2 & 0 \\\\ 0 & 2 \end{pmatrix} = \begin{pmatrix} 2 & 2 \\\\ 4 & 4 \end{pmatrix}
$$

* **The Eloquent View:** Think of $W_V$ as a **feature amplifier and modifier**. By multiplying the input features by $W_V$, the model selectively amplifies certain characteristics of the word while attenuating others. For example, it takes `"I"` $(1, 1)$ and transforms it into a richer content payload $(2, 2)$, packaging up the word's actual semantic meaning ready to be shared with other tokens.
* **The Caveman Speed-Dating Analogy:** Think of the Value matrix as the **facade or outfit change** before stepping into the room. If a token is a rich kid going to a speed-dating event, the Value matrix gets him to wear an Armani suit and a Rolex to shape what he actually brings to the table. 

### Step B: The Query ($Q$) and Key ($K$) Vectors — *The Address Tags and Search Radar*

$$
Q = X_{\text{head1}} W_Q = \begin{pmatrix} 1 & 1 \\\\ 2 & 2 \end{pmatrix} \begin{pmatrix} 1 & 0 \\\\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 1 & 1 \\\\ 2 & 2 \end{pmatrix}
$$

$$
K = X_{\text{head1}} W_K = \begin{pmatrix} 1 & 1 \\\\ 2 & 2 \end{pmatrix} \begin{pmatrix} 1 & 1 \\\\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 1 & 2 \\\\ 2 & 4 \end{pmatrix}
$$

* **The Eloquent View for Keys ($K$):** Think of the Key matrix ($W_K$) as an **id badge generator**. When token 2 (`"love"`) multiplies against $W_K$, it produces a Key vector $(2, 4)$ that acts like a public broadcast tag: *"Hey, I am a verb, I represent an emotion, and I link well with subject pronouns."*
* **The Caveman Speed-Dating Analogy for Keys ($K$):** This is the **public shouting part**. The superficial rich kid token yells out loud across the room: *"Hey girls, I am driving an Aston Martin today!"* 
* **The Eloquent View for Queries ($Q$):** Think of the Query matrix ($W_Q$) as a **radar scanner or a question generator**. When token 1 (`"I"`) multiplies against $W_Q$, it produces a Query vector $(1, 1)$ that acts like a question: *"I am looking for actions or verbs that connect back to me as a subject."*
* **The Caveman Speed-Dating Analogy for Queries ($Q$):** This is your **picker criteria**. The token scans the room saying: *"Just interested in blonde hair and blue eyes, intelligence is not required :D"* (or conversely, a token looking for depth might scan for substance, while someone else turns away from superficial flash).

---

> **A Crucial Engineering Note (What Actually Gets Cached?):**
> When people talk about saving compute time and storing things in the **KV cache** during the prefill phase, notice that **only the Keys ($K$) and Values ($V$) get saved**.
> * Why not Queries ($Q$)? Because Queries are prompt-specific questions used right now to figure out immediate relationships.
> * Once the model moves to the decode phase and starts generating new tokens one by one (like predicting `"you"`), it doesn't need old Queries. It only needs the pre-computed **Keys and Values** of the past tokens (`"I"` and `"love"`) so the new token can instantly shoot out its own Query and cross-compare against them without recalculating the past from scratch!

---

Naturally, this is the linear algebra taking place in Head 1 of global attention layer 1, but Head 2 would do similar parallel algebra for the remaining dimensions.

---

## 5. The Query-Key Voodoo ($QK^T$)

To find out how much attention token 1 should pay to token 2 (and itself), we take the dot product of Queries and transposed Keys ($QK^T$). This creates an $N \times N$ correlation grid ($2 \times 2$ in our case):

$$
QK^T = \begin{pmatrix} 1 & 1 \\\\ 2 & 2 \end{pmatrix} \begin{pmatrix} 1 & 2 \\\\ 2 & 4 \end{pmatrix}^T = \begin{pmatrix} 1 & 1 \\\\ 2 & 2 \end{pmatrix} \begin{pmatrix} 1 & 2 \\\\ 2 & 4 \end{pmatrix} = \begin{pmatrix} 3 & 6 \\\\ 6 & 12 \end{pmatrix}
$$

> **A Quick University Flashback (What is a Dot Product?):**
> If you haven't touched linear algebra since university, a dot product is simply a mathematical way to measure how much two vectors "point in the same direction" or agree with each other. Given two vectors $\mathbf{v}_1 = (a, b)$ and $\mathbf{v}_2 = (c, d)$, you compute it by multiplying corresponding elements and adding them together: $(a \cdot c) + (b \cdot d)$. 
> * **Geometrical Intuition:** If two vectors point in similar directions, their dot product yields a large positive number. If they point in opposite directions, it yields a negative number. If they are completely orthogonal (perpendicular/unrelated), it yields zero. That is why it is the ultimate matching tool for AI!

---

### The Speed-Dating Reality TV Show Analogy
Let's bring this down to our caveman speed-dating session to see how this matrix multiplication actually plays out:

* **The Queries ($Q$ Rows):** Row 1 represents the query of our superficial rich kid token (*"Are you a blue-eyed blonde babe? Intelligence is not required :D"*). Row 2 might represent a token looking for true substance (*"Looking for a guy with actual depth, not a troglodyte with cash"*).
* **The Transposed Keys ($K^T$ Columns):** Column 1 of our transposed key matrix represents the rich kid shouting his advertisement (*"Hey girls, I came in an expensive Aston Martin!"*). Column 2 represents someone else's advertisement (*"I am a quiet, deep bookworm"*).
* **Cross-Multiplying ($QK^T$):** 
  * When **Row 1** (the rich kid's query) multiplies against **Column 1** (his own advertisement), it evaluates compatibility with himself. The low compatibility score says: *Yeah, I didn't come to a speed dating session to date myself, hard pass.* But when Row 1 multiplies against **Column 2**, he spots the blonde babe and his interest spikes.
  * When **Row 2** (the deep bookworm's query) multiplies against Column 1 (the rich kid in the Aston Martin), the result reflects her reaction: *I like the car, but I hate the dude.* 

---

### Scaling and Softmax: Turning Scores Into Percentages

Next, the model scales this grid down by dividing by $\sqrt{d_k}$ (here $\sqrt{2} \approx 1.41$) to keep numbers stable, and runs it through a **softmax** function. 

> **The Magic of Softmax:** Look at how softmax cleans up the raw math: **every single row of this new matrix adds up exactly to 1 (100%)**. 
> * For **Row 1**, the scores normalize so that the weights sum to $0.2 + 0.8 = 1.0$.
> * For **Row 2**, the scores normalize so that the weights sum to $0.1 + 0.9 = 1.0$.

Let's look at our final normalized attention weights matrix:

$$
\text{Attention Scores} = \begin{pmatrix} 0.2 & 0.8 \\\\ 0.1 & 0.9 \end{pmatrix}
$$

*Read this grid like this:*
* Token 1 (`"I"`) spends 20% of its attention on itself and 80% on `"love"`.
* Token 2 (`"love"`) spends 10% of its attention on `"I"` and 90% on itself.

*Read this grid like this:*
* Token 1 (`"I"`) spends 20% of its attention on itself and 80% on `"love"`.
* Token 2 (`"love"`) spends 10% of its attention on `"I"` and 90% on itself.

> **The Speed-Dating Reality Check:** 
> But looking at it purely as academic percentages is too dry—let's translate that right back to our speed-dating event:
> * **Row 1 (The Rich Kid):** That 20% self-attention cell is essentially the rich kid looking at himself and thinking, *"Yeah, I didn't come to speed-dating to meet myself."* But look at that heavy **80%** allocation on column 2: his search radar for the blonde babe hit the jackpot, and he is fully locked in.
> * **Row 2 (The Bookworm Girl):** That tiny **10%** score pointing toward the rich kid shows she is totally put off by him—not even the Aston Martin could salvage the situation. Meanwhile, her **90%** self-attention score shows she walks out of the event thinking, *"Apparently my best match tonight was myself; better alone than in bad company."* :D


> **Why This Matrix is Thrown Away (The Secret to the Decode Phase):**
> Look closely at what just happened: this $N \times N$ compatibility matrix tells us how every token in the current prompt relates to every other token. But once that new token (`"you"`) is conjured and added to the sentence, **this entire speed-dating score matrix is thrown away**. 
> * **The Old Queries Are Gone:** We no longer need the Queries ($Q$) of past tokens like `"I"` and `"love"` because their historical peer-to-peer relationships are already locked in stone.
> * **The Newcomer Arrives:** When the decode phase begins, a brand-new token enters the speed-dating party. It generates its *own* fresh Query ($Q$), and it shoots that query straight at the **permanently cached Keys ($K$)** of all the past tokens. 
> * **The Next Round:** Instead of re-running the whole party from scratch, the model only computes how the new token's Query matches against everyone already in the room, blending their **cached Values ($V$)** on the fly. That is why caching $K$ and $V$ saves GPUs from melting down!



---

## 6. The Grand Finale: Multiplying by Values ($V$)

The final step of Head 1 takes our attention percentage grid and multiplies it by our Value matrix ($V$):

$$\text{Output} = (\text{Attention Scores}) \times V = \begin{pmatrix} 0.2 & 0.8 \\\\ 0.1 & 0.9 \end{pmatrix} \begin{pmatrix} 2 & 2 \\\\ 4 & 4 \end{pmatrix} = \begin{pmatrix} 3.6 & 3.6 \\\\ 3.8 & 3.8 \end{pmatrix}$$

Head 1 has successfully taken raw input tokens, routed them through trained weight matrices, cross-compared them via Queries and Keys, and outputted brand-new, context-infused representations. Head 2 is doing the exact same thing in parallel on dimensions 3 and 4.

During this prefill warmup, these computed Key and Value vectors are saved into the **KV cache**, completing our gym session and leaving the model fully prepared to predict `"you"` next!


---

A Quick Side Note: This guide was co-authored with Gemini, who displayed the infinite patience of a saint while translating sterile transformer math into a world of speed-dating rich kids, Armani suits, and Aston Martins. If the equations look pristine, blame the framework; if it feels like a bizarre reality TV show, blame the human co-pilot.
