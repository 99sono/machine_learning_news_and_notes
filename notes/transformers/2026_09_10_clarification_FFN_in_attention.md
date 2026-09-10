Yes, you have the core mechanics right: prefill is never just the attention matrix.
What Happens in Every Transformer Layer
Every single layer in a transformer stack is a two-step block: Attention followed by an FFN.
 * Attention Block: Mixes information between tokens (Q \times K^T \times V).
 * FFN Block: Processes the output of the attention block for each token individually, projecting the features into a higher-dimensional space, applying non-linearity, and projecting back down.
During prefill, every token in your prompt flows through the Attention mechanism and the FFN in Layer 1, then the Attention mechanism and the FFN in Layer 2, and so on all the way to the final layer.
Standard FFN vs. MoE FFN
 * Dense Models: Every layer has a standard, fixed Feed-Forward Network.
 * Mixture of Experts (MoE) Models: That exact same FFN slot is simply replaced by a Router plus a pool of Expert FFNs. Instead of running all FFN weights, the router picks a couple of experts for each token. But an FFN computation still happens in that layer.
Do Attention Blocks Have Their Own FFNs?
Yes. Every layer pairs one attention mechanism with its own dedicated FFN. Layer 1 has Attention 1 and FFN 1. Layer 2 has Attention 2 and FFN 2. They do not share FFN weights across layers.
Clearing Up the KV Cache and Training
There are no "weights for the KV cache" to train.
 * Training: The model learns fixed weights—like the projection matrices (W_q, W_k, W_v) and the FFN weights.
 * KV Cache: This is purely a runtime memory buffer. When tokens pass through the layers during prefill, the model computes their Keys and Values using those trained W_k and W_v weights, and saves those resulting vectors into the cache so it doesn't have to recalculate them later during decoding.

