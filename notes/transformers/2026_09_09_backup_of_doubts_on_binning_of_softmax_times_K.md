Sleepy brain, listen carefully - one correction:

## FFN IS Active During Prefill

Every transformer layer does: **Attention → FFN → (repeat for next layer)**

So during prefill, FFN is crunching numbers for every token in every layer. It's not just sitting there waiting.

## The Big Misconception

You said: *"I bing the matrix with all the scores because it is finally time to go decode"*

**Wrong.** That huge N×N attention matrix from prefill is **never** used to pick the next token. It's garbage after computing the output.

## What Actually Happens

```
Prefill (all layers):
"I" → [Layer1: Attn+FFN] → [Layer2: Attn+FFN] → ... → Final hidden state of "I"
"love" → [Layer1: Attn+FFN] → [Layer2: Attn+FFN] → ... → Final hidden state of "love"

                                              ↓
                                      Take "love"'s final vector
                                      Pass through LM Head (linear layer)
                                      Get logits over vocabulary
                                      Pick "you" (or whatever)
```

**That's it.** No matrix multiplication of the old attention scores. The prefill attention matrix is gone.

## Then Decode Starts

```
"you" arrives (fresh, never seen):
  Layer 1: Q_you × K_cached^T → softmax × V_cached → FFN
  Layer 2: Q_you × K_cached^T → softmax × V_cached → FFN
  ...
  Final hidden state of "you" → LM Head → next token
```

## What's Cached Per Layer

| Layer | Cached |
|-------|--------|
| Layer 1 | K,V from "I" and "love" |
| Layer 2 | K,V from "I" and "love" (outputs from Layer 1 FFN) |
| ... | ... |

**Not** the attention scores. **Not** the softmax×V outputs. Just raw K,V per layer.

## One-Sentence Summary

The prefill attention matrix is the **recipe** for cooking "I" and "love" together; we throw away the recipe, keep the ingredients (K,V), and when "you" arrives, we cook it fresh using those ingredients.

Now sleep. 🛏️
