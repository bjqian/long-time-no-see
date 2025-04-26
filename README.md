# Large Language Model
The LLM is leading the AI revolution, and it’s interesting to see what’s going on under the hood.

Intuitively, the math and ideas behind an LLM are not too hard to understand.

## Tokenization
Tokenization is the process of grouping **characters** into **tokens**. For an LLM, this compacts the input and reduces the context size. It also helps the model learn concepts during training.

A common tokenization method is **BPE** (Byte-Pair Encoding). It was first used to compress text; a revised version is used in LLMs to map text into a fixed length of tokens.

*(I’ll come back and add an implementation.)*

## Transformer
I have a rough understanding of the Transformer now. It’s a multi-layer neural network. Each layer has several “heads” that do self-attention.

### Flow of prediction
Assume the input is `X1`:

```
X1 -> TransformerModel.forward -> X2
```

`X2` is the prediction for `X1`, so `X2(i,j)` is the next token of `X1(i,j)`. (In practice, tokenization happens before the Transformer.)

### Inside the Transformer
```
X1 -> Layer1.forward -> Layer2.forward -> … -> LayerN.forward -> X2
```
Layers run one after another.

### Inside each layer
Self-attention is the key idea.

* Let `X` have shape `[B, T, C]`.
* Each layer has **m** heads that run in parallel:

```
head_out_i = head_i(X)  for i in 1..m
```

* Each `head_out_i` has shape `[B, T, head_size]`.
* Concatenate all `head_out_i` to get the layer output.

### Inside each head
Each head has three matrices:

| Name  | Shape             |
|-------|-------------------|
| Key   | `[C, head_size]`  |
| Query | `[C, head_size]`  |
| Value | `[C, head_size]`  |

Given `X`:

```text
key   = X @ Key
query = X @ Query
value = X @ Value
```

My current (rough) understanding:

* **Key** says what to look for.
* **Query** says how to match.
* **Value** carries the context.

The head’s output comes from combining `key`, `query`, and `value`.

### Why the Transformer scales
All heads are independent, so they can train in parallel and scale up easily.
