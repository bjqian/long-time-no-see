# Large Language Model
The LLM is leading the AI revolution, it would be interesting to understand the underhood of the model itself. 

Intuitively, the math and concepts of LLM should be easy to understand. 

## Tokenization
Tokenization is a process of grouping charactors into tokens. For LLM, this process could compact the input and reduce the context size. It enhances the model's capability of understanding concepts during the training process.

A well adopted tokenization algorithm is BPE. It is originally introduced to compact text. A revised version is used in LLM to tokenize the input to a desired length of lookup table.

It won't take long to implement it. Will come back and add an implementation.

## Transformer
I have got a rough concept of the transformer. It's a multi layer neuro network. Each layer contains multiple "head" to do self-attention prediction.
### Let's assume the input is X1, the flow of prediction is like this:
X1 -> TransformerModel.Forward -> X2 . X2 is the prediction of X1 which means X2(i,j) is the next token of X1(i,j). Actually not exactlly like this as there is a tokenization process during the TransformerModel. 
### Inside the TransformerModel:
X1 -> Layer1.Forward -> Layer2.Forward ... -> LayerN.Forward -> X2. (This is a classical multilayer neuro network). The forward is done sequencially.
### Inside each Layer:
This is the key of self-attention:
Assume X1.shape = [B, T, C]. 
Each layer will have m heads which could run parallel.
So, we have:
Parallel( head_i(x) for i in m). 
The output of head(x) is of shape [B, T, head_size].
Concat all the output of the heads, we get the output of this layer.
### Inside each header:
Each header has three matrixs called **Key**, **Query**, **Value**. Each of them have shape [C, head_size]
Multiple them each with input X1, we have:
key = Key(X1)  // [B,T, head_size]
query = Query(X1)
value = Value(X1)

I haven't quite got how this [K,Q,V] mechanism comes out. But literally, we could imaging 
Key is asking what to find,
Query is asking how to find,
Value is giving the current context.
Then the result is a partial prediction.
The output = key@query@value.

### The power of transformer
The heads are all independent so they could be trained parallelly. So it could be easily scaled. 
