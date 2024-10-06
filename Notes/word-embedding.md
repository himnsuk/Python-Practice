Word Embedding
---

Word embeddings are a technique to represent words in a continuous vector space where similar words have similar vector representations. These embeddings capture semantic relationships and are fundamental in many natural language processing (NLP) tasks.

Here are the main types of word embeddings:

### 1. **One-Hot Encoding (Baseline Approach)**
This is the simplest form of word representation where each word in the vocabulary is represented by a binary vector with only one "1" and the rest "0s". Each word gets a unique index.

- **Mathematics**:
    - Suppose your vocabulary $( V )$ has $( n )$ words. For each word $( w_i )$, its one-hot vector is $( O_i \in \mathbb{R}^n )$, where $( O_i )$ has all zeroes except at the $( i )$-th position:
    $$
    O_i = [0, 0, ..., 1, ..., 0]
    $$
    - **Problem**: One-hot encoding does not capture any semantic similarity. For example, the vectors for "king" and "queen" are orthogonal (completely dissimilar), even though the two words are semantically related.

### 2. **TF-IDF (Term Frequency-Inverse Document Frequency)**
TF-IDF is a statistical measure that represents the importance of a word in a document relative to the corpus.

- **Mathematics**:
    - **Term Frequency (TF)**: Measures how frequently a term appears in a document.
      $$
      \text{TF}(t, d) = \frac{\text{Count of term } t \text{ in document } d}{\text{Total terms in document } d}
      $$
    - **Inverse Document Frequency (IDF)**: Reduces the weight of common words across many documents.
      $$
      \text{IDF}(t) = \log\left(\frac{\text{Total number of documents}}{\text{Number of documents containing } t}\right)
      $$
    - **TF-IDF**: Combines TF and IDF to create the final weight.
      $$
      \text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)
      $$
    - **Problem**: TF-IDF is sparse and does not capture semantic similarity between words.

### 3. **Word2Vec (Continuous Word Embeddings)**

Word2Vec is one of the most popular word embedding techniques, developed by Google. It comes in two variants: Continuous Bag of Words (CBOW) and Skip-gram. Both models aim to learn distributed vector representations for words, capturing the context and meaning in a dense vector space.

#### (a) **Skip-gram Model**:
The **skip-gram** model predicts the context words given a target word. It aims to maximize the probability of context words appearing around the target word within a fixed window size.

- **Mathematics**:
    - Let the target word be $( w_t )$ and the context words be $( w_{t-k}, ..., w_{t+k} )$ (within a window size $( k )$).
    - The objective is to maximize the probability of the context words given the target word:
    $$
    \prod_{-k \leq j \leq k, j \neq 0} P(w_{t+j} | w_t)
    $$
    - The probability $( P(w_{t+j} | w_t) )$ is calculated using the softmax function over the dot product of the word vectors:
    $$
    P(w_{t+j} | w_t) = \frac{\exp(v_{w_{t+j}}^\top v_{w_t})}{\sum_{w \in V} \exp(v_w^\top v_{w_t})}
    $$
    Where:
    - $( v_w )$ is the vector representation of word $( w )$,
    - $( V )$ is the vocabulary.
  
#### (b) **CBOW Model**:
The **CBOW** model, in contrast, predicts the target word given the context words. The objective is to maximize the probability of a target word given its surrounding context.

- **Mathematics**:
    - Given the context words $( w_{t-k}, ..., w_{t+k} )$, predict the target word $( w_t )$.
    - The objective is to maximize:
    $$
    P(w_t | w_{t-k}, ..., w_{t+k}) = \frac{\exp(v_{w_t}^\top \left(\sum_{-k \leq j \leq k, j \neq 0} v_{w_{t+j}}\right))}{\sum_{w \in V} \exp(v_w^\top \left(\sum_{-k \leq j \leq k, j \neq 0} v_{w_{t+j}}\right))}
    $$
    - The sum of the context vectors is used to predict the target word.

**Advantages**: Word2Vec learns dense, continuous word representations that capture syntactic and semantic relationships (e.g., **king** - **man** + **woman** = **queen**).

### 4. **GloVe (Global Vectors for Word Representation)**

GloVe combines the strengths of Word2Vec and matrix factorization methods. It constructs a word co-occurrence matrix and factors it to learn word embeddings. GloVe focuses on capturing the global statistical information of words, not just the local context.

- **Mathematics**:
    - Let $( X_{ij} )$ be the co-occurrence count of words $( i )$ and $( j )$.
    - GloVe models the ratio of co-occurrences rather than directly maximizing a probability:
    $$
    f(w_i^\top w_j) = \log(X_{ij})
    $$
    - The objective is to minimize the following cost function:
    $$
    J = \sum_{i,j} f(X_{ij})(w_i^\top w_j + b_i + b_j - \log(X_{ij}))^2
    $$
    Where:
    - $( w_i )$ and $( w_j )$ are word vectors for words $( i )$ and $( j )$,
    - $( b_i )$ and $( b_j )$ are biases,
    - $( f(X_{ij}) )$ is a weighting function that reduces the impact of very frequent word pairs.

**Advantages**: GloVe learns embeddings that reflect the co-occurrence statistics of words across the corpus, capturing both local and global patterns.

### 5. **FastText**

FastText, developed by Facebook, extends Word2Vec by considering subword information. It represents each word as a bag of character n-grams. This makes FastText particularly useful for morphologically rich languages and rare words.

- **Mathematics**:
    - Let $( n )$-grams be subsequences of $( n )$ characters from the word. For instance, for the word "apple" and $( n = 3 )$, we have the character n-grams: `app`, `ppl`, `ple`.
    - Instead of learning a vector for the entire word, FastText learns vectors for each $( n )$-gram and sums them to represent the word:
    $$
    v_{\text{word}} = \sum_{\text{ngram} \in \text{word}} v_{\text{ngram}}
    $$
    - The skip-gram or CBOW model can then be applied to these word vectors as in Word2Vec.

**Advantages**: FastText captures subword information, enabling it to handle rare words, out-of-vocabulary words, and morphologically rich languages better than Word2Vec.

### 6. **BERT (Bidirectional Encoder Representations from Transformers)**

BERT is a contextual word embedding model that uses the Transformer architecture to generate word embeddings by considering both the left and right contexts of a word. BERT embeddings are dynamic and depend on the surrounding words, unlike static embeddings such as Word2Vec or GloVe.

- **Mathematics**:
    - BERT uses a **bidirectional transformer** architecture that applies attention mechanisms. The attention score for a word $( w_i )$ with respect to another word $( w_j )$ is calculated as:
    $$
    \text{Attention}(w_i, w_j) = \frac{\exp(e_{ij})}{\sum_{k=1}^{n} \exp(e_{ik})}
    $$
    Where:
    - $( e_{ij} = w_i^\top W w_j )$ is the score between words $( i )$ and $( j )$ (based on their dot product in a learned space),
    - $( W )$ is the weight matrix.
    - BERT learns the relationship between words in both directions (i.e., left-to-right and right-to-left context) by using **masked language modeling** and **next sentence prediction** tasks.
  
**Advantages**: BERT embeddings capture context dynamically, meaning that the same word can have different embeddings depending on its context in a sentence.

---

### Conclusion

In an interview, you can explain word embeddings as the transformation of words into continuous vector spaces where words with similar meanings are close to each other. Highlight the differences between methods like **Word2Vec**, **GloVe**, and **BERT**:

- **Word2Vec**: Learns embeddings based on local context.
- **GloVe**: Captures global co-occurrence statistics.
- **BERT**: Provides contextual embeddings using the Transformer architecture.

You can further emphasize the mathematical intuition behind each method by explaining the objective functions and how embeddings are learned based on word contexts


---
Attention Model
---

### Introduction to the Attention Mechanism

The **Attention Mechanism** is a core component in modern deep learning models, especially in Natural Language Processing (NLP). It was introduced to address the limitations of traditional sequence models like RNNs and LSTMs, particularly when dealing with long-range dependencies in sequences.

The core idea of attention is to allow the model to focus on specific parts of the input sequence when making predictions. In traditional models like LSTMs, every word in a sentence is processed sequentially, and all information gets compressed into a fixed-size vector (context vector). This can make it hard for the model to retain important information when dealing with long sequences. **Attention** solves this by assigning different weights to different parts of the input, allowing the model to focus more on the most relevant information.

---

### Basic Concept of Attention Mechanism

In an **attention model**, we compute a weighted sum of **values** based on **attention scores** (or weights) assigned to each input element (called **keys**), where the scores depend on some relationship (computed through a **query**) between the current word and every other word in the input sequence.

- **Query**: What you're trying to match (the input sequence's current state).
- **Key**: Represents the input sequence you're searching through.
- **Value**: The actual data you're trying to retrieve or focus on, which is often the same as the keys.

These three components—queries, keys, and values—are used to compute attention scores that indicate how much "attention" the model should pay to each part of the input.

---

### Mathematical Explanation of Attention

The attention mechanism computes a **weighted sum** of the **values** based on attention scores, which are derived from the **queries** and **keys**.

#### 1. **Dot-Product Attention** (Simplest Form)
Let:
- $( Q \in \mathbb{R}^{d_k} )$ be the **query vector** (current word representation),
- $( K \in \mathbb{R}^{d_k} )$ be the **key vector** (representation of a word in the input sequence),
- $( V \in \mathbb{R}^{d_v} )$ be the **value vector** (what we’re trying to retrieve, often the same as the key).

The attention score between a query $( Q )$ and key $( K )$ is computed as the dot product between them:
$$
\text{Attention}(Q, K) = Q^\top K = \sum_{i=1}^{d_k} Q_i K_i
$$
This measures how much one word relates to another based on their vector representations.

#### 2. **Scaled Dot-Product Attention**

To prevent large dot products that might make gradients too small or too large, the dot product is scaled by the square root of the dimension of the key vectors $( d_k )$:
$$
\text{Attention}(Q, K) = \frac{Q^\top K}{\sqrt{d_k}}
$$

#### 3. **Softmax Function**:
Once we compute the dot-product between the query and keys, we use the **softmax** function to normalize these scores into probabilities (attention weights):
$$
\alpha_{ij} = \frac{\exp\left(\frac{Q_i^\top K_j}{\sqrt{d_k}}\right)}{\sum_{j=1}^{n} \exp\left(\frac{Q_i^\top K_j}{\sqrt{d_k}}\right)}
$$
Where:
- $( \alpha_{ij} )$ is the attention weight for the $( i )$-th query and $( j )$-th key.

#### 4. **Weighted Sum** (Output of Attention):
Once we have the attention weights, we compute a **weighted sum** of the value vectors $( V )$. The output of the attention mechanism is:
$$
\text{Output}_i = \sum_{j=1}^{n} \alpha_{ij} V_j
$$
This gives the final weighted sum of values, where each value $( V_j )$ is weighted by how much attention $( \alpha_{ij} )$ it receives.

---

### Self-Attention

The attention mechanism can be applied to the same sequence, allowing each word to "attend" to every other word in the sequence, including itself. This is called **self-attention**.

In self-attention:
- The **query, key, and value** vectors are derived from the same input sequence.
- Every word in the sequence looks at every other word and computes attention scores, which tell the model which words are most relevant to focus on when processing the current word.

Self-attention is the foundation of **Transformer** models, which have largely replaced RNNs and LSTMs in NLP tasks because they allow parallelization and can handle long-range dependencies efficiently.

---

### Transformer Architecture (Advance Attention)

The **Transformer** architecture, introduced in the paper "Attention is All You Need," uses attention as the core mechanism to process sequences without the need for recurrent layers (like RNNs or LSTMs).

#### 1. **Multi-Head Self-Attention**
The Transformer extends the attention mechanism with **multi-head self-attention**, which allows the model to look at different parts of the input sequence simultaneously from multiple "perspectives."

- **Multi-head attention** involves running several independent self-attention mechanisms in parallel, each with its own learned set of query, key, and value matrices.
- The outputs of these attention heads are then concatenated and linearly transformed.

For $( h )$ attention heads:
$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, ..., \text{head}_h)W^O
$$
Where:
- $( \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) )$,
- $( W_i^Q, W_i^K, W_i^V )$ are learned projection matrices for queries, keys, and values,
- $( W^O )$ is a learned projection matrix for the final output.

#### 2. **Position Encoding**:
Since Transformers process all words simultaneously (in parallel), they lack an inherent understanding of the word order. To overcome this, **positional encodings** are added to the input embeddings to give the model a sense of word positions in the sequence. These encodings are often computed using sinusoidal functions.

- Position encodings are added to the word embeddings before applying attention:
$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\right)
$$
$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\right)
$$
Where:
- $( pos )$ is the position,
- $( i )$ is the dimension.

---

### Mathematical Summary of Attention in Transformers

1. **Self-Attention**:
    - Compute attention scores: $( \frac{Q^\top K}{\sqrt{d_k}} )$.
    - Normalize scores with softmax: $( \alpha_{ij} )$.
    - Compute the weighted sum of values: $( \sum_{j} \alpha_{ij} V_j )$.

2. **Multi-Head Attention**:
    - Split the query, key, and value matrices into multiple heads.
    - Perform self-attention independently for each head.
    - Concatenate the results and apply a linear transformation.

3. **Positional Encoding**:
    - Add positional information to embeddings using sinusoidal functions.

---

### How to Explain Attention in a Data Science Interview

1. **Start with the Problem**:
    - Mention how traditional models like RNNs struggle with long-range dependencies and compress all information into a fixed-length vector.
    - Explain that attention solves this problem by dynamically focusing on different parts of the input sequence.

2. **Explain the Concept**:
    - Introduce the idea of queries, keys, and values. For example, "Attention is like a search process: the query is what we are looking for, the keys are the candidates, and the values are what we retrieve."
    - Explain that attention computes a relevance score for each key based on the query and retrieves the most relevant values.

3. **Mathematics**:
    - Mention that attention scores are computed as the dot product between queries and keys, scaled by $( \sqrt{d_k} )$, and converted into probabilities using softmax.
    - Show how these scores are used to take a weighted sum of the value vectors, focusing on the most relevant information.

4. **Move to Self-Attention**:
    - Explain how self-attention allows each word in a sentence to focus on every other word, including itself.
    - This helps capture relationships between words in a sentence regardless of their distance in the sequence.

5. **Explain Transformers**:
    - Transition to how attention forms the foundation of Transformers, with multi-head self-attention allowing the model to look at different aspects of the input.
    - Mention how this parallel processing makes Transformers more efficient than RNNs for handling large sequences.

6. **Real-World Relevance**:
    - Discuss how attention models have transformed NLP tasks, enabling state-of-the-art performance in tasks like machine translation, text summarization, and question answering.
    - If possible, relate it to tasks you've worked on or seen in real-world applications.

This structure will help you convey the concept of attention from basic to advanced in a clear, data scientist-friendly manner during an interview.