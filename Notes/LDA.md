### **Latent Dirichlet Allocation (LDA)**: Mathematical Explanation

LDA is a generative probabilistic model used to discover topics in a corpus of documents. It assumes that:

1. Documents are mixtures of topics.
2. Topics are distributions over words.

#### **Mathematical Representation**
1. **Generative Process**:
   For each document $(d)$ in the corpus:
   1. Draw a distribution over topics: $(\theta_d \sim \text{Dirichlet}(\alpha))$, where $(\alpha)$ is the hyperparameter for topic proportions.
   2. For each word $(w_{d,n})$ (the $(n)$-th word in document $(d)$):
      - Draw a topic $(z_{d,n} \sim \text{Categorical}(\theta_d))$.
      - Draw a word $(w_{d,n} \sim \text{Categorical}(\phi_{z_{d,n}}))$, where $(\phi_k \sim \text{Dirichlet}(\beta))$ represents the distribution of words for topic $(k)$.

Here:
- $(w_{d,n})$: $(n)$-th word in the document $(d)$.
- $(z_{d,n})$: Topic assignment for $(w_{d,n})$.
- $(\theta_d)$: Document-topic distribution.
- $(\phi_k)$: Topic-word distribution.

2. **Joint Probability**:
   The joint probability of the observed words $(W)$, topic assignments $(Z)$, and parameters $(\theta, \phi)$ is:

  $$
   P(W, Z, \theta, \phi | \alpha, \beta) = \prod_{d=1}^D P(\theta_d | \alpha) \prod_{k=1}^K P(\phi_k | \beta) \prod_{n=1}^{N_d} P(z_{d,n} | \theta_d) P(w_{d,n} | \phi_{z_{d,n}})
   $$

3. **Inference**:
   The objective is to estimate the posterior distribution:

  $$
   P(\theta, \phi, Z | W, \alpha, \beta)
   $$

   Exact inference is intractable, so algorithms like **variational inference** or **Gibbs sampling** are used.

---

### **Hyperparameters in LDA**
1. **$(\alpha)$**: Controls the sparsity of the document-topic distribution ($(\theta_d)$).
   - Smaller $(\alpha)$: Sparse distributions (fewer topics per document).
   - Larger $(\alpha)$: Dense distributions (many topics per document).

2. **$(\beta)$**: Controls the sparsity of the topic-word distribution ($(\phi_k)$).
   - Smaller $(\beta)$: Sparse distributions (fewer words per topic).
   - Larger $(\beta)$: Dense distributions (many words per topic).

---

### **Calculating Accuracy**
Accuracy in LDA is not as straightforward as classification tasks. Instead, quality is evaluated using the following metrics:

1. **Perplexity**:
   Measures how well the model predicts a set of unseen documents. Lower perplexity indicates better generalization.

  $$
   \text{Perplexity} = \exp \left( -\frac{\sum_{d=1}^D \log P(w_d | \alpha, \beta)}{\sum_{d=1}^D N_d} \right)
   $$

   Here $(N_d)$ is the number of words in document $(d)$, and $(P(w_d | \alpha, \beta))$ is the likelihood of the document.

2. **Coherence Score**:
   Measures the interpretability of topics by evaluating the semantic similarity of top words within topics.

   Methods include:
   - $(C_v)$: Based on a sliding window and normalized Pointwise Mutual Information (NPMI).
   - $(C_{UMass})$: Based on co-occurrence statistics from a reference corpus.

3. **Human Judgment**:
   Human experts evaluate the interpretability and coherence of topics.

4. **Classification Tasks**:
   Use the document-topic distributions as features in downstream classification tasks and measure the classifier's accuracy.

---

Let me know if you want a detailed explanation of inference methods like Gibbs sampling or a Python implementation!