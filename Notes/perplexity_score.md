Perplexity
----
**Perplexity** is a measurement used to evaluate the performance of a probabilistic model, especially in natural language processing (NLP). It measures how well a probability distribution or model predicts a sample. Lower perplexity indicates a better model that predicts the data more accurately.

### Mathematical Definition of Perplexity:

For a language model, perplexity is defined as the exponentiation of the average negative log-likelihood of the true word sequence, given by:

$$
\text{Perplexity}(P) = 2^{H(P)}
$$

Where $(H(P))$ is the entropy of the model, and it is calculated as:

$$
H(P) = - \frac{1}{N} \sum_{i=1}^{N} \log_2 P(w_i)
$$

Here:
- $(N)$ is the number of words in the test set.
- $(w_i)$ represents the $(i)$-th word in the sequence.
- $(P(w_i))$ is the probability assigned by the model to the word $(w_i)$.

Alternatively, in a more generalized form using natural logarithms (base $(e)$):

$$
\text{Perplexity}(P) = e^{-\frac{1}{N} \sum_{i=1}^{N} \log P(w_i)}
$$

### Steps to Calculate Perplexity:
1. **Obtain the Probability Distribution**: Use the trained language model to obtain the probability $(P(w_i))$ of each word in a sequence or corpus.
   
2. **Calculate the Log Probability**: For each word in the sequence, take the logarithm of its predicted probability.

3. **Compute the Average Log Probability**: Sum up all the log probabilities and divide by the total number of words in the sequence ($(N)$).

4. **Exponentiate the Result**: Raise $(e)$ to the negative of the average log probability (or 2 in the case of log base 2).

### Example:

Suppose we have a sentence with 4 words, and the model assigns the following probabilities to the words:

$$
P(w_1) = 0.2, \quad P(w_2) = 0.3, \quad P(w_3) = 0.1, \quad P(w_4) = 0.4
$$

1. **Calculate the log probabilities**:
   $$
   \log P(w_1) = \log 0.2, \quad \log P(w_2) = \log 0.3, \quad \log P(w_3) = \log 0.1, \quad \log P(w_4) = \log 0.4
   $$
   
2. **Average log probability**:
   $$
   \frac{1}{4} \left( \log 0.2 + \log 0.3 + \log 0.1 + \log 0.4 \right)
   $$
   
3. **Exponentiate**:
   $$
   \text{Perplexity} = e^{-\frac{1}{4} (\log 0.2 + \log 0.3 + \log 0.1 + \log 0.4)}
   $$

In practice, the lower the perplexity, the better the model is at predicting the next word in a sequence, meaning the model is less "perplexed" by the data.