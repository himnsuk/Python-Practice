AdaBoost, short for **Adaptive Boosting**, is an ensemble learning algorithm designed to improve the accuracy of weak classifiers by combining them into a strong classifier. Below is a step-by-step mathematical explanation:

---

### **1. Problem Setup**
- You are given a labeled dataset:  
  $$
  D = \{(x_1, y_1), (x_2, y_2), \dots, (x_n, y_n)\}
  $$
  where $( x_i )$ is the input, $( y_i \in \{-1, +1\} )$ is the label, and $( n )$ is the number of samples.

- Initialize weights for all data points:
  $$
  w_i^{(1)} = \frac{1}{n}, \quad \text{for } i = 1, 2, \dots, n
  $$
  The weights $( w_i^{(t)} )$ indicate the importance of each data point at iteration $( t )$.

---

### **2. Iterative Training Process**
AdaBoost trains a sequence of $( T )$ weak classifiers $( h_t(x) )$, each focusing on correcting errors made by previous classifiers.

#### **Step 1: Train a Weak Classifier**
- At iteration $( t )$, train a weak classifier $( h_t(x) )$ (e.g., a decision stump) to minimize the weighted classification error:
  $$
  \epsilon_t = \sum_{i=1}^n w_i^{(t)} \cdot \mathbb{I}(h_t(x_i) \neq y_i)
  $$
  where $( \mathbb{I} )$ is an indicator function that equals 1 if $( h_t(x_i) \neq y_i )$, and 0 otherwise.

#### **Step 2: Compute Classifier's Weight**
- Compute the weight of the weak classifier, which reflects its accuracy:
  $$
  \alpha_t = \frac{1}{2} \ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right)
  $$
  - If $( \epsilon_t )$ is small (high accuracy), $( \alpha_t )$ is large.
  - If $( \epsilon_t = 0.5 )$, $( \alpha_t = 0 )$ (classifier performs no better than random chance).

#### **Step 3: Update Sample Weights**
- Update the weights of the data points:
  $$
  w_i^{(t+1)} = w_i^{(t)} \cdot \exp(-\alpha_t \cdot y_i \cdot h_t(x_i))
  $$
  - Correctly classified samples ($( y_i = h_t(x_i) )$) have reduced weights.
  - Misclassified samples ($( y_i \neq h_t(x_i) )$) have increased weights.

- Normalize weights to ensure they sum to 1:
  $$
  w_i^{(t+1)} \leftarrow \frac{w_i^{(t+1)}}{\sum_{j=1}^n w_j^{(t+1)}}
  $$

---

### **3. Final Strong Classifier**
- After $( T )$ iterations, combine the weak classifiers into a weighted majority vote:
  $$
  H(x) = \text{sign}\left(\sum_{t=1}^T \alpha_t \cdot h_t(x)\right)
  $$
  - $( \alpha_t )$ serves as the weight for each weak classifier.
  - The final prediction is the sign of the weighted sum.

---

### **Key Intuition**
- AdaBoost focuses on misclassified points by increasing their weights.
- The weak classifiers are adaptively improved to reduce errors on hard-to-classify samples.
- The final classifier balances the contributions of all weak classifiers.

---

### **Advantages**
- Works well with weak classifiers, such as decision stumps.
- Robust to overfitting for many practical problems.

### **Limitations**
- Sensitive to noisy data and outliers, as these points can dominate the weight updates.