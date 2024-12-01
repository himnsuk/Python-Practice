### Logistic Regression: Detailed Explanation

**Logistic regression** is a statistical method used for binary classification problems. It predicts the probability of a binary outcome (e.g., yes/no, true/false, 0/1) based on one or more independent variables. Unlike linear regression, it models a non-linear relationship between the dependent and independent variables to constrain predictions to a range between 0 and 1.

---

### 1. **Key Concepts**

- **Dependent Variable (Output)**: Binary (e.g., $( y = 0 )$ or $( y = 1 )$).
- **Independent Variables (Features)**: Continuous or categorical variables used to make predictions.
- **Goal**: Estimate the probability $( P(y=1 \mid x) )$ and classify based on a threshold (usually 0.5).

---

### 2. **Mathematical Formulation**

#### 2.1 **Linear Regression Hypothesis**
In linear regression, the hypothesis is:
$$
h(x) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n
$$
However, this can yield predictions outside the range $([0, 1])$, which is unsuitable for probabilities.

#### 2.2 **Logistic Regression Hypothesis**
To map predictions to a probability range $([0, 1])$, logistic regression applies the **sigmoid function** (or logistic function) to the linear equation:
$$
P(y=1 \mid x) = \sigma(h(x)) = \frac{1}{1 + e^{-z}}, \quad \text{where } z = \beta_0 + \beta_1 x_1 + \dots + \beta_n x_n
$$

- **Sigmoid Function**: $( \sigma(z) )$ squashes any real number into a probability value between 0 and 1.
  
$$
\sigma(z) =
\begin{cases} 
\approx 1, & \text{if } z \gg 0 \\
\approx 0, & \text{if } z \ll 0
\end{cases}
$$

---

### 3. **Cost Function**

Instead of using Mean Squared Error (MSE), logistic regression uses a **log-loss (cross-entropy loss)** to evaluate the model. The cost function for a single data point is:
$$
\text{Cost}(h(x), y) = 
\begin{cases} 
- \log(h(x)), & \text{if } y = 1 \\
- \log(1 - h(x)), & \text{if } y = 0
\end{cases}
$$

This can be combined into a single equation:
$$
\text{Cost}(h(x), y) = - \big[ y \log(h(x)) + (1-y) \log(1 - h(x)) \big]
$$

For the entire dataset:
$$
J(\beta) = \frac{1}{m} \sum_{i=1}^{m} \Big[- y_i \log(h(x_i)) - (1 - y_i) \log(1 - h(x_i))\Big]
$$

- Minimizing this cost function ensures that the predicted probabilities closely match the true labels.

---

### 4. **Optimization**

To minimize the cost function, **Gradient Descent** is used:
$$
\beta_j \leftarrow \beta_j - \alpha \frac{\partial J(\beta)}{\partial \beta_j}
$$
Where:
- $( \alpha )$: Learning rate
- $( \frac{\partial J(\beta)}{\partial \beta_j} )$: Gradient of the cost function w.r.t $( \beta_j )$

---

### 5. **Why Use Logistic Regression Instead of Linear Regression?**

1. **Prediction Range**:
   - Linear regression can produce predictions outside $([0, 1])$.
   - Logistic regression constrains predictions to probabilities in $([0, 1])$.

2. **Interpretability**:
   - Logistic regression directly models probabilities, making it easier to interpret outcomes.

3. **Appropriate Loss Function**:
   - Linear regression minimizes squared errors, which isn't suitable for classification problems.
   - Logistic regression uses log-loss, which penalizes incorrect classifications more effectively.

4. **Decision Boundary**:
   - Logistic regression defines a decision boundary using the sigmoid function. For example, if $( P(y=1 \mid x) > 0.5 )$, classify as $( 1 )$, else classify as $( 0 )$.

5. **Linear Assumptions**:
   - Linear regression assumes a linear relationship between $( x )$ and $( y )$, which isn't valid for binary outcomes.
   - Logistic regression assumes a linear relationship between $( x )$ and the log-odds of $( y )$.

---

### 6. **Example**

**Problem**: Predict whether a student passes ($( y=1 )$) or fails ($( y=0 )$) based on study hours ($( x )$).

#### Dataset
| Hours Studied ($( x )$) | Pass ($( y )$) |
|-------------------------|----------------|
| 1                       | 0              |
| 2                       | 0              |
| 3                       | 0              |
| 4                       | 1              |
| 5                       | 1              |
| 6                       | 1              |

#### Steps:
1. **Hypothesis**:
   $$
   P(y=1 \mid x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x)}}
   $$

2. **Fit Parameters ($( \beta_0, \beta_1 )$)**:
   Using optimization (e.g., gradient descent), we estimate $( \beta_0 )$ and $( \beta_1 )$. Assume:
   $$
   \beta_0 = -3, \, \beta_1 = 1
   $$

3. **Prediction**:
   For $( x = 4 )$:
   $$
   P(y=1 \mid x=4) = \frac{1}{1 + e^{-(-3 + 1 \cdot 4)}} = \frac{1}{1 + e^{-1}} \approx 0.73
   $$
   Since $( 0.73 > 0.5 )$, predict $( y = 1 )$ (pass).

---

### 7. **Extensions**
- **Multinomial Logistic Regression**: For multi-class classification.
- **Regularization**: Add terms like $( L_1 )$ (Lasso) or $( L_2 )$ (Ridge) penalties to prevent overfitting.

---

Logistic regression remains popular because of its simplicity, interpretability, and effectiveness in scenarios where linear models are sufficient for classification tasks.