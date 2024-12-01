**AIC (Akaike Information Criterion), BIC (Bayesian Information Criterion), and AICc (Corrected AIC)** are statistical tools used to compare models and select the one that best explains the data while penalizing for model complexity to avoid overfitting. Here’s a detailed mathematical explanation:

---

### **1. Akaike Information Criterion (AIC)**

AIC estimates the quality of a model by balancing goodness-of-fit with model complexity.

#### **Formula for AIC:**
$$
\text{AIC} = -2 \ln(\hat{L}) + 2k
$$
Where:
- $( \hat{L} )$: Maximum likelihood of the model (fit to the data).
- $( k )$: Number of parameters in the model (including intercept).
- $( -2 \ln(\hat{L}) )$: Deviance or negative log-likelihood of the model.

#### **Interpretation:**
- The first term ($( -2 \ln(\hat{L}) )$) rewards goodness-of-fit (lower values indicate better fit).
- The second term ($( 2k )$) penalizes for the number of parameters to discourage overfitting.

---

### **2. Bayesian Information Criterion (BIC)**

BIC is similar to AIC but introduces a stronger penalty for model complexity, especially for larger datasets.

#### **Formula for BIC:**
$$
\text{BIC} = -2 \ln(\hat{L}) + k \ln(n)
$$
Where:
- $( n )$: Number of data points (sample size).
- Other terms ($( \hat{L} )$, $( k )$) are as defined for AIC.

#### **Interpretation:**
- The penalty term ($( k \ln(n) )$) increases with sample size, making BIC more conservative than AIC for large datasets.
- Prefer models with lower BIC values.

---

### **3. Corrected Akaike Information Criterion (AICc)**

AICc is a corrected version of AIC designed for small sample sizes, where AIC may overfit.

#### **Formula for AICc:**
$$
\text{AICc} = \text{AIC} + \frac{2k(k+1)}{n - k - 1}
$$
Where:
- $( n )$: Number of data points.
- $( k )$: Number of parameters in the model.
- Other terms are as defined for AIC.

#### **Interpretation:**
- The correction term ($( \frac{2k(k+1)}{n - k - 1} )$) increases the penalty when $( n )$ is small relative to $( k )$.
- As $( n \to \infty )$, AICc converges to AIC.

---

### **4. Model Selection Criteria**
- **AIC**: Prefer smaller AIC values. Use for general-purpose model selection.
- **BIC**: Prefer smaller BIC values. Use when you want a stronger penalty for complexity.
- **AICc**: Prefer smaller AICc values. Use for small sample sizes ($( n/k < 40 )$).

---

### **5. Log-Likelihood and Deviance**
To calculate $( \ln(\hat{L}) )$:
$$
\ln(\hat{L}) = -\frac{n}{2} \ln(2\pi\sigma^2) - \frac{\text{RSS}}{2\sigma^2}
$$
Where:
- $( \sigma^2 )$: Variance of residuals.
- $( \text{RSS} )$: Residual Sum of Squares.

---

### **6. Comparison of Criteria**
- AIC and AICc focus on prediction accuracy and trade off goodness-of-fit and complexity.
- BIC adds a stronger penalty for complexity, favoring simpler models as $( n )$ increases.
- **Relationship:**  
  $$
  \text{AICc} > \text{AIC}, \quad \text{BIC} > \text{AIC} \text{ (for large $( n )$)}.
  $$

---

### **7. Practical Example**
For $( k = 3 )$, $( n = 100 )$, and $( \hat{L} = 0.8 )$:
1. $( \ln(\hat{L}) = \ln(0.8) = -0.223 )$.
2. Calculate:
   - $( \text{AIC} = -2(-0.223) + 2(3) = 6.446 )$.
   - $( \text{BIC} = -2(-0.223) + 3\ln(100) = 14.006 )$.
   - $( \text{AICc} = 6.446 + \frac{2(3)(3+1)}{100 - 3 - 1} = 6.698 )$.

---

### **Key Points**
- Lower AIC, BIC, or AICc indicates a better model.
- Use **AIC** for prediction-oriented tasks, **BIC** for model parsimony, and **AICc** for small datasets.


#### Parsimony
>A parsimonious model aims to explain the data with the minimum number of variables or parameters necessary, avoiding unnecessary complexity.