### **Standardization vs. Normalization**

Both standardization and normalization are techniques used in data preprocessing to adjust the scale of the data. These methods are particularly useful in machine learning and statistical modeling.

---

### **Standardization**

- **Definition**: It rescales data to have a mean of **0** and a standard deviation of **1** (unit variance). 
- **Formula**:
  $$
  z = \frac{x - \mu}{\sigma}
  $$
  where:
  - $( x )$ is the data point,
  - $( \mu )$ is the mean of the dataset,
  - $( \sigma )$ is the standard deviation of the dataset.

- **Use case**: Standardization is often used when the dataset contains features with different units (e.g., age in years and income in dollars) and algorithms like SVMs or logistic regression assume features are normally distributed.

**Example**:
Suppose we have a dataset with a feature:
$$
[10, 20, 30, 40, 50]
$$
- Mean ($( \mu )$) = $( 30 )$,
- Standard deviation ($( \sigma )$) = $( 14.14 )$.

Standardized values:
$$
\text{Standardized values} = \left[ \frac{10-30}{14.14}, \frac{20-30}{14.14}, \frac{30-30}{14.14}, \frac{40-30}{14.14}, \frac{50-30}{14.14} \right]
$$
Result:
$$
[-1.41, -0.71, 0, 0.71, 1.41]
$$

---

### **Normalization**

- **Definition**: It rescales the data to fit within a specific range, typically **[0, 1]**. 
- **Formula**:
  $$
  x_{\text{norm}} = \frac{x - x_{\text{min}}}{x_{\text{max}} - x_{\text{min}}}
  $$
  where:
  - $( x )$ is the data point,
  - $( x_{\text{min}} )$ and $( x_{\text{max}} )$ are the minimum and maximum values in the dataset.

- **Use case**: Normalization is used when the data needs to be constrained within a fixed range, such as for neural networks or distance-based algorithms like k-NN.

**Example**:
Using the same dataset:
$$
[10, 20, 30, 40, 50]
$$
- Minimum ($( x_{\text{min}} )$) = $( 10 )$,
- Maximum ($( x_{\text{max}} )$) = $( 50 )$.

Normalized values:
$$
\text{Normalized values} = \left[ \frac{10-10}{50-10}, \frac{20-10}{50-10}, \frac{30-10}{50-10}, \frac{40-10}{50-10}, \frac{50-10}{50-10} \right]
$$
Result:
$$
[0, 0.25, 0.5, 0.75, 1]
$$

---

### **Key Differences**

| **Aspect**              | **Standardization**                        | **Normalization**                      |
|--------------------------|--------------------------------------------|----------------------------------------|
| **Range**               | No fixed range (mean = 0, SD = 1)          | Fixed range, typically [0, 1]          |
| **Effect on outliers**  | Less sensitive to outliers                 | Highly sensitive to outliers           |
| **When to use**         | When features follow Gaussian distribution | When scaling to a specific range is needed |

---

### **Summary**
- **Standardization** focuses on centering data (mean=0, SD=1).
- **Normalization** rescales data to a bounded range, such as [0, 1].
