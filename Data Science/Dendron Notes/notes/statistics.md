---
id: 5xveh2uwkrfkloni4tqrfwo
title: Statistics
desc: ''
updated: 1669803521506
created: 1667440395385
---


### Covariance and correlation

Covariance is an indicator of the extent to which 2 random variables are dependent on each other. A higher number denotes higher dependency. Correlation is a statistical measure that indicates how strongly two variables are related. The value of covariance lies in the range of -∞ and +∞

Covariance indicates the direction of the linear relationship between variables while correlation measures both the strength and direction of the linear relationship between two variables. Correlation is a function of the covariance

![Covariance](assets/images/statistics/2022-11-03-07-35-32.png)

![Correlation](assets/images/statistics/2022-11-03-07-36-34.png)

### Standard Error and margin of error

A margin of error is a statistical measure that accounts for the degree of error received from the outcome of your research sample. On the other hand, standard error measures the accuracy of the representation of the population sample to the mean using the standard deviation of the data set.

How do you find margin of error from standard error?
It is calculated as:
$$
Standard Error = s / √n.
$$

$$
Margin of Error = z*(s/√n)
$$

$$
Confidence Interval = x +/- z*(s/√n)
$$

### Type 1 and Type 2 error
A type I error (false-positive) occurs if an investigator rejects a null hypothesis that is actually true in the population; a type II error (false-negative) occurs if the investigator fails to reject a null hypothesis that is actually false in the population.

### What is Type 1 and Type 2 error example?

Type I error (false positive): the test result says you have coronavirus, but you actually don't. Type II error (false negative): the test result says you don't have coronavirus, but you actually do


The **standard error (SE)** is a statistical term that measures the accuracy with which a sample distribution represents a population. It indicates how much variation or "error" exists between the sample mean and the true population mean. Here's a breakdown:

### 1. **Definition**
The standard error of a statistic (usually the sample mean) is the **standard deviation** of the sampling distribution of that statistic. It quantifies the expected variability of a sample mean (or other statistic) if you took multiple samples from the same population.

### 2. **Formula**
For the sample mean, the standard error is calculated as:
$$
SE = \frac{\sigma}{\sqrt{n}}
$$

- **σ (sigma)**: The population standard deviation (if known).
- **n**: The sample size.

If the population standard deviation is not known, it's often estimated using the sample standard deviation, $(s):$

$$
SE = \frac{s}{\sqrt{n}}
$$

### 3. **Key Concepts**
- **Smaller SE**: A smaller standard error means the sample mean is a more precise estimate of the population mean. This happens when the sample size \(n\) is large or the variability (standard deviation, \(σ\)) in the population is small.
  
- **Larger SE**: A larger standard error indicates more variability in the sample mean, meaning that the sample mean is a less reliable estimate of the population mean. This happens when the sample size \(n\) is small or the variability in the population is high.

### 4. **Standard Deviation vs. Standard Error**
- **Standard Deviation (SD)** measures the spread of individual data points in a population or sample.
- **Standard Error (SE)** measures the spread of sample means, i.e., how much sample means would vary if you repeated your sampling multiple times.

### 5. **Importance**
- **Confidence Intervals**: The SE is used to calculate confidence intervals for a sample mean. A smaller SE results in a narrower confidence interval, indicating more certainty about the population parameter.
- **Hypothesis Testing**: In many statistical tests (like t-tests), SE is used to determine how likely it is that the sample data came from a particular population.

### Example:
If you take multiple samples from a population and compute the mean of each sample, the standard error tells you how much the sample means are likely to differ from the actual population mean. A larger sample size will decrease the standard error, making your sample mean a more accurate estimate of the population mean.

---
---
**Normalization** and **standardization** are techniques used to adjust data into a consistent format, especially before applying machine learning models, ensuring different variables are comparable and models perform optimally. Here's a brief overview of each:

### Normalization
Normalization scales data to a specific range, usually between 0 and 1. It is often used when you want to ensure that all features are on the same scale, especially when the algorithm you're using is sensitive to relative magnitudes (e.g., K-nearest neighbors, neural networks).

**Formula:**

$$
X_{\text{norm}} = \frac{X - X_{\min}}{X_{\max} - X_{\min}}
$$

Where:
- $(X)$ is the original data point,
- $(X_{\min})$ is the minimum value of the feature,
- $(X_{\max})$ is the maximum value of the feature.

### When to use Normalization:
- When the features have different ranges but no known normal distribution.
- Often for algorithms that calculate distances or require bounded input values (like neural networks).

### Standardization
Standardization (or Z-score normalization) rescales data to have a mean of 0 and a standard deviation of 1. This is useful when the data is assumed to have a Gaussian (normal) distribution or when many algorithms (e.g., linear regression, SVMs) require centered and scaled input.

**Formula:**

$$
X_{\text{std}} = \frac{X - \mu}{\sigma}
$$

Where:
- $(X)$ is the original data point,
- $(\mu)$ is the mean of the feature,
- $(\sigma)$ is the standard deviation of the feature.

### When to use Standardization:
- When data follows a normal distribution or can be assumed to.
- When working with algorithms like logistic regression, linear regression, or SVM that rely on normally distributed data for optimal performance.

### Summary:
- **Normalization**: Scales values to a range [0, 1], often used in distance-based algorithms.
- **Standardization**: Transforms data to have a mean of 0 and a standard deviation of 1, suitable for normally distributed data.

The choice between normalization and standardization depends on the model and the nature of the data.