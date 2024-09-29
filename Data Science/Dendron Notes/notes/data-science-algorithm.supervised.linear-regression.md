---
id: 6d3jemvwcvgnpb6by2cv0ie
title: Linear Regression
desc: ''
updated: 1682766582543
created: 1667044145410
---


### R-Square formula

![Linear Regression](./assets/images/data-science-algos/supervised/linear-regression/r-square.png)

![](assets/images/data-science-algos/supervised/linear-regression/2022-11-02-10-14-17.png)

![](assets/images/data-science-algos/supervised/linear-regression/2022-11-02-10-15-18.png)

![VIF](assets/images/data-science-algos/supervised/linear-regression/2022-11-16-14-13-40.png)

### Mean Absolute Error

---

![Mean Absolute Error](assets/images/data-science-algos/supervised/linear-regression/2022-11-16-14-43-38.png)

### Mean Squared Error

represents the average of the squared difference between the original and predicted values in the data set. It measures the variance of the residuals.

![Mean Squared Error](assets/images/data-science-algos/supervised/linear-regression/2022-11-16-14-45-03.png)

### Root Mean Squared Error

is the square root of Mean Squared error. It measures the standard deviation of residuals.

![Root Mean Squared Error](assets/images/data-science-algos/supervised/linear-regression/2022-11-16-14-46-01.png)

### Regression Model Accuracy Metrics

Model performance metrics

In regression model, the most commonly known evaluation metrics include:

1. **R-squared (R2)**, which is the proportion of variation in the outcome that is explained by the predictor variables. In multiple regression models, R2 corresponds to the squared correlation between the observed outcome values and the predicted values by the model. The Higer the R-squared, the better the model.

2. **Root Mean Squared Error (RMSE)**, which measures the average error performed by the model in the predicting the outcome for an observation. Mathematically, the RMSE is the square root of the mean squared error (MSE), which is the average squared difference between the observed actual outcome values and the values predicted by the model. So, MSE = mean((observeds - predicteds)^2) and RMSE = sqrt(MSE). The lower the RMSE, the better the model.

3. **Residual Standard Error (RSE)**, also known as the model sigma, is a variant of the RMSE adjusted for the number of predictors in the model. The lower the RSE, the better the model. In practice, the difference between RMSE and RSE is very small, particularly for large multivariate data.

4. **Mean Absolute Error (MAE)**, like the RMSE, the MAE measures the prediction error. Mathematically, it is the average absolute difference between obsered and predicted out-comes, MAE = mean(abs(observeds - predicteds)) . MAE is less sensitive to outliers compared to RMSE.

The problem with the above metrics, is that they are sensible to the inclusion of additional variables in the model, even if those variables don’t have significant contribution in explaining the outcome. Put in other words, including additional variables in the model will always increase the R2 and reduce the RMSE. So, we need a more robust metric to guide the model choice.

Concerning R2, there is an adjusted version, called Adjusted R-squared, which adjusts the R2 for having too many variables in the model.

Additionally, there are four other important metrics - AIC, AICc, BIC and Mallows Cp - tha are commonly used for model evaluation and selection. These are an unbiased estimate of the model prediction error MSE. The lower these metrics, he better the model.

1. **AIC stands for (Akike’s Information Criteria)**, a metric developeed by the Japanese Statistician, Hirotugu Akaike, 1970. The basic idea of AIC is to penalize the inclusion of additional variables to a model. It adds a penalty that increases the error when including additional terms. The lowwer the AIC, the better the model.

2. **AICc** is a version of AIC corrected for small sample sizes.

3. **BIC (or Bayesian information criteria)** is a variant of AIC with a strong penalty for including additional variables to the model.

4. **Mallows Cp**: Avariant of AIC developed by Colin Mallows.

Linear Regression
---

1. Linear Regression Equation
2. Dependent and Independent variable
3. 

---
---

Akaike's Information Criterion (AIC) is a metric used for model selection, particularly in the context of statistical and machine learning models. It provides a balance between the goodness of fit of the model and the complexity of the model to avoid overfitting.

### Key Concept:
- **Goodness of fit**: Models that better explain the data tend to have a higher likelihood.
- **Model complexity**: More complex models (i.e., models with more parameters) tend to fit the data better, but they can overfit by capturing noise as well.

AIC attempts to quantify this trade-off, penalizing models that are overly complex (i.e., have too many parameters) while rewarding models that fit the data well.

### Formula:

The formula for AIC is:

$$
AIC = 2k - 2\ln(L)
$$

Where:
- $( k )$: Number of parameters in the model (including intercept and variance, if applicable).
- $( L )$: The maximum likelihood of the model (i.e., how well the model fits the data).

### Explanation of the Formula:

1. **Log-likelihood ($(\ln(L))$)**: Measures how well the model fits the data. A higher value indicates a better fit.
2. **Penalty Term (2k)**: Penalizes the complexity of the model. As the number of parameters increases, this term grows, discouraging overfitting by preferring simpler models.

### Interpretation:

- A lower AIC value indicates a better model, considering both fit and complexity.
- When comparing multiple models, the model with the **lowest AIC** is typically preferred.
- **Trade-off**: AIC balances the trade-off between model complexity and goodness of fit, aiming to find a model that explains the data well without overfitting.

### Example:

If you're comparing two regression models:
- Model 1 has fewer parameters but fits the data poorly.
- Model 2 has more parameters and fits the data well.

AIC will account for both the increased complexity (via $(2k)$) and the improved fit (via $(-2\ln(L))$), helping you choose the model that strikes the best balance.


---
---

To calculate the **log-likelihood** of a regression model, you'll need to use the likelihood function, which measures how likely the observed data is, given the model's parameters. In the case of **linear regression** with normally distributed errors, the log-likelihood can be computed based on the residuals (differences between the observed values and the predicted values).

### Steps to Calculate Log-Likelihood for a Regression Model:

#### 1. **Assumptions**:
   - The residuals (errors) are normally distributed with mean 0 and variance $(\sigma^2)$.
   - The observations $( y_i )$ (dependent variable values) are independent.

#### 2. **Formula for Log-Likelihood**:
For a regression model, the log-likelihood ($( \ln(L) )$) is given by:

$$
\ln(L) = -\frac{n}{2} \ln(2\pi) - \frac{n}{2} \ln(\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

Where:
- $( n )$: Number of data points (observations).
- $( \hat{y}_i )$: Predicted value for the $(i)$-th observation.
- $( y_i )$: Actual value for the $(i)$-th observation.
- $( \sigma^2 )$: Variance of the residuals (errors), calculated as:
 $$
  \sigma^2 = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
  $$
  which is the **mean squared error (MSE)** of the model.

#### 3. **Breaking Down the Formula**:
- $( \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 )$ is the sum of squared residuals (SSR), which measures how well the model fits the data.
- The term $( \ln(\sigma^2) )$ penalizes the model for larger variance, as it indicates more spread or error in the model’s predictions.

#### 4. **Steps to Calculate Log-Likelihood**:
1. **Fit the regression model** to obtain the predicted values $( \hat{y}_i )$.
2. **Calculate the residuals**: $( e_i = y_i - \hat{y}_i )$ for each observation.
3. **Compute the sum of squared residuals (SSR)**: 
  $$
   SSR = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
   $$
4. **Estimate the residual variance** $( \sigma^2 )$: 
  $$
   \sigma^2 = \frac{SSR}{n}
   $$
5. Plug the values into the log-likelihood formula.

### Example of Log-Likelihood Calculation:

Suppose you have a regression model with:
- 100 observations ($(n = 100)$).
- Sum of squared residuals (SSR) = 500.
  
1. Compute the residual variance:
  $$
   \sigma^2 = \frac{500}{100} = 5
   $$
   
2. Calculate the log-likelihood:
  $$
   \ln(L) = -\frac{100}{2} \ln(2\pi) - \frac{100}{2} \ln(5) - \frac{1}{2 \times 5} \times 500
   $$
   
   Simplifying the terms:
  $$
   \ln(L) = -50 \ln(2\pi) - 50 \ln(5) - \frac{500}{10}
   $$
  $$
   \ln(L) = -50 \ln(2\pi) - 50 \ln(5) - 50
   $$

Thus, you get the log-likelihood value for this regression model.

### Summary:
To compute the log-likelihood for AIC, you:
1. Fit the regression model.
2. Compute the sum of squared residuals (SSR).
3. Estimate the residual variance ($(\sigma^2)$).
4. Plug these into the log-likelihood formula.

This log-likelihood is then used in the AIC formula:
$$
AIC = 2k - 2\ln(L)
$$ 
where $(k)$ is the number of parameters in the model.