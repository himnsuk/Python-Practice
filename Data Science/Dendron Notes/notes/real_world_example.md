### Real-World Project Example: **Customer Purchase Behavior Analysis**

In this example, you work for a retail company and are tasked with analyzing customer purchase behavior using both **descriptive** and **inferential statistics** to help the company make data-driven decisions.

---

### **Problem:**
The company wants to understand its customers' purchasing habits and determine whether a recent promotional campaign has significantly increased the average purchase amount. They also want insights on general customer purchase trends for better business strategies.

---

### **Step 1: Data Collection**

The company collects data from **1,000 customers** who made purchases over the past three months. The dataset includes:
- **Customer ID**
- **Purchase Amount (in $)**
- **Age**
- **Gender**
- **Loyalty Program Membership (Yes/No)**
- **Promotion Used (Yes/No)**
- **Date of Purchase**

---

### **Descriptive Statistics:**

You start by using **descriptive statistics** to summarize the data and provide insights into customer purchasing behavior.

#### **1. Central Tendency:**
- **Mean Purchase Amount**: The average purchase amount is **$75.50**.
- **Median Purchase Amount**: The median purchase amount is **$72.00** (half of the customers spent less than this, and half spent more).
- **Mode Purchase Amount**: The most frequent purchase amount is **$50** (many customers make purchases around this value).

#### **2. Dispersion:**
- **Standard Deviation**: The standard deviation of the purchase amounts is **$20.75**. This means that on average, customer purchases vary by about $21 from the mean.
- **Range**: The minimum purchase amount is **$10**, and the maximum is **$200**, so the **range** is $190.
- **Interquartile Range (IQR)**: The **IQR** is **$60** (Q3 = $100, Q1 = $40), which shows the spread of the middle 50% of purchase amounts.

#### **3. Distribution Shape:**
- The purchase amounts are **right-skewed**, meaning a few customers spent much more than the average, pulling the distribution’s tail to the right.

#### **4. Segmentation Analysis:**
- **By Gender**: 
  - Average purchase by male customers: **$80**
  - Average purchase by female customers: **$72**
- **By Age Group**:
  - Customers aged 18-25: Average purchase is **$65**.
  - Customers aged 26-40: Average purchase is **$80**.
  - Customers aged 41-60: Average purchase is **$90**.
- **Loyalty Program Members**: 
  - Members spend on average **$90**.
  - Non-members spend on average **$65**.

#### **Visualizations**:
- **Histograms** of purchase amounts to show the distribution.
- **Box plots** segmented by gender and age group to highlight variations in purchasing behavior.
- **Pie chart** of promotion usage to show the percentage of customers who used the promotional offer.

#### **Descriptive Insights**:
- Most customers spend around **$50-$100**.
- **Older customers** (41-60) tend to spend more than younger customers.
- Customers who are part of the **loyalty program** spend significantly more than those who aren’t.
- **Right-skewness** indicates a few high-spending customers are inflating the mean purchase amount.

---

### **Inferential Statistics:**

Now, you want to make inferences beyond the data sample and test if the promotional campaign has led to a significant increase in the average purchase amount.

#### **1. Hypothesis Testing:**
**Objective:** To determine if the promotion led to a higher average purchase amount.

##### **Setup:**
- **Null Hypothesis (H₀):** The promotion has **no effect** on the average purchase amount (i.e., the average purchase amount with and without the promotion is the same).
- **Alternative Hypothesis (H₁):** The promotion has **increased** the average purchase amount.

##### **T-Test (Two-Sample t-test):**
You can perform a **two-sample t-test** to compare the mean purchase amounts of two groups:
- Group 1: Customers who used the promotion.
- Group 2: Customers who didn’t use the promotion.

##### **Results:**
- **Mean Purchase (Promotion Users)**: $85
- **Mean Purchase (Non-Promotion Users)**: $70
- **p-value**: 0.01 (this means there’s a 1% chance the observed difference is due to random chance).

Since the **p-value** is less than 0.05, you **reject the null hypothesis** and conclude that the promotion **significantly increased** the average purchase amount.

#### **2. Confidence Intervals:**
To estimate the true difference in average purchase amounts between customers who used the promotion and those who didn’t, you calculate a **95% confidence interval**.
- Confidence interval for the difference in means: **[8, 20]**
- This means you’re 95% confident that the true increase in purchase amount due to the promotion lies between $8 and $20.

#### **3. Regression Analysis:**
You can build a **multiple linear regression** model to predict purchase amounts based on various factors:
- **Purchase Amount** = β₀ + β₁(Age) + β₂(Gender) + β₃(Loyalty Program) + β₄(Promotion) + ε

##### **Interpretation of Regression Coefficients:**
- **β₁ (Age)**: Older customers tend to spend more, with every additional year increasing purchase amount by approximately **$0.80**.
- **β₃ (Loyalty Program)**: Loyalty program members spend **$25 more** than non-members, on average.
- **β₄ (Promotion)**: Customers who used the promotion spend **$15 more** than those who didn’t, even after controlling for other factors.

#### **4. Chi-Square Test for Independence:**
You perform a **chi-square test** to see if loyalty program membership is associated with using the promotion:
- **Null Hypothesis (H₀):** There is **no relationship** between loyalty program membership and promotion usage.
- **Chi-Square Statistic**: 10.5
- **p-value**: 0.03 (significant at the 5% level).

You reject the null hypothesis and conclude that there is a significant association between being a loyalty program member and using the promotion.

---

### **Inferential Insights:**
- The promotional campaign **significantly increased** the average purchase amount.
- **Loyalty program members** are more likely to use the promotion and spend more overall.
- **Age** and **membership** status are significant predictors of purchase behavior.

---

### **Summary of Project Using Descriptive and Inferential Statistics:**

#### **Descriptive Statistics**:
- Summarized customer purchase behavior (mean, median, standard deviation, etc.).
- Analyzed purchasing patterns by gender, age, loyalty membership, and promotion usage.
- Visualized the distribution of purchase amounts and customer segmentation.

#### **Inferential Statistics**:
- Used hypothesis testing (t-test) to conclude that the promotion **increased** the average purchase amount.
- Built a regression model to predict purchase behavior based on various factors.
- Conducted a chi-square test to find that **loyalty program members** were more likely to use promotions.

---

### **Real-World Impact:**
- The company learns that their promotion is successful and worth continuing or scaling up.
- They can further target **older customers** and **loyalty program members** who are more likely to spend more.
- Based on the regression results, the company could **personalize promotions** to drive further engagement and sales.

---

This project provides a great balance between **descriptive statistics** to summarize the data and **inferential statistics** to draw meaningful conclusions that inform business strategy.