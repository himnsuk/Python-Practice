---
id: 3azv17wvqge748xau54y2p6
title: Chi Square Test
desc: ''
updated: 1697460117433
created: 1697460048959
---

The chi-square test is a statistical test used to determine if there is a significant association between two categorical variables. It is commonly used in various fields, including biology, social sciences, and market research. The test helps to assess whether the observed distribution of data differs from the expected distribution, assuming that there is no relationship between the variables. The result of the test indicates whether the relationship between the variables is statistically significant or if any observed differences are likely due to chance.

There are two main types of chi-square tests: the chi-square goodness-of-fit test and the chi-square test of independence. I'll explain both types with examples:

1. Chi-Square Goodness-of-Fit Test:
   This test is used to determine if an observed categorical frequency distribution fits a hypothesized or expected distribution. It is often used in scenarios where you want to test whether the observed data follows a specific pattern or distribution.

   Example:
   Let's say you are working in a candy factory, and you expect that the distribution of colors for a particular type of candy should be 30% red, 40% blue, and 30% green. You collect a sample of 200 candies and observe the following distribution:
   - Red: 50 candies
   - Blue: 70 candies
   - Green: 80 candies

   To test whether the observed distribution matches the expected distribution, you can perform a chi-square goodness-of-fit test.

   Null Hypothesis (H0): The observed distribution matches the expected distribution.
   Alternative Hypothesis (Ha): The observed distribution does not match the expected distribution.

   By conducting the test, you calculate a chi-square statistic and compare it to a critical value from the chi-square distribution. If the calculated chi-square value is greater than the critical value, you reject the null hypothesis, indicating a significant difference between the observed and expected distributions.

2. Chi-Square Test of Independence:
   This test is used to determine whether there is a statistically significant association between two categorical variables. It helps you understand if changes in one variable are dependent on changes in the other variable.

   Example:
   Suppose you want to investigate whether there is a relationship between a person's gender (male or female) and their preference for a specific soda brand (Coke, Pepsi, or Sprite). You collect data from a random sample of 300 people and create a contingency table like this:

   |            | Coke | Pepsi | Sprite |
   |------------|------|-------|--------|
   | Male       | 50   | 40    | 20     |
   | Female     | 30   | 60    | 40     |

   To test the independence of gender and soda preference, you can perform a chi-square test of independence.

   Null Hypothesis (H0): Gender and soda preference are independent (no association).
   Alternative Hypothesis (Ha): Gender and soda preference are not independent (there is an association).

   You calculate the chi-square statistic for this contingency table and compare it to a critical value. If the calculated chi-square value is greater than the critical value, you reject the null hypothesis, indicating that there is a significant association between gender and soda preference.

In both types of chi-square tests, the degrees of freedom depend on the specific problem and the number of categories or levels in the variables. Chi-square tests are widely used for making inferences about categorical data and are valuable tools for detecting relationships and patterns in such data.