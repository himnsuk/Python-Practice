---
id: 6xeuci4n3geuper7ircc4du
title: Feature Importance
desc: ''
updated: 1667374424880
created: 1667374424880
---

[This is the link from where it's taken](https://machinelearningmastery.com/calculate-feature-importance-with-python/)

This tutorial is divided into six parts; they are:

1. Feature Importance
2. Preparation
   1. Check Scikit-Learn Version
   2. Test Datasets
3. Coefficients as Feature Importance
   1. Linear Regression Feature Importance
   2. Logistic Regression Feature Importance
4. Decision Tree Feature Importance
   1. CART Feature Importance
   2. Random Forest Feature Importance
   3. XGBoost Feature Importance
5. Permutation Feature Importance
   1. Permutation Feature Importance for Regression
   2. Permutation Feature Importance for Classification
6. Feature Selection with Importance



### Feature Importance
Feature importance refers to a class of techniques for assigning scores to input features to a predictive model that indicates the relative importance of each feature when making a prediction.

Feature importance scores can be calculated for problems that involve predicting a numerical value, called regression, and those problems that involve predicting a class label, called classification.

The scores are useful and can be used in a range of situations in a predictive modeling problem, such as:

1. Better understanding the data.
1. Better understanding a model.
1. Reducing the number of input features.

**Feature importance scores can provide insight into the dataset**. The relative scores can highlight which features may be most relevant to the target, and the converse, which features are the least relevant. This may be interpreted by a domain expert and could be used as the basis for gathering more or different data.

**Feature importance scores can provide insight into the model**. Most importance scores are calculated by a predictive model that has been fit on the dataset. Inspecting the importance score provides insight into that specific model and which features are the most important and least important to the model when making a prediction. This is a type of model interpretation that can be performed for those models that support it.

**Feature importance can be used to improve a predictive model**. This can be achieved by using the importance scores to select those features to delete (lowest scores) or those features to keep (highest scores). This is a type of feature selection and can simplify the problem that is being modeled, speed up the modeling process (deleting features is called dimensionality reduction), and in some cases, improve the performance of the model.