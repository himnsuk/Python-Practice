---
id: swlsfu71dan4ib7dkpiebm0
title: Classification
desc: ''
updated: 1669018384793
created: 1667387650067
---


**Accuracy** - Accuracy is the most intuitive performance measure and it is simply a ratio of correctly predicted observation to the total observations. One may think that, if we have high accuracy then our model is best. Yes, accuracy is a great measure but only when you have symmetric datasets where values of false positive and false negatives are almost same. Therefore, you have to look at other parameters to evaluate the performance of your model. For our model, we have got 0.803 which means our model is approx. 80% accurate.

**Accuracy** = TP+TN/TP+FP+FN+TN

Precision - Precision is the ratio of correctly predicted positive observations to the total predicted positive observations. The question that this metric answer is of all passengers that labeled as survived, how many actually survived? High precision relates to the low false positive rate. We have got 0.788 precision which is pretty good.

**Precision** = TP/TP+FP

Recall (Sensitivity) - Recall is the ratio of correctly predicted positive observations to the all observations in actual class - yes. The question recall answers is: Of all the passengers that truly survived, how many did we label? We have got recall of 0.631 which is good for this model as it’s above 0.5.

**Recall** = TP/TP+FN

F1 score - F1 Score is the weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account. Intuitively it is not as easy to understand as accuracy, but F1 is usually more useful than accuracy, especially if you have an uneven class distribution. Accuracy works best if false positives and false negatives have similar cost. If the cost of false positives and false negatives are very different, it’s better to look at both Precision and Recall. In our case, F1 score is 0.701.

F1 Score = 2*(Recall \* Precision) / (Recall + Precision)


---
---

Here are the formulas for **Precision**, **Recall**, and the **F1 Score**, commonly used evaluation metrics for classification tasks, especially in imbalanced datasets.

### 1. **Precision**
Precision measures the accuracy of positive predictions, i.e., out of all the instances classified as positive, how many are actually positive.

$$
\text{Precision} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Positives (FP)}}
$$

- **True Positives (TP)**: Correctly predicted positive instances.
- **False Positives (FP)**: Incorrectly predicted positive instances (actually negative).

### 2. **Recall (Sensitivity or True Positive Rate)**
Recall measures how well the model captures all positive instances, i.e., out of all the actual positive instances, how many were correctly classified.

$$
\text{Recall} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Negatives (FN)}}
$$

- **True Positives (TP)**: Correctly predicted positive instances.
- **False Negatives (FN)**: Instances that are actually positive but were incorrectly classified as negative.

### 3. **F1 Score**
The F1 Score is the harmonic mean of Precision and Recall, providing a balance between the two, especially useful when the data is imbalanced.

$$
\text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

### Summary of Metrics:
- **Precision**: Focuses on the quality of positive predictions (minimizes false positives).
- **Recall**: Focuses on capturing as many actual positives as possible (minimizes false negatives).
- **F1 Score**: Balances precision and recall.

Each of these metrics serves different purposes depending on the problem and the costs of false positives and false negatives.


---
---

The **harmonic mean** is used in the calculation of the **F1 score** because it effectively balances **Precision** and **Recall**, especially when there is a large disparity between the two. Here's why the harmonic mean is preferred:

### 1. **Balances Precision and Recall**
The harmonic mean tends to be lower than the arithmetic mean, particularly when there is a significant difference between Precision and Recall. This ensures that the F1 score will only be high if both Precision and Recall are reasonably high. In contrast, the arithmetic mean could give a misleadingly high result if one value is much higher than the other.

For example:
- If Precision = 0.9 and Recall = 0.1, their arithmetic mean is (0.9 + 0.1) / 2 = 0.5.
- But the harmonic mean (F1 score) is:

$$
  F1 = 2 \times \frac{0.9 \times 0.1}{0.9 + 0.1} = 0.18
$$

This reflects the fact that one of the metrics (Recall in this case) is very low, pulling the F1 score down.

### 2. **Penalizes Large Disparities**
The harmonic mean emphasizes the lower of the two values, so if one metric (either Precision or Recall) is very low, the F1 score will be low as well. This is important because in many cases, both Precision and Recall are needed to assess the overall performance of a model. A model with very high Precision but very low Recall (or vice versa) may not be useful, and the F1 score penalizes such imbalanced performance.

### 3. **Suited for Imbalanced Data**
In many real-world situations, particularly with imbalanced datasets (e.g., spam detection, fraud detection), one class may be much larger than the other. The harmonic mean helps to ensure that both classes are treated fairly, and neither Precision nor Recall is disproportionately favored. This makes the F1 score a better representation of model performance in such cases.

### Summary:
- **Harmonic mean** ensures that the F1 score only becomes high when **both** Precision and Recall are high.
- It penalizes large disparities between Precision and Recall, making it a better measure when the two are imbalanced.
- It provides a more cautious and balanced metric for evaluating model performance, especially with imbalanced datasets.


---
---

**SMOTE (Synthetic Minority Over-sampling Technique)** is a popular method used to handle imbalanced datasets in classification problems by generating synthetic samples for the minority class. Here's an overview of how to use SMOTE and why it works well:

### Steps to Handle Imbalanced Data using SMOTE:

1. **Understand the Imbalance**:
   - First, evaluate the class distribution in your dataset. If the minority class is underrepresented, your classifier may be biased toward the majority class, leading to poor performance on the minority class.

2. **Apply SMOTE**:
   SMOTE generates synthetic data points for the minority class by creating new samples that are interpolated between existing minority class samples.

   **How it works**:
   - For each sample in the minority class, SMOTE selects one or more nearest neighbors (based on distance metrics like Euclidean distance).
   - It then creates new synthetic samples along the line connecting the sample and its neighbors.
   - This effectively "fills in" the gaps in the feature space where the minority class is under-represented.

   **SMOTE Formula** (simplified):
$$
   x_{\text{new}} = x_i + \lambda \times (x_{nn} - x_i)
$$
   Where:
   - $(x_i)$ is a sample from the minority class.
   - $(x_{nn})$ is one of the nearest neighbors.
   - $(\lambda)$ is a random number between 0 and 1.

   This formula creates a synthetic point along the line connecting \(x_i\) and \(x_{nn}\).

3. **Implementing SMOTE** in Python:
   The `imbalanced-learn` library provides an easy-to-use implementation of SMOTE.

   ```python
   from imblearn.over_sampling import SMOTE
   from sklearn.model_selection import train_test_split

   # Split the dataset into features (X) and labels (y)
   X, y = your_dataset_features, your_dataset_labels

   # Split into training and testing sets
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

   # Apply SMOTE to the training data
   smote = SMOTE(random_state=42)
   X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

   # Now use X_train_smote and y_train_smote for training your model
   ```

4. **Train Your Classifier**:
   After applying SMOTE, train your model on the resampled (balanced) dataset. This allows the model to learn more effectively from the minority class and reduce bias.

5. **Evaluate Performance**:
   It's crucial to assess the performance of your model after applying SMOTE. While SMOTE improves performance on the minority class, it's important to ensure that it doesn't lead to overfitting.

   - **Use appropriate metrics** like **Precision, Recall, F1-Score**, and **ROC-AUC** to evaluate the classifier, especially when dealing with imbalanced datasets. Accuracy alone may be misleading.

### Key Considerations:

1. **SMOTE Only Applied to Training Data**:
   - SMOTE should be applied only on the **training** data, not on the test or validation set. This ensures that your test data remains representative of the original distribution.

2. **Combination with Undersampling**:
   - In highly imbalanced datasets, SMOTE can be combined with **undersampling** of the majority class (known as SMOTE-Tomek or SMOTE-ENN) to create a more balanced dataset.

3. **Potential for Overfitting**:
   - Since SMOTE generates synthetic data, it may sometimes introduce overfitting, especially if the synthetic samples are too similar to existing ones. This is why it’s important to monitor model performance and avoid generating too many synthetic samples.

4. **Use with Care on High-Dimensional Data**:
   - SMOTE can struggle with high-dimensional datasets, as it becomes harder to find meaningful nearest neighbors. Dimensionality reduction techniques (like PCA) can help in such cases.

### Summary:
- **SMOTE** is a powerful tool for handling imbalanced datasets by generating synthetic samples for the minority class.
- It helps improve classification performance by creating more balanced training data.
- Ensure proper model evaluation using metrics like Precision, Recall, F1-Score, and ROC-AUC, and apply SMOTE only to the training data.