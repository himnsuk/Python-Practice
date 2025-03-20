### **Measuring Goodness of Fit for Classification Problems**

For classification problems, the goodness of fit measures how well the predicted class labels align with the actual class labels. The most common measures include **accuracy, precision, recall, F1-score, and area under the ROC curve (AUC-ROC)**. Here's the mathematical basis for each:

---

#### **1. Confusion Matrix Elements:**
Given $( n )$ samples:
- $( TP )$: True Positives — correctly predicted positives.
- $( TN )$: True Negatives — correctly predicted negatives.
- $( FP )$: False Positives — negatives incorrectly predicted as positives.
- $( FN )$: False Negatives — positives incorrectly predicted as negatives.

#### **2. Accuracy:**
Proportion of correctly classified samples.
$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

#### **3. Precision:**
Proportion of correctly predicted positive samples out of all samples predicted as positive.
$$
\text{Precision} = \frac{TP}{TP + FP} = \frac{\text{Predected Right True}}{\text{Total Predicted True}}
$$

#### **4. Recall (Sensitivity or True Positive Rate):**
Proportion of actual positives that are correctly predicted.
$$
\text{Recall} = \frac{TP}{TP + FN} = \frac{\text{Predected Right True}}{\text{Total True}}
$$

#### **5. F1-Score:**
Harmonic mean of precision and recall.
$$
\text{F1-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

#### **6. Specificity:**
Proportion of actual negatives that are correctly predicted.
$$
\text{Specificity} = \frac{TN}{TN + FP}
$$

#### **7. ROC Curve and AUC:**

The **Receiver Operating Characteristic (ROC) Curve** is a graphical representation of the performance of a binary classifier as the discrimination threshold varies.

- **Axes of ROC Curve:**
  - $( x )$-axis: False Positive Rate (FPR).
    $$
    \text{FPR} = \frac{FP}{FP + TN}
    $$
  - $( y )$-axis: True Positive Rate (TPR), also known as Recall.
    $$
    \text{TPR} = \frac{TP}{TP + FN}
    $$

- **Steps to Plot the ROC Curve:**
  1. Compute the predicted probabilities from the classifier.
  2. Vary the decision threshold from 0 to 1.
  3. For each threshold, calculate $( TPR )$ and $( FPR )$.
  4. Plot $( TPR )$ vs. $( FPR )$.

- **Area Under the Curve (AUC):**
  The AUC quantifies the overall ability of the model to distinguish between positive and negative classes. A perfect model has an AUC of 1.0, while a random model has an AUC of 0.5.

$$
\text{AUC} = \int_0^1 TPR \, d(\text{FPR})
$$

---

### **Example of ROC Curve:**

#### **Scenario:**
Consider a classifier predicting whether a patient has a disease (Positive) or not (Negative). The model outputs probabilities for 6 patients: $( [0.9, 0.8, 0.6, 0.4, 0.2, 0.1] )$.

#### **Actual Labels:**
- Patients 1, 2, 3: Positive ($( 1 )$)
- Patients 4, 5, 6: Negative ($( 0 )$).

#### **Step 1: Threshold Evaluation**
For different thresholds ($( T )$), classify samples as Positive if $( P > T )$, then compute $( TP )$, $( FP )$, $( TN )$, and $( FN )$.

| Threshold $( T )$ | TP | FP | TN | FN | TPR ($( \frac{TP}{TP + FN} )$) | FPR ($( \frac{FP}{FP + TN} )$) |
|--------------------|----|----|----|----|-------------------------------|-------------------------------|
| 0.9                | 1  | 0  | 3  | 2  | $( 0.33 )$                   | $( 0.00 )$                   |
| 0.8                | 2  | 0  | 3  | 1  | $( 0.67 )$                   | $( 0.00 )$                   |
| 0.6                | 3  | 0  | 3  | 0  | $( 1.00 )$                   | $( 0.00 )$                   |
| 0.4                | 3  | 1  | 2  | 0  | $( 1.00 )$                   | $( 0.33 )$                   |
| 0.2                | 3  | 2  | 1  | 0  | $( 1.00 )$                   | $( 0.67 )$                   |
| 0.1                | 3  | 3  | 0  | 0  | $( 1.00 )$                   | $( 1.00 )$                   |

#### **Step 2: Plot the ROC Curve**
- Points: $( (0, 0), (0, 0.33), (0, 1.00), (0.33, 1.00), (0.67, 1.00), (1.00, 1.00) )$.
- Connect the points to form the curve.

#### **Step 3: Compute AUC**
Using the trapezoidal rule:
$$
\text{AUC} = (0.33 \times 0.33) + (0.33 \times 0.67) + (0.33 \times 1.00) + (0.67 \times 1.00) = 0.89.
$$

Thus, the classifier has an AUC of 0.89, indicating good discrimination.

---

Would you like help visualizing the ROC curve or implementing these calculations in Python?


